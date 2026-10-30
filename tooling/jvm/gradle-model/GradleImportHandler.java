package dev.made.groovy;

import java.net.URI;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.nio.file.Path;
import java.nio.file.LinkOption;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.eclipse.buildship.core.GradleCore;
import org.eclipse.core.resources.IProject;
import org.eclipse.core.resources.IProjectDescription;
import org.eclipse.core.resources.IWorkspace;
import org.eclipse.core.resources.IWorkspaceRunnable;
import org.eclipse.core.resources.ResourcesPlugin;
import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.core.runtime.jobs.ISchedulingRule;
import org.eclipse.core.runtime.jobs.Job;
import org.eclipse.core.runtime.OperationCanceledException;
import org.eclipse.jdt.ls.core.internal.IDelegateCommandHandler;
import org.eclipse.jdt.ls.core.internal.JavaLanguageServerPlugin;
import org.eclipse.jdt.ls.core.internal.ProjectUtils;
import org.eclipse.jdt.ls.core.internal.managers.BasicFileDetector;
import org.eclipse.jdt.ls.core.internal.managers.GradleProjectImporter;

public final class GradleImportHandler implements IDelegateCommandHandler {
    @Override
    public Object executeCommand(String command, List<Object> arguments, IProgressMonitor monitor) throws Exception {
        if ("groovy.gradle.moduleStates.v2".equals(command)) {
            return readModuleStatesWithinWorkspace(arguments, monitor);
        }
        if ("groovy.gradle.applyModuleStates.v2".equals(command)) {
            return applyModuleStatesWithinWorkspace(arguments, monitor);
        }
        if ("groovy.gradle.rootState.v2".equals(command)) {
            return readRootStateWithinWorkspace(arguments, monitor);
        }
        if ("groovy.gradle.applyRootState.v2".equals(command)) {
            return applyRootStateWithinWorkspace(arguments, monitor);
        }
        if ("groovy.gradle.defaultImportRoots".equals(command)) {
            return defaultImportRoots(arguments, monitor);
        }
        if ("groovy.gradle.applyRootState".equals(command)) {
            return applyRootState(arguments, monitor);
        }
        if ("groovy.gradle.rootState".equals(command)) {
            return readRootState(arguments, monitor);
        }
        if ("groovy.gradle.moduleStates".equals(command)) {
            return readModuleStates(arguments, monitor);
        }
        if ("groovy.gradle.applyModuleStates".equals(command)) {
            return applyModuleStates(arguments, monitor);
        }
        try {
            checkCancelled(monitor);
            if ("groovy.gradle.importSelected.v2".equals(command)) {
                return importWithinWorkspace(arguments, monitor);
            }
            if (!"groovy.gradle.importSelected".equals(command) || arguments == null || arguments.size() != 1
                    || !(arguments.get(0) instanceof Map<?, ?> request)
                    || !request.keySet().equals(Set.of("rootUri", "offline"))
                    || !(request.get("rootUri") instanceof String rootUri) || rootUri.length() > 4096
                    || !Boolean.TRUE.equals(request.get("offline"))) {
                throw new IllegalArgumentException();
            }
            Path root = GradleModelHandler.initializedRoot(rootUri, "Gradle");
            String canonicalRootUri = root.toUri().toString();
            if (canonicalRootUri.length() > 4096) {
                throw new IllegalArgumentException();
            }
            performSelectedImport(root, () -> {
                if (!GradleModelHandler.initializedRoot(rootUri, "Gradle").equals(root)) {
                    throw new OperationCanceledException();
                }
            }, monitor);
            Map<String, Object> receipt = new LinkedHashMap<>();
            receipt.put("version", 1);
            receipt.put("rootUri", canonicalRootUri);
            receipt.put("selectedImportPhaseComplete", true);
            receipt.put("fullSynchronizationKnown", false);
            return receipt;
        } catch (LinkageError unavailable) {
            throw new IllegalStateException("Selected Gradle import unavailable. Check the pinned provider SDK and initialized build selection, then refresh.");
        } catch (OperationCanceledException cancelled) {
            throw new IllegalStateException("Selected Gradle import cancelled and may be partial. Refresh before retrying.");
        } catch (Exception failure) {
            if (monitor.isCanceled()) {
                throw new IllegalStateException("Selected Gradle import cancelled and may be partial. Refresh before retrying.");
            }
            throw new IllegalStateException("Selected Gradle import failed or may be partial. Check initialized build selection, host offline settings and cached Gradle installation; fix the build and refresh before retrying.");
        }
    }

    @FunctionalInterface
    private interface ImportAuthority {
        void validate() throws Exception;
    }

    private static void performSelectedImport(Path root, ImportAuthority authority, IProgressMonitor monitor)
            throws Exception {
        validateBuildInputs(root);
        List<Object> configuration = offlineConfiguration();
        checkCancelled(monitor);
        authority.validate();
        var build = GradleCore.getWorkspace().createBuild(GradleProjectImporter.getBuildConfiguration(root));
        var result = build.synchronize(monitor);
        checkCancelled(monitor);
        if (result == null || result.getStatus() == null || !result.getStatus().isOK()) {
            if (result != null && result.getStatus() != null
                    && result.getStatus().matches(org.eclipse.core.runtime.IStatus.CANCEL)) {
                throw new OperationCanceledException();
            }
            throw new IllegalStateException();
        }
        authority.validate();
        if (!offlineConfiguration().equals(configuration)) {
            throw new OperationCanceledException();
        }
        refreshCollidingClosedArtifacts(build, root, monitor);
        authority.validate();
        if (!offlineConfiguration().equals(configuration)) {
            throw new OperationCanceledException();
        }
    }

    private record BuildInputIdentity(Object fileKey, java.nio.file.attribute.FileTime modified,
            long size, String sha256) {}

    private static Map<String, BuildInputIdentity> selectedBuildInputs(Path root) throws Exception {
        validateBuildInputs(root);
        Map<String, BuildInputIdentity> inputs = new LinkedHashMap<>();
        for (String name : BUILD_DESCRIPTORS) {
            Path path = root.resolve(name);
            if (!Files.exists(path, LinkOption.NOFOLLOW_LINKS)) {
                continue;
            }
            var attributes = Files.readAttributes(path, java.nio.file.attribute.BasicFileAttributes.class,
                    LinkOption.NOFOLLOW_LINKS);
            if (!attributes.isRegularFile() || attributes.fileKey() == null || !path.equals(path.toRealPath())
                    || attributes.size() > 4 * 1024 * 1024) {
                throw new IllegalArgumentException();
            }
            byte[] bytes;
            try (var stream = Files.newInputStream(path, LinkOption.NOFOLLOW_LINKS)) {
                bytes = stream.readNBytes(4 * 1024 * 1024 + 1);
            }
            if (bytes.length > 4 * 1024 * 1024 || bytes.length != attributes.size()) {
                throw new IllegalArgumentException();
            }
            var current = Files.readAttributes(path, java.nio.file.attribute.BasicFileAttributes.class,
                    LinkOption.NOFOLLOW_LINKS);
            if (!attributes.fileKey().equals(current.fileKey())
                    || !attributes.lastModifiedTime().equals(current.lastModifiedTime())
                    || attributes.size() != current.size()) {
                throw new OperationCanceledException();
            }
            inputs.put(name, new BuildInputIdentity(attributes.fileKey(), attributes.lastModifiedTime(),
                    attributes.size(), java.util.HexFormat.of().formatHex(
                            java.security.MessageDigest.getInstance("SHA-256").digest(bytes))));
        }
        return inputs;
    }

    private static Object importWithinWorkspace(List<Object> arguments, IProgressMonitor monitor) throws Exception {
        if (arguments == null || arguments.size() != 1 || !(arguments.get(0) instanceof Map<?, ?> request)
                || !request.keySet().equals(Set.of("workspaceRootUri", "buildRootUri", "offline"))
                || !(request.get("workspaceRootUri") instanceof String workspaceUri) || !boundedText(workspaceUri)
                || !(request.get("buildRootUri") instanceof String buildUri) || !boundedText(buildUri)
                || !Boolean.TRUE.equals(request.get("offline"))) {
            throw new IllegalArgumentException();
        }
        ScopedBuild scope = scopedBuild(workspaceUri, buildUri, monitor);
        Map<Path, String> modules = freshModuleNames(scope.build(), monitor);
        validateScopedImportOwnership(scope.build(), modules, monitor);
        scope.validate(monitor);
        performSelectedImport(scope.build(), () -> scope.validate(monitor), monitor);
        return Map.of("version", 2, "workspaceRootUri", scope.workspace().toUri().toString(),
                "buildRootUri", scope.build().toUri().toString(), "selectedImportPhaseComplete", true,
                "fullSynchronizationKnown", false);
    }

    record ScopedBuild(Path workspace, Path build, String workspaceUri, String buildUri,
            Object workspaceIdentity, Object buildIdentity, Map<String, BuildInputIdentity> inputs,
            List<Object> configuration) {
        void validate(IProgressMonitor monitor) throws Exception {
            checkCancelled(monitor);
            if (!GradleModelHandler.initializedRoot(workspaceUri, "Gradle").equals(workspace)
                    || !canonicalDirectory(workspaceUri, workspace).equals(workspace)
                    || !canonicalDirectory(buildUri, workspace).equals(build)) {
                throw new OperationCanceledException();
            }
            requireRootIdentity(workspace, workspaceIdentity);
            requireRootIdentity(build, buildIdentity);
            if (!selectedBuildInputs(build).equals(inputs) || !offlineConfiguration().equals(configuration)) {
                throw new OperationCanceledException();
            }
        }
    }

    static ScopedBuild scopedBuild(String workspaceUri, String buildUri, IProgressMonitor monitor)
            throws Exception {
        checkCancelled(monitor);
        Path workspace = GradleModelHandler.initializedRoot(workspaceUri, "Gradle");
        canonicalDirectory(workspaceUri, workspace);
        Path build = canonicalDirectory(buildUri, workspace);
        if (!boundedText(workspace.toUri().toString()) || !boundedText(build.toUri().toString())) {
            throw new IllegalArgumentException();
        }
        return new ScopedBuild(workspace, build, workspaceUri, buildUri, rootFileKey(workspace),
                rootFileKey(build), selectedBuildInputs(build), offlineConfiguration());
    }

    record ProjectStateContext(Path root, List<Object> configuration,
            ImportAuthority authority, Object rootIdentity) {
        void validate() {
            try {
                authority.validate();
            } catch (RuntimeException failure) {
                throw failure;
            } catch (Exception failure) {
                throw new OperationCanceledException();
            }
        }
    }

    private static ProjectStateContext projectStateContext(Path root, String rootUri,
            List<Object> configuration, Object rootIdentity) {
        return new ProjectStateContext(root, configuration, () -> {
            if (!GradleModelHandler.initializedRoot(rootUri, "Gradle").equals(root)
                    || !offlineConfiguration().equals(configuration)) {
                throw new OperationCanceledException();
            }
        }, rootIdentity);
    }

    private static ScopedBuild rootStateScope(List<Object> arguments, boolean mutation,
            IProgressMonitor monitor) throws Exception {
        if (arguments == null || arguments.size() != 1 || !(arguments.get(0) instanceof Map<?, ?> request)
                || !request.keySet().equals(mutation
                        ? Set.of("workspaceRootUri", "buildRootUri", "offline", "open")
                        : Set.of("workspaceRootUri", "buildRootUri", "offline"))
                || !(request.get("workspaceRootUri") instanceof String workspaceUri) || !boundedText(workspaceUri)
                || !(request.get("buildRootUri") instanceof String buildUri) || !boundedText(buildUri)
                || !Boolean.TRUE.equals(request.get("offline"))
                || mutation && !(request.get("open") instanceof Boolean)) {
            throw new IllegalArgumentException();
        }
        return scopedBuild(workspaceUri, buildUri, monitor);
    }

    private static Object readRootStateWithinWorkspace(List<Object> arguments, IProgressMonitor monitor) {
        try {
            ScopedBuild scope = rootStateScope(arguments, false, monitor);
            Map<Path, String> modules = freshModuleNames(scope.build(), monitor);
            scope.validate(monitor);
            if (!modules.containsKey(scope.build())) {
                throw new IllegalArgumentException();
            }
            ProjectStateContext context = new ProjectStateContext(scope.build(), scope.configuration(), () -> scope.validate(monitor), scope.buildIdentity());
            Object result = readProjectStates(context, Map.of(scope.build(), scope.build()), true, monitor);
            scope.validate(monitor);
            if (!(result instanceof Map<?, ?> response) || !(response.get("modules") instanceof List<?> records)
                    || records.size() != 1 || !(records.get(0) instanceof Map<?, ?> root)
                    || !scope.build().toUri().toString().equals(root.get("moduleUri"))) {
                throw new IllegalArgumentException();
            }
            return Map.of("version", 2, "workspaceRootUri", scope.workspace().toUri().toString(),
                    "buildRootUri", scope.build().toUri().toString(), "workspaceState", root.get("workspaceState"),
                    "descriptionAvailable", root.get("descriptionAvailable"), "fullSynchronizationKnown", false);
        } catch (LinkageError unavailable) {
            throw new IllegalStateException("Scoped Gradle root state unavailable. Check pinned provider SDK setup and refresh.");
        } catch (Exception failure) {
            if (monitor.isCanceled() || failure instanceof OperationCanceledException) {
                throw new IllegalStateException("Scoped Gradle root state read cancelled. Refresh before retrying.");
            }
            throw new IllegalStateException("Scoped Gradle root state unavailable. Check initialized workspace authority, current build inputs, exclusive description and cached host settings; refresh before retrying.");
        }
    }

    private static Object applyRootStateWithinWorkspace(List<Object> arguments, IProgressMonitor monitor) {
        try {
            ScopedBuild scope = rootStateScope(arguments, true, monitor);
            Map<?, ?> request = (Map<?, ?>) arguments.get(0);
            boolean open = (Boolean) request.get("open");
            Map<Path, String> modules = freshModuleNames(scope.build(), monitor);
            scope.validate(monitor);
            if (!modules.containsKey(scope.build())) {
                throw new IllegalArgumentException();
            }
            ProjectStateContext context = new ProjectStateContext(scope.build(), scope.configuration(), () -> scope.validate(monitor), scope.buildIdentity());
            applyProjectStates(context, Map.of(scope.build(), open), modules.keySet(), true, monitor);
            scope.validate(monitor);
            return Map.of("version", 2, "workspaceRootUri", scope.workspace().toUri().toString(),
                    "buildRootUri", scope.build().toUri().toString(), "open", open,
                    "rootStatePhaseComplete", true, "fullSynchronizationKnown", false);
        } catch (LinkageError unavailable) {
            throw new IllegalStateException("Scoped Gradle root state change unavailable. Check pinned provider SDK setup and refresh.");
        } catch (Exception failure) {
            if (monitor.isCanceled() || failure instanceof OperationCanceledException) {
                throw new IllegalStateException("Scoped Gradle root state change cancelled and may be partial. Refresh before retrying.");
            }
            throw new IllegalStateException("Scoped Gradle root state change failed or may be partial. Check initialized workspace authority, retained exclusive description, closed child projects, current inputs and cached host settings; refresh before retrying.");
        }
    }

    private static ScopedBuild moduleStateScope(List<Object> arguments, boolean mutation,
            IProgressMonitor monitor) throws Exception {
        if (arguments == null || arguments.size() != 1 || !(arguments.get(0) instanceof Map<?, ?> request)
                || !request.keySet().equals(mutation
                        ? Set.of("workspaceRootUri", "buildRootUri", "offline", "closedProjectUris", "restoredProjectUris")
                        : Set.of("workspaceRootUri", "buildRootUri", "offline"))
                || !(request.get("workspaceRootUri") instanceof String workspaceUri) || !boundedText(workspaceUri)
                || !(request.get("buildRootUri") instanceof String buildUri) || !boundedText(buildUri)
                || !Boolean.TRUE.equals(request.get("offline"))) {
            throw new IllegalArgumentException();
        }
        return scopedBuild(workspaceUri, buildUri, monitor);
    }

    private static ModelProjects scopedModuleProjects(ScopedBuild scope, IProgressMonitor monitor) throws Exception {
        ModelProjects projects = freshModuleProjects(scope.build(), monitor);
        scope.validate(monitor);
        if (!projects.names().containsKey(scope.build())) {
            throw new IllegalArgumentException();
        }
        for (var entry : projects.buildRoots().entrySet()) {
            if (!canonicalDirectory(entry.getKey().toUri().toString(), scope.build()).equals(entry.getKey())
                    || !canonicalDirectory(entry.getValue().toUri().toString(), scope.build()).equals(entry.getValue())
                    || !entry.getKey().startsWith(entry.getValue())) {
                throw new IllegalArgumentException();
            }
        }
        return projects;
    }

    private record ModuleInputIdentity(Object directoryIdentity, Map<String, BuildInputIdentity> inputs) {}

    private static Map<String, BuildInputIdentity> optionalModuleInputs(Path module) throws Exception {
        for (String name : BUILD_DESCRIPTORS) {
            if (Files.exists(module.resolve(name), LinkOption.NOFOLLOW_LINKS)) {
                return selectedBuildInputs(module);
            }
        }
        return Map.of();
    }

    private static void captureModuleInputs(Map<Path, ModuleInputIdentity> identities, Set<Path> modules,
            IProgressMonitor monitor) throws Exception {
        long inputBytes = 0;
        for (ModuleInputIdentity identity : identities.values()) {
            for (BuildInputIdentity input : identity.inputs().values()) {
                inputBytes += input.size();
            }
        }
        for (Path module : modules) {
            checkCancelled(monitor);
            if (!identities.containsKey(module)) {
                if (identities.size() >= 128) {
                    throw new IllegalArgumentException();
                }
                Object directoryIdentity = rootFileKey(module);
                Map<String, BuildInputIdentity> inputs = optionalModuleInputs(module);
                for (BuildInputIdentity input : inputs.values()) {
                    inputBytes += input.size();
                    if (inputBytes > 16 * 1024 * 1024) {
                        throw new IllegalArgumentException();
                    }
                }
                requireRootIdentity(module, directoryIdentity);
                identities.put(module, new ModuleInputIdentity(directoryIdentity, inputs));
            }
        }
    }

    private static Map<Path, ModuleInputIdentity> registeredModuleInputs(ScopedBuild scope,
            IProgressMonitor monitor) throws Exception {
        IProject[] resources = ResourcesPlugin.getWorkspace().getRoot().getProjects();
        if (resources.length > 1024) {
            throw new IllegalArgumentException();
        }
        Set<Path> modules = new LinkedHashSet<>();
        for (IProject project : resources) {
            checkCancelled(monitor);
            if (!project.exists() || project.getLocation() == null) {
                continue;
            }
            Path location = project.getLocation().toFile().toPath().toAbsolutePath().normalize();
            if (location.startsWith(scope.build()) && !location.equals(scope.build())) {
                modules.add(canonicalDirectory(location.toUri().toString(), scope.build()));
            }
        }
        Map<Path, ModuleInputIdentity> identities = new LinkedHashMap<>();
        captureModuleInputs(identities, modules, monitor);
        return identities;
    }

    private static void validateModuleInputs(Map<Path, ModuleInputIdentity> identities,
            IProgressMonitor monitor) throws Exception {
        for (var entry : identities.entrySet()) {
            checkCancelled(monitor);
            requireRootIdentity(entry.getKey(), entry.getValue().directoryIdentity());
            if (!optionalModuleInputs(entry.getKey()).equals(entry.getValue().inputs())) {
                throw new OperationCanceledException();
            }
        }
    }

    static ProjectStateContext scopedMetadataContext(ScopedBuild scope, IProgressMonitor monitor)
            throws Exception {
        Map<Path, ModuleInputIdentity> identities = registeredModuleInputs(scope, monitor);
        return new ProjectStateContext(scope.build(), scope.configuration(), () -> {
            scope.validate(monitor);
            validateModuleInputs(identities, monitor);
        }, scope.buildIdentity());
    }

    private static ProjectStateContext scopedModuleContext(ScopedBuild scope, ModelProjects projects,
            Map<Path, ModuleInputIdentity> identities, IProgressMonitor monitor) throws Exception {
        validateModuleInputs(identities, monitor);
        Set<Path> modules = new LinkedHashSet<>(projects.names().keySet());
        modules.remove(scope.build());
        captureModuleInputs(identities, modules, monitor);
        return new ProjectStateContext(scope.build(), scope.configuration(), () -> {
            scope.validate(monitor);
            validateModuleInputs(identities, monitor);
        }, scope.buildIdentity());
    }

    private static Object readModuleStatesWithinWorkspace(List<Object> arguments, IProgressMonitor monitor) {
        try {
            ScopedBuild scope = moduleStateScope(arguments, false, monitor);
            Map<Path, ModuleInputIdentity> identities = registeredModuleInputs(scope, monitor);
            ModelProjects projects = scopedModuleProjects(scope, monitor);
            ProjectStateContext context = scopedModuleContext(scope, projects, identities, monitor);
            Object result = readProjectStates(context, projects.buildRoots(), false, monitor);
            context.validate();
            if (!(result instanceof Map<?, ?> response) || !(response.get("modules") instanceof List<?> records)) {
                throw new IllegalArgumentException();
            }
            return Map.of("version", 2, "workspaceRootUri", scope.workspace().toUri().toString(),
                    "buildRootUri", scope.build().toUri().toString(), "modules", records,
                    "fullSynchronizationKnown", false);
        } catch (LinkageError unavailable) {
            throw new IllegalStateException("Scoped Gradle module states unavailable. Check pinned provider SDK setup and refresh.");
        } catch (Exception failure) {
            if (monitor.isCanceled() || failure instanceof OperationCanceledException) {
                throw new IllegalStateException("Scoped Gradle module state read cancelled. Refresh before retrying.");
            }
            throw new IllegalStateException("Scoped Gradle module states unavailable. Check initialized workspace authority, current modules, exclusive descriptions and cached host settings; refresh before retrying.");
        }
    }

    private static Object applyModuleStatesWithinWorkspace(List<Object> arguments, IProgressMonitor monitor) {
        try {
            ScopedBuild scope = moduleStateScope(arguments, true, monitor);
            Map<?, ?> request = (Map<?, ?>) arguments.get(0);
            if (!(request.get("closedProjectUris") instanceof List<?> closed)
                    || !(request.get("restoredProjectUris") instanceof List<?> restored)
                    || closed.size() + restored.size() < 1 || closed.size() + restored.size() > 128) {
                throw new IllegalArgumentException();
            }
            Map<Path, Boolean> requested = new LinkedHashMap<>();
            for (List<?> values : List.of(closed, restored)) {
                for (Object value : values) {
                    checkCancelled(monitor);
                    if (!(value instanceof String uri) || !boundedText(uri)) {
                        throw new IllegalArgumentException();
                    }
                    Path module = canonicalDirectory(uri, scope.build());
                    if (module.equals(scope.build()) || requested.putIfAbsent(module, values == restored) != null) {
                        throw new IllegalArgumentException();
                    }
                }
            }
            Map<Path, ModuleInputIdentity> identities = registeredModuleInputs(scope, monitor);
            captureModuleInputs(identities, requested.keySet(), monitor);
            ModelProjects projects = scopedModuleProjects(scope, monitor);
            ProjectStateContext context = scopedModuleContext(scope, projects, identities, monitor);
            Object result = applyProjectStates(context, requested, projects.names().keySet(), false, monitor);
            context.validate();
            if (!(result instanceof Map<?, ?> response)) {
                throw new IllegalArgumentException();
            }
            return Map.of("version", 2, "workspaceRootUri", scope.workspace().toUri().toString(),
                    "buildRootUri", scope.build().toUri().toString(), "closedProjectUris", response.get("closedProjectUris"),
                    "restoredProjectUris", response.get("restoredProjectUris"), "moduleStatePhaseComplete", true,
                    "fullSynchronizationKnown", false);
        } catch (LinkageError unavailable) {
            throw new IllegalStateException("Scoped Gradle module state change unavailable. Check pinned provider SDK setup and refresh.");
        } catch (Exception failure) {
            if (monitor.isCanceled() || failure instanceof OperationCanceledException) {
                throw new IllegalStateException("Scoped Gradle module state change cancelled and may be partial. Refresh before retrying.");
            }
            throw new IllegalStateException("Scoped Gradle module state change failed or may be partial. Check initialized workspace authority, current non-root modules, exclusive descriptions and cached host settings; refresh before retrying.");
        }
    }

    private static void validateScopedImportOwnership(Path root, Map<Path, String> modules,
            IProgressMonitor monitor) throws Exception {
        IWorkspace workspace = ResourcesPlugin.getWorkspace();
        IProject[] resources = workspace.getRoot().getProjects();
        if (resources.length > 1024 || modules.isEmpty() || modules.size() > 128) {
            throw new IllegalArgumentException();
        }
        for (var entry : modules.entrySet()) {
            checkCancelled(monitor);
            Path module = entry.getKey();
            Path descriptor = module.resolve(".project");
            if (Files.exists(descriptor, LinkOption.NOFOLLOW_LINKS)) {
                byte[] bytes = descriptionBytes(descriptor);
                var description = workspace.loadProjectDescription(new java.io.ByteArrayInputStream(bytes));
                Set<String> natures = new LinkedHashSet<>(Arrays.asList(description.getNatureIds()));
                if (!description.getName().equals(entry.getValue())
                        || !natures.contains("org.eclipse.buildship.core.gradleprojectnature")
                        || natures.contains("org.eclipse.m2e.core.maven2Nature")) {
                    throw new IllegalArgumentException();
                }
            }
            for (IProject project : resources) {
                if (project.exists() && project.getLocation() != null) {
                    Path location = project.getLocation().toFile().toPath().toAbsolutePath().normalize();
                    if (Files.exists(location) && location.toRealPath().startsWith(root)
                            && !location.equals(location.toRealPath())) {
                        throw new IllegalArgumentException();
                    }
                    if (project.getName().equals(entry.getValue()) && !location.equals(module)
                            || location.equals(module) && project.isOpen()
                                    && !ProjectUnlinkHandler.matches(project, module, "Gradle")) {
                        throw new IllegalArgumentException();
                    }
                    if (location.startsWith(root) && !modules.containsKey(location)) {
                        throw new IllegalArgumentException();
                    }
                }
            }
        }
    }

    private record ModuleState(Path path, byte[] descriptionBytes, IProjectDescription description,
            IProject project, boolean open) {}

    private static Object applyRootState(List<Object> arguments, IProgressMonitor monitor) {
        try {
            checkCancelled(monitor);
            if (arguments == null || arguments.size() != 1 || !(arguments.get(0) instanceof Map<?, ?> request)
                    || !request.keySet().equals(Set.of("rootUri", "offline", "open"))
                    || !(request.get("rootUri") instanceof String rootUri) || !boundedText(rootUri)
                    || !Boolean.TRUE.equals(request.get("offline")) || !(request.get("open") instanceof Boolean open)) {
                throw new IllegalArgumentException();
            }
            Path root = GradleModelHandler.initializedRoot(rootUri, "Gradle");
            if (!canonicalDirectory(rootUri, root).equals(root)) {
                throw new IllegalArgumentException();
            }
            Object identity = rootFileKey(root);
            List<Object> configuration = offlineConfiguration();
            applyProjectStates(List.of(Map.of("rootUri", rootUri, "offline", true,
                    "closedProjectUris", open ? List.of() : List.of(rootUri),
                    "restoredProjectUris", open ? List.of(rootUri) : List.of())), monitor, true, identity);
            checkCancelled(monitor);
            requireRootIdentity(root, identity);
            if (!GradleModelHandler.initializedRoot(rootUri, "Gradle").equals(root)
                    || !offlineConfiguration().equals(configuration)) {
                throw new OperationCanceledException();
            }
            return Map.of("version", 1, "rootUri", root.toUri().toString(), "open", open,
                    "rootStatePhaseComplete", true, "fullSynchronizationKnown", false);
        } catch (LinkageError unavailable) {
            throw new IllegalStateException("Gradle root state change unavailable. Check pinned provider SDK setup and refresh.");
        } catch (Exception failure) {
            if (monitor.isCanceled() || failure instanceof OperationCanceledException) {
                throw new IllegalStateException("Gradle root state change cancelled and may be partial. Refresh before retrying.");
            }
            throw new IllegalStateException("Gradle root state change failed or may be partial. Check retained exclusive Gradle description, closed child projects, host file identity support and cached host settings; refresh before retrying.");
        }
    }

    private static final List<String> BUILD_DESCRIPTORS = List.of(
            "build.gradle", "settings.gradle", "build.gradle.kts", "settings.gradle.kts");

    private static Object defaultImportRoots(List<Object> arguments, IProgressMonitor monitor) {
        try {
            checkCancelled(monitor);
            if (arguments == null || arguments.size() != 1 || !(arguments.get(0) instanceof Map<?, ?> request)
                    || !request.keySet().equals(Set.of("rootUri", "candidateBuildFileUris", "cachedOnly"))
                    || !(request.get("rootUri") instanceof String rootUri) || !boundedText(rootUri)
                    || !Boolean.TRUE.equals(request.get("cachedOnly"))
                    || !(request.get("candidateBuildFileUris") instanceof List<?> candidates)
                    || candidates.size() > 128) {
                throw new IllegalArgumentException();
            }
            Path root = GradleModelHandler.initializedRoot(rootUri, "Gradle");
            canonicalDirectory(rootUri, root);
            if (!boundedText(root.toUri().toString())) {
                throw new IllegalArgumentException();
            }
            Set<Path> files = new LinkedHashSet<>();
            Set<Path> directories = new LinkedHashSet<>();
            for (Object value : candidates) {
                checkCancelled(monitor);
                if (!(value instanceof String text) || !boundedText(text)) {
                    throw new IllegalArgumentException();
                }
                URI uri = new URI(text);
                if (!"file".equals(uri.getScheme()) || uri.getQuery() != null || uri.getFragment() != null) {
                    throw new IllegalArgumentException();
                }
                Path path = Path.of(uri);
                if (!path.isAbsolute() || !path.equals(path.normalize()) || !path.startsWith(root)
                        || !path.equals(path.toRealPath()) || !boundedText(path.toString())
                        || !Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS) || !Files.isReadable(path)
                        || path.getFileName() == null || !BUILD_DESCRIPTORS.contains(path.getFileName().toString())
                        || !files.add(path)) {
                    throw new IllegalArgumentException();
                }
                if (!boundedText(path.getParent().toUri().toString())) {
                    throw new IllegalArgumentException();
                }
                directories.add(path.getParent());
            }
            var configured = JavaLanguageServerPlugin.getPreferencesManager().getPreferences().getJavaImportExclusions();
            List<String> exclusions = configured == null ? List.of() : List.copyOf(configured);
            if (exclusions.size() > 128) {
                throw new IllegalArgumentException();
            }
            int bytes = 0;
            for (String pattern : exclusions) {
                if (!boundedText(pattern)) {
                    throw new IllegalArgumentException();
                }
                bytes += pattern.getBytes(StandardCharsets.UTF_8).length;
            }
            if (bytes > 64 * 1024) {
                throw new IllegalArgumentException();
            }
            // The automatic importer is disabled by saved policy; evaluate its
            // detector against visible candidates without scanning or toggling it.
            BasicFileDetector detector = new BasicFileDetector(root, BUILD_DESCRIPTORS.toArray(String[]::new))
                    .includeNested(false).addExclusions("**/build", "**/bin");
            Method predicate = GradleModelHandler.detectorPredicate();
            int maximumDepth = GradleModelHandler.detectorDepth(detector);
            boolean hasInclusions = exclusions.stream().anyMatch(pattern -> pattern.startsWith("!"));
            Set<Path> roots = new LinkedHashSet<>();
            List<ModuleState> retained = new ArrayList<>();
            IWorkspace workspace = ResourcesPlugin.getWorkspace();
            ISchedulingRule current = Job.getJobManager().currentRule();
            ISchedulingRule rule = current != null && current.contains(workspace.getRoot()) ? null : workspace.getRoot();
            workspace.run((IWorkspaceRunnable) progress -> {
                IProject[] projects = ProjectUtils.getAllProjects();
                if (projects.length > 1024) {
                    throw new IllegalArgumentException();
                }
                long retainedBytes = 0;
                for (IProject project : projects) {
                    checkCancelled(progress);
                    if (project.getLocation() == null) {
                        throw new IllegalArgumentException();
                    }
                    boolean gradle = ProjectUtils.isGradleProject(project)
                            || ProjectUtils.hasNature(project, GradleProjectImporter.GRADLE_BUILD_SERVER_NATURE);
                    // Closed resources do not expose natures through ProjectUtils;
                    // excluding them here would prevent default-policy restoration.
                    Path location = project.getLocation().toFile().toPath().toAbsolutePath().normalize();
                    if (!gradle && project.exists() && !project.isOpen() && location.startsWith(root)) {
                        try {
                            if (!location.equals(location.toRealPath())) {
                                throw new IllegalArgumentException();
                            }
                            byte[] descriptionData = descriptionBytes(location.resolve(".project"));
                            retainedBytes += descriptionData.length;
                            if (retainedBytes > 8 * 1024 * 1024 || retained.size() >= 128) {
                                throw new IllegalArgumentException();
                            }
                            var description = workspace.loadProjectDescription(new java.io.ByteArrayInputStream(descriptionData));
                            Set<String> natures = new LinkedHashSet<>(Arrays.asList(description.getNatureIds()));
                            gradle = natures.contains("org.eclipse.buildship.core.gradleprojectnature")
                                    || natures.contains(GradleProjectImporter.GRADLE_BUILD_SERVER_NATURE);
                            if (gradle) {
                                if (!project.getName().equals(description.getName())
                                        || natures.contains("org.eclipse.m2e.core.maven2Nature")) {
                                    throw new IllegalArgumentException();
                                }
                                var state = new ModuleState(location, descriptionData, description, project, false);
                                validateState(state, root, progress);
                                retained.add(state);
                            }
                        } catch (java.io.IOException failure) {
                            throw new IllegalArgumentException();
                        }
                    }
                    if (!gradle) {
                        String exclusion = project.getLocation().toOSString().replace("\\", "\\\\");
                        if (!boundedText(exclusion)) {
                            throw new IllegalArgumentException();
                        }
                        detector.addExclusions(exclusion);
                    }
                }
                try {
                    for (Path directory : directories) {
                        checkCancelled(progress);
                        if ((!directory.equals(root) && root.relativize(directory).getNameCount() > maximumDepth)
                                || GradleModelHandler.detectorExcludes(detector, predicate, directory)) {
                            continue;
                        }
                        boolean shadowed = false;
                        for (Path ancestor = directory.getParent(); ancestor != null && ancestor.startsWith(root); ancestor = ancestor.getParent()) {
                            checkCancelled(progress);
                            boolean excluded = GradleModelHandler.detectorExcludes(detector, predicate, ancestor);
                            if (excluded && !hasInclusions) {
                                shadowed = true;
                                break;
                            }
                            if (!excluded) {
                                for (String name : BUILD_DESCRIPTORS) {
                                    Path descriptor = ancestor.resolve(name);
                                    if (Files.isSymbolicLink(descriptor)) {
                                        throw new IllegalArgumentException();
                                    }
                                    if (Files.isRegularFile(descriptor, LinkOption.NOFOLLOW_LINKS)) {
                                        if (!files.contains(descriptor)) {
                                            throw new IllegalArgumentException();
                                        }
                                        shadowed = true;
                                    }
                                }
                                if (shadowed) {
                                    break;
                                }
                            }
                        }
                        if (!shadowed) {
                            roots.add(directory);
                        }
                    }
                    for (ModuleState state : retained) {
                        validateState(state, root, progress);
                        if (!state.project().exists() || state.project().isOpen()) {
                            throw new OperationCanceledException();
                        }
                    }
                } catch (ReflectiveOperationException failure) {
                    throw new IllegalStateException();
                }
            }, rule, IWorkspace.AVOID_UPDATE, monitor);
            checkCancelled(monitor);
            for (Path file : files) {
                checkCancelled(monitor);
                if (!file.equals(file.toRealPath()) || !Files.isRegularFile(file, LinkOption.NOFOLLOW_LINKS)
                        || !Files.isReadable(file)) {
                    throw new IllegalArgumentException();
                }
            }
            var currentExclusions = JavaLanguageServerPlugin.getPreferencesManager().getPreferences().getJavaImportExclusions();
            if (!exclusions.equals(currentExclusions == null ? List.of() : currentExclusions)
                    || !GradleModelHandler.initializedRoot(rootUri, "Gradle").equals(root)) {
                throw new OperationCanceledException();
            }
            return Map.of("version", 1, "rootUri", root.toUri().toString(),
                    "buildRootUris", roots.stream().map(path -> path.toUri().toString()).toList(),
                    "cachedOnly", true, "readOnly", true, "partial", false, "fullSynchronizationKnown", false);
        } catch (OperationCanceledException cancelled) {
            throw new IllegalStateException("Gradle default-root discovery cancelled; no projects were imported. Refresh before retrying.");
        } catch (Exception | LinkageError failure) {
            throw new IllegalStateException("Gradle default-root discovery failed; no projects were imported. Check current visible build files, initialized host and compatible extension/JDTLS settings, then refresh and retry.");
        }
    }

    static Object rootFileKey(Path root) {
        try {
            // SDK resource paths do not distinguish a directory replaced at the same path.
            var attributes = Files.readAttributes(root, java.nio.file.attribute.BasicFileAttributes.class,
                    LinkOption.NOFOLLOW_LINKS);
            Object identity = attributes.fileKey();
            if (!attributes.isDirectory() || identity == null) {
                throw new IllegalArgumentException();
            }
            return identity;
        } catch (java.io.IOException failure) {
            throw new OperationCanceledException();
        }
    }

    static void requireRootIdentity(Path root, Object identity) {
        if (!identity.equals(rootFileKey(root))) {
            throw new OperationCanceledException();
        }
    }

    private static void requireClosedGradleChildren(Path root, IProgressMonitor monitor)
            throws org.eclipse.core.runtime.CoreException {
        IProject[] projects = ResourcesPlugin.getWorkspace().getRoot().getProjects();
        if (projects.length > 1024) {
            throw new IllegalArgumentException();
        }
        try {
            for (IProject project : projects) {
                checkCancelled(monitor);
                if (!project.exists() || !project.isOpen() || project.getLocation() == null) {
                    continue;
                }
                Path location = project.getLocation().toFile().toPath().toAbsolutePath().normalize();
                Path real = Files.exists(location) ? location.toRealPath() : location;
                if (real.startsWith(root) && !real.equals(root)
                        && Arrays.asList(project.getDescription().getNatureIds())
                                .contains("org.eclipse.buildship.core.gradleprojectnature")) {
                    throw new IllegalArgumentException();
                }
            }
        } catch (java.io.IOException failure) {
            throw new IllegalArgumentException();
        }
    }

    private static Object applyModuleStates(List<Object> arguments, IProgressMonitor monitor) {
        return applyProjectStates(arguments, monitor, false, null);
    }

    private static Object applyProjectStates(List<Object> arguments, IProgressMonitor monitor,
            boolean rootOnly, Object rootIdentity) {
        try {
            checkCancelled(monitor);
            if (arguments == null || arguments.size() != 1 || !(arguments.get(0) instanceof Map<?, ?> request)
                    || !request.keySet().equals(Set.of("rootUri", "offline", "closedProjectUris", "restoredProjectUris"))
                    || !(request.get("rootUri") instanceof String rootUri) || !boundedText(rootUri)
                    || !Boolean.TRUE.equals(request.get("offline"))
                    || !(request.get("closedProjectUris") instanceof List<?> closed)
                    || !(request.get("restoredProjectUris") instanceof List<?> restored)
                    || closed.size() + restored.size() < 1 || closed.size() + restored.size() > 128) {
                throw new IllegalArgumentException();
            }
            Path root = GradleModelHandler.initializedRoot(rootUri, "Gradle");
            if (!canonicalDirectory(rootUri, root).equals(root)) {
                throw new IllegalArgumentException();
            }
            Map<Path, Boolean> requested = new LinkedHashMap<>();
            for (List<?> values : List.of(closed, restored)) {
                for (Object value : values) {
                    checkCancelled(monitor);
                    if (!(value instanceof String text) || !boundedText(text)) {
                        throw new IllegalArgumentException();
                    }
                    Path module = canonicalDirectory(text, root);
                    if (module.equals(root) != rootOnly || requested.putIfAbsent(module, values == restored) != null) {
                        throw new IllegalArgumentException();
                    }
                }
            }
            validateBuildInputs(root);
            List<Object> configuration = offlineConfiguration();
            List<?> projects = currentModelProjects(rootUri, root, monitor);
            Set<Path> modules = new LinkedHashSet<>();
            for (Object value : projects) {
                if (!(value instanceof Map<?, ?> project) || !(project.get("moduleUri") instanceof String uri)
                        || !modules.add(canonicalDirectory(uri, root))) {
                    throw new IllegalArgumentException();
                }
            }
            ProjectStateContext context = projectStateContext(root, rootUri, configuration, rootIdentity);
            return applyProjectStates(context, requested, modules, rootOnly, monitor);
        } catch (Exception failure) {
            if (rootOnly && (monitor.isCanceled() || failure instanceof OperationCanceledException)) {
                throw new OperationCanceledException();
            }
            if (monitor.isCanceled() || failure instanceof OperationCanceledException) {
                throw new IllegalStateException("Gradle module state change cancelled and may be partial. Refresh before retrying.");
            }
            throw new IllegalStateException("Gradle module state change failed or may be partial. Check current module selection, exclusive Gradle ownership, pinned provider SDK, project descriptions and cached host Gradle settings; refresh before retrying.");
        }
    }

    private static Object applyProjectStates(ProjectStateContext context, Map<Path, Boolean> requested,
            Set<Path> modules, boolean rootOnly, IProgressMonitor monitor) throws Exception {
        if (!modules.containsAll(requested.keySet())) {
            throw new IllegalArgumentException();
        }
        Path root = context.root();
        List<Object> configuration = context.configuration();
        Set<Path> missingDescriptions = new LinkedHashSet<>();
        for (var entry : requested.entrySet()) {
            if (!Files.exists(entry.getKey().resolve(".project"), LinkOption.NOFOLLOW_LINKS)) {
                if (entry.getValue() || rootOnly) {
                    throw new IllegalArgumentException();
                }
                missingDescriptions.add(entry.getKey());
            }
        }
        Map<Path, String> freshNames = Map.of();
        if (!missingDescriptions.isEmpty()) {
            if (!offlineConfiguration().equals(configuration)) {
                throw new OperationCanceledException();
            }
            freshNames = freshModuleNames(root, monitor);
            if (!freshNames.keySet().containsAll(missingDescriptions)) {
                throw new IllegalArgumentException();
            }
        }
        IWorkspace workspace = ResourcesPlugin.getWorkspace();
        List<ModuleState> states = new ArrayList<>();
        Set<String> names = new LinkedHashSet<>();
        for (var entry : requested.entrySet()) {
            checkCancelled(monitor);
            Path descriptor = entry.getKey().resolve(".project");
            byte[] bytes;
            IProjectDescription description;
            if (missingDescriptions.contains(entry.getKey())) {
                if (Files.exists(descriptor, LinkOption.NOFOLLOW_LINKS)) {
                    throw new OperationCanceledException();
                }
                bytes = null;
                description = workspace.newProjectDescription(freshNames.get(entry.getKey()));
                description.setNatureIds(new String[] { "org.eclipse.buildship.core.gradleprojectnature" });
            } else {
                bytes = descriptionBytes(descriptor);
                description = workspace.loadProjectDescription(new java.io.ByteArrayInputStream(bytes));
            }
            String name = description.getName();
            Set<String> natures = new LinkedHashSet<>(Arrays.asList(description.getNatureIds()));
            if (!boundedText(name) || name.length() > 256 || !names.add(name)
                    || !workspace.validateName(name, org.eclipse.core.resources.IResource.PROJECT).isOK()
                    || !natures.contains("org.eclipse.buildship.core.gradleprojectnature")
                    || natures.contains("org.eclipse.m2e.core.maven2Nature")) {
                throw new IllegalArgumentException();
            }
            IProject project = workspace.getRoot().getProject(name);
            description.setLocationURI(entry.getKey().toUri());
            ModuleState state = new ModuleState(entry.getKey(), bytes, description, project, entry.getValue());
            validateState(state, root, monitor);
            states.add(state);
        }
        ISchedulingRule current = Job.getJobManager().currentRule();
        ISchedulingRule rule = workspace.getRoot();
        if (current != null && current.contains(rule)) {
            rule = null;
        }
        workspace.run((IWorkspaceRunnable) progress -> {
            context.validate();
            if (rootOnly) {
                requireRootIdentity(root, context.rootIdentity());
                if (Boolean.FALSE.equals(requested.get(root))) {
                    requireClosedGradleChildren(root, progress);
                }
            }
            for (ModuleState state : states) {
                validateState(state, root, progress);
            }
            for (ModuleState state : states) {
                checkCancelled(progress);
                if (!state.project().exists()) {
                    state.project().create(state.description(), progress);
                }
                if (state.open() && !state.project().isOpen()) {
                    state.project().open(progress);
                } else if (!state.open() && state.project().isOpen()) {
                    ProjectUnlinkHandler.clearDiagnostics(state.project(), root, progress);
                    state.project().close(progress);
                }
                checkCancelled(progress);
                if (!state.project().exists() || state.project().isOpen() != state.open()) {
                    throw new OperationCanceledException();
                }
            }
        }, rule, IWorkspace.AVOID_UPDATE, monitor);
        checkCancelled(monitor);
        Map<String, Object> receipt = new LinkedHashMap<>();
        receipt.put("version", 1);
        receipt.put("rootUri", root.toUri().toString());
        receipt.put("closedProjectUris", states.stream().filter(state -> !state.open()).map(state -> state.path().toUri().toString()).toList());
        receipt.put("restoredProjectUris", states.stream().filter(ModuleState::open).map(state -> state.path().toUri().toString()).toList());
        receipt.put("moduleStatePhaseComplete", true);
        receipt.put("fullSynchronizationKnown", false);
        return receipt;
    }

    private static List<?> currentModelProjects(String rootUri, Path root, IProgressMonitor monitor) throws Exception {
        Object model = new GradleModelHandler().executeCommand("groovy.gradle.projects",
                List.of(Map.of("rootUri", rootUri, "offline", true)), monitor);
        if (!(model instanceof Map<?, ?> response) || !Integer.valueOf(1).equals(response.get("version"))
                || !(response.get("projects") instanceof List<?> projects) || projects.size() > 128
                || !root.toUri().toString().equals(response.get("rootUri"))) {
            throw new IllegalArgumentException();
        }
        return projects;
    }

    private static Object readRootState(List<Object> arguments, IProgressMonitor monitor) {
        try {
            Object result = readProjectStates(arguments, monitor, true);
            if (!(result instanceof Map<?, ?> response) || !(response.get("modules") instanceof List<?> records)
                    || records.size() != 1 || !(records.get(0) instanceof Map<?, ?> root)
                    || !response.get("rootUri").equals(root.get("moduleUri"))) {
                throw new IllegalArgumentException();
            }
            return Map.of("version", 1, "rootUri", response.get("rootUri"),
                    "workspaceState", root.get("workspaceState"),
                    "descriptionAvailable", root.get("descriptionAvailable"), "fullSynchronizationKnown", false);
        } catch (LinkageError unavailable) {
            throw new IllegalStateException("Gradle root state unavailable. Check pinned provider SDK setup and refresh.");
        } catch (Exception failure) {
            if (monitor.isCanceled() || failure instanceof OperationCanceledException) {
                throw new IllegalStateException("Gradle root state read cancelled. Refresh before retrying.");
            }
            throw new IllegalStateException("Gradle root state unavailable. Check current build selection, exclusive Gradle ownership, SDK description and cached host settings; refresh before retrying.");
        }
    }

    private static Object readModuleStates(List<Object> arguments, IProgressMonitor monitor) {
        return readProjectStates(arguments, monitor, false);
    }

    private static Object readProjectStates(List<Object> arguments, IProgressMonitor monitor, boolean rootOnly) {
        try {
            checkCancelled(monitor);
            if (arguments == null || arguments.size() != 1 || !(arguments.get(0) instanceof Map<?, ?> request)
                    || !request.keySet().equals(Set.of("rootUri", "offline"))
                    || !(request.get("rootUri") instanceof String rootUri) || !boundedText(rootUri)
                    || !Boolean.TRUE.equals(request.get("offline"))) {
                throw new IllegalArgumentException();
            }
            Path root = GradleModelHandler.initializedRoot(rootUri, "Gradle");
            if (!canonicalDirectory(rootUri, root).equals(root) || !boundedText(root.toUri().toString())) {
                throw new IllegalArgumentException();
            }
            validateBuildInputs(root);
            List<Object> configuration = offlineConfiguration();
            List<?> models = currentModelProjects(rootUri, root, monitor);
            Map<Path, Path> modules = new LinkedHashMap<>();
            for (Object value : models) {
                checkCancelled(monitor);
                if (!(value instanceof Map<?, ?> project) || !(project.get("moduleUri") instanceof String uri)
                        || !(project.get("buildRootUri") instanceof String buildUri)) {
                    throw new IllegalArgumentException();
                }
                Path module = canonicalDirectory(uri, root);
                Path build = canonicalDirectory(buildUri, root);
                if (!module.startsWith(build) || modules.putIfAbsent(module, build) != null) {
                    throw new IllegalArgumentException();
                }
            }
            return readProjectStates(projectStateContext(root, rootUri, configuration, null), modules,
                    rootOnly, monitor);
        } catch (LinkageError unavailable) {
            throw new IllegalStateException("Gradle module states unavailable. Check pinned provider SDK setup and refresh.");
        } catch (Exception failure) {
            if (monitor.isCanceled() || failure instanceof OperationCanceledException) {
                throw new IllegalStateException("Gradle module state read cancelled. Refresh before retrying.");
            }
            throw new IllegalStateException("Gradle module states unavailable. Check current build selection, exclusive Gradle ownership, SDK descriptions and cached host settings; refresh before retrying.");
        }
    }

    private static Object readProjectStates(ProjectStateContext context, Map<Path, Path> modules,
            boolean rootOnly, IProgressMonitor monitor) throws Exception {
        Path root = context.root();
        IWorkspace workspace = ResourcesPlugin.getWorkspace();
        List<Map<String, Object>> records = new ArrayList<>();
        ISchedulingRule current = Job.getJobManager().currentRule();
        ISchedulingRule rule = workspace.getRoot();
        if (current != null && current.contains(rule)) {
            rule = null;
        }
        workspace.run((IWorkspaceRunnable) progress -> {
            try {
                context.validate();
                IProject[] resources = workspace.getRoot().getProjects();
                if (resources.length > 1024) {
                    throw new IllegalArgumentException();
                }
                Map<Path, IProject> registered = new LinkedHashMap<>();
                for (IProject project : resources) {
                    checkCancelled(progress);
                    if (!project.exists() || project.getLocation() == null) {
                        continue;
                    }
                    Path location = project.getLocation().toFile().toPath().toAbsolutePath().normalize();
                    if (!Files.exists(location)) {
                        continue;
                    }
                    Path real = location.toRealPath();
                    if (modules.containsKey(real)) {
                        if (!location.equals(real) || registered.putIfAbsent(real, project) != null) {
                            throw new IllegalArgumentException();
                        }
                    }
                }
                List<ModuleState> descriptions = new ArrayList<>();
                Set<IProject> existing = new LinkedHashSet<>();
                Set<Path> fresh = new LinkedHashSet<>();
                long descriptionSize = 0;
                for (var entry : modules.entrySet()) {
                    checkCancelled(progress);
                    Path module = entry.getKey();
                    if (module.equals(root) != rootOnly) {
                        continue;
                    }
                    Path file = module.resolve(".project");
                    IProject project = registered.get(module);
                    boolean descriptionAvailable = Files.exists(file, LinkOption.NOFOLLOW_LINKS);
                    if (descriptionAvailable) {
                        byte[] bytes = descriptionBytes(file);
                        descriptionSize += bytes.length;
                        if (descriptionSize > 8 * 1024 * 1024) {
                            throw new IllegalArgumentException();
                        }
                        var description = workspace.loadProjectDescription(new java.io.ByteArrayInputStream(bytes));
                        String name = description.getName();
                        Set<String> natures = new LinkedHashSet<>(Arrays.asList(description.getNatureIds()));
                        if (!boundedText(name) || name.length() > 256
                                || !workspace.validateName(name, org.eclipse.core.resources.IResource.PROJECT).isOK()
                                || !natures.contains("org.eclipse.buildship.core.gradleprojectnature")
                                || natures.contains("org.eclipse.m2e.core.maven2Nature")) {
                            throw new IllegalArgumentException();
                        }
                        IProject named = workspace.getRoot().getProject(name);
                        if (project != null && !project.equals(named)) {
                            throw new IllegalArgumentException();
                        }
                        project = named;
                        ModuleState state = new ModuleState(module, bytes, description, project, project.isOpen());
                        validateState(state, root, progress);
                        descriptions.add(state);
                        if (project.exists()) {
                            existing.add(project);
                        }
                    } else {
                        if (project != null) {
                            throw new IllegalArgumentException();
                        }
                        fresh.add(module);
                    }
                    records.add(Map.of("moduleUri", module.toUri().toString(),
                            "buildRootUri", entry.getValue().toUri().toString(),
                            "workspaceState", project == null || !project.exists() ? "absent" : project.isOpen() ? "open" : "closed",
                            "descriptionAvailable", descriptionAvailable));
                }
                for (ModuleState state : descriptions) {
                    validateState(state, root, progress);
                    if (state.project().exists() != existing.contains(state.project())
                            || state.project().isOpen() != state.open()) {
                        throw new OperationCanceledException();
                    }
                }
                for (Path module : fresh) {
                    checkCancelled(progress);
                    if (!module.toRealPath().equals(module)
                            || Files.exists(module.resolve(".project"), LinkOption.NOFOLLOW_LINKS)) {
                        throw new OperationCanceledException();
                    }
                }
            } catch (org.eclipse.core.runtime.CoreException | java.io.IOException failure) {
                throw new IllegalArgumentException();
            } catch (RuntimeException failure) {
                throw failure;
            } catch (Exception failure) {
                throw new IllegalArgumentException();
            }
        }, rule, IWorkspace.AVOID_UPDATE, monitor);
        checkCancelled(monitor);
        context.validate();
        return Map.of("version", 1, "rootUri", root.toUri().toString(),
                "modules", records, "fullSynchronizationKnown", false);
    }

    static Path canonicalDirectory(String text, Path root) throws Exception {
        if (!boundedText(text)) {
            throw new IllegalArgumentException();
        }
        URI uri = new URI(text);
        if (!"file".equals(uri.getScheme()) || uri.getQuery() != null || uri.getFragment() != null) {
            throw new IllegalArgumentException();
        }
        Path requested = Path.of(uri);
        Path real = requested.toRealPath();
        if (!requested.equals(requested.normalize()) || !requested.equals(real)
                || !real.startsWith(root) || !Files.isDirectory(real)) {
            throw new IllegalArgumentException();
        }
        return real;
    }

    private static byte[] descriptionBytes(Path path) throws java.io.IOException {
        if (!Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS) || !path.toRealPath().equals(path)
                || Files.size(path) > 1024 * 1024) {
            throw new IllegalArgumentException();
        }
        byte[] bytes;
        try (var stream = Files.newInputStream(path, LinkOption.NOFOLLOW_LINKS)) {
            bytes = stream.readNBytes(1024 * 1024 + 1);
        }
        if (bytes.length > 1024 * 1024) {
            throw new IllegalArgumentException();
        }
        return bytes;
    }

    private static void validateState(ModuleState state, Path root, IProgressMonitor monitor)
            throws org.eclipse.core.runtime.CoreException {
        try {
            checkCancelled(monitor);
            if (!state.path().startsWith(root) || !state.path().toRealPath().equals(state.path())) {
                throw new IllegalArgumentException();
            }
            Path descriptor = state.path().resolve(".project");
            if (state.descriptionBytes() == null) {
                if (state.open() || Files.exists(descriptor, LinkOption.NOFOLLOW_LINKS) || state.project().exists()) {
                    throw new IllegalArgumentException();
                }
            } else if (!Arrays.equals(descriptionBytes(descriptor), state.descriptionBytes())) {
                throw new IllegalArgumentException();
            }
            IProject[] projects = ResourcesPlugin.getWorkspace().getRoot().getProjects();
            if (projects.length > 1024) {
                throw new IllegalArgumentException();
            }
            for (IProject project : projects) {
                if (project.getLocation() != null) {
                    Path location = project.getLocation().toFile().toPath().toAbsolutePath().normalize();
                    boolean sameLocation = location.equals(state.path())
                            || Files.exists(location) && location.toRealPath().equals(state.path());
                    if (sameLocation && !project.equals(state.project())) {
                        throw new IllegalArgumentException();
                    }
                }
            }
            if (state.project().exists()) {
                if (state.project().getLocation() == null
                        || !state.project().getLocation().toFile().toPath().toAbsolutePath().normalize().equals(state.path())
                        || !state.project().getLocation().toFile().toPath().toRealPath().equals(state.path())
                        || state.project().isOpen() && !ProjectUnlinkHandler.matches(state.project(), state.path(), "Gradle")) {
                    throw new IllegalArgumentException();
                }
            }
        } catch (java.io.IOException unavailable) {
            throw new IllegalArgumentException();
        }
    }

    private record ModelProjects(Map<Path, String> names, Map<Path, Path> buildRoots) {}

    private static Map<Path, String> freshModuleNames(Path root, IProgressMonitor monitor) throws Exception {
        return freshModuleProjects(root, monitor).names();
    }

    private static ModelProjects freshModuleProjects(Path root, IProgressMonitor monitor) throws Exception {
        try {
            checkCancelled(monitor);
            var build = GradleCore.getWorkspace().createBuild(GradleProjectImporter.getBuildConfiguration(root));
            var models = build.withConnection(
                    org.eclipse.buildship.core.internal.workspace.EclipseModelUtils::queryModels, monitor);
            return eclipseModuleProjects(models, root, monitor);
        } catch (LinkageError unavailable) {
            throw new IllegalStateException("The pinned Buildship Eclipse model API is unavailable. Check provider setup and refresh.");
        }
    }

    private static Map<Path, String> eclipseModuleNames(
            Map<String, org.gradle.tooling.model.eclipse.EclipseProject> models, Path root,
            IProgressMonitor monitor) throws Exception {
        return eclipseModuleProjects(models, root, monitor).names();
    }

    private static ModelProjects eclipseModuleProjects(
            Map<String, org.gradle.tooling.model.eclipse.EclipseProject> models, Path root,
            IProgressMonitor monitor) throws Exception {
        if (models == null || models.size() > 128) {
            throw new IllegalArgumentException();
        }
        Map<Path, String> names = new LinkedHashMap<>();
        Map<Path, Path> buildRoots = new LinkedHashMap<>();
        Set<String> uniqueNames = new LinkedHashSet<>();
        for (var model : models.values()) {
            checkCancelled(monitor);
            Path buildRoot = model.getProjectDirectory().toPath().toRealPath();
            if (!buildRoot.startsWith(root)) {
                throw new IllegalArgumentException();
            }
            var pending = new java.util.ArrayDeque<org.gradle.tooling.model.eclipse.EclipseProject>();
            pending.add(model);
            while (!pending.isEmpty()) {
                checkCancelled(monitor);
                if (names.size() >= 128 || pending.size() > 128) {
                    throw new IllegalArgumentException();
                }
                var project = pending.removeFirst();
                Path module = project.getProjectDirectory().toPath().toRealPath();
                String name = project.getName();
                if (!module.startsWith(root) || !boundedText(module.toUri().toString())
                        || !boundedText(name) || name.length() > 256 || !uniqueNames.add(name)
                        || !ResourcesPlugin.getWorkspace().validateName(name,
                                org.eclipse.core.resources.IResource.PROJECT).isOK()
                        || names.putIfAbsent(module, name) != null) {
                    throw new IllegalArgumentException();
                }
                buildRoots.put(module, buildRoot);
                for (var child : project.getChildren()) {
                    if (pending.size() >= 128) {
                        throw new IllegalArgumentException();
                    }
                    pending.addLast(child);
                }
            }
        }
        checkCancelled(monitor);
        return new ModelProjects(names, buildRoots);
    }

    private static void refreshCollidingClosedArtifacts(org.eclipse.buildship.core.GradleBuild build,
            Path root, IProgressMonitor monitor) throws Exception {
        try {
            checkCancelled(monitor);
            IWorkspace workspace = ResourcesPlugin.getWorkspace();
            IProject[] resources = workspace.getRoot().getProjects();
            if (resources.length > 1024) {
                throw new IllegalArgumentException();
            }
            List<ModuleState> closed = new ArrayList<>();
            for (IProject project : resources) {
                checkCancelled(monitor);
                if (!project.exists() || project.isOpen() || project.getLocation() == null) {
                    continue;
                }
                Path module = project.getLocation().toFile().toPath().toRealPath();
                if (!module.startsWith(root)) {
                    continue;
                }
                byte[] bytes = descriptionBytes(module.resolve(".project"));
                var description = workspace.loadProjectDescription(new java.io.ByteArrayInputStream(bytes));
                Set<String> natures = new LinkedHashSet<>(Arrays.asList(description.getNatureIds()));
                if (!natures.contains("org.eclipse.buildship.core.gradleprojectnature")
                        || natures.contains("org.eclipse.m2e.core.maven2Nature")) {
                    continue;
                }
                if (!project.getName().equals(description.getName()) || closed.size() >= 128) {
                    throw new IllegalArgumentException();
                }
                var state = new ModuleState(module, bytes, description, project, false);
                validateState(state, root, monitor);
                closed.add(state);
            }
            if (closed.isEmpty()) {
                return;
            }
            var models = build.withConnection(connection -> connection.action(
                    org.eclipse.buildship.core.internal.util.gradle.IdeFriendlyClassLoading
                            .loadCompositeModelQuery(org.gradle.tooling.model.eclipse.EclipseProject.class))
                    .run(), monitor);
            Map<Path, String> unreservedNames = eclipseModuleNames(models, root, monitor);
            Set<String> referenced = eclipseDependencyNames(models, monitor);
            resources = workspace.getRoot().getProjects();
            if (resources.length > 1024) {
                throw new IllegalArgumentException();
            }
            Map<String, IProject> open = new LinkedHashMap<>();
            for (IProject project : resources) {
                if (project.exists() && project.isOpen()) {
                    open.put(project.getName(), project);
                }
            }
            Set<Path> affected = new LinkedHashSet<>();
            for (ModuleState state : closed) {
                validateState(state, root, monitor);
                if (state.project().isOpen()) {
                    throw new OperationCanceledException();
                }
                String unreserved = unreservedNames.get(state.path());
                IProject namespaceOwner = unreserved == null ? null : open.get(unreserved);
                if (unreserved == null || !referenced.contains(unreserved)
                        || unreserved.equals(state.project().getName()) || namespaceOwner == null) {
                    continue;
                }
                if (namespaceOwner.getLocation() == null) {
                    throw new IllegalArgumentException();
                }
                Path namespaceLocation = namespaceOwner.getLocation().toFile().toPath().toRealPath();
                if (!unreservedNames.containsKey(namespaceLocation)) {
                    affected.add(state.path());
                }
            }
            if (affected.isEmpty()) {
                return;
            }
            var selected = new ArrayList<org.gradle.tooling.model.eclipse.EclipseWorkspaceProject>();
            for (IProject project : resources) {
                checkCancelled(monitor);
                if (!project.exists() || project.getLocation() == null) {
                    continue;
                }
                Path location = project.getLocation().toFile().toPath().toRealPath();
                if (project.isOpen() || affected.contains(location)) {
                    selected.add(new org.eclipse.buildship.core.internal.workspace.DefaultEclipseWorkspaceProject(
                            project.getName(), location.toFile(), project.isOpen()));
                }
            }
            var actual = org.eclipse.buildship.core.internal.workspace.EclipseModelUtils.buildEclipseRuntimeConfigurer();
            var selection = new org.eclipse.buildship.core.internal.workspace.EclipseRuntimeConfigurer(
                    new org.eclipse.buildship.core.internal.workspace.DefaultEclipseWorkspace(
                            workspace.getRoot().getLocation().toFile(), selected));
            var names = org.eclipse.buildship.core.internal.util.gradle.IdeFriendlyClassLoading.loadCompositeModelQuery(
                    org.gradle.tooling.model.eclipse.EclipseProject.class,
                    org.gradle.tooling.model.eclipse.EclipseRuntime.class, actual);
            var artifacts = org.eclipse.buildship.core.internal.util.gradle.IdeFriendlyClassLoading.loadCompositeModelQuery(
                    org.gradle.tooling.model.eclipse.RunClosedProjectBuildDependencies.class,
                    org.gradle.tooling.model.eclipse.EclipseRuntime.class, selection);
            // Gradle8.13 selects closed dependency tasks before applying reserved workspace names.
            var sequence = new org.eclipse.buildship.core.internal.workspace.BuildActionSequence(
                    names, artifacts, new GradleDiscoveryAction.VerifyClosedArtifactSelection());
            Path script = GradleModelHandler.createModelInitScript();
            try {
                build.withConnection(connection -> {
                    var operation = connection.action().projectsLoaded(sequence,
                            new org.eclipse.buildship.core.internal.util.gradle.SimpleIntermediateResultHandler<Void>())
                            .build().forTasks();
                    operation.addArguments("--init-script", script.toString());
                    operation.run();
                    return null;
                }, monitor);
            } finally {
                GradleModelHandler.removeInitScript(script);
            }
            checkCancelled(monitor);
            var javaProjects = new ArrayList<org.eclipse.jdt.core.IJavaElement>();
            for (IProject project : workspace.getRoot().getProjects()) {
                if (project.exists() && project.isOpen() && project.getLocation() != null
                        && unreservedNames.containsKey(project.getLocation().toFile().toPath().toRealPath())
                        && project.hasNature(org.eclipse.jdt.core.JavaCore.NATURE_ID)) {
                    javaProjects.add(org.eclipse.jdt.core.JavaCore.create(project));
                }
            }
            org.eclipse.jdt.core.JavaCore.create(workspace.getRoot()).refreshExternalArchives(
                    javaProjects.toArray(org.eclipse.jdt.core.IJavaElement[]::new), monitor);
            checkCancelled(monitor);
        } catch (LinkageError unavailable) {
            throw new IllegalStateException("The pinned Buildship closed-dependency API is unavailable. Check provider setup and refresh.");
        }
    }

    private static Set<String> eclipseDependencyNames(
            Map<String, org.gradle.tooling.model.eclipse.EclipseProject> models,
            IProgressMonitor monitor) {
        Set<String> names = new LinkedHashSet<>();
        int projects = 0;
        int dependencies = 0;
        for (var model : models.values()) {
            var pending = new java.util.ArrayDeque<org.gradle.tooling.model.eclipse.EclipseProject>();
            pending.add(model);
            while (!pending.isEmpty()) {
                checkCancelled(monitor);
                if (++projects > 128 || pending.size() > 128) {
                    throw new IllegalArgumentException();
                }
                var project = pending.removeFirst();
                for (var dependency : project.getProjectDependencies()) {
                    if (++dependencies > 4096 || !boundedText(dependency.getPath())) {
                        throw new IllegalArgumentException();
                    }
                    names.add(dependency.getPath());
                }
                for (var child : project.getChildren()) {
                    if (pending.size() >= 128) {
                        throw new IllegalArgumentException();
                    }
                    pending.addLast(child);
                }
            }
        }
        return names;
    }

    private static void validateBuildInputs(Path root) throws java.io.IOException {
        boolean buildInputPresent = false;
        for (String name : List.of("settings.gradle", "settings.gradle.kts", "build.gradle", "build.gradle.kts")) {
            Path input = root.resolve(name);
            if (Files.exists(input, LinkOption.NOFOLLOW_LINKS)) {
                if (!Files.isRegularFile(input) || !input.toRealPath().startsWith(root)) {
                    throw new IllegalArgumentException();
                }
                buildInputPresent = true;
            }
        }
        if (!buildInputPresent) {
            throw new IllegalArgumentException();
        }
    }

    private static List<Object> offlineConfiguration() {
        var preferences = JavaLanguageServerPlugin.getPreferencesManager().getPreferences();
        String home = preferences.getGradleHome();
        String version = preferences.getGradleVersion();
        if (!preferences.isImportGradleOfflineEnabled() || preferences.isGradleWrapperEnabled()
                || version != null && !version.isBlank() || home == null || home.isBlank()
                || !Files.isDirectory(Path.of(home))) {
            throw new IllegalArgumentException();
        }
        return Arrays.asList(home, version, preferences.getGradleUserHome(), preferences.getGradleJavaHome(),
                preferences.isGradleWrapperEnabled(), preferences.isImportGradleOfflineEnabled(),
                List.copyOf(preferences.getGradleArguments()), List.copyOf(preferences.getGradleJvmArguments()));
    }

    private static boolean boundedText(String text) {
        return text != null && !text.isBlank() && text.length() <= 4096
                && text.chars().noneMatch(Character::isISOControl);
    }

    private static void checkCancelled(IProgressMonitor monitor) {
        if (monitor.isCanceled()) {
            throw new OperationCanceledException();
        }
    }
}
