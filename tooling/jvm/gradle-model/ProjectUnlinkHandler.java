package dev.made.groovy;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.eclipse.core.resources.IProject;
import org.eclipse.core.resources.IResource;
import org.eclipse.core.resources.IWorkspace;
import org.eclipse.core.resources.IWorkspaceRunnable;
import org.eclipse.core.resources.ResourcesPlugin;
import org.eclipse.core.runtime.CoreException;
import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.core.runtime.OperationCanceledException;
import org.eclipse.core.runtime.Platform;
import org.eclipse.core.runtime.jobs.ISchedulingRule;
import org.eclipse.core.runtime.jobs.Job;
import org.eclipse.jdt.core.JavaCore;
import org.eclipse.jdt.ls.core.internal.EventNotification;
import org.eclipse.jdt.ls.core.internal.EventType;
import org.eclipse.jdt.ls.core.internal.IDelegateCommandHandler;
import org.eclipse.jdt.ls.core.internal.JavaLanguageServerPlugin;
import org.eclipse.jdt.ls.core.internal.ProjectUtils;
import org.eclipse.jdt.ls.core.internal.ResourceUtils;
import org.eclipse.lsp4j.PublishDiagnosticsParams;

public final class ProjectUnlinkHandler implements IDelegateCommandHandler {
    private static final int MAX_PROJECTS = 128;
    private static final int MAX_TEXT = 4096;
    private static final int MAX_MARKERS = 10000;
    private static final int MAX_INSPECTED_PROJECTS = 1024;

    @Override
    public Object executeCommand(String command, List<Object> arguments, IProgressMonitor monitor) throws Exception {
        if ("groovy.gradle.importedBuildRoots".equals(command)) {
            return importedProjects("Gradle", arguments, true, monitor);
        }
        if ("groovy.maven.importedProjects".equals(command) || "groovy.gradle.importedProjects".equals(command)) {
            return importedProjects(command.startsWith("groovy.maven.") ? "Maven" : "Gradle", arguments, false, monitor);
        }
        String buildSystem = "groovy.maven.unlinkSelected".equals(command) ? "Maven"
                : "groovy.gradle.unlinkSelected".equals(command) ? "Gradle" : "JVM";
        List<URI> removed = new ArrayList<>();
        try {
            checkCancelled(monitor);
            if ("JVM".equals(buildSystem) || arguments == null || arguments.size() != 1
                    || !(arguments.get(0) instanceof Map<?, ?> request)
                    || !request.keySet().equals(Set.of("rootUri", "projectUris"))
                    || !(request.get("rootUri") instanceof String rootUri) || !boundedText(rootUri)
                    || !(request.get("projectUris") instanceof List<?> selected) || selected.isEmpty()
                    || selected.size() > MAX_PROJECTS) {
                throw new IllegalArgumentException();
            }
            Path root = GradleModelHandler.initializedRoot(rootUri, buildSystem);
            String canonicalRootUri = root.toUri().toString();
            if (!boundedText(canonicalRootUri)) {
                throw new IllegalArgumentException();
            }
            Map<Path, IProject> projects = new LinkedHashMap<>();
            for (Object value : selected) {
                checkCancelled(monitor);
                if (!(value instanceof String text) || !boundedText(text)) {
                    throw new IllegalArgumentException();
                }
                URI uri = new URI(text);
                if (!"file".equals(uri.getScheme()) || uri.getQuery() != null || uri.getFragment() != null) {
                    throw new IllegalArgumentException();
                }
                Path requested = Path.of(uri);
                Path module = requested.toRealPath();
                if (!requested.normalize().equals(requested) || !requested.equals(module)
                        || !module.startsWith(root) || !Files.isDirectory(module)
                        || !boundedText(module.toUri().toString()) || projects.containsKey(module)) {
                    throw new IllegalArgumentException();
                }
                IProject match = null;
                for (IProject project : ProjectUtils.getAllProjects()) {
                    checkCancelled(monitor);
                    if (matches(project, module, buildSystem)) {
                        if (match != null) {
                            throw new IllegalArgumentException();
                        }
                        match = project;
                    }
                }
                if (match == null) {
                    throw new IllegalArgumentException();
                }
                projects.put(module, match);
            }
            ISchedulingRule current = Job.getJobManager().currentRule();
            ISchedulingRule rule = ResourcesPlugin.getWorkspace().getRoot();
            if (current != null && current.contains(rule)) {
                rule = null;
            }
            ResourcesPlugin.getWorkspace().run((IWorkspaceRunnable) progress -> {
                if (!GradleModelHandler.initializedRoot(rootUri, buildSystem).equals(root)) {
                    throw new OperationCanceledException();
                }
                for (Map.Entry<Path, IProject> entry : projects.entrySet()) {
                    checkCancelled(progress);
                    if (!matches(entry.getValue(), entry.getKey(), buildSystem)) {
                        throw new OperationCanceledException();
                    }
                }
                for (Map.Entry<Path, IProject> entry : projects.entrySet()) {
                    checkCancelled(progress);
                    clearDiagnostics(entry.getValue(), root, progress);
                    checkCancelled(progress);
                    // Unlink removes Eclipse membership; deleting source content is never authorized.
                    entry.getValue().delete(false, false, progress);
                    removed.add(entry.getKey().toUri());
                    checkCancelled(progress);
                    if (entry.getValue().exists() || !Files.isDirectory(entry.getKey())) {
                        throw new OperationCanceledException();
                    }
                }
            }, rule, IWorkspace.AVOID_UPDATE, monitor);
            checkCancelled(monitor);
            Map<String, Object> receipt = new LinkedHashMap<>();
            receipt.put("version", 1);
            receipt.put("rootUri", canonicalRootUri);
            receipt.put("unlinkedProjectUris", removed.stream().map(URI::toString).toList());
            receipt.put("selectedUnlinkPhaseComplete", true);
            receipt.put("fullSynchronizationKnown", false);
            return receipt;
        } catch (OperationCanceledException cancelled) {
            throw new IllegalStateException("Selected " + buildSystem + " unlink cancelled and may be partial. Refresh before retrying.");
        } catch (Exception failure) {
            if (monitor.isCanceled()) {
                throw new IllegalStateException("Selected " + buildSystem + " unlink cancelled and may be partial. Refresh before retrying.");
            }
            throw new IllegalStateException("Selected " + buildSystem + " unlink failed or may be partial. Check initialized root and current imported project selection; refresh before retrying.");
        } finally {
            if (!removed.isEmpty()) {
                try {
                    JavaLanguageServerPlugin.getProjectsManager().getConnection().sendEventNotification(
                            new EventNotification().withType(EventType.ProjectsDeleted).withData(removed));
                } catch (RuntimeException failure) {
                    throw new IllegalStateException("Selected " + buildSystem + " unlink may be partial; project notification failed. Refresh before retrying.");
                }
            }
        }
    }

    private static Object importedProjects(String buildSystem, List<Object> arguments, boolean includeBuildRoots,
            IProgressMonitor monitor) {
        try {
            checkCancelled(monitor);
            if (arguments == null || arguments.size() != 1 || !(arguments.get(0) instanceof Map<?, ?> request)
                    || !request.keySet().equals(Set.of("rootUri", "cachedOnly"))
                    || !Boolean.TRUE.equals(request.get("cachedOnly"))
                    || !(request.get("rootUri") instanceof String rootUri) || !boundedText(rootUri)) {
                throw new IllegalArgumentException();
            }
            Path root = GradleModelHandler.initializedRoot(rootUri, buildSystem);
            String canonicalRootUri = root.toUri().toString();
            if (!boundedText(canonicalRootUri)) {
                throw new IllegalArgumentException();
            }
            if (!includeBuildRoots) {
                return readImportedProjects(buildSystem, rootUri, root, canonicalRootUri, false, monitor);
            }
            GradleImportHandler.canonicalDirectory(rootUri, root);
            Object identity = GradleImportHandler.rootFileKey(root);
            IWorkspace workspace = ResourcesPlugin.getWorkspace();
            ISchedulingRule current = Job.getJobManager().currentRule();
            ISchedulingRule rule = current != null && current.contains(workspace.getRoot()) ? null : workspace.getRoot();
            Map<String, Object> response = new LinkedHashMap<>();
            workspace.run((IWorkspaceRunnable) progress -> {
                GradleImportHandler.requireRootIdentity(root, identity);
                try {
                    response.putAll(readImportedProjects(buildSystem, rootUri, root, canonicalRootUri, true, progress));
                } catch (IOException failure) {
                    throw new IllegalArgumentException();
                }
                GradleImportHandler.requireRootIdentity(root, identity);
            }, rule, IWorkspace.AVOID_UPDATE, monitor);
            checkCancelled(monitor);
            return response;
        } catch (LinkageError unavailable) {
            throw new IllegalStateException(buildSystem + " cached build-root identities unavailable. Check pinned provider SDK setup and refresh before retrying.");
        } catch (Exception failure) {
            if (monitor.isCanceled() || failure instanceof OperationCanceledException) {
                throw new IllegalStateException(buildSystem + " imported project identity read cancelled. Refresh before retrying.");
            }
            throw new IllegalStateException(buildSystem + " imported project identities unavailable. Check initialized root and provider setup; refresh before retrying.");
        }
    }

    private record CachedGradleAssociation(IProject project, Path module, Path build,
            Object moduleIdentity, Object buildIdentity, boolean javaNature) {}

    private static Path cachedGradleBuildRoot(IProject project, Path root) throws IOException {
        var build = org.eclipse.buildship.core.GradleCore.getWorkspace().getBuild(project);
        if (build.isEmpty()
                || !(build.get() instanceof org.eclipse.buildship.core.internal.workspace.InternalGradleBuild internal)) {
            return null;
        }
        Path path = internal.getBuildConfig().getRootProjectDirectory().toPath();
        if (!path.isAbsolute() || !path.equals(path.normalize()) || !path.startsWith(root)
                || !Files.isDirectory(path) || !path.equals(path.toRealPath())
                || !boundedText(path.toUri().toString())) {
            return null;
        }
        return path;
    }

    private static Map<String, Object> readImportedProjects(String buildSystem, String rootUri, Path root,
            String canonicalRootUri, boolean includeBuildRoots, IProgressMonitor monitor)
            throws IOException, CoreException {
        List<Map<String, Object>> projects = new ArrayList<>();
        Set<Path> identities = new LinkedHashSet<>();
        List<CachedGradleAssociation> associations = new ArrayList<>();
        boolean partial = false;
        int inspected = 0;
        for (IProject project : ProjectUtils.getAllProjects()) {
            checkCancelled(monitor);
            if (++inspected > MAX_INSPECTED_PROJECTS) {
                partial = true;
                break;
            }
            if (!project.exists() || !project.isOpen() || project.getLocation() == null) {
                continue;
            }
            boolean maven = ProjectUtils.isMavenProject(project);
            boolean gradle = ProjectUtils.isGradleProject(project);
            if (!("Maven".equals(buildSystem) ? maven : gradle)) {
                continue;
            }
            Path location = project.getLocation().toFile().toPath().toAbsolutePath().normalize();
            if (!location.startsWith(root)) {
                continue;
            }
            if (maven && gradle || !Files.isDirectory(location) || !location.toRealPath().equals(location)) {
                partial = true;
                continue;
            }
            String uri = location.toUri().toString();
            if (projects.size() >= MAX_PROJECTS || !boundedText(uri) || !identities.add(location)) {
                throw new IllegalArgumentException();
            }
            Map<String, Object> record = new LinkedHashMap<>();
            record.put("projectUri", uri);
            record.put("javaNature", project.hasNature(JavaCore.NATURE_ID));
            if (includeBuildRoots) {
                Path build = cachedGradleBuildRoot(project, root);
                if (build == null || !location.startsWith(build)) {
                    partial = true;
                    continue;
                }
                Object moduleIdentity = GradleImportHandler.rootFileKey(location);
                Object buildIdentity = GradleImportHandler.rootFileKey(build);
                associations.add(new CachedGradleAssociation(project, location, build, moduleIdentity, buildIdentity,
                        project.hasNature(JavaCore.NATURE_ID)));
                record.put("buildRootUri", build.toUri().toString());
            }
            projects.add(record);
        }
        checkCancelled(monitor);
        if (!GradleModelHandler.initializedRoot(rootUri, buildSystem).equals(root)) {
            throw new OperationCanceledException();
        }
        for (CachedGradleAssociation association : associations) {
            checkCancelled(monitor);
            GradleImportHandler.requireRootIdentity(association.module(), association.moduleIdentity());
            GradleImportHandler.requireRootIdentity(association.build(), association.buildIdentity());
            if (!matches(association.project(), association.module(), "Gradle")
                    || !association.build().equals(cachedGradleBuildRoot(association.project(), root))
                    || association.javaNature() != association.project().hasNature(JavaCore.NATURE_ID)) {
                throw new OperationCanceledException();
            }
        }
        projects.sort(Comparator.comparing(project -> (String) project.get("projectUri")));
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("version", 1);
        response.put("rootUri", canonicalRootUri);
        response.put("projects", projects);
        if (includeBuildRoots) {
            response.put("cachedOnly", true);
            response.put("readOnly", true);
        }
        response.put("partial", partial);
        response.put("fullSynchronizationKnown", false);
        return response;
    }

    static boolean matches(IProject project, Path module, String buildSystem) {
        if (!project.exists() || !project.isOpen() || project.getLocation() == null) {
            return false;
        }
        Path location = project.getLocation().toFile().toPath().toAbsolutePath().normalize();
        try {
            if (!module.toRealPath().equals(module)) {
                return false;
            }
        } catch (IOException unavailable) {
            return false;
        }
        boolean maven = ProjectUtils.isMavenProject(project);
        boolean gradle = ProjectUtils.isGradleProject(project);
        return location.equals(module) && ("Maven".equals(buildSystem)
                ? maven && !gradle : gradle && !maven);
    }

    static void clearDiagnostics(IProject project, Path root, IProgressMonitor monitor) throws CoreException {
        var markers = project.findMarkers(null, true, IResource.DEPTH_INFINITE);
        if (markers.length > MAX_MARKERS) {
            throw new IllegalArgumentException();
        }
        Set<String> uris = new LinkedHashSet<>();
        for (var marker : markers) {
            checkCancelled(monitor);
            URI uri = marker.getResource().getLocationURI();
            if (uri == null || !"file".equals(uri.getScheme())) {
                continue;
            }
            Path path;
            try {
                path = Path.of(uri).toRealPath();
            } catch (IOException failure) {
                throw new IllegalArgumentException();
            }
            // Linked resources outside this root must not disclose another host selection.
            if (!path.startsWith(root)) {
                continue;
            }
            String text = path.toUri().toString();
            if (new File(uri).isDirectory() && Platform.OS_WIN32.equals(Platform.getOS()) && !text.endsWith("/")) {
                text += "/";
            }
            if (!boundedText(text)) {
                throw new IllegalArgumentException();
            }
            uris.add(ResourceUtils.toClientUri(text));
        }
        for (String uri : uris) {
            checkCancelled(monitor);
            JavaLanguageServerPlugin.getProjectsManager().getConnection().publishDiagnostics(
                    new PublishDiagnosticsParams(uri, Collections.emptyList()));
        }
    }

    private static boolean boundedText(String value) {
        return !value.isBlank() && value.length() <= MAX_TEXT && value.chars().noneMatch(Character::isISOControl);
    }

    private static void checkCancelled(IProgressMonitor monitor) {
        if (monitor.isCanceled()) {
            throw new OperationCanceledException();
        }
    }
}
