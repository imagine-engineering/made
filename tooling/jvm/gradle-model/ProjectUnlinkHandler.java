package dev.made.groovy;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.eclipse.core.resources.IProject;
import org.eclipse.core.resources.IProjectDescription;
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
import org.eclipse.jdt.core.dom.AST;
import org.eclipse.jdt.core.dom.ASTParser;
import org.eclipse.jdt.core.dom.CompilationUnit;
import org.eclipse.jdt.ls.core.internal.EventNotification;
import org.eclipse.jdt.ls.core.internal.EventType;
import org.eclipse.jdt.ls.core.internal.IDelegateCommandHandler;
import org.eclipse.jdt.ls.core.internal.JavaLanguageServerPlugin;
import org.eclipse.jdt.ls.core.internal.ProjectUtils;
import org.eclipse.jdt.ls.core.internal.ResourceUtils;
import org.eclipse.jdt.ls.core.internal.commands.BuildPathCommand;
import org.eclipse.jdt.ls.core.internal.managers.BasicFileDetector;
import org.eclipse.jdt.ls.core.internal.managers.EclipseProjectImporter;
import org.eclipse.lsp4j.PublishDiagnosticsParams;

public final class ProjectUnlinkHandler implements IDelegateCommandHandler {
    private static final int MAX_PROJECTS = 128;
    private static final int MAX_TEXT = 4096;
    private static final int MAX_MARKERS = 10000;
    private static final int MAX_INSPECTED_PROJECTS = 1024;

    @Override
    public Object executeCommand(String command, List<Object> arguments, IProgressMonitor monitor) throws Exception {
        if ("groovy.java.importPlainProjects".equals(command)) {
            return importPlainProjects(arguments, monitor);
        }
        if ("groovy.java.addLooseSource".equals(command)) {
            return addLooseSource(arguments, monitor);
        }
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

    private static Object importPlainProjects(List<Object> arguments, IProgressMonitor monitor) {
        try {
            checkCancelled(monitor);
            if (arguments == null || arguments.size() != 1 || !(arguments.get(0) instanceof Map<?, ?> request)
                    || !request.keySet().equals(Set.of("rootUri", "descriptionUris"))
                    || !(request.get("rootUri") instanceof String rootUri) || !boundedText(rootUri)
                    || !(request.get("descriptionUris") instanceof List<?> descriptions)
                    || descriptions.size() > MAX_PROJECTS) {
                throw new IllegalArgumentException();
            }
            Path root = GradleModelHandler.initializedRoot(rootUri, "Java");
            Set<Path> candidates = new LinkedHashSet<>();
            for (Object description : descriptions) {
                checkCancelled(monitor);
                if (!(description instanceof String text) || !boundedText(text)) {
                    throw new IllegalArgumentException();
                }
                Path file = Path.of(URI.create(text));
                if (!file.startsWith(root) || !file.equals(file.normalize()) || !file.equals(file.toRealPath())
                        || !IProjectDescription.DESCRIPTION_FILE_NAME.equals(file.getFileName().toString())
                        || !Files.isRegularFile(file, LinkOption.NOFOLLOW_LINKS) || !candidates.add(file)) {
                    throw new IllegalArgumentException();
                }
            }
            IWorkspace workspace = ResourcesPlugin.getWorkspace();
            ISchedulingRule rule = workspace.getRoot();
            ISchedulingRule current = Job.getJobManager().currentRule();
            if (current != null && current.contains(rule)) {
                rule = null;
            }
            Set<Path> imported = new LinkedHashSet<>();
            workspace.run((IWorkspaceRunnable) progress -> {
                try {
                    checkCancelled(progress);
                    if (!GradleModelHandler.initializedRoot(rootUri, "Java").equals(root)) {
                        throw new OperationCanceledException();
                    }
                    BasicFileDetector detector = new BasicFileDetector(root, IProjectDescription.DESCRIPTION_FILE_NAME)
                            .addExclusions("**/bin");
                    var predicate = GradleModelHandler.detectorPredicate();
                    int maximumDepth = GradleModelHandler.detectorDepth(detector);
                    var exclusions = JavaLanguageServerPlugin.getPreferencesManager().getPreferences()
                            .getJavaImportExclusions();
                    boolean hasInclusions = exclusions != null && exclusions.stream().anyMatch(pattern -> pattern.startsWith("!"));
                    Set<Path> existing = projectLocations(workspace);
                    for (Path directory : existing) {
                        detector.addExclusions(directory.toString().replace("\\", "\\\\"));
                    }
                    Map<Path, byte[]> eligible = new LinkedHashMap<>();
                    int descriptionBytes = 0;
                    for (Path file : candidates) {
                        checkCancelled(progress);
                        Path directory = file.getParent();
                        if (existing.contains(directory) || (directory.equals(root) ? 0 : root.relativize(directory).getNameCount()) > maximumDepth
                                || GradleModelHandler.detectorExcludes(detector, predicate, directory)) {
                            continue;
                        }
                        boolean excluded = false;
                        for (Path ancestor = directory.getParent(); !hasInclusions && ancestor != null && ancestor.startsWith(root);
                                ancestor = ancestor.getParent()) {
                            if (existing.contains(ancestor) || GradleModelHandler.detectorExcludes(detector, predicate, ancestor)) {
                                excluded = true;
                                break;
                            }
                        }
                        if (excluded) {
                            continue;
                        }
                        Path classpath = directory.resolve(".classpath");
                        if (!Files.isRegularFile(classpath, LinkOption.NOFOLLOW_LINKS) || !classpath.equals(classpath.toRealPath())) {
                            continue;
                        }
                        byte[] bytes = GradleImportHandler.descriptionBytes(file);
                        descriptionBytes += bytes.length;
                        if (descriptionBytes > 8 * 1024 * 1024) {
                            throw new IllegalArgumentException();
                        }
                        IProjectDescription description = workspace.loadProjectDescription(new ByteArrayInputStream(bytes));
                        if (description.hasNature(JavaCore.NATURE_ID)
                                && !description.hasNature("org.eclipse.m2e.core.maven2Nature")
                                && !description.hasNature("org.eclipse.buildship.core.gradleprojectnature")) {
                            eligible.put(file, bytes);
                        }
                    }
                    for (var entry : eligible.entrySet()) {
                        checkCancelled(progress);
                        Path classpath = entry.getKey().getParent().resolve(".classpath");
                        if (!Files.isRegularFile(classpath, LinkOption.NOFOLLOW_LINKS) || !classpath.equals(classpath.toRealPath())
                                || !Arrays.equals(entry.getValue(), GradleImportHandler.descriptionBytes(entry.getKey()))) {
                            throw new OperationCanceledException();
                        }
                    }
                    if (!eligible.isEmpty()) {
                        EclipseProjectImporter importer = new EclipseProjectImporter();
                        importer.initialize(root.toFile());
                        importer.setDirectories(eligible.keySet().stream().map(Path::getParent).toList());
                        importer.importToWorkspace(progress);
                        for (Path file : eligible.keySet()) {
                            Path directory = file.getParent();
                            if (!plainJavaProjectExists(workspace, directory)) {
                                throw new IllegalStateException();
                            }
                            imported.add(directory);
                        }
                    }
                } catch (Exception failure) {
                    if (progress.isCanceled() || failure instanceof OperationCanceledException) {
                        throw new OperationCanceledException();
                    }
                    throw new IllegalStateException();
                }
            }, rule, IWorkspace.AVOID_UPDATE, monitor);
            checkCancelled(monitor);
            return Map.of("version", 1, "rootUri", root.toUri().toString(),
                    "importedProjectUris", imported.stream().map(path -> path.toUri().toString()).toList());
        } catch (OperationCanceledException cancelled) {
            throw new IllegalStateException("Plain Java startup import cancelled and may be partial. Refresh before retrying.");
        } catch (Exception | LinkageError failure) {
            throw new IllegalStateException("Plain Java startup import failed or may be partial. Check initialized folders and current Eclipse metadata; refresh before retrying.");
        }
    }

    private static Object addLooseSource(List<Object> arguments, IProgressMonitor monitor) {
        try {
            checkCancelled(monitor);
            if (arguments == null || arguments.size() != 1 || !(arguments.get(0) instanceof Map<?, ?> request)
                    || !request.keySet().equals(Set.of("rootUri", "sourceUri"))
                    || !(request.get("rootUri") instanceof String rootUri) || !boundedText(rootUri)
                    || !(request.get("sourceUri") instanceof String sourceUri) || !boundedText(sourceUri)) {
                throw new IllegalArgumentException();
            }
            Path root = GradleModelHandler.initializedRoot(rootUri, "Java");
            Path source = Path.of(URI.create(sourceUri));
            if (!source.startsWith(root) || !source.equals(source.toRealPath())
                    || !source.getFileName().toString().endsWith(".java")
                    || !Files.isRegularFile(source, LinkOption.NOFOLLOW_LINKS) || Files.size(source) > 1024 * 1024) {
                throw new IllegalArgumentException();
            }
            IWorkspace workspace = ResourcesPlugin.getWorkspace();
            ISchedulingRule rule = workspace.getRoot();
            ISchedulingRule current = Job.getJobManager().currentRule();
            if (current != null && current.contains(rule)) {
                rule = null;
            }
            var applied = new LinkedHashSet<String>();
            workspace.run(progress -> {
                checkCancelled(progress);
                if (!GradleModelHandler.initializedRoot(rootUri, "Java").equals(root)) {
                    throw new OperationCanceledException();
                }
                var preferences = JavaLanguageServerPlugin.getPreferencesManager().getPreferences();
                if (preferences.getInvisibleProjectSourcePaths() != null) {
                    return;
                }
                try {
                    if (!source.equals(source.toRealPath()) || Files.size(source) > 1024 * 1024) {
                        throw new OperationCanceledException();
                    }
                    Path directory = looseSourceDirectory(source);
                    if (directory == null) {
                        return;
                    }
                    if (!directory.startsWith(root) || directory.equals(root) || !directory.equals(directory.toRealPath())) {
                        return;
                    }
                    if (projectLocations(workspace).stream().anyMatch(
                            project -> directory.startsWith(project) || project.startsWith(directory))) {
                        return;
                    }
                    for (Path ancestor = source.getParent(); ancestor != null && ancestor.startsWith(root);
                            ancestor = ancestor.getParent()) {
                        if (hasProjectDescriptor(ancestor)) {
                            return;
                        }
                    }
                    checkCancelled(progress);
                    if (!BuildPathCommand.addToSourcePath(directory.toUri().toString()).status) {
                        throw new IllegalStateException();
                    }
                    applied.add(directory.toUri().toString());
                } catch (IOException failure) {
                    throw new IllegalStateException(failure);
                }
            }, rule, IWorkspace.AVOID_UPDATE, monitor);
            checkCancelled(monitor);
            return Map.of("version", 1, "sourceRoots", List.copyOf(applied));
        } catch (OperationCanceledException cancelled) {
            throw new IllegalStateException("Loose Java source setup was cancelled. Refresh before retrying.");
        } catch (Exception | LinkageError failure) {
            throw new IllegalStateException("Loose Java source setup failed. Check initialized folders and current project metadata.");
        }
    }

    private static Path looseSourceDirectory(Path source) throws IOException {
        try (var input = Files.newInputStream(source, LinkOption.NOFOLLOW_LINKS)) {
            byte[] bytes = input.readNBytes(1024 * 1024 + 1);
            if (bytes.length > 1024 * 1024) {
                throw new IOException("Java source exceeds the size limit");
            }
            ASTParser parser = ASTParser.newParser(AST.getJLSLatest());
            parser.setIgnoreMethodBodies(true);
            parser.setSource(new String(bytes, java.nio.charset.StandardCharsets.UTF_8).toCharArray());
            var unit = (CompilationUnit) parser.createAST(null);
            Path parent = source.getParent();
            if (unit.getPackage() == null) {
                return parent;
            }
            String[] segments = unit.getPackage().getName().getFullyQualifiedName().split("\\.");
            for (int index = segments.length - 1; index >= 0; index--) {
                if (parent == null || parent.getFileName() == null
                        || !parent.getFileName().toString().equals(segments[index])) {
                    return null;
                }
                parent = parent.getParent();
            }
            return parent;
        }
    }

    private static boolean hasProjectDescriptor(Path directory) {
        return List.of("pom.xml", "settings.gradle", "settings.gradle.kts", "build.gradle",
                "build.gradle.kts", ".project", ".classpath").stream()
                .anyMatch(name -> !Files.notExists(directory.resolve(name), LinkOption.NOFOLLOW_LINKS));
    }

    private static Set<Path> projectLocations(IWorkspace workspace) {
        IProject[] projects = workspace.getRoot().getProjects();
        if (projects.length > MAX_INSPECTED_PROJECTS) {
            throw new IllegalStateException();
        }
        Set<Path> paths = new LinkedHashSet<>();
        for (IProject project : projects) {
            if (project.getLocation() != null) {
                paths.add(project.getLocation().toFile().toPath().toAbsolutePath().normalize());
            }
        }
        return paths;
    }

    private static boolean plainJavaProjectExists(IWorkspace workspace, Path path) throws CoreException {
        for (IProject project : workspace.getRoot().getProjects()) {
            if (project.isOpen() && project.getLocation() != null && path.equals(project.getLocation().toFile().toPath())
                    && project.hasNature(JavaCore.NATURE_ID) && !ProjectUtils.isMavenProject(project)
                    && !ProjectUtils.isGradleProject(project)) {
                return true;
            }
        }
        return false;
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
