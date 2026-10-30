package dev.made.groovy;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.InputStream;
import java.io.IOException;
import java.lang.reflect.Method;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.LinkOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.maven.model.Model;
import org.eclipse.core.resources.IProject;
import org.eclipse.core.resources.IWorkspace;
import org.eclipse.core.resources.IWorkspaceRunnable;
import org.eclipse.core.resources.ResourcesPlugin;
import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.core.runtime.OperationCanceledException;
import org.eclipse.core.runtime.jobs.ISchedulingRule;
import org.eclipse.core.runtime.jobs.Job;
import org.eclipse.jdt.ls.core.internal.EventNotification;
import org.eclipse.jdt.ls.core.internal.EventType;
import org.eclipse.jdt.ls.core.internal.IDelegateCommandHandler;
import org.eclipse.jdt.ls.core.internal.JavaLanguageServerPlugin;
import org.eclipse.jdt.ls.core.internal.ProjectUtils;
import org.eclipse.jdt.ls.core.internal.managers.BasicFileDetector;
import org.eclipse.m2e.core.MavenPlugin;
import org.eclipse.m2e.core.project.IMavenProjectImportResult;
import org.eclipse.m2e.core.project.MavenProjectInfo;
import org.eclipse.m2e.core.project.ProjectImportConfiguration;

public final class MavenImportHandler implements IDelegateCommandHandler {
    private static final int MAX_POMS = 32;
    private static final int MAX_REACTOR_POMS = 128;
    private static final int MAX_REACTOR_BYTES = 8 * 1024 * 1024;
    private static final int MAX_TEXT = 4096;
    private static final int MAX_POM_BYTES = 1024 * 1024;

    @Override
    public Object executeCommand(String command, List<Object> arguments, IProgressMonitor monitor) throws Exception {
        if ("groovy.maven.defaultImportRoots".equals(command)) {
            return defaultImportRoots(arguments, monitor);
        }
        if (monitor.isCanceled()) {
            throw new IllegalStateException("Selected Maven import cancelled. Refresh before retrying.");
        }
        try {
            boolean masked = "groovy.maven.importReactorWithMasks".equals(command);
            boolean reactor = masked || "groovy.maven.importReactor".equals(command);
            if ((!reactor && !"groovy.maven.importSelected".equals(command)) || arguments == null || arguments.size() != 1
                    || !(arguments.get(0) instanceof Map<?, ?> request)
                    || !request.keySet().equals(masked ? Set.of("rootUri", "pomUri", "excludedProjectUris", "offline", "pomMasks")
                            : reactor ? Set.of("rootUri", "pomUri", "excludedProjectUris", "offline")
                            : Set.of("rootUri", "pomUris", "offline"))
                    || !(request.get("rootUri") instanceof String rootUri) || rootUri.length() > MAX_TEXT
                    || !Boolean.TRUE.equals(request.get("offline"))) {
                throw new IllegalArgumentException();
            }
            PomMasks masks = masked ? PomMasks.fromRequest(request.get("pomMasks")) : null;
            Path root = GradleModelHandler.initializedRoot(rootUri, "Maven");
            if (!MavenPlugin.getMavenConfiguration().isOffline()) {
                throw new IllegalStateException("Selected Maven import requires the host Maven offline setting. Check settings and retry.");
            }
            Set<Path> poms = new LinkedHashSet<>();
            Set<Path> excluded = new LinkedHashSet<>();
            Path reactorPom = null;
            if (reactor) {
                reactorPom = selectedPom(request.get("pomUri"), root);
                if (!(request.get("excludedProjectUris") instanceof List<?> exclusions) || exclusions.size() > MAX_REACTOR_POMS) {
                    throw new IllegalArgumentException();
                }
                for (Object value : exclusions) {
                    checkCancelled(monitor);
                    Path directory = localPath(value).toRealPath();
                    if (!directory.startsWith(root) || !Files.isDirectory(directory) || !excluded.add(directory)) {
                        throw new IllegalArgumentException();
                    }
                }
                if (excluded.stream().anyMatch(reactorPom.getParent()::startsWith)) {
                    throw new IllegalArgumentException();
                }
                poms.add(reactorPom);
            } else {
                if (!(request.get("pomUris") instanceof List<?> selected) || selected.isEmpty() || selected.size() > MAX_POMS) {
                    throw new IllegalArgumentException();
                }
                for (Object value : selected) {
                    checkCancelled(monitor);
                    if (!poms.add(selectedPom(value, root))) {
                        throw new IllegalArgumentException();
                    }
                }
            }
            Map<Path, MavenProjectInfo> selectedProjects = new LinkedHashMap<>();
            Map<Path, byte[]> selectedInputs = new LinkedHashMap<>();
            List<Path> pending = new ArrayList<>(poms);
            int reactorBytes = 0;
            for (int index = 0; index < pending.size(); index++) {
                checkCancelled(monitor);
                Path pom = pending.get(index);
                currentProject(pom, monitor);
                byte[] bytes = readPom(pom);
                if (reactor && (reactorBytes += bytes.length) > MAX_REACTOR_BYTES) {
                    throw new IllegalArgumentException();
                }
                Model model = MavenPlugin.getMavenModelManager().readMavenModel(new ByteArrayInputStream(bytes));
                if (model == null || model.getArtifactId() == null) {
                    throw new IllegalArgumentException();
                }
                model.setPomFile(pom.toFile());
                selectedProjects.put(pom, new MavenProjectInfo("pom.xml", pom.toFile(), model, null));
                selectedInputs.put(pom, bytes);
                if (reactor) {
                    // Effective modules come from the host Maven context, including its active profiles and settings.
                    var execution = MavenPlugin.getMaven().createExecutionContext();
                    execution.getExecutionRequest().setOffline(true).setUpdateSnapshots(false);
                    var effective = execution.execute(
                            (context, progress) -> MavenPlugin.getMaven().readProject(pom.toFile(), progress), monitor);
                    if (effective == null || effective.getModules().size() > MAX_REACTOR_POMS
                            || !Arrays.equals(readPom(pom), bytes)) {
                        throw new IllegalArgumentException();
                    }
                    for (String module : effective.getModules()) {
                        checkCancelled(monitor);
                        if (module == null || module.isBlank() || module.length() > MAX_TEXT) {
                            throw new IllegalArgumentException();
                        }
                        Path declaration = pom.getParent().resolve(module).normalize();
                        if (!declaration.startsWith(root)) {
                            throw new IllegalArgumentException();
                        }
                        Path directory = declaration.getFileName() != null && "pom.xml".equals(declaration.getFileName().toString())
                                ? declaration.getParent() : declaration;
                        if (excluded.stream().anyMatch(directory::startsWith)) {
                            continue;
                        }
                        Path child = selectedPom((Files.isDirectory(declaration) ? declaration.resolve("pom.xml") : declaration).toUri().toString(), root);
                        if (excluded.stream().anyMatch(child.getParent()::startsWith)) {
                            continue;
                        }
                        if (poms.size() >= MAX_REACTOR_POMS || !poms.add(child)) {
                            throw new IllegalArgumentException();
                        }
                        pending.add(child);
                    }
                }
            }
            Set<Path> ignored = new LinkedHashSet<>();
            if (masks != null) {
                for (Path pom : poms) {
                    if (masks.matches(portableHostPath(pom), monitor)) {
                        ignored.add(pom);
                    }
                }
                poms.removeAll(ignored);
            }
            checkCancelled(monitor);
            var manager = MavenPlugin.getProjectConfigurationManager();
            ISchedulingRule current = Job.getJobManager().currentRule();
            ISchedulingRule rule = masked ? ResourcesPlugin.getWorkspace().getRoot() : manager.getRule();
            if (current != null && current.contains(rule)) {
                rule = null;
            }
            List<IMavenProjectImportResult> results = new ArrayList<>();
            Set<Path> missing = new LinkedHashSet<>();
            Set<Path> imported = new LinkedHashSet<>();
            List<URI> moduleUris = new ArrayList<>();
            // The JDTLS scanner can mistake an unlinked aggregate-only module for its imported ancestor.
            ResourcesPlugin.getWorkspace().run((IWorkspaceRunnable) progress -> {
                checkCancelled(progress);
                List<MavenProjectInfo> missingProjects = new ArrayList<>();
                List<IProject> existingProjects = new ArrayList<>();
                try {
                    if (!GradleModelHandler.initializedRoot(rootUri, "Maven").equals(root)
                            || !MavenPlugin.getMavenConfiguration().isOffline()) {
                        throw new OperationCanceledException();
                    }
                    for (Path directory : excluded) {
                        if (!directory.toRealPath().equals(directory) || !Files.isDirectory(directory)) {
                            throw new OperationCanceledException();
                        }
                    }
                    for (Map.Entry<Path, byte[]> input : selectedInputs.entrySet()) {
                        checkCancelled(progress);
                        if (!Arrays.equals(readPom(input.getKey()), input.getValue())) {
                            throw new OperationCanceledException();
                        }
                        currentProject(input.getKey(), progress);
                    }
                    List<String> unlink = new ArrayList<>();
                    for (Path pom : ignored) {
                        checkCancelled(progress);
                        if (currentProject(pom, progress) != null) {
                            unlink.add(pom.getParent().toUri().toString());
                        }
                    }
                    if (!unlink.isEmpty()) {
                        try {
                            new ProjectUnlinkHandler().executeCommand("groovy.maven.unlinkSelected", List.of(
                                    Map.of("rootUri", rootUri, "projectUris", unlink)), progress);
                        } catch (OperationCanceledException cancelled) {
                            throw cancelled;
                        } catch (Exception failure) {
                            throw new IllegalStateException("Masked Maven unlink may be partial; refresh before retrying.");
                        }
                    }
                    for (Path pom : poms) {
                        checkCancelled(progress);
                        if (!Arrays.equals(readPom(pom), selectedInputs.get(pom))) {
                            throw new OperationCanceledException();
                        }
                        IProject project = currentProject(pom, progress);
                        if (project == null) {
                            missing.add(pom);
                            missingProjects.add(selectedProjects.get(pom));
                        } else {
                            existingProjects.add(project);
                        }
                    }
                } catch (IOException changedInputs) {
                    throw new OperationCanceledException();
                }
                if (!missingProjects.isEmpty()) {
                    results.addAll(manager.importProjects(missingProjects, new ProjectImportConfiguration(), progress));
                }
                for (IProject project : existingProjects) {
                    checkCancelled(progress);
                    // The pinned per-project API uses host offline mode and does not force dependency updates.
                    manager.updateProjectConfiguration(project, progress);
                }
                for (Path pom : poms) {
                    checkCancelled(progress);
                    IProject project = currentProject(pom, progress);
                    var facade = project == null ? null : MavenPlugin.getMavenProjectRegistry().getProject(project);
                    if (facade == null || facade.getMavenProject() == null || facade.isStale()) {
                        throw new IllegalArgumentException();
                    }
                }
            }, rule, IWorkspace.AVOID_UPDATE, monitor);
            checkCancelled(monitor);
            for (IMavenProjectImportResult result : results) {
                checkCancelled(monitor);
                if (result.getProject() == null || result.getProject().getLocation() == null) {
                    throw new IllegalArgumentException();
                }
                Path module = result.getProject().getLocation().toFile().toPath().toRealPath();
                Path pom = module.resolve("pom.xml");
                if (!missing.contains(pom) || !imported.add(pom)) {
                    throw new IllegalArgumentException();
                }
                moduleUris.add(module.toUri());
            }
            if (!imported.equals(missing)) {
                throw new IllegalArgumentException();
            }
            if (!GradleModelHandler.initializedRoot(rootUri, "Maven").equals(root)
                    || !MavenPlugin.getMavenConfiguration().isOffline()) {
                throw new OperationCanceledException();
            }
            for (Path directory : excluded) {
                if (!directory.toRealPath().equals(directory) || !Files.isDirectory(directory)) {
                    throw new OperationCanceledException();
                }
            }
            for (Map.Entry<Path, byte[]> input : selectedInputs.entrySet()) {
                checkCancelled(monitor);
                IProject project = currentProject(input.getKey(), monitor);
                if (!Arrays.equals(readPom(input.getKey()), input.getValue())
                        || (ignored.contains(input.getKey()) ? project != null : project == null)) {
                    throw new OperationCanceledException();
                }
            }
            if (!moduleUris.isEmpty()) {
                JavaLanguageServerPlugin.getProjectsManager().getConnection().sendEventNotification(
                        new EventNotification().withType(EventType.ProjectsImported).withData(moduleUris));
            }
            Map<String, Object> receipt = new LinkedHashMap<>();
            receipt.put("version", masked ? 2 : 1);
            receipt.put("rootUri", root.toUri().toString());
            receipt.put("importedPomUris", poms.stream().map(pom -> pom.toUri().toString()).toList());
            if (reactor) {
                receipt.put("reactorPomUri", reactorPom.toUri().toString());
                receipt.put("excludedProjectUris", excluded.stream().map(directory -> directory.toUri().toString()).toList());
            }
            if (masked) {
                receipt.put("ignoredPomUris", ignored.stream().map(pom -> pom.toUri().toString()).toList());
            }
            receipt.put("selectedImportPhaseComplete", true);
            receipt.put("fullSynchronizationKnown", false);
            return receipt;
        } catch (OperationCanceledException cancelled) {
            throw new IllegalStateException("Selected Maven import cancelled. Refresh before retrying.");
        } catch (Exception failure) {
            if (monitor.isCanceled()) {
                throw new IllegalStateException("Selected Maven import cancelled. Refresh before retrying.");
            }
            throw new IllegalStateException("Selected Maven import failed or may be partial. Check project selection, host offline settings and cache; refresh before retrying.");
        }
    }

    private static String portableHostPath(Path pom) {
        String path = pom.toString();
        return File.separatorChar == '\\' ? path.replace('\\', '/') : path;
    }

    static final class PomMasks {
        private final List<int[]> patterns;
        private long remainingCells = 64L * 1024 * 1024;

        private PomMasks(List<int[]> patterns) {
            this.patterns = patterns;
        }

        static PomMasks fromRequest(Object value) {
            if (!(value instanceof Map<?, ?> contract)
                    || !contract.keySet().equals(Set.of("version", "syntax", "pathFormat", "patterns"))
                    || !(contract.get("version") instanceof Number version) || version.doubleValue() != 1
                    || !"idea-mask-v1".equals(contract.get("syntax"))
                    || !"portable-absolute-v1".equals(contract.get("pathFormat"))
                    || !(contract.get("patterns") instanceof List<?> masks) || masks.size() > 128) {
                throw new IllegalArgumentException();
            }
            Set<String> distinct = new LinkedHashSet<>();
            List<int[]> patterns = new ArrayList<>();
            int totalBytes = 0;
            for (Object mask : masks) {
                if (!(mask instanceof String text) || text.isEmpty() || text.length() > 4096 || !validText(text)
                        || text.getBytes(StandardCharsets.UTF_8).length > 4096 || !distinct.add(text)) {
                    throw new IllegalArgumentException();
                }
                totalBytes += text.getBytes(StandardCharsets.UTF_8).length;
                if (totalBytes > 65536) {
                    throw new IllegalArgumentException();
                }
                // The reference split drops empty backslash pieces, rather than quoting wildcards.
                String translated = String.join("\\", Arrays.stream(text.split("\\\\"))
                        .filter(piece -> !piece.isEmpty()).toList());
                patterns.add(translated.codePoints().toArray());
            }
            return new PomMasks(patterns);
        }

        boolean matches(String path, IProgressMonitor monitor) {
            validatePath(path, File.separatorChar == '\\');
            int[] points = path.codePoints().toArray();
            for (int[] pattern : patterns) {
                checkCancelled(monitor);
                boolean[] previous = new boolean[points.length + 1];
                boolean[] next = new boolean[points.length + 1];
                previous[0] = true;
                for (int token : pattern) {
                    checkCancelled(monitor);
                    remainingCells -= points.length + 1L;
                    if (remainingCells < 0) {
                        throw new IllegalArgumentException();
                    }
                    next[0] = token == '*' && previous[0];
                    for (int index = 1; index <= points.length; index++) {
                        if ((index & 255) == 0) {
                            checkCancelled(monitor);
                        }
                        next[index] = token == '*' ? previous[index] || next[index - 1]
                                : previous[index - 1] && (token == '?' || token == points[index - 1]);
                    }
                    boolean[] temporary = previous;
                    previous = next;
                    next = temporary;
                }
                if (previous[points.length]) {
                    return true;
                }
            }
            return false;
        }

        static void validatePath(String path, boolean windows) {
            if (path.isEmpty() || path.length() > 4096 || !validText(path)
                    || path.getBytes(StandardCharsets.UTF_8).length > 4096 || path.endsWith("/")) {
                throw new IllegalArgumentException();
            }
            String relative;
            if (windows) {
                if (path.indexOf('\\') >= 0) {
                    throw new IllegalArgumentException();
                }
                if (path.startsWith("//")) {
                    relative = path.substring(2);
                    if (relative.split("/", -1).length < 3) {
                        throw new IllegalArgumentException();
                    }
                } else if (path.length() >= 3 && ((path.charAt(0) >= 'A' && path.charAt(0) <= 'Z')
                        || (path.charAt(0) >= 'a' && path.charAt(0) <= 'z')) && path.startsWith(":/", 1)) {
                    relative = path.substring(3);
                } else {
                    throw new IllegalArgumentException();
                }
            } else {
                if (!path.startsWith("/") || path.startsWith("//")) {
                    throw new IllegalArgumentException();
                }
                relative = path.substring(1);
            }
            for (String piece : relative.split("/", -1)) {
                if (piece.isEmpty() || piece.equals(".") || piece.equals("..")
                        || (windows && (piece.endsWith(".") || piece.endsWith(" ")
                            || piece.chars().anyMatch(point -> ":*?\"<>|".indexOf(point) >= 0)))) {
                    throw new IllegalArgumentException();
                }
            }
        }

        private static boolean validText(String text) {
            for (int index = 0; index < text.length(); index++) {
                char character = text.charAt(index);
                if (Character.isISOControl(character) || character == '\u2028' || character == '\u2029') {
                    return false;
                }
                if (Character.isHighSurrogate(character)) {
                    if (++index >= text.length() || !Character.isLowSurrogate(text.charAt(index))) {
                        return false;
                    }
                } else if (Character.isLowSurrogate(character)) {
                    return false;
                }
            }
            return true;
        }
    }

    private static Object defaultImportRoots(List<Object> arguments, IProgressMonitor monitor) {
        try {
            checkCancelled(monitor);
            if (arguments == null || arguments.size() != 1
                    || !(arguments.get(0) instanceof Map<?, ?> request)
                    || !request.keySet().equals(Set.of("rootUri", "candidatePomUris", "cachedOnly"))
                    || !(request.get("rootUri") instanceof String rootUri) || rootUri.length() > MAX_TEXT
                    || !Boolean.TRUE.equals(request.get("cachedOnly"))
                    || !(request.get("candidatePomUris") instanceof List<?> candidates)
                    || candidates.size() > MAX_POMS) {
                throw new IllegalArgumentException();
            }
            Path root = GradleModelHandler.initializedRoot(rootUri, "Maven");
            String canonicalRootUri = root.toUri().toString();
            if (canonicalRootUri.length() > MAX_TEXT) {
                throw new IllegalArgumentException();
            }
            Set<Path> poms = new LinkedHashSet<>();
            for (Object value : candidates) {
                checkCancelled(monitor);
                Path path = localPath(value);
                Path pom = path.toAbsolutePath().normalize();
                if (!pom.equals(path) || !pom.startsWith(root) || !pom.equals(pom.toRealPath())
                        || !Files.isRegularFile(pom, LinkOption.NOFOLLOW_LINKS) || !Files.isReadable(pom)
                        || pom.getFileName() == null || !"pom.xml".equals(pom.getFileName().toString())
                        || !PomMasks.validText(pom.toString()) || Files.size(pom) > MAX_POM_BYTES
                        || !poms.add(pom)) {
                    throw new IllegalArgumentException();
                }
            }
            var preferences = JavaLanguageServerPlugin.getPreferencesManager().getPreferences();
            List<String> configuredExclusions = preferences.getJavaImportExclusions();
            if (configuredExclusions != null && configuredExclusions.size() > MAX_REACTOR_POMS) {
                throw new IllegalArgumentException();
            }
            List<String> exclusions = configuredExclusions == null ? List.of() : List.copyOf(configuredExclusions);
            int exclusionBytes = 0;
            for (String pattern : exclusions) {
                if (pattern.length() > MAX_TEXT || !PomMasks.validText(pattern)) {
                    throw new IllegalArgumentException();
                }
                exclusionBytes += pattern.getBytes(StandardCharsets.UTF_8).length;
            }
            if (exclusionBytes > 64 * 1024) {
                throw new IllegalArgumentException();
            }
            // Automatic Maven import is disabled for saved policy. Reuse the pinned
            // detector predicate without scanning or toggling provider preferences.
            BasicFileDetector detector = new BasicFileDetector(root, "pom.xml")
                    .includeNested(false).addExclusions("**/target");
            Method predicate = GradleModelHandler.detectorPredicate();
            int maximumDepth = GradleModelHandler.detectorDepth(detector);
            boolean hasInclusions = exclusions.stream().anyMatch(pattern -> pattern.startsWith("!"));
            Set<Path> importedMaven = new LinkedHashSet<>();
            Set<Path> projectLocations = new LinkedHashSet<>();
            IProject[] projects = ProjectUtils.getAllProjects();
            if (projects.length > 1024) {
                throw new IllegalArgumentException();
            }
            for (IProject project : projects) {
                checkCancelled(monitor);
                if (project.getLocation() == null) {
                    throw new IllegalArgumentException();
                }
                Path directory = project.getLocation().toFile().toPath().toAbsolutePath().normalize();
                boolean maven = ProjectUtils.isMavenProject(project);
                if (!maven) {
                    String exclusion = directory.toString().replace("\\", "\\\\");
                    exclusionBytes += exclusion.getBytes(StandardCharsets.UTF_8).length;
                    if (exclusion.length() > MAX_TEXT || !PomMasks.validText(exclusion) || exclusionBytes > 64 * 1024) {
                        throw new IllegalArgumentException();
                    }
                    detector.addExclusions(exclusion);
                }
                if (!directory.startsWith(root)) {
                    continue;
                }
                if (!projectLocations.add(directory) || projectLocations.size() > MAX_REACTOR_POMS) {
                    throw new IllegalArgumentException();
                }
                if (maven) {
                    if (!ProjectUnlinkHandler.matches(project, directory, "Maven")) {
                        throw new IllegalArgumentException();
                    }
                    importedMaven.add(directory);
                }
            }
            Set<Path> roots = new LinkedHashSet<>();
            Set<Path> retained = new LinkedHashSet<>();
            for (Path pom : poms) {
                checkCancelled(monitor);
                Path directory = pom.getParent();
                if (importedMaven.contains(directory)) {
                    roots.add(pom);
                    retained.add(pom);
                    continue;
                }
                int relativeDepth = directory.equals(root) ? 0 : root.relativize(directory).getNameCount();
                if (relativeDepth > maximumDepth || GradleModelHandler.detectorExcludes(detector, predicate, directory)) {
                    continue;
                }
                boolean shadowed = false;
                for (Path ancestor = directory.getParent(); ancestor != null && ancestor.startsWith(root); ancestor = ancestor.getParent()) {
                    checkCancelled(monitor);
                    boolean excluded = GradleModelHandler.detectorExcludes(detector, predicate, ancestor);
                    if (excluded && !hasInclusions) {
                        shadowed = true;
                        break;
                    }
                    if (!excluded) {
                        Path parentPom = ancestor.resolve("pom.xml");
                        if (Files.isSymbolicLink(parentPom)) {
                            throw new IllegalArgumentException();
                        }
                        if (Files.isRegularFile(parentPom, LinkOption.NOFOLLOW_LINKS)) {
                            if (!poms.contains(parentPom)) {
                                throw new IllegalArgumentException();
                            }
                            shadowed = true;
                            break;
                        }
                    }
                }
                if (!shadowed) {
                    roots.add(pom);
                }
            }
            checkCancelled(monitor);
            List<String> currentExclusions = JavaLanguageServerPlugin.getPreferencesManager().getPreferences().getJavaImportExclusions();
            if (!exclusions.equals(currentExclusions == null ? List.of() : currentExclusions)
                    || !GradleModelHandler.initializedRoot(rootUri, "Maven").equals(root)) {
                throw new OperationCanceledException();
            }
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("version", 1);
            result.put("rootUri", canonicalRootUri);
            result.put("rootPomUris", roots.stream().map(pom -> pom.toUri().toString()).toList());
            result.put("retainedPomUris", retained.stream().map(pom -> pom.toUri().toString()).toList());
            result.put("cachedOnly", true);
            result.put("readOnly", true);
            result.put("partial", false);
            result.put("fullSynchronizationKnown", false);
            return result;
        } catch (OperationCanceledException cancelled) {
            throw new IllegalStateException("Maven default-root discovery cancelled; no projects were imported. Refresh before retrying.");
        } catch (Exception | LinkageError failure) {
            throw new IllegalStateException("Maven default-root discovery failed; no projects were imported. Check current POMs, initialized host and compatible extension/JDTLS settings, then refresh and retry.");
        }
    }

    private static Path localPath(Object value) throws Exception {
        if (!(value instanceof String text) || text.length() > MAX_TEXT) {
            throw new IllegalArgumentException();
        }
        URI uri = new URI(text);
        if (!"file".equals(uri.getScheme()) || uri.getQuery() != null || uri.getFragment() != null) {
            throw new IllegalArgumentException();
        }
        return Path.of(uri);
    }

    private static Path selectedPom(Object value, Path root) throws Exception {
        Path pom = localPath(value).toRealPath();
        if (!pom.startsWith(root) || !Files.isRegularFile(pom) || pom.getFileName() == null
                || !"pom.xml".equals(pom.getFileName().toString()) || Files.size(pom) > MAX_POM_BYTES) {
            throw new IllegalArgumentException();
        }
        return pom;
    }

    private static byte[] readPom(Path pom) throws IOException {
        if (!pom.toRealPath().equals(pom) || !Files.isRegularFile(pom)) {
            throw new IOException("Selected Maven POM changed");
        }
        try (InputStream input = Files.newInputStream(pom)) {
            byte[] bytes = input.readNBytes(MAX_POM_BYTES + 1);
            if (bytes.length > MAX_POM_BYTES) {
                throw new IOException("Selected Maven POM exceeds its bound");
            }
            return bytes;
        }
    }

    private static IProject currentProject(Path pom, IProgressMonitor monitor) {
        IProject selected = null;
        IProject[] projects = ProjectUtils.getAllProjects();
        if (projects.length > 1024) {
            throw new IllegalArgumentException();
        }
        for (IProject project : projects) {
            checkCancelled(monitor);
            if (project.getLocation() != null && project.getLocation().toFile().toPath().toAbsolutePath().normalize().equals(pom.getParent())) {
                if (selected != null || !ProjectUnlinkHandler.matches(project, pom.getParent(), "Maven")) {
                    throw new IllegalArgumentException();
                }
                selected = project;
            }
        }
        return selected;
    }

    private static void checkCancelled(IProgressMonitor monitor) {
        if (monitor.isCanceled()) {
            throw new OperationCanceledException();
        }
    }
}
