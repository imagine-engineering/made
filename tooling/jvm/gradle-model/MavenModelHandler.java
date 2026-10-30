package dev.made.groovy;

import java.io.InputStream;
import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.ArrayDeque;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.HashSet;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Set;
import java.util.zip.ZipFile;
import javax.xml.XMLConstants;
import javax.xml.parsers.DocumentBuilderFactory;

import com.google.gson.Gson;
import org.apache.maven.RepositoryUtils;
import org.eclipse.aether.DefaultRepositorySystemSession;
import org.eclipse.aether.RepositorySystem;
import org.eclipse.aether.artifact.Artifact;
import org.eclipse.aether.collection.CollectRequest;
import org.eclipse.aether.collection.CollectResult;
import org.eclipse.aether.collection.DependencyCollectionException;
import org.eclipse.aether.collection.DependencyCollectionContext;
import org.eclipse.aether.collection.DependencyTraverser;
import org.eclipse.aether.graph.Dependency;
import org.eclipse.aether.graph.DependencyNode;
import org.eclipse.aether.util.graph.manager.DependencyManagerUtils;
import org.eclipse.aether.util.graph.transformer.ConflictResolver;
import org.apache.maven.lifecycle.DefaultLifecycles;
import org.apache.maven.lifecycle.Lifecycle;
import org.apache.maven.model.Plugin;
import org.apache.maven.model.Profile;
import org.apache.maven.project.MavenProject;
import org.eclipse.core.runtime.IProgressMonitor;
import org.apache.maven.project.ProjectSorter;
import org.apache.maven.project.DuplicateProjectException;
import org.codehaus.plexus.util.dag.CycleDetectedException;
import org.eclipse.jdt.ls.core.internal.IDelegateCommandHandler;
import org.eclipse.m2e.core.MavenPlugin;
import org.eclipse.m2e.core.project.IMavenProjectFacade;
import org.w3c.dom.Element;
import org.w3c.dom.Node;

public final class MavenModelHandler implements IDelegateCommandHandler {
    private static final int MAX_TEXT = 4096;
    private static final int MAX_XML = 1024 * 1024;

    @Override
    public Object executeCommand(String command, List<Object> arguments, IProgressMonitor monitor) throws Exception {
        if ("groovy.maven.dependencyGraph".equals(command)) {
            return dependencyGraph(arguments, monitor);
        }
        boolean parameters = "groovy.maven.goalDetails".equals(command);
        boolean details = "groovy.maven.projectDetails".equals(command) || parameters;
        if ((!"groovy.maven.projects".equals(command) && !details) || arguments == null || arguments.size() != 1
                || !(arguments.get(0) instanceof Map<?, ?>)) {
            throw new IllegalArgumentException("Invalid Groovy Maven model request.");
        }
        Map<?, ?> request = (Map<?, ?>) arguments.get(0);
        if (!(request.get("rootUri") instanceof String) || ((String) request.get("rootUri")).length() > MAX_TEXT
                || !Boolean.TRUE.equals(request.get("cachedOnly"))) {
            throw new IllegalArgumentException("An initialized Maven project file URI and explicit cached-only mode are required.");
        }
        if (monitor.isCanceled()) {
            throw new IllegalStateException("Maven metadata discovery cancelled. Refresh to retry.");
        }
        try {
            Path root = GradleModelHandler.initializedRoot((String) request.get("rootUri"), "Maven");
            return new Reader().projects(root, monitor, details, parameters);
        } catch (IllegalArgumentException invalid) {
            throw new IllegalArgumentException("The Maven root is unavailable or not an initialized project folder.");
        } catch (Exception failure) {
            if (monitor.isCanceled()) {
                throw new IllegalStateException("Maven metadata discovery cancelled. Refresh to retry.");
            }
            throw new IllegalStateException("Maven cached metadata is unavailable or invalid. Check project import, offline cache and settings; then refresh.");
        }
    }

    private Object dependencyGraph(List<Object> arguments, IProgressMonitor monitor) throws Exception {
        if (monitor.isCanceled()) {
            throw new IllegalStateException("Maven dependency collection cancelled. Refresh to retry.");
        }
        try {
            if (arguments == null || arguments.size() != 1 || !(arguments.get(0) instanceof Map<?, ?> request)
                    || !request.keySet().equals(Set.of("rootUri", "pomUri", "cachedOnly", "offline"))
                    || !(request.get("rootUri") instanceof String rootUri) || rootUri.length() > MAX_TEXT
                    || !(request.get("pomUri") instanceof String pomUri) || pomUri.length() > MAX_TEXT
                    || !Boolean.TRUE.equals(request.get("cachedOnly")) || !Boolean.TRUE.equals(request.get("offline"))) {
                throw new IllegalArgumentException();
            }
            Path root = GradleModelHandler.initializedRoot(rootUri, "Maven");
            java.net.URI uri = new java.net.URI(pomUri);
            if (!"file".equals(uri.getScheme()) || uri.getQuery() != null || uri.getFragment() != null
                    || uri.getAuthority() != null && !uri.getAuthority().isEmpty()) {
                throw new IllegalArgumentException();
            }
            Path pom = Path.of(uri).toRealPath();
            if (!pom.startsWith(root) || !Files.isRegularFile(pom)) {
                throw new IllegalArgumentException();
            }
            return new Reader().dependencyGraph(root, pom, monitor);
        } catch (IllegalArgumentException invalid) {
            throw new IllegalArgumentException("Maven dependency collection requires an initialized root, current cached POM and explicit offline mode.");
        } catch (Exception failure) {
            if (monitor.isCanceled()) {
                throw new IllegalStateException("Maven dependency collection cancelled. Refresh to retry.");
            }
            throw new IllegalStateException("Maven dependency metadata is unavailable, stale or exceeds limits. Check import, offline cache and settings; then refresh.");
        }
    }

    private static final class Reader {
        private int responseCharacters;
        private int responseCharacterLimit = 1024 * 1024;

        private Map<String, Object> dependencyGraph(Path root, Path pom, IProgressMonitor monitor) throws Exception {
            CollectionBudget budget = new CollectionBudget(monitor);
            IMavenProjectFacade facade = graphFacade(root, pom, budget);
            MavenProject project = facade.getMavenProject();
            byte[] inputs = graphInputs(pom);
            if (!MavenPlugin.getMavenConfiguration().isOffline()) {
                throw new IllegalStateException();
            }
            Map<String, Object> result = MavenPlugin.getMavenProjectRegistry().execute(facade, (context, progress) -> {
                budget.check();
                if (context.getSession().getCurrentProject() != project || facade.isStale()) {
                    throw new IllegalStateException();
                }
                var session = new DefaultRepositorySystemSession(context.getRepositorySession());
                session.setOffline(true).setUpdatePolicy("never");
                session.setArtifactDescriptorPolicy(new org.eclipse.aether.util.repository.SimpleArtifactDescriptorPolicy(false, false));
                session.setConfigProperty(ConflictResolver.CONFIG_PROP_VERBOSE, ConflictResolver.Verbosity.FULL);
                session.setConfigProperty(DependencyManagerUtils.CONFIG_PROP_VERBOSE, true);
                session.setDependencyTraverser(new BoundedTraverser(session.getDependencyTraverser(), budget, 0));
                CollectRequest request = new CollectRequest();
                request.setRootArtifact(RepositoryUtils.toArtifact(project.getArtifact()));
                request.setRequestContext("project");
                request.setRepositories(project.getRemoteProjectRepositories());
                if (project.getDependencies().size() > 4096 || project.getRemoteProjectRepositories().size() > 128) {
                    throw new IllegalStateException();
                }
                for (var dependency : project.getDependencies()) {
                    budget.check();
                    graphDependencyInput(dependency);
                    request.addDependency(RepositoryUtils.toDependency(dependency, session.getArtifactTypeRegistry()));
                }
                if (project.getDependencyManagement() != null) {
                    if (project.getDependencyManagement().getDependencies().size() > 4096) {
                        throw new IllegalStateException();
                    }
                    for (var dependency : project.getDependencyManagement().getDependencies()) {
                        budget.check();
                        graphDependencyInput(dependency);
                        request.addManagedDependency(RepositoryUtils.toDependency(dependency, session.getArtifactTypeRegistry()));
                    }
                }
                RepositorySystem repository = context.getComponentLookup().lookup(RepositorySystem.class);
                CollectResult collected;
                try {
                    collected = repository.collectDependencies(session, request);
                } catch (DependencyCollectionException unavailable) {
                    collected = unavailable.getResult();
                }
                budget.check();
                return graphProjection(root, pom, collected, budget);
            }, monitor);
            budget.check();
            if (graphFacade(root, pom, budget) != facade || facade.getMavenProject() != project
                    || !java.util.Arrays.equals(inputs, graphInputs(pom))) {
                throw new IllegalStateException();
            }
            return result;
        }

        private void graphDependencyInput(org.apache.maven.model.Dependency dependency) {
            text(dependency.getGroupId());
            text(dependency.getArtifactId());
            text(dependency.getType());
            text(dependency.getClassifier());
            text(dependency.getVersion());
            text(dependency.getScope());
            if (dependency.getExclusions().size() > 128) {
                throw new IllegalStateException();
            }
            for (var exclusion : dependency.getExclusions()) {
                text(exclusion.getGroupId());
                text(exclusion.getArtifactId());
            }
        }

        private byte[] graphInputs(Path pom) throws Exception {
            try (InputStream stream = Files.newInputStream(pom)) {
                byte[] bytes = stream.readNBytes(MAX_XML + 1);
                if (bytes.length > MAX_XML) {
                    throw new IllegalStateException();
                }
                return bytes;
            }
        }

        private IMavenProjectFacade graphFacade(Path root, Path pom, CollectionBudget budget) throws Exception {
            IMavenProjectFacade selected = null;
            int count = 0;
            for (IMavenProjectFacade facade : MavenPlugin.getMavenProjectRegistry().getProjects()) {
                budget.check();
                if (facade.getPom().getLocation() == null) {
                    continue;
                }
                Path cachedPom = facade.getPom().getLocation().toFile().toPath().toAbsolutePath().normalize();
                if (!cachedPom.startsWith(root)) {
                    continue;
                }
                cachedPom = cachedPom.toRealPath();
                if (!cachedPom.startsWith(root) || ++count > 128) {
                    throw new IllegalStateException();
                }
                if (cachedPom.equals(pom)) {
                    if (selected != null || !facade.getProject().isAccessible() || facade.isStale()
                            || facade.getMavenProject() == null
                            || !org.eclipse.jdt.ls.core.internal.ProjectUtils.isMavenProject(facade.getProject())) {
                        throw new IllegalStateException();
                    }
                    selected = facade;
                }
            }
            if (selected == null) {
                throw new IllegalStateException();
            }
            return selected;
        }

        private Map<String, Object> graphProjection(Path root, Path pom, CollectResult collected,
                CollectionBudget budget) {
            if (collected == null || collected.getRoot() == null) {
                throw new IllegalStateException();
            }
            Map<String, Map<String, Object>> components = new LinkedHashMap<>();
            List<Map<String, Object>> edges = new ArrayList<>();
            ArrayDeque<DependencyNode> pending = new ArrayDeque<>();
            Map<DependencyNode, String> occurrenceIds = new IdentityHashMap<>();
            List<DependencyNode> orderedNodes = new ArrayList<>();
            List<Map<String, Object>> nodes = new ArrayList<>();
            DependencyNode graphRoot = collected.getRoot();
            String rootComponent = graphComponent(graphRoot.getArtifact(), components);
            String rootNode = graphNodeIdentity(pom.toUri().toString(), 0, rootComponent);
            occurrenceIds.put(graphRoot, rootNode);
            nodes.add(Map.of("id", text(rootNode), "component", text(rootComponent)));
            pending.add(graphRoot);
            int edgeCount = 0;
            while (!pending.isEmpty()) {
                budget.check();
                DependencyNode parent = pending.removeFirst();
                orderedNodes.add(parent);
                edgeCount = Math.addExact(edgeCount, parent.getChildren().size());
                if (edgeCount > 16384) {
                    throw new IllegalStateException();
                }
                String fromNode = occurrenceIds.get(parent);
                int position = 0;
                for (DependencyNode child : parent.getChildren()) {
                    budget.check();
                    if (child.getDependency() == null) {
                        throw new IllegalStateException();
                    }
                    if (!occurrenceIds.containsKey(child)) {
                        if (occurrenceIds.size() >= 4096) {
                            throw new IllegalStateException();
                        }
                        String component = graphComponent(child.getArtifact(), components);
                        String node = graphNodeIdentity(fromNode, position, component);
                        occurrenceIds.put(child, node);
                        nodes.add(Map.of("id", text(node), "component", text(component)));
                        pending.addLast(child);
                    }
                    position++;
                }
            }
            for (DependencyNode parent : orderedNodes) {
                budget.check();
                String fromNode = occurrenceIds.get(parent);
                String from = graphComponent(parent.getArtifact(), components);
                int position = 0;
                for (DependencyNode child : parent.getChildren()) {
                    budget.check();
                    Dependency dependency = child.getDependency();
                    String component = graphComponent(child.getArtifact(), components);
                    Object winnerData = child.getData().get(ConflictResolver.NODE_DATA_WINNER);
                    if (winnerData != null && !(winnerData instanceof DependencyNode)) {
                        throw new IllegalStateException();
                    }
                    DependencyNode winner = (DependencyNode) winnerData;
                    String targetNode = occurrenceIds.get(child);
                    String selectedNode = winner == null ? targetNode : occurrenceIds.get(winner);
                    if (targetNode == null || selectedNode == null) {
                        throw new IllegalStateException();
                    }
                    String selected = winner == null ? component : graphComponent(winner.getArtifact(), components);
                    Map<String, Object> edge = new LinkedHashMap<>();
                    edge.put("from", text(from));
                    edge.put("fromNode", text(fromNode));
                    edge.put("targetNode", text(targetNode));
                    edge.put("selectedNode", text(selectedNode));
                    edge.put("position", position++);
                    edge.put("component", text(component));
                    edge.put("selected", text(selected));
                    edge.put("omitted", winner != null);
                    edge.put("direct", parent == graphRoot);
                    edge.put("versionConstraint", text(child.getVersionConstraint() == null ? null : child.getVersionConstraint().toString()));
                    edge.put("premanagedVersion", text(DependencyManagerUtils.getPremanagedVersion(child)));
                    edge.put("premanagedScope", text(DependencyManagerUtils.getPremanagedScope(child)));
                    edge.put("premanagedOptional", DependencyManagerUtils.getPremanagedOptional(child));
                    edge.put("managedBits", child.getManagedBits());
                    Object originalScope = child.getData().get(ConflictResolver.NODE_DATA_ORIGINAL_SCOPE);
                    Object originalOptional = child.getData().get(ConflictResolver.NODE_DATA_ORIGINAL_OPTIONALITY);
                    if (originalScope != null && !(originalScope instanceof String)
                            || originalOptional != null && !(originalOptional instanceof Boolean)) {
                        throw new IllegalStateException();
                    }
                    edge.put("originalScope", text((String) originalScope));
                    edge.put("originalOptional", originalOptional);
                    edge.put("scope", text(dependency.getScope()));
                    edge.put("optional", dependency.isOptional());
                    edge.put("exclusions", graphExclusions(dependency.getExclusions()));
                    var premanagedExclusions = DependencyManagerUtils.getPremanagedExclusions(child);
                    edge.put("premanagedExclusions", premanagedExclusions == null ? null : graphExclusions(premanagedExclusions));
                    edges.add(edge);
                }
            }
            List<Map<String, Object>> cycles = new ArrayList<>();
            int cycleSteps = 0;
            if (collected.getCycles().size() > 4096) {
                throw new IllegalStateException();
            }
            // Resolver reports cut cycles separately; their coordinates cannot identify node occurrences.
            for (var cycle : collected.getCycles()) {
                budget.check();
                cycleSteps = Math.addExact(cycleSteps, cycle.getPrecedingDependencies().size());
                cycleSteps = Math.addExact(cycleSteps, cycle.getCyclicDependencies().size());
                if (cycleSteps > 16384) {
                    throw new IllegalStateException();
                }
                cycles.add(Map.of("preceding", graphCycleDependencies(cycle.getPrecedingDependencies(), components),
                        "cyclic", graphCycleDependencies(cycle.getCyclicDependencies(), components)));
            }
            List<Map<String, Object>> unavailableDescriptors = new ArrayList<>();
            if (collected.getExceptions().size() > 4096) {
                throw new IllegalStateException();
            }
            for (Exception failure : collected.getExceptions()) {
                budget.check();
                if (failure instanceof org.eclipse.aether.resolution.ArtifactDescriptorException unavailable
                        && unavailable.getResult() != null && unavailable.getResult().getArtifact() != null) {
                    unavailableDescriptors.add(Map.of("component", text(graphComponent(unavailable.getResult().getArtifact(), components)),
                            "problem", "offline-metadata-unavailable-check-cache-and-refresh"));
                }
            }
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("version", 2);
            result.put("occurrencesKnown", true);
            result.put("nodes", nodes);
            result.put("rootNode", text(rootNode));
            result.put("rootUri", text(root.toUri().toString()));
            result.put("pomUri", text(pom.toUri().toString()));
            result.put("rootComponent", text(rootComponent));
            result.put("components", new ArrayList<>(components.values()));
            result.put("edges", edges);
            result.put("unavailableDescriptors", unavailableDescriptors);
            result.put("collectionComplete", collected.getExceptions().isEmpty());
            result.put("structuralCoverageKnown", collected.getExceptions().isEmpty());
            result.put("problem", collected.getExceptions().isEmpty() ? null : "offline-metadata-unavailable-check-cache-and-refresh");
            result.put("originsKnown", false);
            result.put("artifactsKnown", false);
            result.put("offline", true);
            result.put("cachedOnly", true);
            result.put("cycleCount", collected.getCycles().size());
            result.put("cycles", cycles);
            result.put("cycleOccurrencesKnown", false);
            if (new Gson().toJson(result).getBytes(StandardCharsets.UTF_8).length > 8 * 1024 * 1024) {
                throw new IllegalStateException();
            }
            budget.check();
            return result;
        }

        private List<Map<String, Object>> graphCycleDependencies(List<Dependency> dependencies,
                Map<String, Map<String, Object>> components) {
            if (dependencies.size() > 256) {
                throw new IllegalStateException();
            }
            List<Map<String, Object>> result = new ArrayList<>();
            for (Dependency dependency : dependencies) {
                Map<String, Object> record = new LinkedHashMap<>();
                record.put("component", text(graphComponent(dependency.getArtifact(), components)));
                record.put("scope", text(dependency.getScope()));
                record.put("optional", dependency.isOptional());
                record.put("exclusions", graphExclusions(dependency.getExclusions()));
                result.add(record);
            }
            return result;
        }

        private String graphNodeIdentity(String parent, int position, String component) {
            byte[] path = new Gson().toJson(List.of(parent, position, component)).getBytes(StandardCharsets.UTF_8);
            try {
                return "node:" + java.util.HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256").digest(path));
            } catch (java.security.GeneralSecurityException unavailable) {
                throw new IllegalStateException();
            }
        }

        private List<Map<String, Object>> graphExclusions(java.util.Collection<org.eclipse.aether.graph.Exclusion> source) {
            if (source.size() > 128) {
                throw new IllegalStateException();
            }
            List<Map<String, Object>> exclusions = new ArrayList<>();
            for (var exclusion : source) {
                exclusions.add(Map.of("groupId", text(exclusion.getGroupId()), "artifactId", text(exclusion.getArtifactId()),
                        "extension", text(exclusion.getExtension()), "classifier", text(exclusion.getClassifier())));
            }
            return exclusions;
        }

        private String graphComponent(Artifact artifact, Map<String, Map<String, Object>> components) {
            if (artifact == null) {
                throw new IllegalStateException();
            }
            String id = new Gson().toJson(List.of(text(artifact.getGroupId()), text(artifact.getArtifactId()), text(artifact.getExtension()),
                    text(artifact.getClassifier()), text(artifact.getVersion())));
            text(id);
            if (!components.containsKey(id)) {
                if (components.size() >= 4096) {
                    throw new IllegalStateException();
                }
                Map<String, Object> record = new LinkedHashMap<>();
                record.put("id", text(id));
                record.put("groupId", text(artifact.getGroupId()));
                record.put("artifactId", text(artifact.getArtifactId()));
                record.put("extension", text(artifact.getExtension()));
                record.put("classifier", text(artifact.getClassifier()));
                record.put("version", text(artifact.getVersion()));
                components.put(id, record);
            }
            return id;
        }

        private static final class CollectionBudget {
            private final IProgressMonitor monitor;
            private final long deadline = System.nanoTime() + java.util.concurrent.TimeUnit.SECONDS.toNanos(60);
            private final java.util.concurrent.atomic.AtomicInteger traversals = new java.util.concurrent.atomic.AtomicInteger();
            private volatile boolean exceeded;

            private CollectionBudget(IProgressMonitor monitor) {
                this.monitor = monitor;
            }

            private void check() {
                if (monitor.isCanceled() || System.nanoTime() > deadline || exceeded || traversals.get() > 16384) {
                    throw new IllegalStateException();
                }
            }

            private void traverse() {
                check();
                if (traversals.incrementAndGet() > 16384) {
                    throw new IllegalStateException();
                }
            }
        }

        private static final class BoundedTraverser implements DependencyTraverser {
            private final DependencyTraverser delegate;
            private final CollectionBudget budget;
            private final int depth;

            private BoundedTraverser(DependencyTraverser delegate, CollectionBudget budget, int depth) {
                this.delegate = delegate;
                this.budget = budget;
                this.depth = depth;
            }

            @Override
            public boolean traverseDependency(Dependency dependency) {
                budget.traverse();
                return delegate == null || delegate.traverseDependency(dependency);
            }

            @Override
            public DependencyTraverser deriveChildTraverser(DependencyCollectionContext context) {
                budget.check();
                if (depth >= 256) {
                    budget.exceeded = true;
                    throw new IllegalStateException();
                }
                return new BoundedTraverser(delegate == null ? null : delegate.deriveChildTraverser(context), budget, depth + 1);
            }
        }

        private void appendCachedDescendants(Path unavailablePom, Set<Path> cachedPoms,
                ArrayDeque<Path> pending, Set<Path> visited, IProgressMonitor monitor) {
            checkCancelled(monitor);
            Path directory = unavailablePom.getParent();
            if (directory == null || cachedPoms.size() > 128 || pending.size() > 128 || visited.size() > 128) {
                throw new IllegalStateException();
            }
            Set<Path> scheduled = new HashSet<>(visited);
            scheduled.addAll(pending);
            if (scheduled.size() > 128) {
                throw new IllegalStateException();
            }
            List<Path> additions = new ArrayList<>();
            // An unlinked aggregate can still have imported descendants with cached models.
            for (Path candidate : cachedPoms.stream().sorted().toList()) {
                checkCancelled(monitor);
                if (candidate.startsWith(directory) && !scheduled.contains(candidate)) {
                    if (scheduled.size() >= 128 || pending.size() + additions.size() >= 128) {
                        throw new IllegalStateException();
                    }
                    scheduled.add(candidate);
                    additions.add(candidate);
                }
            }
            checkCancelled(monitor);
            pending.addAll(additions);
        }

        private Map<String, Object> projects(Path root, IProgressMonitor monitor, boolean details,
                boolean parameters) throws Exception {
            responseCharacterLimit = parameters ? 4 * 1024 * 1024 : 1024 * 1024;
            List<Map<String, Object>> projects = new ArrayList<>();
            List<String> unavailable = new ArrayList<>();
            List<MavenProject> reactor = new ArrayList<>();
            String repositoryPath = MavenPlugin.getMaven().getLocalRepositoryPath();
            Path repository = repositoryPath == null ? null : Path.of(repositoryPath).toAbsolutePath().normalize();
            Map<Path, IMavenProjectFacade> cached = new LinkedHashMap<>();
            for (IMavenProjectFacade facade : MavenPlugin.getMavenProjectRegistry().getProjects()) {
                checkCancelled(monitor);
                Path pom = facade.getPom().getLocation().toFile().toPath().toRealPath();
                if (pom.startsWith(root)) {
                    if (cached.size() >= 128 || cached.put(pom, facade) != null) {
                        throw new IllegalStateException();
                    }
                }
            }
            ArrayDeque<Path> pending = new ArrayDeque<>();
            Set<Path> visited = new HashSet<>();
            Path rootPom = root.resolve("pom.xml");
            if (Files.isRegularFile(rootPom)) {
                Path pom = rootPom.toRealPath();
                if (!pom.startsWith(root)) {
                    throw new IllegalStateException();
                }
                pending.add(pom);
            } else {
                // Cached facades let container reads avoid discovering or importing new POMs.
                pending.addAll(cached.keySet().stream().sorted().toList());
            }
            while (!pending.isEmpty()) {
                checkCancelled(monitor);
                Path pom = pending.removeFirst();
                if (!visited.add(pom)) {
                    continue;
                }
                if (visited.size() > 128) {
                    throw new IllegalStateException();
                }
                // The monitored overload can rebuild/resolve the model; this read must use the cached overload.
                IMavenProjectFacade facade = cached.get(pom);
                MavenProject project = facade == null ? null : facade.getMavenProject();
                if (project == null) {
                    unavailable.add(text(pom.toUri().toString()));
                    appendCachedDescendants(pom, cached.keySet(), pending, visited, monitor);
                    continue;
                }
                reactor.add(project);
                Map<String, Object> record = new LinkedHashMap<>();
                record.put("pomUri", text(pom.toUri().toString()));
                record.put("groupId", text(project.getGroupId()));
                record.put("artifactId", text(project.getArtifactId()));
                record.put("version", text(project.getVersion()));
                record.put("packaging", text(project.getPackaging()));
                List<String> modules = new ArrayList<>();
                for (String module : project.getModules()) {
                    if (modules.size() >= 128) {
                        throw new IllegalStateException();
                    }
                    Path path = pom.getParent().resolve(module).normalize();
                    if (!path.startsWith(root) || Files.exists(path) && !path.toRealPath().startsWith(root)) {
                        throw new IllegalStateException();
                    }
                    modules.add(text(path.toUri().toString()));
                    Path modulePom = Files.isRegularFile(path) ? path : path.resolve("pom.xml");
                    if (Files.exists(modulePom)) {
                        modulePom = modulePom.toRealPath();
                    }
                    if (!modulePom.startsWith(root) || pending.size() >= 128) {
                        throw new IllegalStateException();
                    }
                    pending.addLast(modulePom);
                }
                record.put("moduleUris", modules);
                record.put("profiles", profileStatuses(project, monitor));
                record.put("inheritedProfileDefinitionsKnown", false);
                if (details) {
                    profileDefinitions(record, project, root, monitor);
                }
                Map<String, Plugin> plugins = new LinkedHashMap<>();
                Set<String> buildPluginKeys = new HashSet<>();
                if (project.getPluginManagement() != null) {
                    for (Plugin plugin : project.getPluginManagement().getPlugins()) {
                        plugins.put(plugin.getKey(), plugin);
                    }
                }
                for (Plugin plugin : project.getBuildPlugins()) {
                    buildPluginKeys.add(plugin.getKey());
                    Plugin managed = plugins.get(plugin.getKey());
                    if (plugin.getVersion() == null && managed != null) {
                        Plugin effective = plugin.clone();
                        effective.setVersion(managed.getVersion());
                        plugins.put(plugin.getKey(), effective);
                    } else {
                        plugins.put(plugin.getKey(), plugin);
                    }
                }
                if (plugins.size() > 128) {
                    throw new IllegalStateException();
                }
                List<Map<String, Object>> pluginRecords = new ArrayList<>();
                for (Plugin plugin : plugins.values()) {
                    checkCancelled(monitor);
                    pluginRecords.add(plugin(plugin, repository, monitor, !buildPluginKeys.contains(plugin.getKey()), parameters));
                }
                pluginRecords.sort(Comparator.comparing(plugin -> (String) plugin.get("artifactId")));
                record.put("plugins", pluginRecords);
                projects.add(record);
            }
            projects.sort(Comparator.comparing(project -> (String) project.get("pomUri")));
            unavailable.sort(String::compareTo);
            List<Map<String, Object>> lifecycles = new ArrayList<>();
            for (Lifecycle lifecycle : MavenPlugin.getMaven().lookup(DefaultLifecycles.class).getLifeCycles()) {
                Map<String, Object> record = new LinkedHashMap<>();
                record.put("id", text(lifecycle.getId()));
                if (lifecycle.getPhases().size() > 128 || lifecycles.size() >= 32) {
                    throw new IllegalStateException();
                }
                List<String> phases = new ArrayList<>();
                for (String phase : lifecycle.getPhases()) {
                    phases.add(text(phase));
                }
                record.put("phases", phases);
                lifecycles.add(record);
            }
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("version", parameters ? 3 : details ? 2 : 1);
            result.put("rootUri", root.toUri().toString());
            result.put("projects", projects);
            result.put("unavailablePomUris", unavailable);
            result.put("lifecycles", lifecycles);
            result.put("cachedOnly", true);
            result.put("reactorOrderKnown", false);
            if (details) {
                reactorOrder(result, reactor, unavailable, root, monitor);
            }
            if (new Gson().toJson(result).getBytes(StandardCharsets.UTF_8).length > 8 * 1024 * 1024) {
                throw new IllegalStateException();
            }
            checkCancelled(monitor);
            return result;
        }

        private Map<String, Boolean> profileStatuses(MavenProject project, IProgressMonitor monitor) {
            checkCancelled(monitor);
            Map<String, Boolean> profiles = new LinkedHashMap<>();
            if (project.getOriginalModel() != null) {
                for (Profile profile : project.getOriginalModel().getProfiles()) {
                    addProfileStatus(profiles, profile.getId(), false, monitor);
                }
            }
            for (Profile profile : project.getActiveProfiles()) {
                addProfileStatus(profiles, profile.getId(), true, monitor);
            }
            return profiles;
        }

        private void addProfileStatus(Map<String, Boolean> profiles, String identifier, boolean active,
                IProgressMonitor monitor) {
            checkCancelled(monitor);
            if (!profiles.containsKey(identifier) && profiles.size() >= 128) {
                throw new IllegalStateException();
            }
            profiles.put(text(identifier), active);
        }

        private void profileDefinitions(Map<String, Object> record, MavenProject project, Path root,
                IProgressMonitor monitor) throws Exception {
            List<Map<String, Object>> definitions = new ArrayList<>();
            Set<MavenProject> visited = Collections.newSetFromMap(new IdentityHashMap<>());
            MavenProject source = project;
            boolean complete = true;
            while (source != null) {
                checkCancelled(monitor);
                if (!visited.add(source) || visited.size() > 32 || source.getOriginalModel() == null) {
                    complete = false;
                    break;
                }
                Path sourcePom = source.getFile() == null ? null : source.getFile().toPath().toRealPath();
                if (source.getActiveProfiles().size() > 128) {
                    throw new IllegalStateException();
                }
                Set<String> active = new HashSet<>();
                for (Profile profile : source.getActiveProfiles()) {
                    active.add(profile.getId());
                }
                Set<String> declared = new HashSet<>();
                for (Profile profile : source.getOriginalModel().getProfiles()) {
                    if (definitions.size() >= 128 || !declared.add(profile.getId())) {
                        throw new IllegalStateException();
                    }
                    Map<String, Object> definition = new LinkedHashMap<>();
                    definition.put("id", text(profile.getId()));
                    definition.put("sourceGroupId", text(source.getGroupId()));
                    definition.put("sourceArtifactId", text(source.getArtifactId()));
                    definition.put("sourceVersion", text(source.getVersion()));
                    definition.put("sourcePomUri", sourcePom != null && sourcePom.startsWith(root)
                            ? text(sourcePom.toUri().toString()) : null);
                    definition.put("inherited", source != project);
                    definition.put("activeAtSource", active.contains(profile.getId()));
                    definitions.add(definition);
                }
                MavenProject parent = source.getParent();
                if (parent == null && source.getOriginalModel().getParent() != null) {
                    complete = false;
                }
                source = parent;
            }
            record.put("profileDefinitions", definitions);
            record.put("inheritedProfileDefinitionsKnown", complete);
        }

        private void reactorOrder(Map<String, Object> result, List<MavenProject> reactor,
                List<String> unavailable, Path root, IProgressMonitor monitor) throws Exception {
            List<String> order = new ArrayList<>();
            String problem = unavailable.isEmpty() ? null : "missing-cached-model";
            Set<String> coordinates = new HashSet<>();
            int edges = 0;
            for (MavenProject project : reactor) {
                checkCancelled(monitor);
                edges = Math.addExact(edges, project.getDependencies().size());
                edges = Math.addExact(edges, project.getBuildExtensions().size());
                edges = Math.addExact(edges, project.getBuildPlugins().size());
                for (Plugin plugin : project.getBuildPlugins()) {
                    edges = Math.addExact(edges, plugin.getDependencies().size());
                }
                if (edges > 4096) {
                    throw new IllegalStateException();
                }
                if (!coordinates.add(project.getGroupId() + ":" + project.getArtifactId() + ":" + project.getVersion())) {
                    problem = "duplicate-coordinates";
                }
            }
            if (problem == null) {
                try {
                    Set<Path> expected = new HashSet<>();
                    for (MavenProject project : reactor) {
                        expected.add(project.getFile().toPath().toRealPath());
                    }
                    Set<Path> seen = new HashSet<>();
                    List<MavenProject> copies = new ArrayList<>();
                    for (MavenProject project : reactor) {
                        checkCancelled(monitor);
                        // The sorter adds project references; constructor copies isolate the shared M2E cache.
                        copies.add(new MavenProject(project));
                    }
                    for (MavenProject project : new ProjectSorter(copies).getSortedProjects()) {
                        checkCancelled(monitor);
                        Path pom = project.getFile().toPath().toRealPath();
                        if (!pom.startsWith(root) || !seen.add(pom)) {
                            throw new IllegalStateException();
                        }
                        order.add(text(pom.toUri().toString()));
                    }
                    if (order.size() != reactor.size() || !seen.equals(expected)) {
                        throw new IllegalStateException();
                    }
                } catch (CycleDetectedException | DuplicateProjectException invalidGraph) {
                    problem = "invalid-cached-reactor";
                    order.clear();
                }
            }
            result.put("reactorOrderKnown", problem == null);
            result.put("reactorPomUris", order);
            result.put("reactorOrderProblem", problem);
        }

        private Map<String, Object> plugin(Plugin plugin, Path repository, IProgressMonitor monitor,
                boolean managedOnly, boolean parameters) throws Exception {
            Map<String, Object> record = new LinkedHashMap<>();
            record.put("groupId", text(plugin.getGroupId()));
            record.put("artifactId", text(plugin.getArtifactId()));
            record.put("version", text(plugin.getVersion()));
            record.put("managedOnly", managedOnly);
            record.put("goalsKnown", false);
            record.put("goalPrefix", null);
            record.put("goals", List.of());
            if (repository == null || !coordinate(plugin.getGroupId()) || !coordinate(plugin.getArtifactId())
                    || !coordinate(plugin.getVersion())) {
                return record;
            }
            Path jar = repository.resolve(plugin.getGroupId().replace('.', '/')).resolve(plugin.getArtifactId())
                    .resolve(plugin.getVersion()).resolve(plugin.getArtifactId() + "-" + plugin.getVersion() + ".jar");
            if (!Files.isRegularFile(jar)) {
                return record;
            }
            if (!jar.toRealPath().startsWith(repository.toRealPath())) {
                throw new IllegalStateException();
            }
            try (ZipFile archive = new ZipFile(jar.toFile())) {
                java.util.zip.ZipEntry entry = archive.getEntry("META-INF/maven/plugin.xml");
                if (entry == null) {
                    return record;
                }
                if (entry.getSize() > MAX_XML) {
                    throw new IllegalStateException();
                }
                byte[] bytes;
                try (InputStream stream = archive.getInputStream(entry)) {
                    bytes = stream.readNBytes(MAX_XML + 1);
                }
                if (bytes.length > MAX_XML) {
                    throw new IllegalStateException();
                }
                DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
                factory.setFeature(XMLConstants.FEATURE_SECURE_PROCESSING, true);
                factory.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);
                factory.setFeature("http://xml.org/sax/features/external-general-entities", false);
                factory.setFeature("http://xml.org/sax/features/external-parameter-entities", false);
                factory.setAttribute(XMLConstants.ACCESS_EXTERNAL_DTD, "");
                factory.setAttribute(XMLConstants.ACCESS_EXTERNAL_SCHEMA, "");
                factory.setXIncludeAware(false);
                factory.setExpandEntityReferences(false);
                Element descriptor = factory.newDocumentBuilder().parse(new ByteArrayInputStream(bytes)).getDocumentElement();
                if (!"plugin".equals(descriptor.getTagName())
                        || !plugin.getGroupId().equals(childText(descriptor, "groupId"))
                        || !plugin.getArtifactId().equals(childText(descriptor, "artifactId"))
                        || !plugin.getVersion().equals(childText(descriptor, "version"))) {
                    throw new IllegalStateException();
                }
                List<Map<String, Object>> goals = new ArrayList<>();
                Element mojos = child(descriptor, "mojos");
                if (mojos != null) {
                    for (Node node = mojos.getFirstChild(); node != null; node = node.getNextSibling()) {
                        checkCancelled(monitor);
                        if (node instanceof Element mojo && "mojo".equals(mojo.getTagName())) {
                            if (goals.size() >= 4096) {
                                throw new IllegalStateException();
                            }
                            Map<String, Object> goal = new LinkedHashMap<>();
                            goal.put("name", text(childText(mojo, "goal")));
                            goal.put("phase", text(childText(mojo, "phase")));
                            goal.put("description", text(childText(mojo, "description")));
                            if (parameters) {
                                goal.put("parameters", parameters(mojo, monitor));
                                goal.put("parametersKnown", true);
                            }
                            goals.add(goal);
                        }
                    }
                }
                goals.sort(Comparator.comparing(goal -> (String) goal.get("name")));
                record.put("goalPrefix", text(childText(descriptor, "goalPrefix")));
                record.put("goals", goals);
                record.put("goalsKnown", true);
            }
            return record;
        }

        private List<Map<String, Object>> parameters(Element mojo, IProgressMonitor monitor) {
            List<Map<String, Object>> result = new ArrayList<>();
            Set<String> names = new HashSet<>();
            Element parameters = child(mojo, "parameters");
            if (parameters == null) {
                return result;
            }
            for (Node node = parameters.getFirstChild(); node != null; node = node.getNextSibling()) {
                checkCancelled(monitor);
                if (node instanceof Element parameter && "parameter".equals(parameter.getTagName())) {
                    if (result.size() >= 256) {
                        throw new IllegalStateException();
                    }
                    String name = text(childText(parameter, "name"));
                    if (name == null || !name.matches("[A-Za-z_$][A-Za-z0-9_$.-]*") || !names.add(name)) {
                        throw new IllegalStateException();
                    }
                    Map<String, Object> declaration = new LinkedHashMap<>();
                    declaration.put("name", name);
                    declaration.put("alias", text(childText(parameter, "alias")));
                    declaration.put("declaredType", text(childText(parameter, "type")));
                    declaration.put("required", booleanValue(parameter, "required", false));
                    declaration.put("editable", booleanValue(parameter, "editable", true));
                    declaration.put("description", text(childText(parameter, "description")));
                    result.add(declaration);
                }
            }
            return result;
        }

        private boolean booleanValue(Element parameter, String name, boolean defaultValue) {
            String value = childText(parameter, name);
            if (value == null) {
                return defaultValue;
            }
            return switch (value.trim()) {
                case "true" -> true;
                case "false" -> false;
                default -> throw new IllegalStateException();
            };
        }

        private Element child(Element parent, String name) {
            for (Node node = parent.getFirstChild(); node != null; node = node.getNextSibling()) {
                if (node instanceof Element element && name.equals(element.getTagName())) {
                    return element;
                }
            }
            return null;
        }

        private String childText(Element parent, String name) {
            Element element = child(parent, name);
            return element == null ? null : element.getTextContent();
        }

        private boolean coordinate(String value) {
            return value != null && value.length() <= 256 && value.matches("[A-Za-z0-9_][A-Za-z0-9_.-]*")
                    && !value.contains("..");
        }

        private String text(String value) {
            responseCharacters = Math.addExact(responseCharacters, (value == null ? 0 : value.length()) + 80);
            if (responseCharacters > responseCharacterLimit) {
                throw new IllegalStateException();
            }
            if (value != null && (value.length() > MAX_TEXT || value.chars().anyMatch(character ->
                    Character.isISOControl(character) && character != '\n' && character != '\t' && character != '\r'))) {
                throw new IllegalStateException();
            }
            return value;
        }

        private void checkCancelled(IProgressMonitor monitor) {
            if (monitor.isCanceled()) {
                throw new IllegalStateException();
            }
        }
    }
}
