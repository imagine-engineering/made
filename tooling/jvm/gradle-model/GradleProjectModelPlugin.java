package dev.made.groovy;

import java.io.File;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.lang.reflect.WildcardType;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.HashSet;
import javax.inject.Inject;

import groovy.json.JsonOutput;
import org.gradle.api.Plugin;
import org.gradle.api.Project;
import org.gradle.api.provider.Property;
import org.gradle.api.provider.ListProperty;
import org.gradle.api.Task;
import org.gradle.api.artifacts.Configuration;
import org.gradle.api.artifacts.component.ComponentIdentifier;
import org.gradle.api.artifacts.component.ComponentSelector;
import org.gradle.api.artifacts.component.ModuleComponentIdentifier;
import org.gradle.api.artifacts.component.ModuleComponentSelector;
import org.gradle.api.artifacts.component.ProjectComponentIdentifier;
import org.gradle.api.artifacts.component.ProjectComponentSelector;
import org.gradle.api.artifacts.result.ResolvedComponentResult;
import org.gradle.api.artifacts.result.ResolvedDependencyResult;
import org.gradle.api.artifacts.result.UnresolvedDependencyResult;
import org.gradle.api.plugins.JavaPluginExtension;
import org.gradle.api.tasks.SourceSet;
import org.gradle.api.tasks.compile.JavaCompile;
import org.gradle.api.tasks.options.Option;
import org.gradle.tooling.provider.model.ToolingModelBuilder;
import org.gradle.tooling.provider.model.ToolingModelBuilderRegistry;
import org.gradle.tooling.provider.model.ParameterizedToolingModelBuilder;

public final class GradleProjectModelPlugin implements Plugin<Project> {
    private final ToolingModelBuilderRegistry registry;

    @Inject
    public GradleProjectModelPlugin(ToolingModelBuilderRegistry registry) {
        this.registry = registry;
    }

    @Override
    public void apply(Project project) {
        registry.register(new Builder());
        registry.register(new GraphBuilder());
        registry.register(new ClosedArtifactGuardBuilder());
    }

    public static final class ClosedArtifactGuardBuilder implements ToolingModelBuilder {
        @Override
        public boolean canBuild(String modelName) {
            return GradleClosedArtifactSelection.class.getName().equals(modelName);
        }

        @Override
        public Object buildAll(String modelName, Project project) {
            List<String> tasks = project.getRootProject().getGradle().getStartParameter().getTaskNames();
            if (!canBuild(modelName) || tasks.isEmpty() || tasks.size() > 128
                    || tasks.stream().distinct().count() != tasks.size()
                    || tasks.stream().anyMatch(name -> name == null || name.length() > 256
                            || !name.matches("eclipseClosedDependencies_*"))) {
                throw new IllegalStateException("Closed Gradle artifact task selection is empty or unsupported. Refresh before retrying.");
            }
            return new ClosedArtifactSelection();
        }
    }

    public static final class ClosedArtifactSelection implements GradleClosedArtifactSelection {
        @Override
        public String getJson() {
            return "{}";
        }
    }

    public static final class GraphDetails implements GradleDependencyGraph {
        private final String json;

        GraphDetails(String json) {
            this.json = json;
        }

        @Override
        public String getJson() {
            return json;
        }
    }

    private static final class GraphBudget {
        private int characters;

        String text(String value) {
            String bounded = Builder.text(value);
            characters = Math.addExact(characters, bounded.length() + 80);
            if (characters > 1024 * 1024) {
                throw new IllegalStateException("Gradle graph response budget exceeded.");
            }
            return bounded;
        }
    }

    public static final class GraphBuilder implements ParameterizedToolingModelBuilder<GradleDependencyGraph.Parameters> {
        private static final int MAX_COMPONENTS = 4096;
        private static final int MAX_EDGES = 16384;

        @Override
        public Class<GradleDependencyGraph.Parameters> getParameterType() {
            return GradleDependencyGraph.Parameters.class;
        }

        @Override
        public boolean canBuild(String name) {
            return name.equals(GradleDependencyGraph.class.getName());
        }

        @Override
        public Object buildAll(String name, Project project) {
            throw new IllegalStateException("An explicit Gradle configuration is required.");
        }

        @Override
        public Object buildAll(String name, GradleDependencyGraph.Parameters parameters, Project project) {
            String requested = parameters == null ? null : parameters.getConfiguration();
            if (requested == null || requested.isEmpty()
                    || requested.getBytes(java.nio.charset.StandardCharsets.UTF_8).length > 4096
                    || requested.codePoints().anyMatch(codePoint -> Character.isISOControl(codePoint)
                            || codePoint >= Character.MIN_SURROGATE && codePoint <= Character.MAX_SURROGATE)) {
                throw new IllegalStateException("Invalid Gradle graph configuration.");
            }
            Configuration configuration = project.getConfigurations().findByName(requested);
            if (configuration == null || !configuration.isCanBeResolved()) {
                throw new IllegalStateException("The Gradle graph configuration is unavailable or not resolvable.");
            }
            GraphBudget budget = new GraphBudget();
            ResolvedComponentResult root = configuration.getIncoming().getResolutionResult().getRootComponent().get();
            ArrayDeque<ResolvedComponentResult> pending = new ArrayDeque<>();
            HashSet<ComponentIdentifier> seen = new HashSet<>();
            List<Map<String, Object>> components = new ArrayList<>();
            List<Map<String, Object>> edges = new ArrayList<>();
            pending.add(root);
            seen.add(root.getId());
            boolean complete = true;
            while (!pending.isEmpty()) {
                ResolvedComponentResult component = pending.removeFirst();
                Map<String, Object> record = identifier(component.getId(), budget);
                String from = (String) record.get("id");
                var reason = component.getSelectionReason();
                record.put("conflictResolution", reason.isConflictResolution());
                record.put("forced", reason.isForced());
                record.put("selectedByRule", reason.isSelectedByRule());
                record.put("compositeSubstitution", reason.isCompositeSubstitution());
                record.put("constrained", reason.isConstrained());
                List<String> causes = new ArrayList<>();
                for (var descriptor : reason.getDescriptions()) {
                    if (causes.size() >= 128) {
                        throw new IllegalStateException("Gradle selection-cause limit exceeded.");
                    }
                    causes.add(budget.text(descriptor.getCause().name()));
                }
                causes.sort(String::compareTo);
                record.put("selectionCauses", causes);
                components.add(record);
                for (var dependency : component.getDependencies()) {
                    if (edges.size() >= MAX_EDGES) {
                        throw new IllegalStateException("Gradle graph-edge limit exceeded.");
                    }
                    Map<String, Object> edge = new LinkedHashMap<>();
                    edge.put("from", from);
                    edge.put("requested", selector(dependency.getRequested(), budget));
                    edge.put("constraint", dependency.isConstraint());
                    edge.put("originKnown", false);
                    if (dependency instanceof ResolvedDependencyResult resolved) {
                        ResolvedComponentResult selected = resolved.getSelected();
                        edge.put("resolved", true);
                        edge.put("selected", identifier(selected.getId(), budget).get("id"));
                        edge.put("selectedVariant", budget.text(resolved.getResolvedVariant().getDisplayName()));
                        if (seen.add(selected.getId())) {
                            if (seen.size() > MAX_COMPONENTS) {
                                throw new IllegalStateException("Gradle graph-component limit exceeded.");
                            }
                            pending.addLast(selected);
                        }
                    } else if (dependency instanceof UnresolvedDependencyResult) {
                        complete = false;
                        edge.put("resolved", false);
                        edge.put("selected", null);
                        edge.put("selectedVariant", null);
                        edge.put("failureCategory", "Dependency metadata unavailable. Check offline cache and build settings; refresh after repair.");
                    } else {
                        throw new IllegalStateException("Unsupported Gradle graph edge.");
                    }
                    edges.add(edge);
                }
            }
            components.sort(Comparator.comparing(component -> (String) component.get("id")));
            edges.sort(Comparator.comparing(JsonOutput::toJson));
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("configuration", requested);
            result.put("rootComponent", identifier(root.getId(), budget).get("id"));
            result.put("structuralCoverageKnown", true);
            result.put("resolutionComplete", complete);
            result.put("originsKnown", false);
            result.put("artifactsKnown", false);
            result.put("components", components);
            result.put("edges", edges);
            String json = JsonOutput.toJson(result);
            if (json.length() > 2 * 1024 * 1024) {
                throw new IllegalStateException("Gradle graph metadata limit exceeded.");
            }
            return new GraphDetails(json);
        }

        private static Map<String, Object> identifier(ComponentIdentifier identifier, GraphBudget budget) {
            Map<String, Object> record = new LinkedHashMap<>();
            List<String> identity;
            if (identifier instanceof ModuleComponentIdentifier module) {
                record.put("kind", "module");
                record.put("group", budget.text(module.getGroup()));
                record.put("module", budget.text(module.getModule()));
                record.put("version", budget.text(module.getVersion()));
                identity = List.of("module", module.getGroup(), module.getModule(), module.getVersion());
            } else if (identifier instanceof ProjectComponentIdentifier project) {
                record.put("kind", "project");
                record.put("buildTreePath", budget.text(project.getBuildTreePath()));
                record.put("projectPath", budget.text(project.getProjectPath()));
                identity = List.of("project", project.getBuildTreePath());
            } else {
                throw new IllegalStateException("Unsupported Gradle component identity.");
            }
            record.put("id", budget.text(JsonOutput.toJson(identity)));
            return record;
        }

        private static Map<String, Object> selector(ComponentSelector selector, GraphBudget budget) {
            Map<String, Object> record = new LinkedHashMap<>();
            if (selector instanceof ModuleComponentSelector module) {
                record.put("kind", "module");
                record.put("group", budget.text(module.getGroup()));
                record.put("module", budget.text(module.getModule()));
                record.put("version", budget.text(module.getVersion()));
                var constraint = module.getVersionConstraint();
                record.put("requiredVersion", budget.text(constraint.getRequiredVersion()));
                record.put("preferredVersion", budget.text(constraint.getPreferredVersion()));
                record.put("strictVersion", budget.text(constraint.getStrictVersion()));
                if (constraint.getRejectedVersions().size() > 128) {
                    throw new IllegalStateException("Gradle rejected-version limit exceeded.");
                }
                record.put("rejectedVersions", constraint.getRejectedVersions().stream().map(budget::text).toList());
            } else if (selector instanceof ProjectComponentSelector project) {
                record.put("kind", "project");
                record.put("buildPath", budget.text(project.getBuildPath()));
                record.put("projectPath", budget.text(project.getProjectPath()));
            } else {
                throw new IllegalStateException("Unsupported Gradle dependency selector.");
            }
            return record;
        }
    }

    public static final class Details implements GradleProjectDetails, GradleSourceOutputs {
        private final String json;

        Details(String json) {
            this.json = json;
        }

        @Override
        public String getJson() {
            return json;
        }
    }

    public static final class Builder implements ToolingModelBuilder {
        @Override
        public boolean canBuild(String name) {
            return name.equals(GradleProjectDetails.class.getName()) || name.equals(GradleSourceOutputs.class.getName());
        }

        @Override
        public Object buildAll(String name, Project project) {
            if (name.equals(GradleSourceOutputs.class.getName())) {
                return sourceOutputs(project);
            }
            Map<String, Object> result = new LinkedHashMap<>();
            List<Map<String, Object>> sources = new ArrayList<>();
            JavaPluginExtension java = project.getExtensions().findByType(JavaPluginExtension.class);
            result.put("javaSourceSetsKnown", java != null);
            if (java != null) {
                for (SourceSet sourceSet : java.getSourceSets()) {
                    if (sources.size() >= 128) {
                        throw new IllegalStateException("Gradle source-set limit exceeded.");
                    }
                    List<String> roots = new ArrayList<>();
                    for (File root : sourceSet.getAllSource().getSrcDirs()) {
                        if (roots.size() >= 256) {
                            throw new IllegalStateException("Gradle source-root limit exceeded.");
                        }
                        roots.add(text(root.toPath().toAbsolutePath().normalize().toUri().toString()));
                    }
                    roots.sort(String::compareTo);
                    Map<String, Object> source = new LinkedHashMap<>();
                    source.put("name", text(sourceSet.getName()));
                    source.put("roots", roots);
                    source.put("generatednessKnown", false);
                    annotationProcessorOutput(project,
                            project.getTasks().findByName(sourceSet.getCompileJavaTaskName()), source);
                    sources.add(source);
                }
            }
            sources.sort(Comparator.comparing(source -> (String) source.get("name")));
            result.put("sourceSets", sources);
            List<Map<String, Object>> configurations = new ArrayList<>();
            int configurationEdges = 0;
            for (Configuration configuration : project.getConfigurations()) {
                if (configurations.size() >= 512) {
                    throw new IllegalStateException("Gradle configuration limit exceeded.");
                }
                List<String> parents = new ArrayList<>();
                for (Configuration parent : configuration.getExtendsFrom()) {
                    if (++configurationEdges > 4096 || parents.size() >= 128) {
                        throw new IllegalStateException("Gradle configuration edge limit exceeded.");
                    }
                    parents.add(text(parent.getName()));
                }
                parents.sort(String::compareTo);
                Map<String, Object> declaration = new LinkedHashMap<>();
                declaration.put("name", text(configuration.getName()));
                declaration.put("description", configuration.getDescription() == null
                        ? null : text(configuration.getDescription()));
                declaration.put("resolvable", configuration.isCanBeResolved());
                declaration.put("consumable", configuration.isCanBeConsumed());
                declaration.put("extendsFrom", parents);
                configurations.add(declaration);
            }
            configurations.sort(Comparator.comparing(configuration -> (String) configuration.get("name")));
            result.put("configurationsKnown", true);
            result.put("configurations", configurations);
            result.put("dependencyGraphKnown", false);
            List<Map<String, Object>> tasks = new ArrayList<>();
            for (Task task : project.getTasks()) {
                if (tasks.size() >= 4096) {
                    throw new IllegalStateException("Gradle task limit exceeded.");
                }
                Map<String, Map<String, Object>> options = new LinkedHashMap<>();
                Map<String, String> declarationsByName = new LinkedHashMap<>();
                boolean complete = true;
                for (Class<?> type = task.getClass(); type != null; type = type.getSuperclass()) {
                    Method[] methods = type.getDeclaredMethods();
                    Arrays.sort(methods, Comparator.comparing(Method::toGenericString));
                    for (Method method : methods) {
                        Option option = method.getAnnotation(Option.class);
                        if (option != null) {
                            Type declaredType = method.getParameterCount() == 1
                                    ? method.getGenericParameterTypes()[0] : method.getGenericReturnType();
                            complete &= addOption(options, declarationsByName, option, method.getName(),
                                    declaredType, optionMethodIdentity(method));
                        }
                    }
                    Field[] fields = type.getDeclaredFields();
                    Arrays.sort(fields, Comparator.comparing(Field::getName));
                    for (Field field : fields) {
                        Option option = field.getAnnotation(Option.class);
                        if (option != null) {
                            complete &= addOption(options, declarationsByName, option, field.getName(),
                                    field.getGenericType(), "field:"
                                            + field.getDeclaringClass().getName() + ":" + field.getName());
                        }
                    }
                }
                complete &= interfaceOptions(task.getClass(), options, declarationsByName);
                List<Map<String, Object>> declarations = new ArrayList<>(options.values());
                declarations.sort(Comparator.comparing(option -> (String) option.get("name")));
                Map<String, Object> taskRecord = new LinkedHashMap<>();
                taskRecord.put("path", text(task.getPath()));
                taskRecord.put("options", declarations);
                taskRecord.put("optionsComplete", complete);
                tasks.add(taskRecord);
            }
            tasks.sort(Comparator.comparing(task -> (String) task.get("path")));
            result.put("tasks", tasks);
            String json = JsonOutput.toJson(result);
            if (json.length() > 2 * 1024 * 1024) {
                throw new IllegalStateException("Gradle project metadata limit exceeded.");
            }
            return new Details(json);
        }

        private static Details sourceOutputs(Project project) {
            Map<String, Object> result = new LinkedHashMap<>();
            List<Map<String, Object>> sources = new ArrayList<>();
            JavaPluginExtension java = project.getExtensions().findByType(JavaPluginExtension.class);
            result.put("sourceSetsKnown", true);
            result.put("javaPluginPresent", java != null);
            result.put("artifactSelectionKnown", false);
            if (java != null) {
                for (SourceSet sourceSet : java.getSourceSets()) {
                    if (sources.size() >= 128) {
                        throw new IllegalStateException("Gradle source-set limit exceeded.");
                    }
                    Map<String, Object> source = new LinkedHashMap<>();
                    source.put("name", text(sourceSet.getName()));
                    source.put("classesDirs", outputDirectories(project, sourceSet.getOutput().getClassesDirs()));
                    File resources = sourceSet.getOutput().getResourcesDir();
                    source.put("resourcesDir", resources == null ? null : outputUri(project, resources));
                    source.put("additionalDirs", outputDirectories(project, sourceSet.getOutput().getDirs()));
                    sources.add(source);
                }
            }
            sources.sort(Comparator.comparing(source -> (String) source.get("name")));
            result.put("sourceSets", sources);
            String json = JsonOutput.toJson(result);
            if (json.length() > 2 * 1024 * 1024) {
                throw new IllegalStateException("Gradle output metadata limit exceeded.");
            }
            return new Details(json);
        }

        private static void annotationProcessorOutput(Project project, Task task, Map<String, Object> source) {
            boolean known = task instanceof JavaCompile;
            String uri = null;
            if (task instanceof JavaCompile compile) {
                var directory = compile.getOptions().getGeneratedSourceOutputDirectory().getOrNull();
                if (directory != null) {
                    uri = outputUri(project, directory.getAsFile());
                }
            }
            source.put("annotationProcessorOutputKnown", known);
            source.put("annotationProcessorOutputUri", uri);
        }

        private static List<String> outputDirectories(Project project, Iterable<File> directories) {
            List<String> result = new ArrayList<>();
            for (File directory : directories) {
                if (result.size() >= 256) {
                    throw new IllegalStateException("Gradle output-directory limit exceeded.");
                }
                result.add(outputUri(project, directory));
            }
            result.sort(String::compareTo);
            return result;
        }

        private static String outputUri(Project project, File directory) {
            try {
                Path root = project.getRootProject().getProjectDir().toPath().toRealPath();
                Path path = directory.toPath().toAbsolutePath().normalize();
                if (!path.startsWith(root)) {
                    throw new IllegalStateException();
                }
                Path ancestor = path;
                while (ancestor != null && !Files.exists(ancestor, java.nio.file.LinkOption.NOFOLLOW_LINKS)) {
                    ancestor = ancestor.getParent();
                }
                if (ancestor == null || !ancestor.toRealPath().startsWith(root)) {
                    throw new IllegalStateException();
                }
                String uri = path.toUri().toString();
                // Path.toUri adds a slash for existing directories, which must not change declaration identity.
                return text(uri.endsWith("/") ? uri.substring(0, uri.length() - 1) : uri);
            } catch (Exception unavailable) {
                throw new IllegalStateException("Gradle output declaration is outside its build root or unavailable.");
            }
        }

        private static boolean interfaceOptions(Class<?> taskType,
                Map<String, Map<String, Object>> options, Map<String, String> declarationsByName) {
            ArrayDeque<Class<?>> pending = new ArrayDeque<>();
            HashSet<Class<?>> visited = new HashSet<>();
            HashSet<String> classDeclarations = new HashSet<>(options.keySet());
            for (Class<?> type = taskType; type != null; type = type.getSuperclass()) {
                Class<?>[] interfaces = type.getInterfaces();
                Arrays.sort(interfaces, Comparator.comparing(Class::getName));
                pending.addAll(Arrays.asList(interfaces));
            }
            boolean complete = true;
            while (!pending.isEmpty()) {
                Class<?> implemented = pending.removeFirst();
                if (!visited.add(implemented)) {
                    continue;
                }
                if (visited.size() > 128) {
                    throw new IllegalStateException("Gradle task interface limit exceeded.");
                }
                Method[] methods = implemented.getDeclaredMethods();
                Arrays.sort(methods, Comparator.comparing(Method::toGenericString));
                for (Method method : methods) {
                    Option annotation = method.getAnnotation(Option.class);
                    if (annotation != null) {
                        String name = annotation.option().isEmpty() ? method.getName() : annotation.option();
                        String identity = optionMethodIdentity(method);
                        Map<String, Object> previous = options.get(name);
                        boolean sameDeclaration = identity.equals(declarationsByName.get(name));
                        // Conflicting sibling interface metadata has no documented deterministic precedence.
                        boolean resolved = previous == null || sameDeclaration
                                && (classDeclarations.contains(name)
                                        || annotation.description().equals(previous.get("description")));
                        complete &= resolved && !Modifier.isStatic(method.getModifiers())
                                && Modifier.isPublic(method.getModifiers());
                        Type declaredType = method.getParameterCount() == 1
                                ? method.getGenericParameterTypes()[0] : method.getGenericReturnType();
                        complete &= addOption(options, declarationsByName, annotation, method.getName(),
                                declaredType, identity);
                    }
                }
                Class<?>[] parents = implemented.getInterfaces();
                Arrays.sort(parents, Comparator.comparing(Class::getName));
                pending.addAll(Arrays.asList(parents));
            }
            return complete;
        }

        private static String optionMethodIdentity(Method method) {
            String identity = "method:" + method.getName() + ":"
                    + Arrays.toString(method.getGenericParameterTypes()) + ":"
                    + method.getGenericReturnType().getTypeName();
            int modifiers = method.getModifiers();
            if (Modifier.isStatic(modifiers)
                    || !Modifier.isPublic(modifiers) && !Modifier.isProtected(modifiers)) {
                identity += ":" + method.getDeclaringClass().getName();
            }
            return identity;
        }

        private static boolean addOption(Map<String, Map<String, Object>> options,
                Map<String, String> declarationsByName, Option annotation,
                String defaultName, Type declaredType, String declarationIdentity) {
            String name = text(annotation.option().isEmpty() ? defaultName : annotation.option());
            if (options.size() >= 128 && !options.containsKey(name)) {
                throw new IllegalStateException("Gradle task option limit exceeded.");
            }
            Map<String, Object> option = new LinkedHashMap<>();
            option.put("name", name);
            option.put("description", text(annotation.description()));
            option.put("declaredType", text(declaredType.getTypeName()));
            Class<?> enumType = null;
            if (declaredType instanceof Class<?> type && type.isEnum()) {
                enumType = type;
            } else if (declaredType instanceof ParameterizedType parameterized
                    && parameterized.getActualTypeArguments().length == 1) {
                boolean list = parameterized.getRawType().equals(List.class)
                        || parameterized.getRawType().equals(ListProperty.class);
                Type element = parameterized.getActualTypeArguments()[0];
                if (list && element instanceof WildcardType wildcard
                        && wildcard.getLowerBounds().length == 0
                        && wildcard.getUpperBounds().length == 1) {
                    element = wildcard.getUpperBounds()[0];
                }
                if ((list || parameterized.getRawType().equals(Property.class))
                        && element instanceof Class<?> type && type.isEnum()) {
                    enumType = type;
                }
            }
            List<String> constants = new ArrayList<>();
            if (enumType != null) {
                // Reading enum instances would initialize project-defined classes during declaration projection.
                for (Field field : enumType.getDeclaredFields()) {
                    if (field.isEnumConstant()) {
                        if (constants.size() >= 128) {
                            throw new IllegalStateException("Gradle enum option constant limit exceeded.");
                        }
                        constants.add(text(field.getName()));
                    }
                }
                constants.sort(String::compareTo);
            }
            option.put("enumConstantsKnown", enumType != null);
            option.put("enumConstants", constants);
            Map<String, Object> previous = options.putIfAbsent(name, option);
            String previousIdentity = declarationsByName.putIfAbsent(name, declarationIdentity);
            return previous == null || declarationIdentity.equals(previousIdentity);
        }

        private static String text(String value) {
            if (value.length() > 4096 || value.chars().anyMatch(character -> Character.isISOControl(character)
                    && character != '\n' && character != '\t')) {
                throw new IllegalStateException("Gradle metadata text limit exceeded.");
            }
            return value;
        }
    }
}
