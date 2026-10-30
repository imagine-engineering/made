package dev.made.groovy;

import java.io.File;
import java.io.Serializable;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.gradle.tooling.BuildAction;
import org.gradle.tooling.BuildController;
import org.gradle.tooling.model.GradleProject;
import org.gradle.tooling.model.GradleTask;
import org.gradle.tooling.model.gradle.BasicGradleProject;
import org.gradle.tooling.model.gradle.GradleBuild;

public final class GradleDiscoveryAction implements BuildAction<Map<String, Object>>, Serializable {
    private static final long serialVersionUID = 1L;
    private static final int MAX_PROJECTS = 128;
    private static final int MAX_TASKS = 4096;
    private static final int MAX_TEXT = 4096;
    private final String root;
    private final boolean includeDetails;
    private final boolean includeOutputs;
    private final String resolutionModule;
    private final String resolutionConfiguration;
    private transient int responseCharacters;

    public GradleDiscoveryAction(String root) {
        this(root, false);
    }

    public GradleDiscoveryAction(String root, boolean includeDetails) {
        this(root, includeDetails, false);
    }

    public GradleDiscoveryAction(String root, boolean includeDetails, boolean includeOutputs) {
        this(root, includeDetails, includeOutputs, null, null);
    }

    private GradleDiscoveryAction(String root, boolean includeDetails, boolean includeOutputs,
            String resolutionModule, String resolutionConfiguration) {
        this.root = root;
        this.includeDetails = includeDetails;
        this.includeOutputs = includeOutputs;
        this.resolutionModule = resolutionModule;
        this.resolutionConfiguration = resolutionConfiguration;
    }

    public static GradleDiscoveryAction forResolution(String root, String module, String configuration) {
        return new GradleDiscoveryAction(root, false, false, module, configuration);
    }

    @Override
    public Map<String, Object> execute(BuildController controller) {
        responseCharacters = 0;
        Path rootPath = Path.of(root);
        ArrayDeque<GradleBuild> pending = new ArrayDeque<>();
        Set<String> visited = new HashSet<>();
        List<Map<String, Object>> projects = new ArrayList<>();
        int taskCount = 0;
        int projectCount = 0;
        pending.add(controller.getBuildModel());
        while (!pending.isEmpty()) {
            GradleBuild build = pending.removeFirst();
            String buildRoot = confined(rootPath, build.getBuildIdentifier().getRootDir());
            if (!visited.add(buildRoot)) {
                continue;
            }
            if (visited.size() > MAX_PROJECTS || pending.size() > MAX_PROJECTS) {
                throw new IllegalStateException("Gradle build limit exceeded.");
            }
            for (BasicGradleProject basic : build.getProjects()) {
                if (++projectCount > MAX_PROJECTS) {
                    throw new IllegalStateException("Gradle project limit exceeded.");
                }
                String module = confined(rootPath, basic.getProjectDirectory());
                if (resolutionModule != null && !resolutionModule.equals(module)) {
                    continue;
                }
                if (resolutionModule != null) {
                    GradleDependencyGraph graph = controller.getModel(basic, GradleDependencyGraph.class,
                            GradleDependencyGraph.Parameters.class,
                            parameters -> parameters.setConfiguration(resolutionConfiguration));
                    String json = graph.getJson();
                    if (json.length() > 2 * 1024 * 1024) {
                        throw new IllegalStateException("Gradle graph metadata size limit exceeded.");
                    }
                    account(json.length());
                    Map<String, Object> record = new LinkedHashMap<>();
                    record.put("buildRootUri", text(new File(buildRoot).toURI().toString()));
                    record.put("moduleUri", text(new File(module).toURI().toString()));
                    record.put("projectPath", text(basic.getPath()));
                    record.put("name", text(basic.getName()));
                    record.put("graphJson", json);
                    projects.add(record);
                    continue;
                }
                GradleProject model = controller.getModel(basic, GradleProject.class);
                GradleProject project = model.findByPath(basic.getPath());
                if (project == null) {
                    throw new IllegalStateException("Gradle project model is unavailable.");
                }
                Map<String, Object> record = new LinkedHashMap<>();
                record.put("buildRootUri", text(new File(buildRoot).toURI().toString()));
                record.put("moduleUri", text(new File(confined(rootPath, basic.getProjectDirectory())).toURI().toString()));
                record.put("projectPath", text(basic.getPath()));
                record.put("name", text(basic.getName()));
                if (includeOutputs) {
                    String json = controller.getModel(basic, GradleSourceOutputs.class).getJson();
                    if (json.length() > 2 * 1024 * 1024) {
                        throw new IllegalStateException("Gradle output metadata size limit exceeded.");
                    }
                    account(json.length());
                    record.put("outputsJson", json);
                    projects.add(record);
                    continue;
                }
                List<Map<String, Object>> tasks = new ArrayList<>();
                for (GradleTask task : project.getTasks()) {
                    if (++taskCount > MAX_TASKS) {
                        throw new IllegalStateException("Gradle task limit exceeded.");
                    }
                    Map<String, Object> taskRecord = new LinkedHashMap<>();
                    taskRecord.put("path", text(task.getPath()));
                    taskRecord.put("name", text(task.getName()));
                    taskRecord.put("group", text(task.getGroup()));
                    taskRecord.put("description", text(task.getDescription()));
                    // The standard GradleProject model has no task option metadata.
                    taskRecord.put("optionsKnown", false);
                    tasks.add(taskRecord);
                }
                record.put("tasks", tasks);
                if (includeDetails) {
                    GradleProjectDetails details = controller.getModel(basic, GradleProjectDetails.class);
                    String json = details.getJson();
                    if (json.length() > 2 * 1024 * 1024) {
                        throw new IllegalStateException("Gradle metadata size limit exceeded.");
                    }
                    account(json.length());
                    record.put("detailsJson", json);
                }
                projects.add(record);
            }
            for (GradleBuild included : build.getIncludedBuilds()) {
                if (pending.size() >= MAX_PROJECTS) {
                    throw new IllegalStateException("Gradle included-build limit exceeded.");
                }
                pending.addLast(included);
            }
        }
        Map<String, Object> result = new LinkedHashMap<>();
        if (resolutionModule != null && projects.size() != 1) {
            throw new IllegalStateException("The selected Gradle graph module is unavailable or ambiguous.");
        }
        result.put("version", includeDetails ? 2 : 1);
        result.put("rootUri", rootPath.toUri().toString());
        result.put("projects", projects);
        if (!includeOutputs && resolutionModule == null) {
            result.put("sourceSetsKnown", false);
        }
        return result;
    }

    public static final class VerifyClosedArtifactSelection implements BuildAction<Void>, Serializable {
        private static final long serialVersionUID = 1L;

        @Override
        public Void execute(BuildController controller) {
            controller.getModel(GradleClosedArtifactSelection.class);
            return null;
        }
    }

    private static String confined(Path root, File file) {
        try {
            Path path = file.toPath().toRealPath();
            if (!path.startsWith(root)) {
                throw new IllegalStateException();
            }
            return path.toString();
        } catch (Exception error) {
            throw new IllegalStateException("Gradle model path is outside the project folder or unavailable.");
        }
    }

    private String text(String value) {
        if (value == null) {
            return null;
        }
        if (value.length() > MAX_TEXT || value.chars().anyMatch(character -> Character.isISOControl(character)
                && character != '\n' && character != '\t')) {
            throw new IllegalStateException("Gradle model text limit exceeded.");
        }
        account(value.length() + 80);
        return value;
    }

    private void account(int characters) {
        responseCharacters = Math.addExact(responseCharacters, characters);
        // Reserve room for JSON escaping, keys and UTF-8 rather than allocate an oversized reply.
        if (responseCharacters > 1024 * 1024) {
            throw new IllegalStateException("Gradle response size limit exceeded.");
        }
    }
}
