package dev.made.groovy;

import java.io.File;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.io.OutputStream;
import java.net.URI;
import java.nio.file.Path;
import java.nio.file.Files;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.jdt.ls.core.internal.managers.BasicFileDetector;
import org.eclipse.jdt.ls.core.internal.IDelegateCommandHandler;
import org.eclipse.jdt.ls.core.internal.JavaLanguageServerPlugin;
import org.eclipse.jdt.ls.core.internal.preferences.Preferences;
import org.gradle.tooling.CancellationTokenSource;
import org.gradle.tooling.GradleConnectionException;
import org.gradle.tooling.GradleConnector;
import org.gradle.tooling.ProjectConnection;
import org.gradle.tooling.ResultHandler;
import com.google.gson.Gson;

public final class GradleModelHandler implements IDelegateCommandHandler {
    private static final int MAX_TEXT = 4096;
    private static final long TIMEOUT_SECONDS = 60;

    @Override
    public Object executeCommand(String command, List<Object> arguments, IProgressMonitor monitor)
            throws Exception {
        if ("groovy.gradle.projectDetails.v3".equals(command)) {
            return scopedProjectDetails(arguments, monitor, false);
        }
        if ("groovy.gradle.projectDetails.v4".equals(command)) {
            return scopedProjectDetails(arguments, monitor, true);
        }
        boolean includeDetails = "groovy.gradle.projectDetails".equals(command);
        boolean includeOutputs = "groovy.gradle.sourceOutputs".equals(command);
        boolean includeGraph = "groovy.gradle.dependencyGraph".equals(command);
        if ((!"groovy.gradle.projects".equals(command) && !includeDetails && !includeOutputs && !includeGraph) || arguments == null || arguments.size() != 1
                || !(arguments.get(0) instanceof Map<?, ?>)) {
            throw new IllegalArgumentException("Invalid Groovy Gradle model request.");
        }
        Map<?, ?> request = (Map<?, ?>) arguments.get(0);
        if (includeGraph && (!request.keySet().equals(java.util.Set.of("rootUri", "moduleUri", "configuration", "offline"))
                || !Boolean.TRUE.equals(request.get("offline"))
                || !(request.get("moduleUri") instanceof String moduleUri) || moduleUri.length() > MAX_TEXT
                || !(request.get("configuration") instanceof String configuration) || configuration.isEmpty()
                || configuration.getBytes(java.nio.charset.StandardCharsets.UTF_8).length > MAX_TEXT
                || configuration.codePoints().anyMatch(codePoint -> Character.isISOControl(codePoint)
                        || codePoint >= Character.MIN_SURROGATE && codePoint <= Character.MAX_SURROGATE))) {
            throw new IllegalArgumentException("Gradle graphs require an initialized root, module, supported configuration name and explicit offline mode.");
        }
        if (includeOutputs && (!request.keySet().equals(java.util.Set.of("rootUri", "offline"))
                || !Boolean.TRUE.equals(request.get("offline")))) {
            throw new IllegalArgumentException("Gradle output declarations require an initialized root and explicit offline mode.");
        }
        if (!(request.get("rootUri") instanceof String) || ((String) request.get("rootUri")).length() > MAX_TEXT
                || !(request.get("offline") instanceof Boolean)) {
            throw new IllegalArgumentException("A project file URI and explicit offline mode are required.");
        }
        Path root = initializedRoot((String) request.get("rootUri"), "Gradle");
        return queryModel(request, monitor, root, includeDetails, includeOutputs, includeGraph);
    }

    private static Object scopedProjectDetails(List<Object> arguments, IProgressMonitor monitor,
            boolean moduleFenced) {
        try {
            if (arguments == null || arguments.size() != 1 || !(arguments.get(0) instanceof Map<?, ?> request)
                    || !request.keySet().equals(java.util.Set.of("workspaceRootUri", "buildRootUri", "offline"))
                    || !(request.get("workspaceRootUri") instanceof String workspaceUri) || workspaceUri.length() > MAX_TEXT
                    || !(request.get("buildRootUri") instanceof String buildUri) || buildUri.length() > MAX_TEXT
                    || !Boolean.TRUE.equals(request.get("offline"))) {
                throw new IllegalArgumentException();
            }
            GradleImportHandler.ScopedBuild scope = GradleImportHandler.scopedBuild(workspaceUri, buildUri, monitor);
            scope.validate(monitor);
            var moduleContext = moduleFenced ? GradleImportHandler.scopedMetadataContext(scope, monitor) : null;
            if (moduleContext != null) {
                moduleContext.validate();
            }
            Object model = queryModel(request, monitor, scope.build(), true, false, false);
            scope.validate(monitor);
            if (moduleContext != null) {
                moduleContext.validate();
            }
            Map<String, Object> response = Map.of("version", moduleFenced ? 4 : 3,
                    "workspaceRootUri", scope.workspace().toUri().toString(),
                    "buildRootUri", scope.build().toUri().toString(), "model", model,
                    "modelQueryComplete", true, "fullSynchronizationKnown", false);
            if (new Gson().toJson(response).getBytes(StandardCharsets.UTF_8).length > 8 * 1024 * 1024) {
                throw new IllegalArgumentException();
            }
            return response;
        } catch (org.eclipse.core.runtime.OperationCanceledException cancelled) {
            throw new IllegalStateException("Scoped Gradle model discovery cancelled or authority changed. Check current inputs and host settings, then refresh and retry.");
        } catch (Exception | LinkageError failure) {
            if (monitor.isCanceled()) {
                throw new IllegalStateException("Scoped Gradle model discovery cancelled. Refresh to retry.");
            }
            if ("Gradle model discovery timed out. Check the build and refresh.".equals(failure.getMessage())) {
                throw new IllegalStateException("Scoped Gradle model discovery timed out. Check the build and refresh.");
            }
            throw new IllegalStateException("Scoped Gradle model discovery failed. Check initialized folders, visible build inputs, cached Gradle installation, JDK and host settings, then refresh and retry.");
        }
    }

    private static Object queryModel(Map<?, ?> request, IProgressMonitor monitor, Path root,
            boolean includeDetails, boolean includeOutputs, boolean includeGraph) throws Exception {
        String resolutionModule = null;
        if (includeGraph) {
            try {
                URI uri = new URI((String) request.get("moduleUri"));
                if (!"file".equals(uri.getScheme()) || uri.getQuery() != null || uri.getFragment() != null) {
                    throw new IllegalArgumentException();
                }
                Path module = Path.of(uri).toRealPath();
                if (!Files.isDirectory(module) || !module.startsWith(root)) {
                    throw new IllegalArgumentException();
                }
                resolutionModule = module.toString();
            } catch (Exception unavailable) {
                throw new IllegalArgumentException("The Gradle graph module is outside its initialized root or unavailable.");
            }
        }
        Preferences preferences = JavaLanguageServerPlugin.getPreferencesManager().getPreferences();
        if (!root.resolve("settings.gradle").toFile().isFile()
                && !root.resolve("settings.gradle.kts").toFile().isFile()
                && !root.resolve("build.gradle").toFile().isFile()
                && !root.resolve("build.gradle.kts").toFile().isFile()) {
            throw new IllegalArgumentException("No Gradle build file exists in the project folder.");
        }
        List<String> properties = new ArrayList<>();
        Object requestedProperties = request.get("properties");
        if (requestedProperties != null) {
            if (!(requestedProperties instanceof List<?>) || ((List<?>) requestedProperties).size() > 32) {
                throw new IllegalArgumentException("Invalid Gradle project properties.");
            }
            for (Object property : (List<?>) requestedProperties) {
                if (!(property instanceof String) || ((String) property).length() > MAX_TEXT
                        || !((String) property).matches("-P[A-Za-z0-9_.-]+=[^\\p{Cntrl}]*")) {
                    throw new IllegalArgumentException("Invalid Gradle project property.");
                }
                properties.add((String) property);
            }
        }
        List<String> operationArguments = new ArrayList<>(preferences.getGradleArguments());
        // The request controls offline mode; an inherited argument cannot reverse it.
        operationArguments.removeIf(argument -> "--offline".equals(argument));
        if ((Boolean) request.get("offline")) {
            operationArguments.add("--offline");
        }
        operationArguments.addAll(properties);
        if (monitor.isCanceled()) {
            throw new IllegalStateException("Gradle model discovery cancelled. Refresh to retry.");
        }
        GradleConnector connector = GradleConnector.newConnector().forProjectDirectory(root.toFile());
        // --offline applies to the build, not Tooling API's distribution bootstrap.
        if ((Boolean) request.get("offline") && (preferences.getGradleHome() == null
                || preferences.getGradleHome().isEmpty())) {
            throw new IllegalStateException("Offline model discovery requires a configured cached Gradle installation. Set java.import.gradle.home and refresh.");
        }
        if (preferences.getGradleHome() != null && !preferences.getGradleHome().isEmpty()) {
            connector.useInstallation(new File(preferences.getGradleHome()));
        } else if (preferences.getGradleVersion() != null && !preferences.getGradleVersion().isEmpty()) {
            connector.useGradleVersion(preferences.getGradleVersion());
        }
        if (preferences.getGradleUserHome() != null && !preferences.getGradleUserHome().isEmpty()) {
            connector.useGradleUserHomeDir(new File(preferences.getGradleUserHome()));
        }
        CancellationTokenSource cancellation = GradleConnector.newCancellationTokenSource();
        Path initScript = null;
        if (includeDetails || includeOutputs || includeGraph) {
            try {
                initScript = createModelInitScript();
                operationArguments.add("--init-script");
                operationArguments.add(initScript.toString());
            } catch (Exception unavailable) {
                if (initScript != null) {
                    removeInitScript(initScript);
                }
                throw new IllegalStateException("The Gradle metadata provider artifact is unavailable. Check extension setup and refresh.");
            }
        }
        try (ProjectConnection connection = connector.connect()) {
            try {
                CompletableFuture<Map<String, Object>> response = new CompletableFuture<>();
                GradleDiscoveryAction action = includeGraph
                        ? GradleDiscoveryAction.forResolution(root.toString(), resolutionModule, (String) request.get("configuration"))
                        : new GradleDiscoveryAction(root.toString(), includeDetails, includeOutputs);
                org.gradle.tooling.BuildActionExecuter<Map<String, Object>> operation = connection
                        .action(action)
                        .withArguments(operationArguments)
                        .withCancellationToken(cancellation.token())
                        .setStandardOutput(OutputStream.nullOutputStream())
                        .setStandardError(OutputStream.nullOutputStream());
                if (preferences.getGradleJavaHome() != null && !preferences.getGradleJavaHome().isEmpty()) {
                    operation.setJavaHome(new File(preferences.getGradleJavaHome()));
                }
                operation.setJvmArguments(preferences.getGradleJvmArguments());
                operation.run(new ResultHandler<Map<String, Object>>() {
                    @Override
                    public void onComplete(Map<String, Object> value) {
                        response.complete(value);
                    }

                    @Override
                    public void onFailure(GradleConnectionException failure) {
                        response.completeExceptionally(failure);
                    }
                });
                long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(TIMEOUT_SECONDS);
                while (System.nanoTime() < deadline) {
                    if (monitor.isCanceled()) {
                        cancellation.cancel();
                        throw new IllegalStateException("Gradle model discovery cancelled. Refresh to retry.");
                    }
                    try {
                        Map<String, Object> value = response.get(100, TimeUnit.MILLISECONDS);
                        if (new Gson().toJson(value).getBytes(StandardCharsets.UTF_8).length > 8 * 1024 * 1024) {
                            throw new IllegalStateException("Gradle model response size limit exceeded.");
                        }
                        return value;
                    } catch (TimeoutException waiting) {
                        continue;
                    } catch (ExecutionException failure) {
                        // Gradle exceptions may contain secrets from user build scripts or repository URLs.
                        throw new IllegalStateException("Gradle model discovery failed. Check build settings, JDK and cached dependencies; fix the build and refresh.");
                    }
                }
                cancellation.cancel();
                throw new IllegalStateException("Gradle model discovery timed out. Check the build and refresh.");
            } finally {
                // Closing a cancelled connection can wait for user configuration code to unwind.
                if (initScript != null) {
                    removeInitScript(initScript);
                }
            }
        } catch (InterruptedException interrupted) {
            cancellation.cancel();
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Gradle model discovery interrupted. Refresh to retry.");
        } catch (GradleConnectionException failure) {
            throw new IllegalStateException("Gradle connection failed. Check Gradle installation, JDK and network permissions.");
        } finally {
            if (initScript != null) {
                removeInitScript(initScript);
            }
        }
    }

    static Method detectorPredicate() throws ReflectiveOperationException {
        Method predicate = BasicFileDetector.class.getDeclaredMethod("isExcluded", Path.class);
        if (!predicate.trySetAccessible()) {
            throw new IllegalStateException();
        }
        return predicate;
    }

    static int detectorDepth(BasicFileDetector detector) throws ReflectiveOperationException {
        Field depth = BasicFileDetector.class.getDeclaredField("maxDepth");
        if (!depth.trySetAccessible()) {
            throw new IllegalStateException();
        }
        int maximumDepth = depth.getInt(detector);
        if (maximumDepth < 1 || maximumDepth > 32) {
            throw new IllegalStateException();
        }
        return maximumDepth;
    }

    static boolean detectorExcludes(BasicFileDetector detector, Method predicate, Path directory) throws ReflectiveOperationException {
        Object result = predicate.invoke(detector, directory);
        if (!(result instanceof Boolean excluded)) {
            throw new IllegalStateException();
        }
        return excluded;
    }

    static Path initializedRoot(String value, String buildSystem) {
        Path root;
        try {
            URI uri = new URI(value);
            if (!"file".equals(uri.getScheme()) || uri.getQuery() != null || uri.getFragment() != null) {
                throw new IllegalArgumentException();
            }
            root = Path.of(uri).toRealPath();
        } catch (Exception unavailable) {
            throw new IllegalArgumentException("The " + buildSystem + " project root is unavailable.");
        }
        try {
            for (org.eclipse.core.runtime.IPath initialized : JavaLanguageServerPlugin.getPreferencesManager()
                    .getPreferences().getRootPaths()) {
                if (root.equals(initialized.toFile().toPath().toRealPath())) {
                    return root;
                }
            }
        } catch (java.io.IOException unavailable) {
            throw new IllegalArgumentException("An initialized project folder is unavailable. Reopen it and refresh.");
        }
        throw new IllegalArgumentException("The " + buildSystem + " root must match an initialized project folder.");
    }

    static Path createModelInitScript() throws Exception {
        URI location = GradleModelHandler.class.getProtectionDomain().getCodeSource().getLocation().toURI();
        if (!"file".equals(location.getScheme())) {
            throw new IllegalStateException();
        }
        Path providerJar = Path.of(location).toRealPath();
        if (!Files.isRegularFile(providerJar)) {
            throw new IllegalStateException();
        }
        String escaped = providerJar.toString().replace("\\", "\\\\").replace("'", "\\'");
        Path script = Files.createTempFile("groovy-gradle-model-", ".gradle");
        try {
            Files.writeString(script, "initscript { dependencies { classpath files('" + escaped
                    + "') } }\nallprojects { apply plugin: dev.made.groovy.GradleProjectModelPlugin }\n");
            return script;
        } catch (Exception failure) {
            try {
                Files.deleteIfExists(script);
            } catch (java.io.IOException cleanup) {
                failure.addSuppressed(cleanup);
            }
            throw failure;
        }
    }

    static void removeInitScript(Path script) {
        try {
            Files.deleteIfExists(script);
        } catch (java.io.IOException failure) {
            throw new IllegalStateException("Gradle metadata temporary script cleanup failed. Check host permissions.");
        }
    }

}
