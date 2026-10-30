#!/usr/bin/env python3

import argparse
from contextlib import contextmanager
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time


REPOSITORY = Path(__file__).resolve().parent.parent


def jdtls_probe():
    specification = importlib.util.spec_from_file_location(
        "groovy_jdtls_probe", REPOSITORY / "script/test-groovy-jdtls-provider.py"
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Cannot load existing JDTLS fixture client")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def client_class():
    return jdtls_probe().JdtlsClient


def discover(client, root, properties=None, details=False):
    return client.request("workspace/executeCommand", {
        "command": "groovy.gradle.projectDetails" if details else "groovy.gradle.projects",
        "arguments": [{"rootUri": root.as_uri(), "offline": True, "properties": properties or []}],
    }, timeout=80)


def expect_failure(client, request, expected, forbidden=None):
    try:
        client.request("workspace/executeCommand", request, timeout=80)
    except RuntimeError as failure:
        message = str(failure)
        if expected not in message or forbidden and forbidden in message:
            raise AssertionError("Provider failure did not match its sanitized category") from failure
        return message
    raise AssertionError("Provider accepted an invalid request")


def probe_enum_declarations(arguments):
    provider = arguments.provider.resolve()
    if hashlib.sha256(provider.read_bytes()).hexdigest() != arguments.sha256:
        raise AssertionError("Provider binary differs from supplied build")
    evidence = arguments.receipt.parent
    evidence.mkdir(parents=True, exist_ok=True)
    classpath = os.pathsep.join([str(provider), str(arguments.gradle_home.resolve() / "lib/*"),
                                str(arguments.gradle_home.resolve() / "lib/plugins/*")])
    source = evidence / "EnumDeclarationProbe.java"
    source.write_text(r"""
import java.lang.reflect.*;
import java.util.*;
import org.gradle.api.provider.Property;
import org.gradle.api.provider.ListProperty;
import org.gradle.api.provider.SetProperty;
import org.gradle.api.tasks.options.Option;

public final class EnumDeclarationProbe {
    enum Mode { ZULU, ALPHA; static { System.setProperty("groovy.enum.initialized", "true"); } }
    enum Empty {}
    enum AtLimit { AT_LIMIT }
    enum OverLimit { OVER_LIMIT }
    static class Declarations<T> {
        @Option(option="scalar", description="Enum") public void scalar(Mode value) {}
        @Option(option="property", description="Property") public Property<Mode> property() { throw new AssertionError("getter invoked"); }
        @Option(option="wildcard", description="Wildcard") public Property<? extends Mode> wildcard() { throw new AssertionError("getter invoked"); }
        @SuppressWarnings("rawtypes") @Option(option="raw", description="Raw") public void raw(Property value) {}
        @Option(option="variable", description="Variable") public void variable(T value) {}
        @Option(option="collection", description="Collection") public void collection(List<Mode> value) {}
        @Option(option="listProperty", description="List property") public ListProperty<Mode> listProperty() { throw new AssertionError("getter invoked"); }
        @Option(option="listEmpty", description="Empty list") public void listEmpty(List<Empty> value) {}
        @Option(option="listLimit", description="List limit") public void listLimit(List<AtLimit> value) {}
        @Option(option="listOverflow", description="List overflow") public void listOverflow(List<OverLimit> value) {}
        @Option(option="propertyOverflow", description="Property list overflow") public ListProperty<OverLimit> propertyOverflow() { throw new AssertionError("getter invoked"); }
        @Option(option="listWildcard", description="List wildcard") public void listWildcard(List<? extends Mode> value) {}
        @Option(option="listLower", description="List lower bound") public void listLower(List<? super Mode> value) {}
        @Option(option="propertyLower", description="Property lower bound") public ListProperty<? super Mode> propertyLower() { throw new AssertionError("getter invoked"); }
        @Option(option="listNonEnum", description="List non enum bound") public void listNonEnum(List<? extends String> value) {}
        @Option(option="listVariable", description="List variable") public void listVariable(List<T> value) {}
        @Option(option="nestedList", description="Nested list") public void nestedList(List<List<Mode>> value) {}
        @Option(option="rawList", description="Raw list") @SuppressWarnings("rawtypes") public void rawList(List value) {}
        @Option(option="propertyWildcard", description="Property wildcard") public ListProperty<? extends Mode> propertyWildcard() { throw new AssertionError("getter invoked"); }
        @Option(option="set", description="Set") public void set(Set<Mode> value) {}
        @Option(option="setProperty", description="Set property") public SetProperty<Mode> setProperty() { throw new AssertionError("getter invoked"); }
        @Option(option="listString", description="List string") public void listString(List<String> value) {}
        @Option(option="string", description="String") public void string(String value) {}
        @Option(option="empty", description="Empty") public void empty(Empty value) {}
        @Option(option="limit", description="Limit") public void limit(AtLimit value) {}
        @Option(option="overflow", description="Overflow") public void overflow(OverLimit value) {}
        @Option(option="field", description="Field") public Mode field;
        @Option(option="listField", description="List field") public List<Mode> listField;
    }
    public static void main(String[] arguments) throws Exception {
        Method projection = Class.forName("dev.made.groovy.GradleProjectModelPlugin$Builder")
            .getDeclaredMethod("addOption", Map.class, Map.class, Option.class, String.class, Type.class, String.class);
        projection.setAccessible(true);
        Map<String, Map<String, Object>> options = new LinkedHashMap<>();
        Map<String, String> identities = new LinkedHashMap<>();
        for (Method method : Declarations.class.getDeclaredMethods()) {
            Option annotation = method.getAnnotation(Option.class);
            Type type = method.getParameterCount() == 1 ? method.getGenericParameterTypes()[0] : method.getGenericReturnType();
            if (List.of("overflow", "listOverflow", "propertyOverflow").contains(method.getName())) {
                try {
                    projection.invoke(null, options, identities, annotation, method.getName(), type, method.toGenericString());
                    throw new AssertionError("enum budget accepted");
                } catch (InvocationTargetException failure) {
                    if (!(failure.getCause() instanceof IllegalStateException)
                        || !failure.getCause().getMessage().equals("Gradle enum option constant limit exceeded.")) throw failure;
                }
                System.out.println("PASS: enum overflow rejected " + method.getName());
                continue;
            }
            projection.invoke(null, options, identities, annotation, method.getName(), type, method.toGenericString());
        }
        for (String name : List.of("field", "listField")) {
            Field field = Declarations.class.getDeclaredField(name);
            projection.invoke(null, options, identities, field.getAnnotation(Option.class), field.getName(), field.getGenericType(), name);
        }
        for (String name : List.of("scalar", "property", "field", "collection", "listProperty", "listField", "listWildcard", "propertyWildcard")) {
            Map<String, Object> option = options.get(name);
            if (!Boolean.TRUE.equals(option.get("enumConstantsKnown"))
                || !List.of("ALPHA", "ZULU").equals(option.get("enumConstants"))) throw new AssertionError(name);
            System.out.println("PASS: exact enum " + name);
        }
        for (String name : List.of("wildcard", "raw", "variable", "string", "listLower", "propertyLower", "listNonEnum", "listVariable", "nestedList", "rawList", "set", "setProperty", "listString")) {
            Map<String, Object> option = options.get(name);
            if (!Boolean.FALSE.equals(option.get("enumConstantsKnown"))
                || !List.of().equals(option.get("enumConstants"))) throw new AssertionError(name);
            System.out.println("PASS: unknown enum " + name);
        }
        for (String name : List.of("empty", "listEmpty")) {
            if (!Boolean.TRUE.equals(options.get(name).get("enumConstantsKnown"))
                || !List.of().equals(options.get(name).get("enumConstants"))) throw new AssertionError(name);
            System.out.println("PASS: known empty enum " + name);
        }
        for (String name : List.of("limit", "listLimit")) {
            if (!(options.get(name).get("enumConstants") instanceof List<?> constants) || constants.size() != 128)
                throw new AssertionError(name);
            System.out.println("PASS: exact enum budget boundary " + name);
        }
        if (System.getProperty("groovy.enum.initialized") != null) throw new AssertionError("enum initialized");
        System.out.println("PASS: projection does not initialize enum or invoke getter");
        Mode.class.getEnumConstants();
        if (!"true".equals(System.getProperty("groovy.enum.initialized"))) throw new AssertionError("canary ineffective");
        System.out.println("PASS: enum initialization positive control");
    }
}
""".replace("AT_LIMIT", ",".join(f"V{index}" for index in range(128)))
       .replace("OVER_LIMIT", ",".join(f"V{index}" for index in range(129))))
    commands = [[str(arguments.java_home.resolve() / "bin/javac"), "--release", "17", "-Xlint:all", "-Werror",
                 "-classpath", classpath, "-d", str(evidence), str(source)],
                [str(arguments.java_home.resolve() / "bin/java"), "-Xmx256m", "-classpath", str(evidence) + os.pathsep + classpath,
                 "EnumDeclarationProbe"]]
    receipt = {"provider_sha256": arguments.sha256, "commands": commands, "checks": [],
               "limits": ["Reflection boundary only; no JDTLS, actual Gradle discovery, product UI, Tasks, native or remote proof"]}
    try:
        for index, command in enumerate(commands):
            result = subprocess.run(command, capture_output=True, text=True)
            (evidence / f"reflection-{index}.log").write_text(result.stdout + result.stderr)
            receipt.setdefault("exit_codes", []).append(result.returncode)
            if result.returncode:
                raise RuntimeError("Pinned enum reflection qualification failed")
            if index == 1:
                receipt["checks"] = [line.removeprefix("PASS: ") for line in result.stdout.splitlines() if line.startswith("PASS: ")]
    finally:
        arguments.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def probe_generated_source_declarations(arguments):
    provider = arguments.provider.resolve()
    if hashlib.sha256(provider.read_bytes()).hexdigest() != arguments.sha256:
        raise AssertionError("Provider binary differs from supplied build")
    gradle = arguments.gradle_home.resolve()
    specification = importlib.util.spec_from_file_location(
        "groovy_provider_build", REPOSITORY / "script/build-groovy-gradle-provider.py")
    if specification is None or specification.loader is None:
        raise RuntimeError("Cannot inspect the existing provider SDK pins")
    build_module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(build_module)
    for name, digest in build_module.PINNED_GRADLE_ARTIFACTS.items():
        if hashlib.sha256((gradle / name).read_bytes()).hexdigest() != digest:
            raise AssertionError("Pinned generated-source fixture SDK differs: " + name)
    fixture_jar = gradle / "lib/plugins/gradle-unit-test-fixtures-8.13.jar"
    if hashlib.sha256(fixture_jar.read_bytes()).hexdigest() != "a6d897306cdfb62a0150c0161a9037b69331d6f02adf34f912e5508738f0eb52":
        raise AssertionError("Pinned Gradle ProjectBuilder fixture artifact differs")
    evidence = arguments.receipt.parent
    evidence.mkdir(parents=True, exist_ok=True)
    classpath = os.pathsep.join([str(provider), str(gradle / "lib/*"), str(gradle / "lib/plugins/*")])
    source = evidence / "GeneratedSourceDeclarationProbe.java"
    source.write_text(r"""
import java.io.File;
import java.lang.reflect.*;
import java.nio.file.*;
import java.util.*;
import org.gradle.api.Project;
import org.gradle.api.Task;
import org.gradle.api.tasks.compile.JavaCompile;
import org.gradle.testfixtures.ProjectBuilder;
import org.gradle.internal.concurrent.CompositeStoppable;
import org.gradle.testfixtures.internal.ProjectBuilderImpl;

public final class GeneratedSourceDeclarationProbe {
    private static Method projection;

    private static Map<String, Object> project(Project project, Task task) throws Exception {
        Map<String, Object> record = new LinkedHashMap<>();
        record.put("generatednessKnown", false);
        record.put("roots", List.of("unchanged"));
        projection.invoke(null, project, task, record);
        if (!Boolean.FALSE.equals(record.get("generatednessKnown"))
            || !List.of("unchanged").equals(record.get("roots"))) throw new AssertionError("legacy roots changed");
        return record;
    }

    private static void rejected(Project project, JavaCompile task, File path) throws Exception {
        task.getOptions().getGeneratedSourceOutputDirectory().set(path);
        Map<String, Object> record = new LinkedHashMap<>(Map.of("sentinel", "unchanged"));
        try {
            projection.invoke(null, project, task, record);
            throw new AssertionError("foreign output accepted");
        } catch (InvocationTargetException failure) {
            if (!(failure.getCause() instanceof IllegalStateException)
                || !"Gradle output declaration is outside its build root or unavailable.".equals(failure.getCause().getMessage())) throw failure;
        }
        if (!record.equals(Map.of("sentinel", "unchanged"))) throw new AssertionError("partial projection mutated");
    }

    public static void main(String[] arguments) throws Exception {
        Path root = Path.of(arguments[0]);
        Files.createDirectories(root);
        Project project = ProjectBuilder.builder().withProjectDir(root.toFile())
            .withGradleUserHomeDir(root.resolveSibling("gradle-home").toFile()).build();
        try {
            probe(project, root);
        } finally {
            CompositeStoppable.stoppable(
                (org.gradle.internal.concurrent.Stoppable) () -> ProjectBuilderImpl.stop(project),
                ProjectBuilderImpl.getGlobalServices()).stop();
        }
    }

    private static void probe(Project project, Path root) throws Exception {
        projection = Class.forName("dev.made.groovy.GradleProjectModelPlugin$Builder")
            .getDeclaredMethod("annotationProcessorOutput", Project.class, Task.class, Map.class);
        projection.setAccessible(true);
        Task unsupported = (Task) Proxy.newProxyInstance(Task.class.getClassLoader(), new Class<?>[]{Task.class},
            (proxy, method, values) -> { throw new AssertionError("unknown task callback invoked"); });
        Task ordinary = project.getTasks().register("ordinary").get();
        for (Task task : Arrays.asList(null, unsupported, ordinary)) {
            Map<String, Object> record = project(project, task);
            if (!Boolean.FALSE.equals(record.get("annotationProcessorOutputKnown"))
                || !record.containsKey("annotationProcessorOutputUri") || record.get("annotationProcessorOutputUri") != null)
                throw new AssertionError("unknown task coverage");
        }
        System.out.println("PASS: unknown or absent compile task invokes no callback");
        JavaCompile unconfigured = project.getTasks().register("unconfigured", JavaCompile.class).get();
        Map<String, Object> absent = project(project, unconfigured);
        if (!Boolean.TRUE.equals(absent.get("annotationProcessorOutputKnown"))
            || absent.get("annotationProcessorOutputUri") != null) throw new AssertionError("known absent output");
        System.out.println("PASS: known absent compiler output differs from unknown task");
        project.getPluginManager().apply("java");
        JavaCompile compile = (JavaCompile) project.getTasks().getByName("compileJava");
        Path canary = root.resolve("compiler-action-ran");
        compile.doLast(ignored -> { throw new AssertionError("compiler action ran"); });
        Path output = root.resolve("declared/not-created/generated");
        compile.getOptions().getGeneratedSourceOutputDirectory().set(output.toFile());
        Map<String, Object> missing = project(project, compile);
        String expected = output.toUri().toString();
        if (expected.endsWith("/")) expected = expected.substring(0, expected.length() - 1);
        if (!Boolean.TRUE.equals(missing.get("annotationProcessorOutputKnown"))
            || !expected.equals(missing.get("annotationProcessorOutputUri"))) throw new AssertionError("declared missing output");
        System.out.println("PASS: configured output is reported without creating or compiling sources");
        Files.createDirectories(output);
        Map<String, Object> present = project(project, compile);
        if (!missing.equals(present)) throw new AssertionError("directory existence changes identity");
        System.out.println("PASS: missing and existing output directory identity is stable");
        Path outside = root.resolveSibling("outside-private-canary");
        Files.createDirectories(outside);
        rejected(project, compile, outside.resolve("generated").toFile());
        System.out.println("PASS: foreign declaration rejects before mutating projection");
        Path link = root.resolve("linked-output");
        Files.createSymbolicLink(link, outside);
        try {
            rejected(project, compile, link.resolve("generated").toFile());
        } finally {
            Files.delete(link);
        }
        System.out.println("PASS: symlink escape rejects with the sanitized category");
        compile.getOptions().getGeneratedSourceOutputDirectory().set(output.toFile());
        if (!missing.equals(project(project, compile))) throw new AssertionError("declaration recovery");
        System.out.println("PASS: configured compiler output recovers after failed declarations");
        Object builder = Class.forName("dev.made.groovy.GradleProjectModelPlugin$Builder").getConstructor().newInstance();
        Object details = builder.getClass().getMethod("buildAll", String.class, Project.class)
            .invoke(builder, "dev.made.groovy.GradleProjectDetails", project);
        String json = (String) details.getClass().getMethod("getJson").invoke(details);
        if (!json.contains("annotationProcessorOutputKnown") || !json.contains(expected)) throw new AssertionError("actual source-set projection omitted output");
        if (project.getPlugins().hasPlugin("idea") || Files.exists(canary)
            || Files.exists(root.resolve("build/classes"))) throw new AssertionError("unexpected plugin or compiler action");
        System.out.println("PASS: actual source-set builder reuses the fact without an IDEA plugin or compiler action");
        System.out.println("OUTPUT_URI: " + expected);
    }
}
""")
    subprocess.run([str(arguments.java_home.resolve() / "bin/javac"), "--release", "17", "-Xlint:all", "-Werror",
                    "-classpath", classpath, "-d", str(evidence), str(source)], check=True)
    checks = []
    identities = []
    with tempfile.TemporaryDirectory(prefix="groovy-generated-declarations-") as directory:
        project_root = Path(directory).resolve() / "project"
        project_root.mkdir()
        environment = dict(os.environ, JAVA_HOME=str(arguments.gradle_java_home.resolve()), GRADLE_USER_HOME=str(project_root.parent / "gradle-home"))
        for label in ["fresh", "restart"]:
            # Each JVM has fresh Gradle objects; the fixture directory persists across the restart.
            log_path = evidence / (label + ".log")
            with log_path.open("w") as output:
                try:
                    result = subprocess.run([str(arguments.gradle_java_home.resolve() / "bin/java"),
                                             "--add-opens=java.base/java.lang=ALL-UNNAMED",
                                             "-Dorg.gradle.native=false", "-classpath", os.pathsep.join([str(evidence), classpath]),
                                             "GeneratedSourceDeclarationProbe", str(project_root)],
                                            env=environment, text=True, stdout=output, stderr=subprocess.STDOUT, timeout=90)
                except subprocess.TimeoutExpired as error:
                    raise RuntimeError("Generated-source declaration probe timed out; inspect retained " + label + ".log") from error
            output_text = log_path.read_text()
            if result.returncode:
                raise RuntimeError("Generated-source declaration probe failed; inspect retained " + label + ".log")
            names = [line.removeprefix("PASS: ") for line in output_text.splitlines() if line.startswith("PASS: ")]
            if len(names) != 8:
                raise AssertionError("Generated-source declaration probe did not run all eight cases")
            checks.extend(label + ": " + name for name in names)
            identities.append(next(line.removeprefix("OUTPUT_URI: ") for line in output_text.splitlines() if line.startswith("OUTPUT_URI: ")))
        if identities[0] != identities[1]:
            raise AssertionError("Fresh and restarted compiler declaration identities differ")
    receipt = {"provider_sha256": arguments.sha256, "checks": checks,
               "compiler_java_home": str(arguments.java_home.resolve()),
               "runtime_java_home": str(arguments.gradle_java_home.resolve()),
               "limits": ["Actual Gradle8.13 ProjectBuilder/production Builder boundary only; no JDTLS/Tooling API worker, installed provider or native/remote qualification",
                          "Configured Java annotation-processor output only; no successful generation or complete Java/Kotlin generator coverage"]}
    arguments.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def probe(arguments):
    provider = arguments.provider.resolve()
    if hashlib.sha256(provider.read_bytes()).hexdigest() != arguments.sha256:
        raise AssertionError("Provider binary differs from the supplied source build")
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"provider_sha256": arguments.sha256, "checks": [], "limits": [
        "No installed extension/default download, mounted UI, native app or remote qualification",
        "Declared roots, Java compiler annotation-processor output and method/field/enum declarations only; successful generation, other Java/Kotlin generators, custom accepted values and dynamic suggestions remain unqualified",
    ]}
    gradle_environment = dict(os.environ, JAVA_HOME=str(arguments.gradle_java_home.resolve()),
                              GRADLE_USER_HOME=str(arguments.gradle_user_home.resolve()))
    gradle_environment["JAVA_TOOL_OPTIONS"] = (
        gradle_environment.get("JAVA_TOOL_OPTIONS", "") + " -Djava.net.preferIPv4Stack=true"
    ).strip()
    with tempfile.TemporaryDirectory(prefix="groovy-gradle-model-") as directory:
        root = (Path(directory) / "project").resolve()
        shutil.copytree(REPOSITORY / "crates/project/test_data/groovy_jvm_gradle", root,
                        ignore=shutil.ignore_patterns("build", ".gradle", ".git"))
        application_build = root / "application/build.gradle.kts"
        application_build.write_text(application_build.read_text() + '''
val parityDeclarationParent = configurations.create("parityDeclarationParent") {
    isCanBeResolved = false
    isCanBeConsumed = false
}
configurations.create("parityUnresolved") {
    description = "Declared fixture; resolution must not run"
    isCanBeResolved = true
    isCanBeConsumed = false
    extendsFrom(parityDeclarationParent)
    incoming.beforeResolve { throw GradleException("GROOVY_CONFIGURATION_RESOLUTION_CANARY") }
}
dependencies { add("parityUnresolved", "evidence:deliberately-missing:1.0") }
''')
        host_temporary = Path(directory) / "host-temp"
        host_temporary.mkdir()
        previous_java_options = os.environ.get("JAVA_TOOL_OPTIONS")
        os.environ["JAVA_TOOL_OPTIONS"] = f"{previous_java_options or ''} -Djava.io.tmpdir={host_temporary}".strip()
        try:
            client = client_class()(
                arguments.launcher.resolve(), arguments.java_home.resolve(), root,
                Path(directory) / "jdtls-data", deny_external_network=True,
                gradle_user_home=arguments.gradle_user_home.resolve(),
            )
        finally:
            if previous_java_options is None:
                os.environ.pop("JAVA_TOOL_OPTIONS", None)
            else:
                os.environ["JAVA_TOOL_OPTIONS"] = previous_java_options
        settings = {"java": {"import": {
            "maven": {"enabled": False},
            "gradle": {
                "enabled": False,
                "java": {"home": str(arguments.gradle_java_home.resolve())},
                "user": {"home": str(arguments.gradle_user_home.resolve())},
                "home": str(arguments.gradle_home.resolve()),
            },
        }}}
        try:
            initialized = client.request("initialize", {
                "processId": os.getpid(), "rootUri": root.as_uri(),
                "workspaceFolders": [{"uri": root.as_uri(), "name": root.name}],
                "capabilities": {"workspace": {"workspaceFolders": True, "configuration": True}},
                "initializationOptions": {"bundles": [str(provider)], "settings": settings},
            }, timeout=90)
            commands = initialized["capabilities"]["executeCommandProvider"]["commands"]
            if not {"groovy.gradle.projects", "groovy.gradle.projectDetails"}.issubset(commands):
                raise AssertionError("JDTLS did not register the model command")
            client.notify("initialized", {})
            response = discover(client, root, ["-PgroovyParity=true"])
            (arguments.receipt.parent / "model-response.json").write_text(json.dumps(response, indent=2) + "\n")
            if response["version"] != 1 or response["sourceSetsKnown"]:
                raise AssertionError("Provider protocol or unknown source-set claim differs")
            projects = response["projects"]
            if len(projects) != 4:
                raise AssertionError(f"Expected root, two modules and included build; found {len(projects)}")
            identities = [(project["buildRootUri"], project["projectPath"]) for project in projects]
            if len(set(identities)) != len(identities):
                raise AssertionError("Composite project identities collide")
            task = next(task for project in projects for task in project["tasks"]
                        if task["path"] == ":application:parityArguments")
            if task["group"] != "groovy parity" or task["optionsKnown"]:
                raise AssertionError("Task group or unknown option metadata differs")
            if not any(task["path"] == ":parityIncluded" for project in projects for task in project["tasks"]):
                raise AssertionError("Included build tasks were omitted")
            if list(root.rglob("groovy-parity-message*")) or (root / "application/build/classes").exists():
                raise AssertionError("Discovery executed a fixture build task")
            again = discover(client, root, ["-PgroovyParity=true"])
            if projects != again["projects"]:
                raise AssertionError("Repeat discovery changed stable model identities")
            receipt["checks"].extend(["cached composite model", "stable identities", "task group/path", "no requested tasks"])
            detailed = discover(client, root, ["-PgroovyParity=true"], details=True)
            (arguments.receipt.parent / "details-response.json").write_text(json.dumps(detailed, indent=2) + "\n")
            if detailed["version"] != 2 or len(detailed["projects"]) != 4:
                raise AssertionError("Extended composite model differs")
            application = next(project for project in detailed["projects"] if project["projectPath"] == ":application")
            metadata = json.loads(application["detailsJson"])
            if metadata.get("configurationsKnown") is not True or metadata.get("dependencyGraphKnown") is not False:
                raise AssertionError("Configuration declaration or unknown dependency-graph coverage differs")
            configurations = {configuration["name"]: configuration for configuration in metadata["configurations"]}
            if len(configurations) != len(metadata["configurations"]):
                raise AssertionError("Configuration identities collide")
            unresolved = configurations["parityUnresolved"]
            if unresolved != {
                "name": "parityUnresolved", "description": "Declared fixture; resolution must not run",
                "resolvable": True, "consumable": False, "extendsFrom": ["parityDeclarationParent"],
            }:
                raise AssertionError("Declared custom configuration roles or inheritance differ")
            if configurations["implementation"]["resolvable"] or not configurations["runtimeClasspath"]["resolvable"]:
                raise AssertionError("Java configuration roles differ")
            if "implementation" not in configurations["runtimeClasspath"]["extendsFrom"]:
                raise AssertionError("Java configuration inheritance missing")
            for project in detailed["projects"]:
                if json.loads(project["detailsJson"]).get("configurationsKnown") is not True:
                    raise AssertionError("Composite project declaration coverage missing")
            repeated_details = discover(client, root, ["-PgroovyParity=true"], details=True)
            if detailed != repeated_details:
                raise AssertionError("Repeat detail discovery changed declared configuration identities")
            receipt["checks"].extend([
                "declared configuration roles and immediate inheritance", "composite configuration coverage",
                "stable configuration identities", "unresolved configuration guard; dependency graph stays unknown",
            ])
            if not metadata["javaSourceSetsKnown"]:
                raise AssertionError("Java source sets are unknown for a Java module")
            sources = {source["name"]: source for source in metadata["sourceSets"]}
            if not {"main", "test"}.issubset(sources):
                raise AssertionError("Main/test source sets missing")
            for name in ["main", "test"]:
                expected = (root / f"application/src/{name}/java").as_uri()
                if not any(uri.rstrip("/") == expected for uri in sources[name]["roots"]):
                    raise AssertionError("Source-set root differs")
                if sources[name]["generatednessKnown"]:
                    raise AssertionError("Generatedness was inferred without evidence")
            generated = (root / "application/build/generated/sources/groovyFixture/java").as_uri()
            if not any(uri.rstrip("/") == generated for uri in sources["main"]["roots"]):
                raise AssertionError("Declared generated-source root missing")
            for name in ["main", "test"]:
                expected_output = (root / f"application/build/generated/sources/annotationProcessor/java/{name}").as_uri()
                if sources[name].get("annotationProcessorOutputKnown") is not True or sources[name].get("annotationProcessorOutputUri") != expected_output:
                    raise AssertionError("Configured Java compiler output declaration is missing or wrong")
            receipt["checks"].append("actual cached main/test annotation-processor output declarations")

            def compiler_output(response):
                project = next(project for project in response["projects"] if project["projectPath"] == ":application")
                return next(source for source in json.loads(project["detailsJson"])["sourceSets"] if source["name"] == "main")

            original_generated_build = application_build.read_text()
            declared_output = root / "application/declared-processor-output"
            try:
                application_build.write_text(original_generated_build + '\n' + '''
tasks.named<JavaCompile>("compileJava") {
    options.generatedSourceOutputDirectory.set(layout.projectDirectory.dir("declared-processor-output"))
}
''')
                missing_output = compiler_output(discover(client, root, details=True))
                if missing_output.get("annotationProcessorOutputKnown") is not True or missing_output.get("annotationProcessorOutputUri") != declared_output.as_uri() or declared_output.exists():
                    raise AssertionError("Configured generated-source output was inferred, omitted or executed")
                receipt["checks"].append("configured missing compiler output stays a declaration without task execution")
                declared_output.mkdir()
                existing_output = compiler_output(discover(client, root, details=True))
                if missing_output != existing_output:
                    raise AssertionError("Existing directory changed declared compiler output identity")
                receipt["checks"].append("actual model output identity survives directory creation")
                application_build.write_text(original_generated_build + '\n' + '''
tasks.named<JavaCompile>("compileJava") {
    options.generatedSourceOutputDirectory.convention(null as org.gradle.api.file.Directory?)
    options.generatedSourceOutputDirectory.set(null as java.io.File?)
}
''')
                absent_output = compiler_output(discover(client, root, details=True))
                if absent_output.get("annotationProcessorOutputKnown") is not True or "annotationProcessorOutputUri" not in absent_output or absent_output["annotationProcessorOutputUri"] is not None:
                    raise AssertionError("Known absent compiler output differs from provider reply")
                receipt["checks"].append("known absent compiler declaration differs from unknown coverage")
                outside_output = root.parent / "private-generated-output-canary"
                outside_output.mkdir()
                for label, output in [("outside root", outside_output), ("symlink escape", root / "application/processor-link")]:
                    if label == "symlink escape":
                        output.symlink_to(outside_output, target_is_directory=True)
                    try:
                        application_build.write_text(original_generated_build + '\n' +
                            'tasks.named<JavaCompile>("compileJava") { options.generatedSourceOutputDirectory.set(file('
                            + json.dumps(str(output / "generated")) + ')) }\n')
                        expect_failure(client, {"command": "groovy.gradle.projectDetails", "arguments": [
                            {"rootUri": root.as_uri(), "offline": True}
                        ]}, "Gradle model discovery failed", forbidden="private-generated-output-canary")
                        receipt["checks"].append(label + " compiler output rejected with sanitized failure")
                    finally:
                        if label == "symlink escape":
                            output.unlink()
            finally:
                application_build.write_text(original_generated_build)
            repaired_output = compiler_output(discover(client, root, details=True))
            if repaired_output.get("annotationProcessorOutputKnown") is not True or repaired_output.get("annotationProcessorOutputUri") != (root / "application/build/generated/sources/annotationProcessor/java/main").as_uri():
                raise AssertionError("Compiler output did not recover after repair")
            receipt["checks"].append("compiler output model recovers after declaration repair")
            options = next(task for task in metadata["tasks"] if task["path"] == ":application:parityArguments")
            message = next(option for option in options["options"] if option["name"] == "message")
            if not options["optionsComplete"] or message["description"] != "Fixture message" or "Property" not in message["declaredType"]:
                raise AssertionError("Declared message option differs")
            included = next(project for project in detailed["projects"] if project["moduleUri"].rstrip("/") == (root / "included").as_uri().replace("file:///", "file:/"))
            if "detailsJson" not in included:
                raise AssertionError("Included build metadata missing")
            if list(root.rglob("groovy-parity-message*")) or (root / "application/build/classes").exists():
                raise AssertionError("Extended discovery executed selected tasks")
            receipt["checks"].extend(["extended composite metadata", "named main/test roots", "declared generated root without inferred generatedness", "declared custom option", "metadata query does not execute selected tasks"])
            if list(host_temporary.glob("groovy-gradle-model-*.gradle")):
                raise AssertionError("Successful metadata query retained its temporary script")
            receipt["checks"].append("successful metadata temporary-script cleanup")
            application_build = root / "application/build.gradle.kts"
            original_application = application_build.read_text()
            try:
                application_build.write_text(original_application + '''
interface ParityInterfaceOption {
    @org.gradle.api.tasks.options.Option(option = "mode", description = "Interface mode")
    fun setMode(value: String)
}
abstract class ParityInterfaceTask : DefaultTask(), ParityInterfaceOption {
    override fun setMode(value: String) {}
}
tasks.register<ParityInterfaceTask>("parityInterface")
interface ParityInterfaceChild : ParityInterfaceOption
interface ParityInterfaceOtherChild : ParityInterfaceOption
abstract class ParityDiamondInterfaceTask : DefaultTask(), ParityInterfaceChild, ParityInterfaceOtherChild {
    override fun setMode(value: String) { error("interface method must not run during discovery") }
}
interface ParityConflictingInterfaceOption {
    @org.gradle.api.tasks.options.Option(option = "mode", description = "Other interface mode")
    fun setOtherMode(value: String)
}
abstract class ParityConflictingInterfacesTask : DefaultTask(), ParityInterfaceOption, ParityConflictingInterfaceOption {
    override fun setMode(value: String) {}
    override fun setOtherMode(value: String) {}
}
abstract class ParityClassInterfaceTask : DefaultTask(), ParityInterfaceOption {
    @org.gradle.api.tasks.options.Option(option = "mode", description = "Class mode")
    override fun setMode(value: String) {}
}
tasks.register<ParityDiamondInterfaceTask>("parityDiamondInterface")
tasks.register<ParityConflictingInterfacesTask>("parityConflictingInterfaces")
tasks.register<ParityClassInterfaceTask>("parityClassInterface")
interface ParityEquivalentInterfaceOption {
    @org.gradle.api.tasks.options.Option(option = "mode", description = "Interface mode")
    fun setMode(value: String)
}
abstract class ParityEquivalentInterfacesTask : DefaultTask(), ParityInterfaceOption, ParityEquivalentInterfaceOption {
    override fun setMode(value: String) {}
}
tasks.register<ParityEquivalentInterfacesTask>("parityEquivalentInterfaces")



abstract class ParityDistinctSetters : DefaultTask() {
    @org.gradle.api.tasks.options.Option(option = "mode", description = "Collision mode")
    fun setFirst(value: String) {}
    @org.gradle.api.tasks.options.Option(option = "mode", description = "Collision mode")
    fun setSecond(value: String) {}
}
abstract class ParityOptionBase : DefaultTask() {
    @org.gradle.api.tasks.options.Option(option = "mode", description = "Override mode")
    open fun setMode(value: String) {}
}
abstract class ParityEqualOverride : ParityOptionBase() {
    @org.gradle.api.tasks.options.Option(option = "mode", description = "Override mode")
    override fun setMode(value: String) {}
}
abstract class ParityConflictOverride : ParityOptionBase() {
    @org.gradle.api.tasks.options.Option(option = "mode", description = "Changed mode")
    override fun setMode(value: String) {}
}
abstract class ParityFieldBase : DefaultTask() {
    @field:org.gradle.api.tasks.options.Option(option = "mode", description = "Field mode")
    private var mode: String = "base"
}
abstract class ParityShadowFields : ParityFieldBase() {
    @field:org.gradle.api.tasks.options.Option(option = "mode", description = "Field mode")
    private var mode: String = "derived"
}
tasks.register<ParityDistinctSetters>("parityDistinctSetters")
tasks.register<ParityEqualOverride>("parityEqualOverride")
tasks.register<ParityConflictOverride>("parityConflictOverride")
tasks.register<ParityShadowFields>("parityShadowFields")

abstract class ParityPrivateMethodBase : DefaultTask() {
    @org.gradle.api.tasks.options.Option(option = "mode", description = "Hidden mode")
    private fun setMode(value: String) {}
}
abstract class ParityPrivateMethodHidden : ParityPrivateMethodBase() {
    @org.gradle.api.tasks.options.Option(option = "mode", description = "Hidden mode")
    private fun setMode(value: String) {}
}
abstract class ParityStaticMethodBase : DefaultTask() {
    companion object {
        @JvmStatic
        @org.gradle.api.tasks.options.Option(option = "mode", description = "Static mode")
        fun setMode(value: String) {}
    }
}
abstract class ParityStaticMethodHidden : ParityStaticMethodBase() {
    companion object {
        @JvmStatic
        @org.gradle.api.tasks.options.Option(option = "mode", description = "Static mode")
        fun setMode(value: String) {}
    }
}
tasks.register<ParityPrivateMethodHidden>("parityPrivateMethodHidden")
tasks.register<ParityStaticMethodHidden>("parityStaticMethodHidden")

enum class ParityEnumMode { ZULU, ALPHA;
    companion object { init { File(ENUM_CANARY_MARKER).writeText("initialized") } }
}
abstract class ParityEnumOptions : DefaultTask() {
    @org.gradle.api.tasks.options.Option(option = "scalar", description = "Scalar enum")
    fun setScalar(value: ParityEnumMode) {}
    @get:org.gradle.api.tasks.Input
    @get:org.gradle.api.tasks.options.Option(option = "property", description = "Property enum")
    abstract val propertyMode: Property<ParityEnumMode>
    @org.gradle.api.tasks.options.Option(option = "collection", description = "Collection enum")
    fun setCollection(value: List<ParityEnumMode>) {}
    @get:org.gradle.api.tasks.Input
    @get:org.gradle.api.tasks.options.Option(option = "listProperty", description = "List property enum")
    abstract val listMode: ListProperty<ParityEnumMode>
    @org.gradle.api.tasks.options.Option(option = "string", description = "String declaration")
    fun setString(value: String) {}
    @org.gradle.api.tasks.options.OptionValues("string")
    fun suggestModes(): List<String> = throw GradleException("GROOVY_OPTION_VALUES_CANARY")
}
tasks.register<ParityEnumOptions>("parityEnumOptions")
'''.replace("ENUM_CANARY_MARKER", json.dumps(str(root / "enum-initialized-marker"))))
                budget_methods = "\n".join(
                    '@org.gradle.api.tasks.options.Option(option = "budget' + str(index)
                    + '", description = "Budget") fun setBudget' + str(index) + '(value: String) {}'
                    for index in range(127))
                application_build.write_text(application_build.read_text() + '\n' + '''
abstract class ParityOptionBudgetBase : DefaultTask() {
    @org.gradle.api.tasks.options.Option(option = "mode", description = "Base mode")
    open fun setMode(value: String) {}
''' + budget_methods + '\n}\n' + '''
abstract class ParityOptionBudgetChild : ParityOptionBudgetBase(), ParityInterfaceOption {
    @org.gradle.api.tasks.options.Option(option = "mode", description = "Changed mode")
    override fun setMode(value: String) {}
}
tasks.register<ParityOptionBudgetChild>("parityOptionBudget")
''')
                interface_response = discover(client, root, details=True)
                interface_project = next(project for project in interface_response["projects"]
                                         if project["projectPath"] == ":application")
                interface_task = next(task for task in json.loads(interface_project["detailsJson"])["tasks"]
                                      if task["path"] == ":application:parityInterface")
                if not interface_task["optionsComplete"] or [option["name"] for option in interface_task["options"]] != ["mode"]:
                    raise AssertionError("Unique interface declaration missing or incomplete")
                interface_mode = interface_task["options"][0]
                if interface_mode["description"] != "Interface mode" or interface_mode["declaredType"] != "java.lang.String":
                    raise AssertionError("Unique interface declaration metadata differs")
                receipt["checks"].append("unique interface-only declaration")
                declared_tasks = {task["path"]: task for task in json.loads(interface_project["detailsJson"])["tasks"]}
                budget_task = declared_tasks[":application:parityOptionBudget"]
                budget_options = {option["name"]: option for option in budget_task["options"]}
                if not budget_task["optionsComplete"] or len(budget_options) != 128:
                    raise AssertionError("Exact distinct-name budget rejects repeated override declarations")
                if budget_options["mode"]["description"] != "Changed mode":
                    raise AssertionError("Exact option budget changed nearest-class precedence")
                receipt["checks"].append("exact 128-name budget admits class and interface override declarations")
                receipt["checks"].append("exact option-name budget preserves closest class description")
                diamond_task = declared_tasks[":application:parityDiamondInterface"]
                if not diamond_task["optionsComplete"] or diamond_task["options"] != interface_task["options"]:
                    raise AssertionError("Inherited diamond interface declaration differs")
                receipt["checks"].append("inherited diamond interface deduplicated without invocation")
                collision = declared_tasks[":application:parityConflictingInterfaces"]
                if collision["optionsComplete"] or [option["name"] for option in collision["options"]] != ["mode"]:
                    raise AssertionError("Conflicting sibling interface collision certified or lost declaration")
                receipt["checks"].append("conflicting sibling interface declarations remain incomplete")
                class_interface = declared_tasks[":application:parityClassInterface"]
                if not class_interface["optionsComplete"] or class_interface["options"][0]["description"] != "Class mode":
                    raise AssertionError("Matching class interface override incomplete or wrong precedence")
                receipt["checks"].append("class declaration takes precedence over matching interface")
                equivalent = declared_tasks[":application:parityEquivalentInterfaces"]
                if not equivalent["optionsComplete"] or equivalent["options"] != interface_task["options"]:
                    raise AssertionError("Equivalent interface declarations are not complete")
                receipt["checks"].append("equivalent sibling interface declarations complete")

                for name, complete in [("parityDistinctSetters", False), ("parityEqualOverride", True),
                                       ("parityConflictOverride", True), ("parityShadowFields", False),
                                       ("parityPrivateMethodHidden", False), ("parityStaticMethodHidden", False)]:
                    task = declared_tasks[f":application:{name}"]
                    if task["optionsComplete"] is not complete or len(task["options"]) != 1:
                        raise AssertionError("Gradle option declaration collision coverage differs")
                    if task["options"][0]["name"] != "mode":
                        raise AssertionError("Unknown collision removed its bounded declaration")
                    receipt["checks"].append(f"option declaration identity: {name} coverage {complete}")
                overridden = declared_tasks[":application:parityConflictOverride"]
                if overridden["options"][0]["description"] != "Changed mode":
                    raise AssertionError("Closest class option description not preserved")
                receipt["checks"].append("closest class overrides inherited option description")
                for task_name, description in [("parityConflictOverride", "Changed mode"), ("parityClassInterface", "Class mode")]:
                    command = [str(arguments.gradle_home / "bin/gradle"), "--offline", "--console=plain",
                               "--gradle-user-home", str(arguments.gradle_user_home), "help", "--task", "application:" + task_name]
                    help_result = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE,
                                                 stderr=subprocess.STDOUT, timeout=60,
                                                 env=gradle_environment)
                    (arguments.receipt.parent / (task_name + "-help.log")).write_text(help_result.stdout)
                    if help_result.returncode != 0 or description not in help_result.stdout or "--mode" not in help_result.stdout:
                        raise AssertionError("Pinned Gradle help differs from effective option declaration")
                    receipt["checks"].append("pinned help confirms option precedence: " + task_name)
                (arguments.receipt.parent / "option-enum-response.json").write_text(json.dumps(interface_response, indent=2) + "\n")
                enum_options = {option["name"]: option for option in declared_tasks[":application:parityEnumOptions"]["options"]}
                for name in ["scalar", "property", "collection", "listProperty"]:
                    if enum_options[name].get("enumConstantsKnown") is not True or enum_options[name].get("enumConstants") != ["ALPHA", "ZULU"]:
                        raise AssertionError(f"Bounded enum declaration metadata differs for {name}")
                    receipt["checks"].append(f"bounded enum declaration: {name}")
                for name in ["string"]:
                    if enum_options[name].get("enumConstantsKnown") is not False or enum_options[name].get("enumConstants") != []:
                        raise AssertionError("Unsupported enum declaration falsely known")
                    receipt["checks"].append(f"unsupported enum declaration unknown: {name}")
                if (root / "enum-initialized-marker").exists():
                    raise AssertionError("Enum declaration projection initialized project enum")
                receipt["checks"].append("enum initialization and dynamic OptionValues canaries remain untouched")
                repeated = discover(client, root, details=True)
                if repeated != interface_response:
                    raise AssertionError("Collision option declarations are not stable")
                if list(root.rglob("groovy-parity-message*")) or (root / "application/build/classes").exists():
                    raise AssertionError("Collision discovery executed a requested build task")
                receipt["checks"].append("stable collision declarations without requested task execution")
            finally:
                application_build.write_text(original_application)
            try:
                overflow_methods = "\n".join(
                    '@org.gradle.api.tasks.options.Option(option = "budget' + str(index)
                    + '", description = "Budget") fun setBudget' + str(index) + '(value: String) {}'
                    for index in range(129))
                application_build.write_text(original_application + '\nabstract class ParityPrivateOptionOverflow : DefaultTask() {\n'
                                             + overflow_methods + '\n}\n'
                                             + 'tasks.register<ParityPrivateOptionOverflow>("parityOptionOverflow")\n')
                expect_failure(client, {"command": "groovy.gradle.projectDetails", "arguments": [
                    {"rootUri": root.as_uri(), "offline": True}
                ]}, "Gradle model discovery failed", forbidden="ParityPrivateOptionOverflow")
                receipt["checks"].append("129th distinct option name fails with sanitized provider category")
            finally:
                application_build.write_text(original_application)
            discover(client, root, details=True)
            receipt["checks"].append("recovery after distinct option-name overflow")
            try:
                constants = ", ".join(f"V{index}" for index in range(129))
                application_build.write_text(original_application + "\nenum class ParityOverflowMode { " + constants + " }\n" + '''
abstract class ParityEnumOverflow : DefaultTask() {
    @org.gradle.api.tasks.options.Option(option = "mode", description = "Enum budget")
    fun setMode(value: ParityOverflowMode) {}
}
tasks.register<ParityEnumOverflow>("parityEnumOverflow")
''')
                expect_failure(client, {"command": "groovy.gradle.projectDetails", "arguments": [
                    {"rootUri": root.as_uri(), "offline": True}
                ]}, "Gradle model discovery failed", forbidden="ParityOverflowMode")
                receipt["checks"].append("enum declaration overflow fails with sanitized provider category")
            finally:
                application_build.write_text(original_application)
            discover(client, root, details=True)
            receipt["checks"].append("recovery after enum declaration overflow")
            try:
                interfaces = "\n".join(f"interface ParityPrivateInterface{index}" for index in range(129))
                implemented = ", ".join(f"ParityPrivateInterface{index}" for index in range(129))
                application_build.write_text(original_application + "\n" + interfaces + "\n"
                                             + f"abstract class ParityInterfaceOverflow : DefaultTask(), {implemented}\n"
                                             + 'tasks.register<ParityInterfaceOverflow>("parityInterfaceOverflow")\n')
                expect_failure(client, {"command": "groovy.gradle.projectDetails", "arguments": [
                    {"rootUri": root.as_uri(), "offline": True}
                ]}, "Gradle model discovery failed", forbidden="ParityPrivateInterface")
                receipt["checks"].append("interface declaration overflow fails with sanitized provider category")
            finally:
                application_build.write_text(original_application)
            discover(client, root, details=True)
            receipt["checks"].append("recovery after interface declaration overflow")

            try:
                application_build.write_text(original_application + '''
abstract class ParityGetterCanary : DefaultTask() {
    @get:org.gradle.api.tasks.options.Option(option = "never", description = "Metadata only")
    val never: Property<String>
        get() = throw GradleException("GROOVY_OPTION_GETTER_CANARY")
}
tasks.register<ParityGetterCanary>("parityGetterCanary")
''')
                expect_failure(client, {"command": "groovy.gradle.projectDetails", "arguments": [
                    {"rootUri": root.as_uri(), "offline": True}
                ]}, "Gradle model discovery failed", forbidden="GROOVY_OPTION_GETTER_CANARY")
                receipt["checks"].append("Gradle task creation getter failure is sanitized")
            finally:
                application_build.write_text(original_application)
            discover(client, root, details=True)
            receipt["checks"].append("recovery after Gradle task creation getter failure")
            expect_failure(client, {"command": "groovy.gradle.projects", "arguments": [
                {"rootUri": Path(directory).as_uri(), "offline": True}
            ]}, "must match an initialized")
            expect_failure(client, {"command": "groovy.gradle.projects", "arguments": [
                {"rootUri": root.as_uri(), "offline": True, "properties": ["--init-script=foreign"]}
            ]}, "Invalid Gradle project property")
            receipt["checks"].extend(["foreign root rejected", "non-property argument rejected"])
            canary = "GROOVY_PRIVATE_MODEL_CANARY"
            build_file = root / "build.gradle.kts"
            original = build_file.read_text()
            try:
                build_file.write_text(original + f'\nthrow GradleException("{canary}")\n')
                expect_failure(client, {"command": "groovy.gradle.projectDetails", "arguments": [
                    {"rootUri": root.as_uri(), "offline": True}
                ]}, "Gradle model discovery failed", forbidden=canary)
            finally:
                build_file.write_text(original)
            discover(client, root, details=True)
            receipt["checks"].extend(["build failure sanitized", "recovery after build repair"])
            if list(host_temporary.glob("groovy-gradle-model-*.gradle")):
                raise AssertionError("Failed metadata query retained its temporary script")
            receipt["checks"].append("failed metadata temporary-script cleanup")
            try:
                application_build.write_text(original_application + '\nrepeat(513) { configurations.create("groovyConfigurationLimit$it") }\n')
                expect_failure(client, {"command": "groovy.gradle.projectDetails", "arguments": [
                    {"rootUri": root.as_uri(), "offline": True}
                ]}, "Gradle model discovery failed")
                receipt["checks"].append("configuration count limit rejected")
            finally:
                application_build.write_text(original_application)
            try:
                application_build.write_text(original_application + '''
val parityEdgeLimit = configurations.create("parityEdgeLimit")
repeat(129) { parityEdgeLimit.extendsFrom(configurations.create("parityEdgeParent$it")) }
''')
                expect_failure(client, {"command": "groovy.gradle.projectDetails", "arguments": [
                    {"rootUri": root.as_uri(), "offline": True}
                ]}, "Gradle model discovery failed")
                receipt["checks"].append("configuration parent-edge limit rejected")
            finally:
                application_build.write_text(original_application)
            try:
                application_build.write_text(original_application + '''
val parityManyParents = (0 until 33).map { configurations.create("parityManyParent$it") }
repeat(125) {
    configurations.create("parityManyChild$it").extendsFrom(*parityManyParents.toTypedArray())
}
''')
                expect_failure(client, {"command": "groovy.gradle.projectDetails", "arguments": [
                    {"rootUri": root.as_uri(), "offline": True}
                ]}, "Gradle model discovery failed")
                receipt["checks"].append("configuration total-edge limit rejected")
            finally:
                application_build.write_text(original_application)
            discover(client, root, details=True)
            receipt["checks"].append("recovery after configuration limits")
            try:
                build_file.write_text(original + '\nrepeat(4100) { tasks.register("groovyModelLimit$it") }\n')
                expect_failure(client, {"command": "groovy.gradle.projects", "arguments": [
                    {"rootUri": root.as_uri(), "offline": True}
                ]}, "Gradle model discovery failed")
                receipt["checks"].append("task count limit rejected")
            finally:
                build_file.write_text(original)
            try:
                build_file.write_text(original + '''
if (providers.gradleProperty("groovyModelWait").orNull == "true") {
    file("model-started").writeText("started")
    Thread.sleep(30000)
}
''')
                response = []
                request_id = client.next_id

                def waiting_query():
                    try:
                        response.append(discover(client, root, ["-PgroovyModelWait=true"], details=True))
                    except Exception as failure:
                        response.append(failure)

                query = threading.Thread(target=waiting_query)
                query.start()
                deadline = time.monotonic() + 50
                while not (root / "model-started").exists() and query.is_alive() and time.monotonic() < deadline:
                    time.sleep(0.05)
                if not (root / "model-started").exists():
                    query.join(timeout=35)
                    raise AssertionError("Cancellation query did not reach fixture configuration")
                client.notify("$/cancelRequest", {"id": request_id})
                query.join(timeout=40)
                if query.is_alive() or not response or not isinstance(response[0], RuntimeError):
                    raise AssertionError("Cancelled model query published a successful response")
                if "cancel" not in str(response[0]).lower():
                    raise AssertionError("Cancelled model query lacked a cancellation category")
                receipt["checks"].append("in-flight LSP cancellation")
            finally:
                build_file.write_text(original)
            discover(client, root, details=True)
            receipt["checks"].append("recovery after cancellation")
            if list(host_temporary.glob("groovy-gradle-model-*.gradle")):
                raise AssertionError("Cancelled metadata query retained its temporary script")
            receipt["checks"].append("cancelled metadata temporary-script cleanup")
            offline_settings = json.loads(json.dumps(settings))
            offline_settings["java"]["import"]["gradle"]["home"] = ""
            client.notify("workspace/didChangeConfiguration", {"settings": offline_settings})
            expect_failure(client, {"command": "groovy.gradle.projects", "arguments": [
                {"rootUri": root.as_uri(), "offline": True}
            ]}, "configured cached Gradle installation")
            client.notify("workspace/didChangeConfiguration", {"settings": settings})
            discover(client, root)
            receipt["checks"].extend(["offline bootstrap denied without cached installation", "recovery after home restoration"])
        except Exception:
            evidence = arguments.receipt.parent / "failed-provider"
            evidence.mkdir(parents=True, exist_ok=True)
            logs = Path(directory) / "jdtls-data/.metadata/.log"
            if logs.exists():
                shutil.copy2(logs, evidence / "jdtls.log")
            client.stderr.seek(0)
            (evidence / "stderr.log").write_bytes(client.stderr.read())
            raise
        finally:
            client.close()
            subprocess.run([str(root / "gradlew"), "--offline", "--stop"], cwd=root,
                           env=gradle_environment, check=True, capture_output=True)
    arguments.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def fresh_gradle_fixture_bundle(arguments, directory, qualifiers):
    import zipfile
    bundle, source = qualifiers.unlink_fixture_bundle(arguments, directory, workspace_states=True)
    original = source.read_text()
    anchor = '  if (command.equals("groovy.fixture.unlink.projectState")) {'
    extension = r'''
  if (command.equals("groovy.fixture.unlink.freshGradleModule")) {
   if (!request.keySet().equals(java.util.Set.of("rootUri", "projectUri", "operation")) || monitor.isCanceled()) throw new IllegalArgumentException();
   String operation = (String) request.get("operation");
   if (!java.util.Set.of("query", "seed", "reserve", "release").contains(operation)) throw new IllegalArgumentException();
   Path requested = Path.of(new URI((String) request.get("projectUri")));
   Path module = requested.toRealPath();
   if (!requested.equals(module) || !module.startsWith(root) || module.equals(root) || !Files.isDirectory(module)) throw new IllegalArgumentException();
   var preferences = JavaLanguageServerPlugin.getPreferencesManager().getPreferences();
   if (!preferences.isImportGradleOfflineEnabled() || preferences.isGradleWrapperEnabled()
       || preferences.getGradleVersion() != null && !preferences.getGradleVersion().isBlank()
       || preferences.getGradleHome() == null || !Files.isDirectory(Path.of(preferences.getGradleHome()))) throw new IllegalArgumentException();
   var build = org.eclipse.buildship.core.GradleCore.getWorkspace().createBuild(
       org.eclipse.jdt.ls.core.internal.managers.GradleProjectImporter.getBuildConfiguration(root));
   var models = build.withConnection(org.eclipse.buildship.core.internal.workspace.EclipseModelUtils::queryModels, monitor);
   if (models == null || models.size() > 128) throw new IllegalArgumentException();
   var records = new java.util.ArrayList<Map<String,Object>>();
   var paths = new java.util.LinkedHashSet<Path>();
   var names = new java.util.LinkedHashSet<String>();
   Map<String,Object> selected = null;
   for (var model : models.values()) {
    Path buildRoot = model.getProjectDirectory().toPath().toRealPath();
    if (!buildRoot.startsWith(root)) throw new IllegalArgumentException();
    var pending = new java.util.ArrayDeque<org.gradle.tooling.model.eclipse.EclipseProject>(); pending.add(model);
    while (!pending.isEmpty()) {
     if (monitor.isCanceled() || records.size() >= 128 || pending.size() > 128) throw new IllegalArgumentException();
     var item = pending.removeFirst(); Path location = item.getProjectDirectory().toPath().toRealPath(); String name = item.getName();
     if (!location.startsWith(root) || !paths.add(location) || name == null || name.isBlank() || name.length() > 256
         || !names.add(name) || !ResourcesPlugin.getWorkspace().validateName(name,org.eclipse.core.resources.IResource.PROJECT).isOK()) throw new IllegalArgumentException();
     Map<String,Object> record = Map.of("moduleUri",location.toUri().toString(),"buildRootUri",buildRoot.toUri().toString(),"name",name);
     records.add(record); if (location.equals(module)) selected = record;
     for (var child : item.getChildren()) { if (pending.size() >= 128) throw new IllegalArgumentException(); pending.addLast(child); }
    }
   }
   if (selected == null) throw new IllegalArgumentException();
   var workspace = ResourcesPlugin.getWorkspace();
   Path reservation = root.getParent().resolve("groovy-eclipse-name-reservation");
   Map<String,Object> reservationReceipt = Map.of();
   if (operation.equals("reserve")) {
    String name = (String) selected.get("name"); var project = workspace.getRoot().getProject(name);
    if (project.exists() || Files.exists(reservation)) throw new IllegalArgumentException();
    var description = workspace.newProjectDescription(name); description.setLocationURI(reservation.toUri());
    project.create(description,monitor);
    reservationReceipt = Map.of("name",project.getName(),"location",reservation.toUri().toString(),"open",project.isOpen());
   } else if (operation.equals("release")) {
    org.eclipse.core.resources.IProject selectedReservation = null;
    var projects = workspace.getRoot().getProjects(); if (projects.length > 1024) throw new IllegalArgumentException();
    for (var project : projects) {
     if (project.getLocation() != null && project.getLocation().toFile().toPath().toAbsolutePath().normalize().equals(reservation)) {
      if (selectedReservation != null) throw new IllegalArgumentException(); selectedReservation = project;
     }
    }
    if (selectedReservation == null) throw new IllegalArgumentException();
    String name = selectedReservation.getName(); Path description = reservation.resolve(".project"); byte[] bytes = Files.readAllBytes(description);
    selectedReservation.delete(false,false,monitor);
    if (!java.util.Arrays.equals(bytes,Files.readAllBytes(description))) throw new IllegalStateException();
    reservationReceipt = Map.of("name",name,"location",reservation.toUri().toString(),"sourcePreserved",true);
   } else if (operation.equals("seed")) {
    String name = (String) selected.get("name"); var project = workspace.getRoot().getProject(name);
    if (project.exists() || Files.exists(module.resolve(".project"),java.nio.file.LinkOption.NOFOLLOW_LINKS)) throw new IllegalArgumentException();
    var description = workspace.newProjectDescription(name); description.setLocationURI(module.toUri());
    description.setNatureIds(new String[]{"org.eclipse.buildship.core.gradleprojectnature"});
    workspace.run((org.eclipse.core.resources.IWorkspaceRunnable) progress -> {
     if (progress.isCanceled() || project.exists() || Files.exists(module.resolve(".project"),java.nio.file.LinkOption.NOFOLLOW_LINKS)) throw new IllegalArgumentException();
     project.create(description,progress); if (project.isOpen()) throw new IllegalStateException();
    },workspace.getRoot(),org.eclipse.core.resources.IWorkspace.AVOID_UPDATE,monitor);
   }
   return Map.of("version",1,"rootUri",root.toUri().toString(),"projects",records,"selected",selected,
       "createdClosed",operation.equals("seed"),"readOnly",operation.equals("query"),"reservation",reservationReceipt,"fullSynchronizationKnown",false);
  }
'''
    if arguments.provider_fresh_module_states_only:
        extension = r'''
  if (command.equals("groovy.fixture.unlink.sdkJobs")) {
   var jobs = org.eclipse.core.runtime.jobs.Job.getJobManager().find(null);
   if (jobs.length > 1024) throw new IllegalArgumentException();
   var values = new java.util.ArrayList<Map<String,Object>>();
   for (var job : jobs) {
    String type = job.getClass().getName();
    if (type.startsWith("org.eclipse.buildship.")) values.add(Map.of("type",type,"state",job.getState()));
   }
   return Map.of("jobs",values);
  }
  if (command.equals("groovy.fixture.unlink.compatibilityDiagnostic")) {
   try {
    var type = org.eclipse.core.runtime.Platform.getBundle("dev.made.groovy.gradle.model").loadClass("dev.made.groovy.GradleImportHandler");
    var method = type.getDeclaredMethod("refreshCollidingClosedArtifacts",org.eclipse.buildship.core.GradleBuild.class,Path.class,IProgressMonitor.class);
    method.setAccessible(true);
    var build = org.eclipse.buildship.core.GradleCore.getWorkspace().createBuild(org.eclipse.jdt.ls.core.internal.managers.GradleProjectImporter.getBuildConfiguration(root));
    method.invoke(null,build,root,monitor);
    return Map.of("failed",false);
   } catch (java.lang.reflect.InvocationTargetException failure) {
    var buffer = new java.io.StringWriter(); failure.getCause().printStackTrace(new java.io.PrintWriter(buffer));
    String trace = buffer.toString(); return Map.of("failed",true,"trace",trace.substring(0,Math.min(trace.length(),16384)));
   }
  }
''' + extension
        extension = extension.replace('root.getParent().resolve("groovy-eclipse-name-reservation")', 'root.resolve("groovy-eclipse-name-reservation")')
        extension = extension.replace('    project.create(description,monitor);\n    reservationReceipt', '    project.create(description,monitor); project.open(monitor);\n    JavaLanguageServerPlugin.getProjectsManager().configureFilters(monitor);\n    reservationReceipt')

    if arguments.module_state_snapshots_only:
        extension = r'''
  if (command.equals("groovy.fixture.unlink.moduleSnapshotFault")) {
   if (!request.keySet().equals(java.util.Set.of("rootUri","fault"))) throw new IllegalArgumentException();
   String fault=(String)request.get("fault");
   if (!java.util.Set.of("descriptor","cancel","root","linkage","oversized","dual","missing","alias","directory").contains(fault)) throw new IllegalArgumentException();
   var originals=new java.util.LinkedHashMap<Path,byte[]>();
   for (String name:List.of("application","library","included")) {
    Path file=root.resolve(name).resolve(".project");
    if (Files.isRegularFile(file,java.nio.file.LinkOption.NOFOLLOW_LINKS)) {
     if (!file.toRealPath().equals(file) || Files.size(file)>1024*1024) throw new IllegalArgumentException();
     originals.put(file,Files.readAllBytes(file));
    }
   }
   if (originals.isEmpty()) throw new IllegalArgumentException();
   Path detached=root.resolveSibling("snapshot-fault-detached");
   Path applicationDetached=root.resolve("application-snapshot-detached");
   if (Files.exists(applicationDetached)) throw new IllegalArgumentException();
   if (Files.exists(detached)) throw new IllegalArgumentException();
   boolean[] applied={false};
   Path selected=root.resolve("library/.project");
   Path alias=selected.resolveSibling(".project.snapshot-alias-target");
   if (!originals.containsKey(selected) || Files.exists(alias)) throw new IllegalArgumentException();
   if (java.util.Set.of("oversized","dual","missing","alias").contains(fault)) {
    applied[0]=true;
    if (fault.equals("oversized")) Files.write(selected,new byte[1024*1024+1]);
    else if (fault.equals("missing")) Files.delete(selected);
    else if (fault.equals("alias")) {
     Files.write(alias,originals.get(selected));Files.delete(selected);Files.createSymbolicLink(selected,alias);
    } else {
     String text=new String(originals.get(selected),java.nio.charset.StandardCharsets.UTF_8);
     if (!text.contains("</natures>")) throw new IllegalArgumentException();
     Files.writeString(selected,text.replace("</natures>","<nature>org.eclipse.m2e.core.maven2Nature</nature></natures>"));
    }
   }
   var faultMonitor=new org.eclipse.core.runtime.NullProgressMonitor() {
    @Override public boolean isCanceled() {
     if (!applied[0] && java.util.Arrays.stream(Thread.currentThread().getStackTrace()).anyMatch(frame ->
         frame.getClassName().equals("dev.made.groovy.GradleImportHandler") && frame.getMethodName().equals("validateState"))) {
      applied[0]=true;
      try {
       if (fault.equals("descriptor")) {
        for (var entry:originals.entrySet()) Files.writeString(entry.getKey(),new String(entry.getValue(),java.nio.charset.StandardCharsets.UTF_8)+"\n");
       } else if (fault.equals("directory")) {
        Files.move(root.resolve("application"),applicationDetached);
       } else if (fault.equals("root")) {
        Files.move(root,detached);Files.createSymbolicLink(root,root.resolveSibling("unrelated"));
       } else if (fault.equals("linkage")) throw new NoClassDefFoundError("private-canary");
      } catch (java.io.IOException failure) {throw new IllegalStateException("Fixture fault could not be applied",failure);}
     }
     return applied[0] && fault.equals("cancel");
    }
   };
   try {
    var type=org.eclipse.core.runtime.Platform.getBundle("dev.made.groovy.gradle.model").loadClass("dev.made.groovy.GradleImportHandler");
    var handler=(IDelegateCommandHandler)type.getConstructor().newInstance();
    handler.executeCommand("groovy.gradle.moduleStates",List.of(Map.of("rootUri",root.toUri().toString(),"offline",true)),faultMonitor);
    return Map.of("failed",false,"faultApplied",applied[0]);
   } catch (IllegalStateException failure) {
    return Map.of("failed",true,"faultApplied",applied[0],"message",failure.getMessage());
   } finally {
    if (Files.exists(detached)) {
     if (!Files.isSymbolicLink(root)) throw new IllegalStateException("Unexpected root fault state");
     Files.delete(root);Files.move(detached,root);
    }
    if (Files.exists(applicationDetached)) {
     if (Files.exists(root.resolve("application"))) throw new IllegalStateException("Unexpected application fault state");
     Files.move(applicationDetached,root.resolve("application"));
    }
    for (var entry:originals.entrySet()) {
     if (Files.isSymbolicLink(entry.getKey())) Files.delete(entry.getKey());
     Files.write(entry.getKey(),entry.getValue());
    }
    if (Files.exists(alias)) Files.delete(alias);
   }
  }
''' + extension
    assert anchor in original
    source.write_text(original.replace(anchor, extension + anchor, 1))
    evidence = arguments.receipt.parent
    shutil.copy2(source, evidence / "FreshGradleFixtureHandler.java")
    classes = Path(directory) / "unlink-fixture-classes"
    plugins = arguments.launcher.resolve().parent.parent / "plugins"
    command = [str(arguments.java_home.resolve() / "bin/javac"), "--release", "17", "-Xlint:all", "-Werror",
        "-classpath", os.pathsep.join(str(path) for path in plugins.glob("*.jar")), "-d", str(classes), str(source)]
    compiled = subprocess.run(command, capture_output=True, text=True)
    (evidence / "fresh-fixture-compile.json").write_text(json.dumps({"command":command,"exitCode":compiled.returncode},indent=2)+"\n")
    (evidence / "fresh-fixture-compile.log").write_text(compiled.stdout + compiled.stderr)
    compiled.check_returncode()
    with zipfile.ZipFile(bundle) as archive:
        manifest = archive.read("META-INF/MANIFEST.MF").decode().replace("org.eclipse.core.runtime,org.eclipse.lsp4j\n", "org.eclipse.core.runtime,org.eclipse.lsp4j,\n org.eclipse.buildship.core,org.gradle.toolingapi\n")
        descriptor = archive.read("plugin.xml").decode().replace("</delegateCommandHandler>", '<command id="groovy.fixture.unlink.freshGradleModule"/></delegateCommandHandler>', 1)
    if arguments.module_state_snapshots_only:
        descriptor = descriptor.replace("</delegateCommandHandler>", '<command id="groovy.fixture.unlink.moduleSnapshotFault"/></delegateCommandHandler>',1)
    if arguments.provider_fresh_module_states_only:
        descriptor = descriptor.replace("</delegateCommandHandler>", '<command id="groovy.fixture.unlink.compatibilityDiagnostic"/><command id="groovy.fixture.unlink.sdkJobs"/></delegateCommandHandler>',1)
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("META-INF/MANIFEST.MF",manifest)
        archive.writestr("plugin.xml",descriptor)
        for path in classes.rglob("*.class"):
            archive.write(path,path.relative_to(classes).as_posix())
    return bundle, source


def probe_import_selection(arguments):
    provider = arguments.provider.resolve()
    if hashlib.sha256(provider.read_bytes()).hexdigest() != arguments.sha256:
        raise AssertionError("Provider binary differs from the supplied source build")
    evidence = arguments.receipt.parent
    evidence.mkdir(parents=True, exist_ok=True)
    receipt = {"provider_sha256": arguments.sha256, "checks": [], "journeys": [], "limits": [
        "Isolated headless pinned JDTLS/Buildship contract only; no product controls, native or physical remote qualification",
        "Command acknowledgements/events and membership are distinct from full synchronization",
        "No artifact download, Task, selected-test or Debug qualification" if arguments.resolution_graph_only
            else "No download, dependency resolution, Task, selected-test or Debug qualification",
    ]}
    specification = importlib.util.spec_from_file_location("groovy_provider_build", REPOSITORY / "script/build-groovy-gradle-provider.py")
    if specification is None or specification.loader is None:
        raise RuntimeError("Cannot load pinned provider build inputs")
    build_inputs = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(build_inputs)
    artifacts = [(arguments.launcher.resolve().parent.parent / "plugins" / name, expected)
                 for name, expected in build_inputs.PINNED_ARTIFACTS.items()]
    artifacts.extend((arguments.gradle_home.resolve() / name, expected)
                     for name, expected in build_inputs.PINNED_GRADLE_ARTIFACTS.items())
    receipt["pinned_input_sha256"] = {}
    for artifact, expected in artifacts:
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if digest != expected:
            raise AssertionError("Pinned project mutation artifact differs")
        receipt["pinned_input_sha256"][artifact.name] = digest
    @contextmanager
    def fixture_directory():
        temporary = tempfile.TemporaryDirectory(prefix="groovy-gradle-import-selection-")
        try:
            yield temporary.name
        finally:
            try:
                for attempt in range(15):
                    try:
                        temporary.cleanup()
                        break
                    except OSError as failure:
                        if failure.errno != errno.ENOTEMPTY or attempt == 14:
                            raise
                        receipt.setdefault("temporaryCleanupRetries", []).append({"attempt": attempt + 1, "errno": failure.errno})
                        time.sleep(0.2)
            finally:
                arguments.receipt.write_text(json.dumps(receipt, indent=2) + "\n")

    if arguments.closed_modules_only:
        receipt["limits"].append("Opt-in test-only Eclipse close/open and existing Buildship synchronization can build required artifacts; not a product Ignore action, Product Tasks, native or full parity qualification")
    with fixture_directory() as directory:
        fixture_bundle = None
        qualifiers = None
        if arguments.guarded_unlink_only or arguments.resource_membership_only or arguments.closed_modules_only:
            specification = importlib.util.spec_from_file_location("groovy_unlink_qualification", REPOSITORY / "script/test-groovy-maven-model-provider.py")
            if specification is None or specification.loader is None:
                raise RuntimeError("Cannot load existing unlink qualification fixture")
            qualifiers = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(qualifiers)
            if arguments.guarded_unlink_only or arguments.closed_modules_only:
                if arguments.fresh_closed_modules_only:
                    fixture_bundle, fixture_source = fresh_gradle_fixture_bundle(arguments, directory, qualifiers)
                else:
                    fixture_bundle, fixture_source = qualifiers.unlink_fixture_bundle(arguments, directory, workspace_states=arguments.closed_modules_only, ownership_modes=arguments.provider_module_states_only)
                receipt["testOnlyFixtureSha256"] = hashlib.sha256(fixture_bundle.read_bytes()).hexdigest()
                receipt["testOnlyFixtureSourceSha256"] = hashlib.sha256(fixture_source.read_bytes()).hexdigest()
                shutil.copy2(fixture_source, evidence / fixture_source.name)
                shutil.copy2(fixture_bundle, evidence / fixture_bundle.name)
        root = Path(directory).resolve() / "projects"
        primary, unrelated = root / "primary", root / "unrelated"
        for folder, name in [(primary, "selection-primary"), (unrelated, "selection-unrelated"), (primary / "included", "selection-included")]:
            folder.mkdir(parents=True)
            (folder / "settings.gradle").write_text(f"rootProject.name = '{name}'\n")
            (folder / "build.gradle").write_text("plugins { id 'java' }\n")
        if arguments.resource_membership_only or arguments.source_outputs_only:
            for folder in [primary, primary / "included"]:
                (folder / "build.gradle").write_text("plugins { id 'base' }\n")
        with (primary / "settings.gradle").open("a") as output:
            output.write("include 'application', 'library'\nincludeBuild 'included'\n")
        for name in ["application", "library"]:
            folder = primary / name
            folder.mkdir()
            (folder / "build.gradle").write_text("plugins { id 'java' }\n")
        with (primary / "application/build.gradle").open("a") as output:
            output.write("dependencies { implementation project(':library') }\n")
        for folder in [primary, unrelated, primary / "included", primary / "application", primary / "library"]:
            if (arguments.resource_membership_only or arguments.source_outputs_only) and folder in (primary, primary / "included"):
                continue
            source = folder / "src/main/java/evidence/Main.java"
            source.parent.mkdir(parents=True)
            source.write_text('package evidence; public class Main {}\n')
        graph_repository = primary / "graph-repository"
        def graph_pom(name, version="1.0", dependencies=()):
            import xml.etree.ElementTree as ElementTree
            pom = ElementTree.Element("project", xmlns="http://maven.apache.org/POM/4.0.0")
            for key, value in [("modelVersion", "4.0.0"), ("groupId", "evidence"), ("artifactId", name), ("version", version)]:
                ElementTree.SubElement(pom, key).text = value
            children = ElementTree.SubElement(pom, "dependencies")
            for dependency, requested in dependencies:
                child = ElementTree.SubElement(children, "dependency")
                for key, value in [("groupId", "evidence"), ("artifactId", dependency), ("version", requested)]:
                    ElementTree.SubElement(child, key).text = value
            target = graph_repository / "evidence" / name / version / (name + "-" + version + ".pom")
            target.parent.mkdir(parents=True)
            ElementTree.ElementTree(pom).write(target, encoding="utf-8", xml_declaration=True)
        if arguments.resolution_graph_only:
            for name, version, dependencies in [
                ("shared", "1.0", []), ("shared", "2.0", []),
                ("left", "1.0", [("shared", "1.0")]), ("right", "1.0", [("shared", "2.0")]),
                ("cycle-a", "1.0", [("cycle-b", "1.0")]), ("cycle-b", "1.0", [("cycle-a", "1.0")]),
            ]: graph_pom(name, version, dependencies)
            with (primary / "included/build.gradle").open("a") as output:
                output.write("group = 'evidence'\nversion = '1.0'\n")
            with (primary / "application/build.gradle").open("a") as output:
                output.write("""
repositories { maven { url = uri(rootProject.file('graph-repository')) } }
configurations.create('parityGraph') { canBeResolved = true; canBeConsumed = false }
['härt-α', 'runtime graph', 'runtime📦', 'r' + ('x' * 128), 'r' + ('x' * 4095), 'α' * 2048].each { literalName ->
    configurations.create(literalName) { canBeResolved = true; canBeConsumed = false; extendsFrom configurations.parityGraph }
}
configurations.create('guardedGraph') {
    canBeResolved = true; canBeConsumed = false
    incoming.beforeResolve { throw new GradleException('GROOVY_GRAPH_PRIVATE_CANARY') }
}
dependencies {
    parityGraph 'evidence:left:1.0'
    parityGraph 'evidence:right:1.0'
    parityGraph 'evidence:cycle-a:1.0'
    parityGraph 'evidence:missing:1.0'
    parityGraph 'evidence:selection-included:1.0'
    parityGraph project(':library')
    constraints {
        parityGraph('evidence:shared') {
            version { prefer '1.0'; reject '3.0' }
            because 'GROOVY_GRAPH_REASON_CANARY'
        }
    }
}
""")
        if arguments.source_outputs_only:
            with (primary / "library/build.gradle").open("a") as output:
                output.write("""
sourceSets.main.java.destinationDirectory.set(layout.projectDirectory.dir('custom/classes'))
sourceSets.main.output.resourcesDir = file('custom/resources')
sourceSets.main.output.dir(file('custom/additional'), builtBy: 'outputCanary')
tasks.named('compileJava') { options.generatedSourceOutputDirectory.set(layout.projectDirectory.dir('custom/annotation-sources')) }
tasks.register('outputCanary') { doLast { throw new GradleException('GROOVY_OUTPUT_TASK_CANARY') } }
configurations.create('outputUnresolved') {
    canBeResolved = true
    canBeConsumed = false
    incoming.beforeResolve { throw new GradleException('GROOVY_OUTPUT_RESOLUTION_CANARY') }
}
dependencies { outputUnresolved 'evidence:uncached-output:1.0' }
""")
        if arguments.required_library_only:
            source = primary / "library/src/main/java/library/Required.java"
            source.parent.mkdir(parents=True)
            source.write_text('package library; public class Required { public static String message() { return "required"; } }\n')
            source = primary / "application/src/main/java/application/UsesLibrary.java"
            source.parent.mkdir(parents=True)
            source.write_text('package application; public class UsesLibrary { public static String message() { return library.Required.message(); } }\n')
            with (primary / "library/build.gradle").open("a") as output:
                output.write("tasks.named('jar') { archiveFileName = 'fixture-required-library.jar' }\n")
                if arguments.fresh_closed_modules_only:
                    output.write("apply plugin: 'eclipse'\neclipse.project.name = 'groovy-eclipse-library'\n")
        if arguments.provider_fresh_module_states_only:
            with (primary / "build.gradle").open("a") as output:
                output.write("""
apply plugin: 'eclipse'
def observedSynchronization = tasks.register('evidenceSynchronization') {
    doLast { rootProject.file('sdk-sync-counter').append('sync\\n') }
}
eclipse.synchronizationTasks(observedSynchronization)
tasks.register('forbiddenDefault') {
    doLast { rootProject.file('forbidden-default-task').text = 'executed' }
}
defaultTasks 'forbiddenDefault'
""")
            with (primary / "library/build.gradle").open("a") as output:
                output.write("""
tasks.named('jar') {
    preserveFileTimestamps = false
    reproducibleFileOrder = true
    outputs.upToDateWhen { false }
    doLast { rootProject.file('affected-producer-counter').append('affected\\n') }
}
""")
            with (primary / "included/build.gradle").open("a") as output:
                output.write("""
group = 'evidence'
version = '1.0'
apply plugin: 'eclipse'
eclipse.project.name = 'ordinary-eclipse-included'
tasks.named('jar') {
    preserveFileTimestamps = false
    reproducibleFileOrder = true
    outputs.upToDateWhen { false }
    doLast { rootProject.file('ordinary-producer-counter').append('ordinary\\n') }
}
""")
            with (primary / "application/build.gradle").open("a") as output:
                output.write("dependencies { implementation 'evidence:selection-included:1.0' }\n")
            source = primary / "included/src/main/java/ordinary/Other.java"
            source.parent.mkdir(parents=True,exist_ok=True)
            source.write_text('package ordinary; public class Other { public static String message() { return "other"; } }\n')
            source = primary / "application/src/main/java/application/UsesLibrary.java"
            source.write_text(source.read_text().replace('return library.Required.message();','return library.Required.message() + ordinary.Other.message();'))
        if arguments.guarded_unlink_only or arguments.resource_membership_only or arguments.required_library_only:
            for module in [primary, unrelated, primary / "included", primary / "application", primary / "library"]:
                sentinel = module / "build/generated/sources/unlink/Sentinel.java"
                sentinel.parent.mkdir(parents=True)
                sentinel.write_text("package evidence; public class Sentinel {}\n")
        if arguments.resource_membership_only:
            for module in [primary, unrelated, primary / "included", primary / "application", primary / "library"]:
                with (module / "build.gradle").open("a") as build:
                    build.write("\ntasks.register('unrequestedCompositeDefault') { doLast { file('unrequested-composite-default.marker').text = 'ran' } }\ndefaultTasks 'unrequestedCompositeDefault'\n")
        originals = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in root.rglob("*") if path.is_file()}
        receipt["fixture_sha256"] = originals
        home = Path(directory) / "owned-gradle-home"
        home.mkdir()
        settings = {"java": {
            "autobuild": {"enabled": False},
            "configuration": {"updateBuildConfiguration": "disabled"},
            "import": {"maven": {"enabled": False}, "gradle": {
                "enabled": True, "offline": {"enabled": True}, "wrapper": {"enabled": False},
                "home": str(arguments.gradle_home.resolve()), "java": {"home": str(arguments.gradle_java_home.resolve())},
                "user": {"home": str(home)}, "arguments": ["--offline"],
            }},
        }}
        primary_members = {"primary", "primary/application", "primary/library", "primary/included"}
        if arguments.resource_membership_only or arguments.source_outputs_only:
            primary_members -= {"primary", "primary/included"}
        if arguments.provider_fresh_module_states_only:
            primary_members.remove("primary/included")
        all_members = primary_members | {"unrelated"}

        def preserve_sources(label):
            for relative, digest in originals.items():
                path = root / relative
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise AssertionError("Gradle project selection changed or removed fixture source/build files")
            if list(root.rglob("TEST-*.xml")):
                raise AssertionError("Gradle project selection emitted test reports")
            receipt["checks"].append(label)

        def membership(client):
            values = client.request("workspace/executeCommand", {"command": "java.project.getAll", "arguments": []}, timeout=15)
            result = set()
            for uri in values or []:
                from urllib.parse import urlparse, unquote
                parsed = urlparse(uri)
                if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
                    raise AssertionError("Gradle membership returned a nonlocal URI")
                path = Path(unquote(parsed.path)).resolve()
                if not path.is_relative_to(root):
                    receipt.setdefault("rejectedMembershipUris", []).append(uri[:4096])
                    raise AssertionError("Gradle membership escaped the disposable fixtures")
                result.add(path.relative_to(root).as_posix())
            return result

        def await_membership(client, expected, journey, label):
            deadline = time.monotonic() + 90
            observed = set()
            while time.monotonic() < deadline:
                try:
                    observed = membership(client)
                except TimeoutError:
                    journey.setdefault("readTimeouts", []).append(label)
                    if client.process.poll() is not None:
                        raise AssertionError("Gradle provider exited before membership was available")
                    continue
                if observed == expected or isinstance(expected, tuple) and observed in expected:
                    journey["states"].append({"label": label, "members": sorted(observed)})
                    receipt["checks"].append(label)
                    print(label, flush=True)
                    return
                time.sleep(0.2)
            journey["states"].append({"label": label, "expected": [sorted(value) for value in expected] if isinstance(expected, tuple) else sorted(expected), "members": sorted(observed)})
            raise AssertionError(f"{label}: Gradle membership did not match selected fixtures")

        def composite_cached_ownership(client, journey, stage):
            from urllib.parse import unquote, urlparse

            expected_models = {primary: primary, primary / "application": primary,
                               primary / "library": primary, primary / "included": primary / "included"}
            expected_cached = {module: primary for module in expected_models}
            observed = {"stage": stage, "reads": []}
            journey.setdefault("compositeCachedOwnership", []).append(observed)
            before_java = membership(client)
            before_resources = qualifiers.imported_resource_membership(client, root)
            metadata = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in root.rglob("*") if path.is_file()
                        and (path.name in {".project", ".classpath"} or ".settings" in path.parts)}

            def identity(uri):
                if not isinstance(uri, str) or len(uri) > 4096:
                    raise AssertionError("Composite identity is not bounded text")
                parsed = urlparse(uri)
                if parsed.scheme != "file" or parsed.netloc or parsed.query or parsed.fragment:
                    raise AssertionError("Composite identity is not a confined file URI")
                value = Path(unquote(parsed.path))
                if not value.is_absolute() or value.resolve() != value or not value.is_relative_to(root):
                    raise AssertionError("Composite identity escapes its disposable fixture")
                return value

            def cached(scope, expected):
                request = {"rootUri": scope.as_uri(), "cachedOnly": True}
                reply = client.request("workspace/executeCommand", {
                    "command": "groovy.gradle.importedBuildRoots", "arguments": [request],
                }, timeout=60)
                legacy = client.request("workspace/executeCommand", {
                    "command": "groovy.gradle.importedProjects", "arguments": [request],
                }, timeout=60)
                observed["reads"].append({"scope": scope.relative_to(root).as_posix(),
                                          "reply": reply, "legacy": legacy})
                if (set(reply) != {"version", "rootUri", "projects", "cachedOnly", "readOnly", "partial", "fullSynchronizationKnown"}
                        or type(reply["version"]) is not int or reply["version"] != 1
                        or identity(reply["rootUri"]) != scope or reply["cachedOnly"] is not True
                        or reply["readOnly"] is not True or reply["partial"] is not False
                        or reply["fullSynchronizationKnown"] is not False
                        or set(legacy) != {"version", "rootUri", "projects", "partial", "fullSynchronizationKnown"}
                        or type(legacy["version"]) is not int or legacy["version"] != 1
                        or identity(legacy["rootUri"]) != scope or legacy["partial"] is not False
                        or legacy["fullSynchronizationKnown"] is not False
                        or not isinstance(reply["projects"], list) or len(reply["projects"]) != len(expected)):
                    raise AssertionError("Cached composite ownership envelope differs or overclaims")
                actual = {}
                for record in reply["projects"]:
                    if set(record) != {"projectUri", "buildRootUri", "javaNature"} or type(record["javaNature"]) is not bool:
                        raise AssertionError("Cached composite ownership record differs")
                    module, build = identity(record["projectUri"]), identity(record["buildRootUri"])
                    if module in actual or record["javaNature"] is not (module.relative_to(root).as_posix() in all_members):
                        raise AssertionError("Cached composite Java/resource identity is duplicated or differs")
                    actual[module] = build
                if actual != expected:
                    raise AssertionError("Cached composite connection roots differ from the expected Buildship owner")
                if [{key: record[key] for key in ["projectUri", "javaNature"]}
                        for record in reply["projects"]] != legacy["projects"]:
                    raise AssertionError("Composite cached ownership and independent v1 namespace differ")
                return reply

            initial = cached(primary, expected_cached)
            cached(unrelated, {unrelated: unrelated})
            request = {"workspaceRootUri": primary.as_uri(), "buildRootUri": primary.as_uri(), "offline": True}
            response = client.request("workspace/executeCommand", {
                "command": "groovy.gradle.projectDetails.v4", "arguments": [request],
            }, timeout=90)
            observed["details"] = response
            if (set(response) != {"version", "workspaceRootUri", "buildRootUri", "model", "modelQueryComplete", "fullSynchronizationKnown"}
                    or type(response["version"]) is not int or response["version"] != 4
                    or identity(response["workspaceRootUri"]) != primary
                    or identity(response["buildRootUri"]) != primary
                    or response["modelQueryComplete"] is not True or response["fullSynchronizationKnown"] is not False):
                raise AssertionError("Composite v4 model envelope differs or overclaims")
            model = response["model"]
            if (set(model) != {"version", "rootUri", "projects", "sourceSetsKnown"}
                    or type(model["version"]) is not int or model["version"] != 2
                    or identity(model["rootUri"]) != primary or model["sourceSetsKnown"] is not False
                    or not isinstance(model["projects"], list) or len(model["projects"]) != len(expected_models)):
                raise AssertionError("Composite Tooling API model differs")
            model_roots = {}
            for record in model["projects"]:
                module, build = identity(record["moduleUri"]), identity(record["buildRootUri"])
                if module in model_roots or not isinstance(record.get("detailsJson"), str):
                    raise AssertionError("Composite model identity is duplicated or lacks details")
                model_roots[module] = build
            if model_roots != expected_models:
                raise AssertionError("Composite participant roots were replaced by connection-root ownership")
            if cached(primary, expected_cached) != initial:
                raise AssertionError("Configuring metadata changed cached composite ownership")
            if (membership(client) != before_java
                    or qualifiers.imported_resource_membership(client, root) != before_resources
                    or {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in root.rglob("*") if path.is_file()
                        and (path.name in {".project", ".classpath"} or ".settings" in path.parts)} != metadata):
                raise AssertionError("Composite identity/model reads changed Java/resource membership or metadata")
            if list(root.rglob("unrequested-composite-default.marker")):
                raise AssertionError("Composite identity/model read executed a default task")
            preserve_sources(stage + ": composite read/model source and generated sentinels preserved")
            receipt["checks"].extend([
                stage + ": exact cached composite connection root differs from included Tooling API build identity",
                stage + ": complete independent v1 namespace with Java/non-Java and unrelated ownership",
                stage + ": current cached identities and resource metadata unchanged before/after model; no default task",
            ])

        graph_baseline = None

        def resolution_graph_checks(client, journey):
            nonlocal graph_baseline
            request = {"command": "groovy.gradle.dependencyGraph", "arguments": [{
                "rootUri": primary.as_uri(), "moduleUri": (primary / "application").as_uri(),
                "configuration": "parityGraph", "offline": True,
            }]}
            def query():
                return client.request("workspace/executeCommand", request, timeout=80)
            response = query()
            if set(response) != {"version", "rootUri", "projects"} or response["version"] != 1 or len(response["projects"]) != 1:
                raise AssertionError("Selected configuration graph envelope differs")
            graph = json.loads(response["projects"][0]["graphJson"])
            journey.setdefault("graphResponses", []).append({"label": "baseline", "response": response})
            if graph_baseline is not None:
                if response != graph_baseline:
                    raise AssertionError("Same-profile graph restart changes stable component/edge identities")
                receipt["checks"].append("same-profile graph restart preserves exact bounded resolution facts")
                return
            graph_baseline = response
            if graph["structuralCoverageKnown"] is not True or graph["resolutionComplete"] is not False or graph["originsKnown"] or graph["artifactsKnown"]:
                raise AssertionError("Graph coverage overclaims resolution/origin/artifact readiness")
            nodes = {node["id"]: node for node in graph["components"]}
            if len(nodes) != len(graph["components"]) or graph["rootComponent"] not in nodes:
                raise AssertionError("Component identities collide or lack a root")
            if any(edge["from"] not in nodes or edge["resolved"] and edge["selected"] not in nodes for edge in graph["edges"]):
                raise AssertionError("Resolution edges lack real component endpoints")
            shared = next(node for node in nodes.values() if node.get("module") == "shared")
            if shared["version"] != "2.0" or not shared["conflictResolution"] or "CONFLICT_RESOLUTION" not in shared["selectionCauses"]:
                raise AssertionError("Actual selected version/conflict cause differs")
            diamond = [edge for edge in graph["edges"] if edge["requested"].get("module") == "shared" and not edge["constraint"]]
            if len(diamond) != 2 or {edge["requested"]["version"] for edge in diamond} != {"1.0", "2.0"} or any(edge["selected"] != shared["id"] for edge in diamond):
                raise AssertionError("Diamond requested/selected version edges differ")
            cycle = {node["module"]: node["id"] for node in nodes.values() if node.get("module") in ["cycle-a", "cycle-b"]}
            if len(cycle) != 2 or not all(any(edge["from"] == cycle[left] and edge["selected"] == cycle[right] for edge in graph["edges"]) for left,right in [("cycle-a","cycle-b"),("cycle-b","cycle-a")]):
                raise AssertionError("Real cyclic edges are absent")
            projects = {node["buildTreePath"] for node in nodes.values() if node["kind"] == "project"}
            if not {":application", ":library", ":included"}.issubset(projects):
                raise AssertionError("Main/composite project component identities differ")
            constraint = next(edge for edge in graph["edges"] if edge["constraint"])
            if constraint["requested"].get("preferredVersion") != "1.0" or constraint["requested"].get("rejectedVersions") != ["3.0"] or "GROOVY_GRAPH_REASON_CANARY" in json.dumps(graph):
                raise AssertionError("Rich constraint metadata differs or rule text leaked")
            receipt["checks"].append("rich preferred/rejected constraint versions; private selection descriptions omitted")
            missing = next(edge for edge in graph["edges"] if edge["requested"].get("module") == "missing")
            if missing["resolved"] or missing["selected"] is not None or str(root) in missing["failureCategory"] or "Exception" in missing["failureCategory"]:
                raise AssertionError("Unresolved graph edge category is missing or unsanitized")
            if any(edge["originKnown"] or edge["resolved"] and not edge["selectedVariant"] for edge in graph["edges"]):
                raise AssertionError("Selected variant is absent or source origins invented")
            receipt["checks"].extend(["real diamond requested/selected versions and enumerated conflict cause", "real cyclic dependency edges without recursive looping", "main/composite project identities and selected variants", "unresolved edge sanitized; source origins/artifacts remain unknown"])
            if query() != response:
                raise AssertionError("Repeat graph query changes stable identities or edges")
            receipt["checks"].append("stable repeated selected-configuration graph")
            for literal_name in ["härt-α", "runtime graph", "runtime📦", "r" + "x" * 128, "r" + "x" * 4095, "α" * 2048]:
                selected_request = json.loads(json.dumps(request))
                selected_request["arguments"][0]["configuration"] = literal_name
                selected = client.request("workspace/executeCommand", selected_request, timeout=80)
                selected_graph = json.loads(selected["projects"][0]["graphJson"])
                if selected_graph.pop("configuration") != literal_name or selected_graph != {key: value for key, value in graph.items() if key != "configuration"}:
                    raise AssertionError("Literal configuration selection changes inherited graph facts or loses its exact name")
                if client.request("workspace/executeCommand", selected_request, timeout=80) != selected:
                    raise AssertionError("Repeated literal configuration graph is unstable")
                receipt["checks"].append("exact stable inherited graph for literal configuration: " + (literal_name if len(literal_name) <= 128 else "UTF8-bytes=" + str(len(literal_name.encode("utf-8")))))
            for changed, expected in [
                ({"offline": False}, "explicit offline mode"), ({"configuration": "../private-canary"}, "Gradle model discovery failed"),
                ({"configuration": ""}, "explicit offline mode"),
                ({"configuration": "runtime\nprivate-canary"}, "explicit offline mode"),
                ({"configuration": "runtime\x00private-canary"}, "explicit offline mode"),
                ({"configuration": "runtime\u0085private-canary"}, "explicit offline mode"),
                ({"configuration": "α" * 2049}, "explicit offline mode"),
                ({"configuration": "x" * 4097}, "explicit offline mode"),
                ({"moduleUri": unrelated.as_uri()}, "outside its initialized root"),
                ({"moduleUri": (primary / "application/src").as_uri()}, "Gradle model discovery failed"),
                ({"configuration": "implementation"}, "Gradle model discovery failed"),
                ({"configuration": "not-existing"}, "Gradle model discovery failed"),
                ({"configuration": "runtime:α"}, "Gradle model discovery failed"),
                ({"configuration": "runtime/graph"}, "Gradle model discovery failed"),
                ({"configuration": "runtime\\graph"}, "Gradle model discovery failed"),
                ({"configuration": ".runtime"}, "Gradle model discovery failed"),
                ({"configuration": "runtime."}, "Gradle model discovery failed"),
                ({"configuration": "guardedGraph"}, "Gradle model discovery failed"),
                ({"private-canary": "unadvertised"}, "explicit offline mode"),
            ]:
                invalid = dict(request["arguments"][0], **changed)
                error = expect_failure(client, {"command": request["command"], "arguments": [invalid]}, expected, "GROOVY_GRAPH_PRIVATE_CANARY")
                if str(root) in error or "private-canary" in error:
                    raise AssertionError("Graph rejection exposes private input")
                receipt["checks"].append("wrong/foreign/missing/guarded graph input rejected and sanitized")
            build = primary / "application/build.gradle";original = build.read_bytes()
            for label,count,mutation in [
                ("component count",4097,"dependencies { (0..<4097).each { parityGraph 'evidence:bounded-node' + it + ':1.0' } }"),
                ("edge count",130,"dependencies { (0..<130).each { parityGraph 'evidence:bounded-parent' + it + ':1.0' } }"),
                ("selector text",0,"dependencies { parityGraph('evidence:' + ('x' * 4200) + ':1.0') }"),
                ("build failure",0,"throw new GradleException('GROOVY_GRAPH_PRIVATE_CANARY')"),
            ]:
                created = []
                try:
                    if label == "component count":
                        for index in range(count):
                            name = "bounded-node" + str(index);graph_pom(name);created.append(graph_repository / "evidence" / name)
                    if label == "edge count":
                        for index in range(129):
                            name = "bounded-leaf" + str(index);graph_pom(name);created.append(graph_repository / "evidence" / name)
                        dependencies = [("bounded-leaf" + str(index), "1.0") for index in range(129)]
                        for index in range(count):
                            name = "bounded-parent" + str(index);graph_pom(name,dependencies=dependencies);created.append(graph_repository / "evidence" / name)
                    build.write_bytes(original + ('\n' + mutation + '\n').encode())
                    error = expect_failure(client, request, "Gradle model discovery failed", "GROOVY_GRAPH_PRIVATE_CANARY")
                    journey.setdefault("failures", []).append({"label":label,"error":error})
                    receipt["checks"].append(label + " rejected with sanitized graph failure")
                finally:
                    build.write_bytes(original)
                    for path in created: shutil.rmtree(path)
                if query() != response:
                    raise AssertionError("Graph facts did not recover after " + label)
                receipt["checks"].append(label + " repaired without stale graph facts")
            waiting_file = primary / "graph-query-started"
            try:
                build.write_bytes(original + b"\nfile('../graph-query-started').text = 'started'\nThread.sleep(30000)\n")
                responses = [];request_id = client.next_id
                def waiting():
                    try: responses.append(query())
                    except Exception as failure: responses.append(failure)
                thread=threading.Thread(target=waiting);thread.start();deadline=time.monotonic()+50
                while not waiting_file.exists() and thread.is_alive() and time.monotonic()<deadline:time.sleep(0.05)
                if not waiting_file.exists():
                    thread.join(timeout=35);raise AssertionError("Graph cancellation did not reach fixture configuration")
                client.notify("$/cancelRequest", {"id":request_id});thread.join(timeout=40)
                if thread.is_alive() or not responses or not isinstance(responses[0],RuntimeError) or "cancel" not in str(responses[0]).lower():
                    raise AssertionError("Graph cancellation published success or lacks category")
                receipt["checks"].append("in-flight graph LSP cancellation rejects success")
            finally:
                build.write_bytes(original)
                if waiting_file.exists():waiting_file.unlink()
            if query()!=response:raise AssertionError("Graph recovery after cancellation differs")
            receipt["checks"].append("graph cancellation recovery")
            disabled=json.loads(json.dumps(settings));disabled["java"]["import"]["gradle"]["home"]=""
            client.notify("workspace/didChangeConfiguration",{"settings":disabled});expect_failure(client,request,"configured cached Gradle installation")
            client.notify("workspace/didChangeConfiguration",{"settings":settings})
            if query()!=response:raise AssertionError("Graph home restoration changes facts")
            receipt["checks"].extend(["graph offline bootstrap refusal without cached home","graph home restoration recovery"])
            basic=discover(client,primary);details=discover(client,primary,details=True)
            outputs=client.request("workspace/executeCommand",{"command":"groovy.gradle.sourceOutputs","arguments":[{"rootUri":primary.as_uri(),"offline":True}]},timeout=80)
            if basic["version"]!=1 or details["version"]!=2 or outputs["version"]!=1 or any("graphJson" in record for item in [basic,details,outputs] for record in item["projects"]):
                raise AssertionError("Graph query changes existing discovery/output schemas")
            receipt["checks"].append("v1/v2/output-only compatibility avoids guarded configuration resolution")
            fixture_cache = home / "caches/modules-2/files-2.1/evidence"
            if list(root.rglob("*.jar")) or list(root.rglob("*.class")) or list(root.rglob("TEST-*.xml")) or list(fixture_cache.rglob("*.jar")):
                raise AssertionError("Graph query acquired fixture artifacts or executed output/test Tasks")
            journey["fixtureGroupCacheFiles"] = [str(path.relative_to(fixture_cache)) for path in sorted(fixture_cache.rglob("*")) if path.is_file()]
            receipt["checks"].append("no fixture artifact/classes/test output requested or produced")

        output_baseline = None
        compiler_output_baseline = None

        def compiler_output_checks(client, journey):
            nonlocal compiler_output_baseline
            from urllib.parse import unquote, urlparse
            response = discover(client, primary, details=True)
            facts = []
            declarations = 0
            for project in response["projects"]:
                module = Path(unquote(urlparse(project["moduleUri"]).path)).resolve()
                details = json.loads(project["detailsJson"])
                facts.append({"buildRootUri": project["buildRootUri"], "moduleUri": project["moduleUri"],
                              "projectPath": project["projectPath"], "sourceSets": details["sourceSets"]})
                for source in details["sourceSets"]:
                    expected = module / "build/generated/sources/annotationProcessor/java" / source["name"]
                    if module == primary / "library" and source["name"] == "main":
                        expected = module / "custom/annotation-sources"
                    if source.get("annotationProcessorOutputKnown") is not True or source.get("annotationProcessorOutputUri") != expected.as_uri() or source["generatednessKnown"]:
                        raise AssertionError("Actual configured compiler output or unknown other-generator coverage differs")
                    declarations += 1
            if declarations < 4:
                raise AssertionError("Composite compiler output fixture selected too few source sets")
            journey.setdefault("compilerOutputResponses", []).append(response)
            if compiler_output_baseline is None:
                compiler_output_baseline = facts
                receipt["checks"].append("actual configured/default compiler outputs preserve module/source-set identities without generation")
            else:
                if facts != compiler_output_baseline:
                    raise AssertionError("Same-profile JDTLS restart changed compiler output declarations")
                receipt["checks"].append("same-profile JDTLS restart preserves configured compiler output facts")

        def source_output_checks(client, journey):
            nonlocal output_baseline
            from urllib.parse import unquote, urlparse
            request = {"command": "groovy.gradle.sourceOutputs", "arguments": [{"rootUri": primary.as_uri(), "offline": True}]}
            def query():
                return client.request("workspace/executeCommand", request, timeout=80)
            def retain(label, response):
                journey.setdefault("outputResponses", []).append({"label": label, "response": response})
                receipt["checks"].append(label)
            baseline = query()
            if set(baseline) != {"version", "rootUri", "projects"} or baseline["version"] != 1 or len(baseline["projects"]) != 4:
                raise AssertionError("Output-only composite protocol differs")
            compiler_output_checks(client, journey)
            if output_baseline is not None:
                if baseline != output_baseline:
                    raise AssertionError("Same-profile restart changed output declarations")
                retain("same-profile suppressed-startup output restart preserves declarations", baseline)
                return
            output_baseline = baseline
            identities = {(record["buildRootUri"], record["projectPath"]) for record in baseline["projects"]}
            if len(identities) != 4:
                raise AssertionError("Output module identities collide")
            records = {Path(unquote(urlparse(record["moduleUri"]).path)).resolve(): json.loads(record["outputsJson"]) for record in baseline["projects"]}
            for folder in [primary, primary / "included"]:
                if records[folder] != {"sourceSetsKnown": True, "javaPluginPresent": False, "artifactSelectionKnown": False, "sourceSets": []}:
                    raise AssertionError("Non-Java output declaration is not known empty")
            library = {source["name"]: source for source in records[primary / "library"]["sourceSets"]}
            main = library["main"]
            for key, relative in [("classesDirs", "custom/classes"), ("additionalDirs", "custom/additional")]:
                if main[key] != [(primary / "library" / relative).as_uri()]:
                    raise AssertionError("Authoritative custom output directory differs")
            if main["resourcesDir"] != (primary / "library/custom/resources").as_uri() or (primary / "library/custom").exists():
                raise AssertionError("Missing custom resources declaration differs or task ran")
            application = {source["name"]: source for source in records[primary / "application"]["sourceSets"]}
            if application["main"]["classesDirs"] != [(primary / "application/build/classes/java/main").as_uri()]:
                raise AssertionError("Standard class output declaration differs")
            if any(record["artifactSelectionKnown"] for record in records.values()):
                raise AssertionError("Output declarations claim variant/artifact selection")
            retain("composite outputs: custom/missing/standard/non-Java declarations; artifact selection unknown", baseline)
            if query() != baseline:
                raise AssertionError("Repeat output declarations or stable identities differ")
            receipt["checks"].append("stable output-only identities and no resolution/output-producing task")
            directories = [primary / "library/custom" / name for name in ["classes", "resources", "additional"]]
            try:
                for path in directories: path.mkdir(parents=True)
                if query() != baseline:
                    raise AssertionError("Output declaration identity changed when its directory became present")
                receipt["checks"].append("existing and absent output declarations preserve identical URIs")
            finally:
                for path in directories: path.rmdir()
                (primary / "library/custom").rmdir()
            for invalid in [
                {"rootUri": primary.as_uri(), "offline": False},
                {"rootUri": primary.as_uri(), "offline": True, "properties": ["private-canary"]},
            ]:
                expect_failure(client, {"command": request["command"], "arguments": [invalid]}, "explicit offline mode", "private-canary")
                receipt["checks"].append("strict output request rejects online/unknown input")
            expect_failure(client, {"command": request["command"], "arguments": [{"rootUri": root.as_uri(), "offline": True}]}, "must match an initialized")
            receipt["checks"].append("foreign initialized output root rejected")
            build = primary / "library/build.gradle"
            original = build.read_bytes()
            mutations = [
                ("source-set count", "(1..129).each { sourceSets.create('limit' + it) }"),
                ("output-directory count", "(1..257).each { sourceSets.main.output.dir(file('bounded/' + it)) }"),
                ("output text", "sourceSets.main.output.resourcesDir = file('x' * 4200)"),
                ("private ancestor", "sourceSets.main.output.resourcesDir = file('../..')"),
                ("escaped symlink ancestor", "sourceSets.main.output.resourcesDir = file('escape/not-created')"),
                ("build failure", "throw new GradleException('GROOVY_OUTPUT_PRIVATE_CANARY')"),
            ]
            escaped = primary / "library/escape"
            escaped.symlink_to(unrelated, target_is_directory=True)
            try:
                for label, mutation in mutations:
                    try:
                        build.write_bytes(original + ('\n' + mutation + '\n').encode())
                        expect_failure(client, request, "Gradle model discovery failed", "GROOVY_OUTPUT_PRIVATE_CANARY")
                        receipt["checks"].append(label + " rejected with sanitized output failure")
                    finally:
                        build.write_bytes(original)
                    if query() != baseline:
                        raise AssertionError("Output declaration did not recover after " + label)
                    receipt["checks"].append(label + " repaired without stale output")
            finally:
                escaped.unlink()
            waiting_file = primary / "output-query-started"
            try:
                build.write_bytes(original + b"\nfile('../output-query-started').text = 'started'\nThread.sleep(30000)\n")
                responses = []
                request_id = client.next_id
                def waiting():
                    try: responses.append(query())
                    except Exception as failure: responses.append(failure)
                thread = threading.Thread(target=waiting);thread.start()
                deadline = time.monotonic() + 50
                while not waiting_file.exists() and thread.is_alive() and time.monotonic() < deadline:
                    time.sleep(0.05)
                if not waiting_file.exists():
                    thread.join(timeout=35)
                    raise AssertionError("Output cancellation did not reach fixture configuration")
                client.notify("$/cancelRequest", {"id": request_id});thread.join(timeout=40)
                if thread.is_alive() or not responses or not isinstance(responses[0], RuntimeError) or "cancel" not in str(responses[0]).lower():
                    raise AssertionError("Output cancellation returned success or no cancellation category")
                receipt["checks"].append("in-flight output LSP cancellation rejects success")
            finally:
                build.write_bytes(original)
                if waiting_file.exists(): waiting_file.unlink()
            if query() != baseline:
                raise AssertionError("Output declaration did not recover after cancellation")
            receipt["checks"].append("output recovery after cancellation")
            disabled = json.loads(json.dumps(settings));disabled["java"]["import"]["gradle"]["home"] = ""
            client.notify("workspace/didChangeConfiguration", {"settings": disabled})
            expect_failure(client, request, "configured cached Gradle installation")
            client.notify("workspace/didChangeConfiguration", {"settings": settings})
            if query() != baseline:
                raise AssertionError("Output declaration did not recover after home restoration")
            receipt["checks"].extend(["output offline bootstrap denied without cached installation", "output home restoration recovery"])
            basic = discover(client, primary)
            detailed = discover(client, primary, details=True)
            if basic["version"] != 1 or detailed["version"] != 2 or any("outputsJson" in record for response in [basic,detailed] for record in response["projects"]):
                raise AssertionError("Output command changed existing v1/v2 schema")
            retain("existing v1/v2 discovery compatible and output contract separate", {"basic": basic, "detailed": detailed})
            if list(root.rglob("*.class")) or list(root.rglob("TEST-*.xml")) or (primary / "library/custom").exists():
                raise AssertionError("Output query produced fixture classes/resources/test reports")
            receipt["checks"].append("all output-producing tasks and reports remain absent")

        artifact = primary / "library/build/libs/fixture-required-library.jar"
        prepared_outputs = {}
        if arguments.provider_fresh_module_states_only:
            persistent_namespace = {}
            def producer_counters():
                values = {}
                for name,path,token in [("sync",primary / "sdk-sync-counter","sync"),
                        ("ordinary",primary / "included/ordinary-producer-counter","ordinary"),
                        ("affected",primary / "affected-producer-counter","affected")]:
                    lines = path.read_text().splitlines() if path.exists() else []
                    if any(line != token for line in lines):
                        raise AssertionError("SDK task counter has unexpected content")
                    values[name] = len(lines)
                if (primary / "forbidden-default-task").exists():
                    raise AssertionError("SDK compatibility executed a default task")
                return values
            def verify_producer_counters(before,journey,stage,library_closed):
                after = producer_counters()
                journey.setdefault("producerCounters",[]).append({"stage":stage,"before":before,"after":after,"libraryClosed":library_closed})
                if after["sync"] != before["sync"]+1 or after["ordinary"] != before["ordinary"]+1 or after["affected"] != before["affected"]+int(library_closed):
                    raise AssertionError("SDK sync/closed-artifact producers were skipped or executed more than once")
                receipt["checks"].append(stage + ": single sync/ordinary/affected producer counts and no default task")

        def module_snapshot(client, journey, stage, idle=False, failures=False):
            if not arguments.module_state_snapshots_only:
                return
            paths = [primary / name for name in ["application","library","included"]]
            before = producer_counters() if idle else None
            request = {"rootUri":primary.as_uri(),"offline":True}
            if failures:
                alias = root / "snapshot-root-alias"
                alias.symlink_to(primary,target_is_directory=True)
                try:
                    for invalid in [{**request,"offline":False},{**request,"extra":"private-canary"},
                            {"rootUri":primary.as_uri()},{**request,"rootUri":alias.as_uri()},
                            {**request,"rootUri":root.as_uri()},{**request,"rootUri":(primary / "library").as_uri()},
                            {**request,"rootUri":"https://private-canary.invalid/"}]:
                        expect_failure(client,{"command":"groovy.gradle.moduleStates","arguments":[invalid]},"Gradle module states unavailable",str(root))
                    receipt["checks"].append(stage + ": malformed/foreign/alias/offline snapshot requests refuse")
                finally:alias.unlink()
            if failures:
                empty=client.request("workspace/executeCommand",{"command":"groovy.gradle.moduleStates","arguments":[{"rootUri":unrelated.as_uri(),"offline":True}]},timeout=120)
                if set(empty)!={"version","rootUri","modules","fullSynchronizationKnown"} or empty["version"]!=1 or empty["rootUri"].rstrip("/")!=unrelated.as_uri() or empty["modules"]!=[] or empty["fullSynchronizationKnown"] is not False:
                    raise AssertionError("Root-only build snapshot is not an exact empty module collection")
                receipt["checks"].append(stage + ": root-only initialized build returns an empty bounded module set")
            response = client.request("workspace/executeCommand",{"command":"groovy.gradle.moduleStates","arguments":[request]},timeout=120)
            if (set(response)!={"version","rootUri","modules","fullSynchronizationKnown"} or response["version"]!=1
                    or response["rootUri"].rstrip("/")!=primary.as_uri() or response["fullSynchronizationKnown"] is not False
                    or not isinstance(response["modules"],list) or len(response["modules"])!=3):
                raise AssertionError("Module snapshot envelope differs or overclaims")
            observed={}
            from urllib.parse import unquote,urlparse
            for item in response["modules"]:
                if set(item)!={"moduleUri","buildRootUri","workspaceState","descriptionAvailable"}:
                    raise AssertionError("Module snapshot record fields differ")
                module=Path(unquote(urlparse(item["moduleUri"]).path)).resolve()
                if module not in paths or module in observed:raise AssertionError("Module snapshot identities are foreign/root/duplicate")
                expected_build=module if module.name=="included" else primary
                if item["buildRootUri"].rstrip("/")!=expected_build.as_uri():raise AssertionError("Module snapshot build ownership differs")
                actual=client.request("workspace/executeCommand",{"command":"groovy.fixture.unlink.projectState","arguments":[{"rootUri":primary.as_uri(),"projectUri":module.as_uri(),"operation":"state"}]},timeout=30)
                expected="absent" if actual.get("missing") is True else "open" if actual.get("open") is True else "closed"
                if item["workspaceState"]!=expected or item["descriptionAvailable"] is not (module / ".project").is_file():
                    raise AssertionError("Provider module state differs from actual SDK resource/description")
                observed[module]=item
            if before is not None and producer_counters()!=before:raise AssertionError("Read-only module snapshot produced SDK sync or artifacts")
            journey.setdefault("moduleSnapshots",[]).append({"stage":stage,"request":request,"response":response,"idleCountersUnchanged":idle})
            receipt["checks"].append(stage + ": exact module/build states match independent SDK resource oracle")

        def required_library_observation(client, journey, label, cached, project_name, closed=False):
            from urllib.parse import unquote, urlparse
            application = primary / "application"
            samples = []
            for _ in range(3):
                result = client.request("workspace/executeCommand", {"command": "java.project.getSettings", "arguments": [application.as_uri(), ["org.eclipse.jdt.ls.core.classpathEntries"]]}, timeout=30)
                entries = result.get("org.eclipse.jdt.ls.core.classpathEntries") if isinstance(result, dict) else None
                if not isinstance(entries, list) or len(entries) > 1024 or any(not isinstance(entry, dict) for entry in entries):
                    raise AssertionError("Required-library application classpath is unavailable or exceeds its bound")
                samples.append(entries)
                time.sleep(0.2)
            entries = samples[-1]
            project_reference = any(entry.get("kind") == 2 and entry.get("path") == "/" + project_name for entry in entries)
            binaries = {artifact, primary / "library/build/classes/java/main"}
            replacements = []
            for entry in entries:
                value = entry.get("path")
                if entry.get("kind") == 1 and isinstance(value, str):
                    path = Path(unquote(urlparse(value).path) if value.startswith("file:") else value)
                    if path in binaries:
                        replacements.append({"path": str(path), "exists": path.exists(), "jar": path == artifact})
            for path, digest in prepared_outputs.items():
                if closed:
                    if path.suffix == ".class" and path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                        raise AssertionError("Closed-module compilation differs from the prepared required class")
                elif path.is_file() != cached or cached and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise AssertionError("Prepared library output availability or immutable bytes changed")
            if closed and artifact.is_file():
                import zipfile
                with zipfile.ZipFile(artifact) as archive:
                    expected_class = prepared_outputs[primary / "library/build/classes/java/main/library/Required.class"]
                    if hashlib.sha256(archive.read("library/Required.class")).hexdigest() != expected_class:
                        raise AssertionError("Closed-module replacement JAR has different required class bytes")
            source = application / "src/main/java/application/UsesLibrary.java"
            with client.notifications_lock:
                start = len(client.notifications)
            client.notify("textDocument/didOpen", {"textDocument": {"uri": source.as_uri(), "languageId": "java", "version": 1, "text": source.read_text()}})
            diagnostics = None
            version_known = False
            try:
                for _ in range(30):
                    client.request("textDocument/documentSymbol", {"textDocument": {"uri": source.as_uri()}}, timeout=15)
                    with client.notifications_lock:
                        publications = [message.get("params", {}) for message in list(client.notifications)[start:]
                            if message.get("method") == "textDocument/publishDiagnostics" and message.get("params", {}).get("uri") == source.as_uri()]
                    if publications:
                        publication = publications[-1]
                        diagnostics = publication.get("diagnostics")
                        if not isinstance(diagnostics, list) or len(diagnostics) > 128 or any(not isinstance(item, dict) for item in diagnostics):
                            raise AssertionError("Required-library diagnostics exceed the fixture bound")
                        version_known = publication.get("version") == 1
                        if cached or any(item.get("severity") == 1 for item in diagnostics):
                            break
                    time.sleep(0.2)
            finally:
                client.notify("textDocument/didClose", {"textDocument": {"uri": source.as_uri()}})
            if diagnostics is not None and any(str(root) in item.get("message", "") or str(home) in item.get("message", "") for item in diagnostics):
                raise AssertionError("Required-library diagnostic exposes a private fixture path")
            observed = membership(client)
            errors = None if diagnostics is None else any(item.get("severity") == 1 for item in diagnostics)
            qualified = not project_reference and bool(replacements) and observed == all_members - {"primary/library"} and diagnostics is not None and errors == (False if closed else not cached)
            if closed:
                qualified = qualified and any(item["exists"] for item in replacements)
            journey.setdefault("requiredLibrary", []).append({"label": label, "cachedOutputCell": cached,
                "classpathSamples": samples, "requiredProjectReferenceRemains": project_reference,
                "preparedBinaryReplacements": replacements, "members": sorted(observed),
                "removedLibraryReimported": "primary/library" in observed,
                "sourceDiagnosticsKnown": diagnostics is not None, "sourceDiagnosticVersionKnown": version_known,
                "sourceDiagnostics": None if diagnostics is None else [{"code": item.get("code"), "severity": item.get("severity"), "range": item.get("range")} for item in diagnostics],
                "sourceErrorObserved": errors, "replacementQualified": qualified, "closedModuleCell": closed,
                "fullSynchronizationKnown": False})
            receipt["checks"].append(label + ": bounded classpath, immutable output, membership and diagnostic observation retained")
            if not qualified:
                receipt.setdefault("parityGaps", []).append(label + ": required-library replacement is unsupported or lacks affirmative diagnostic evidence")

        def fresh_closed_startup(client, journey, label, cached):
            from urllib.parse import unquote, urlparse
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if membership(client):
                    raise AssertionError("Fresh module was imported before its explicit policy phase")
                with client.notifications_lock:
                    started = any(message.get("method") == "language/status" and message.get("params", {}).get("type") == "Started" for message in client.notifications)
                if started: break
                time.sleep(0.1)
            else:
                raise AssertionError("Fresh SDK workspace startup did not complete before the seed check")
            def checkpoint(stage):
                presence = {str(path.relative_to(root)):path.is_file() for path in prepared_outputs}
                if any(present != cached for present in presence.values()):
                    raise AssertionError("SDK discovery/seed changed output readiness before explicit synchronization")
                for path,digest in prepared_outputs.items():
                    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                        raise AssertionError("SDK discovery/seed changed a cached required output")
                journey.setdefault("freshOutputCheckpoints",[]).append({"stage":stage,"cachedOutputCell":cached,"filesPresent":presence})
                receipt["checks"].append(label + ": " + stage + " retains cached/missing output status before sync")
            checkpoint("before model discovery")
            descriptor = primary / "library/.project"
            if descriptor.exists():
                raise AssertionError("Fresh module already has an SDK description")
            invalid = {"rootUri":primary.as_uri(),"offline":True,"closedProjectUris":[(primary / "library").as_uri()],"restoredProjectUris":[]}
            if arguments.provider_fresh_module_states_only:
                invalid["closedProjectUris"].append((primary / "included").as_uri())
            if not arguments.provider_fresh_module_states_only:
                expect_failure(client, {"command":"groovy.gradle.applyModuleStates","arguments":[invalid]}, "Gradle module state change failed or may be partial",str(root))
                if descriptor.exists() or membership(client):
                    raise AssertionError("Missing-description refusal imported or seeded a project")
                receipt["checks"].append(label + ": existing production phase refuses missing description with no Java import")
            def operation(name):
                response = client.request("workspace/executeCommand", {"command":"groovy.fixture.unlink.freshGradleModule","arguments":[{"rootUri":primary.as_uri(),"projectUri":(primary / "library").as_uri(),"operation":name}]}, timeout=120)
                journey.setdefault("freshModuleReplies",[]).append({"operation":name,"response":response})
                if (set(response) != {"version","rootUri","projects","selected","createdClosed","readOnly","reservation","fullSynchronizationKnown"}
                        or response["version"] != 1 or response["fullSynchronizationKnown"] is not False
                        or response["readOnly"] is not (name == "query") or response["createdClosed"] is not (name == "seed")
                        or Path(unquote(urlparse(response["rootUri"]).path)).resolve() != primary
                        or len(response["projects"]) != 4):
                    raise AssertionError("SDK fresh-module envelope differs")
                paths = set(); names = set()
                for record in response["projects"]:
                    if set(record) != {"moduleUri","buildRootUri","name"}: raise AssertionError("SDK project identity fields differ")
                    path = Path(unquote(urlparse(record["moduleUri"]).path)).resolve()
                    build_root = Path(unquote(urlparse(record["buildRootUri"]).path)).resolve()
                    if not path.is_relative_to(primary) or not build_root.is_relative_to(primary) or path in paths or record["name"] in names:
                        raise AssertionError("SDK model has foreign/duplicate module or name identities")
                    paths.add(path); names.add(record["name"])
                if paths != {primary,primary / "application",primary / "library",primary / "included"} or Path(unquote(urlparse(response["selected"]["moduleUri"]).path)).resolve() != primary / "library":
                    raise AssertionError("SDK model omits or selects another composite module")
                if membership(client): raise AssertionError("SDK model/seed command imported a Java module")
                return response
            if arguments.module_state_snapshots_only:
                if cached:
                    budget_settings=primary / "settings.gradle"
                    original_settings=budget_settings.read_bytes()
                    directories=[]
                    budget_hashes={}
                    before_budget=producer_counters()
                    registry_before=client.request("workspace/executeCommand",{"command":"groovy.fixture.unlink.projectState","arguments":[{"rootUri":primary.as_uri(),"projectUri":(primary / "library").as_uri(),"operation":"state"}]},timeout=30)
                    try:
                        for count in [8,9]:
                            for index in range(len(directories),count):
                                directory=primary / ("snapshot-budget-"+str(index));directory.mkdir();directories.append(directory)
                                (directory / "build.gradle").write_text("")
                                prefix=("<projectDescription><name>snapshot-budget-"+str(index)+"</name><natures><nature>org.eclipse.buildship.core.gradleprojectnature</nature></natures><!--").encode()
                                suffix=b"--></projectDescription>"
                                data=prefix+b"x"*(1024*1024-len(prefix)-len(suffix))+suffix
                                (directory / ".project").write_bytes(data);budget_hashes[directory]=hashlib.sha256(data).hexdigest()
                            budget_settings.write_bytes(original_settings+("\ninclude "+", ".join("'"+path.name+"'" for path in directories)+"\n").encode())
                            request={"command":"groovy.gradle.moduleStates","arguments":[{"rootUri":primary.as_uri(),"offline":True}]}
                            if count==8:
                                value=client.request("workspace/executeCommand",request,timeout=120)
                                if len(value.get("modules",[]))!=11 or value.get("fullSynchronizationKnown") is not False:
                                    raise AssertionError("Exact8MiB snapshot description bound did not admit complete module data")
                                for path in directories:
                                    item=next((item for item in value["modules"] if item["moduleUri"].rstrip("/")==path.as_uri()),None)
                                    if item is None or item.get("workspaceState")!="absent" or item.get("descriptionAvailable") is not True:
                                        raise AssertionError("Description-only budget module was imported or lost")
                                journey.setdefault("snapshotDescriptionBudgets",[]).append({"bytes":8*1024*1024,"response":value})
                                receipt["checks"].append(label + ": exact8MiB total descriptors admit without SDK import")
                            else:
                                error=expect_failure(client,request,"Gradle module states unavailable",str(root))
                                journey.setdefault("snapshotDescriptionBudgets",[]).append({"bytes":9*1024*1024,"error":error})
                                receipt["checks"].append(label + ": over8MiB descriptors refuse without SDK import")
                            registry_after=client.request("workspace/executeCommand",{"command":"groovy.fixture.unlink.projectState","arguments":[{"rootUri":primary.as_uri(),"projectUri":(primary / "library").as_uri(),"operation":"state"}]},timeout=30)
                            if registry_after!=registry_before or producer_counters()!=before_budget or membership(client):
                                raise AssertionError("Description budget read changed SDK registry, Java membership or outputs")
                    finally:
                        budget_settings.write_bytes(original_settings)
                        for path in directories:
                            if path.is_symlink() or sorted(item.name for item in path.iterdir())!=[".project","build.gradle"] or hashlib.sha256((path / ".project").read_bytes()).hexdigest()!=budget_hashes[path]:
                                raise AssertionError("Owned budget fixture changed unexpectedly during read")
                            (path / ".project").unlink();(path / "build.gradle").unlink();path.rmdir()
                module_snapshot(client,journey,label + ": fresh absent snapshot",idle=True,failures=True)
            baseline = operation("query")
            if baseline["selected"]["name"] != "groovy-eclipse-library":
                raise AssertionError("Authoritative SDK name ignores the explicit Eclipse name")
            reserved = operation("reserve")["reservation"]
            reservation = Path(unquote(urlparse(reserved["location"]).path)).resolve()
            if reservation.parent != (primary if arguments.provider_fresh_module_states_only else root) or reserved["name"] != baseline["selected"]["name"] or reserved["open"] is not arguments.provider_fresh_module_states_only:
                raise AssertionError("SDK namespace reservation escaped its fixture or opened a project")
            reservation_bytes = (reservation / ".project").read_bytes()
            if arguments.provider_fresh_module_states_only:
                (reservation / "evidence-preserve.txt").write_text("unrelated namespace source\n")
            collision = operation("query")
            if collision["selected"]["name"] == baseline["selected"]["name"] or (reservation / ".project").read_bytes() != reservation_bytes:
                raise AssertionError("SDK failed to supply a conflict-free name or changed the unrelated reservation")
            if arguments.provider_fresh_module_states_only:
                persistent_namespace.update(path=reservation,name=reserved["name"],sha256=hashlib.sha256(reservation_bytes).hexdigest())
                restore = {**invalid,"closedProjectUris":[],"restoredProjectUris":[(primary / "library").as_uri()]}
                expect_failure(client,{"command":"groovy.gradle.applyModuleStates","arguments":[restore]},"Gradle module state change failed or may be partial",str(root))
                if descriptor.exists() or membership(client):
                    raise AssertionError("Fresh restore refusal mutated module membership or description")
                receipt["checks"].append(label + ": fresh restore without a description refuses before mutation")
                qualifiers.qualify_unlink_wait(client,primary,invalid,"groovy.gradle.applyModuleStates",receipt,journey)
                if descriptor.exists() or (primary / "included/.project").exists() or membership(client):
                    raise AssertionError("Cancelled fresh state request created or imported a module")
                receipt["checks"].append(label + ": fresh held cancellation preserves absent descriptions and membership")
                held = client.request("workspace/executeCommand", {"command":"groovy.fixture.unlink.hold","arguments":[{"rootUri":primary.as_uri()}]},timeout=30)
                paths = {name:Path(held[name]) for name in ["started","release","finished"]}
                if any(path.parent != primary.parent for path in paths.values()):
                    raise AssertionError("Fresh race gate escaped its disposable fixture")
                injected = b"<projectDescription><name>unexpected-fresh-description</name></projectDescription>"
                try:
                    deadline = time.monotonic()+10
                    while not paths["started"].exists() and time.monotonic()<deadline:time.sleep(0.02)
                    if not paths["started"].exists():raise AssertionError("Fresh race workspace gate did not start")
                    request_id = client.next_id
                    client.next_id += 1
                    client.send({"jsonrpc":"2.0","id":request_id,"method":"workspace/executeCommand","params":{"command":"groovy.gradle.applyModuleStates","arguments":[invalid]}})
                    deadline = time.monotonic()+20
                    while time.monotonic()<deadline:
                        if client.request("workspace/executeCommand",{"command":"groovy.fixture.unlink.stack","arguments":[]},timeout=10).get("unlinkRequestWaiting") is True:break
                        time.sleep(0.02)
                    else:raise AssertionError("Fresh race request never reached the workspace mutation fence")
                    descriptor.write_bytes(injected)
                    paths["release"].write_text("release\n")
                    import queue
                    deadline = time.monotonic()+30
                    reply = None
                    while time.monotonic()<deadline:
                        try:message=client.messages.get(timeout=max(0.01,deadline-time.monotonic()))
                        except queue.Empty:break
                        if isinstance(message,Exception):raise message
                        if message.get("id")==request_id and "method" not in message:
                            reply=message;break
                        if "id" in message and "method" in message:
                            if message["method"]!="workspace/configuration":raise AssertionError("Unexpected client request during fresh race")
                            client.send({"jsonrpc":"2.0","id":message["id"],"result":[{} for _ in message.get("params",{}).get("items",[])]})
                        elif "method" in message:
                            with client.notifications_lock:client.notifications.append(message)
                    if reply is None or "error" not in reply or "Gradle module state change failed or may be partial" not in reply["error"].get("message","") or str(root) in json.dumps(reply):
                        raise AssertionError("Fresh descriptor race did not refuse with a sanitized phase error")
                    if descriptor.read_bytes()!=injected or (primary / "included/.project").exists() or membership(client):
                        raise AssertionError("Fresh descriptor race changed bytes or partially created modules")
                    journey.setdefault("freshDescriptorRaces",[]).append({"error":reply["error"],"preservedSha256":hashlib.sha256(injected).hexdigest(),"noMembership":True})
                    receipt["checks"].append(label + ": fresh descriptor appearance refuses every target before mutation")
                finally:
                    paths["release"].write_text("release\n")
                    deadline=time.monotonic()+10
                    while not paths["finished"].exists() and time.monotonic()<deadline:time.sleep(0.02)
                    if not paths["finished"].exists():raise AssertionError("Fresh race gate did not finish")
                    if descriptor.exists():
                        if descriptor.read_bytes()!=injected:raise AssertionError("Fresh race cleanup found unexpected bytes")
                        descriptor.unlink()
                    for path in paths.values():
                        if path.exists():path.unlink()
                response = client.request("workspace/executeCommand", {"command":"groovy.gradle.applyModuleStates","arguments":[invalid]}, timeout=120)
                journey.setdefault("freshProductionReplies",[]).append(response)
                if (set(response) != {"version","rootUri","closedProjectUris","restoredProjectUris","moduleStatePhaseComplete","fullSynchronizationKnown"}
                        or response["version"] != 1 or response["moduleStatePhaseComplete"] is not True or response["fullSynchronizationKnown"] is not False
                        or response["rootUri"].rstrip("/") != primary.as_uri()
                        or response["closedProjectUris"] != [uri+"/" for uri in invalid["closedProjectUris"]] or response["restoredProjectUris"] != []):
                    raise AssertionError("Production fresh module-state receipt differs or overclaims")
                expected_name = collision["selected"]["name"]
            else:
                released = operation("release")["reservation"]
                if released["sourcePreserved"] is not True or (reservation / ".project").read_bytes() != reservation_bytes:
                    raise AssertionError("SDK namespace release removed source content")
                if operation("query")["selected"] != baseline["selected"]:
                    raise AssertionError("SDK model name did not recover after namespace release")
                if sorted(path.name for path in reservation.iterdir()) != [".project"]:
                    raise AssertionError("Disposable reservation contains unexpected source content")
                (reservation / ".project").unlink()
                reservation.rmdir()
                receipt["checks"].append(label + ": namespace release preserves metadata before explicit owned-fixture cleanup")
                seeded = operation("seed")
                if seeded["selected"] != baseline["selected"] or not descriptor.is_file():
                    raise AssertionError("SDK closed resource seed differs from its authoritative model")
                expected_name = baseline["selected"]["name"]
            state = client.request("workspace/executeCommand", {"command":"groovy.fixture.unlink.projectState","arguments":[{"rootUri":primary.as_uri(),"projectUri":(primary / "library").as_uri(),"operation":"state"}]}, timeout=30)
            if state.get("open") is not False or state.get("name") != expected_name:
                raise AssertionError("Fresh SDK resource was opened or guessed")
            receipt["checks"].append(label + ": fresh custom-name/collision/recovery model stays outside Java import; SDK seed remains closed")
            checkpoint("after SDK name queries and closed seed")
            if arguments.module_state_snapshots_only:
                module_snapshot(client,journey,label + ": production closed snapshot",idle=True)
            if arguments.module_state_snapshots_only:
                for fault in ["descriptor","cancel","root","linkage","oversized","dual","missing","alias","directory"]:
                    response=client.request("workspace/executeCommand",{"command":"groovy.fixture.unlink.moduleSnapshotFault","arguments":[{"rootUri":primary.as_uri(),"fault":fault}]},timeout=120)
                    if response.get("failed") is not True or response.get("faultApplied") is not True or str(root) in json.dumps(response) or "private-canary" in json.dumps(response):
                        raise AssertionError("Snapshot fault did not refuse without leaking fixture inputs")
                    message=response.get("message","")
                    if "refresh" not in message.lower() or ("Gradle module state read cancelled" not in message and "Gradle module states unavailable" not in message):
                        raise AssertionError("Snapshot fault lost bounded recovery guidance")
                    journey.setdefault("moduleSnapshotFaults",[]).append({"fault":fault,"response":response})
                    module_snapshot(client,journey,label + ": recovery after " + fault,idle=True)
                    receipt["checks"].append(label + ": snapshot " + fault + " refuses with preserved SDK state and recovery")

            for build_root in [primary,unrelated]:
                if arguments.provider_fresh_module_states_only and build_root == primary:
                    before = producer_counters()
                try:
                    client.request("workspace/executeCommand", {"command":"groovy.gradle.importSelected","arguments":[{"rootUri":build_root.as_uri(),"offline":True}]}, timeout=180)
                except RuntimeError:
                    if arguments.provider_fresh_module_states_only:
                        journey["compatibilityDiagnostic"] = client.request("workspace/executeCommand", {"command":"groovy.fixture.unlink.compatibilityDiagnostic","arguments":[{"rootUri":primary.as_uri()}]},timeout=180)
                    raise
                if arguments.provider_fresh_module_states_only and build_root == primary:
                    verify_producer_counters(before,journey,label + ": initial sync",True)
            await_membership(client, all_members - {"primary/library"}, journey, label + ": initial explicit sync leaves fresh library unimported")
            preserve_sources(label + ": fresh seed and model discovery preserve fixture sources and reports")

        def closed_module_lifecycle(client, journey, label, cached, project_name, restart):
            from urllib.parse import unquote, urlparse
            def namespace_checkpoint(stage):
                namespace = persistent_namespace["path"]
                response = client.request("workspace/executeCommand", {"command":"groovy.fixture.unlink.projectState","arguments":[{"rootUri":primary.as_uri(),"projectUri":namespace.as_uri(),"operation":"state"}]},timeout=30)
                actual_bytes = (namespace / ".project").read_bytes()
                if (namespace / "evidence-preserve.txt").read_bytes() != b"unrelated namespace source\n":
                    raise AssertionError("SDK synchronization/restart changed unrelated source content")
                journey.setdefault("namespaceObservations",[]).append({"stage":stage,"state":response,"expectedName":persistent_namespace["name"],"actualDescriptionSha256":hashlib.sha256(actual_bytes).hexdigest(),"expectedDescriptionSha256":persistent_namespace["sha256"],"description":actual_bytes.decode()})
                if (response.get("name") != persistent_namespace["name"] or response.get("open") is not True
                        or hashlib.sha256(actual_bytes).hexdigest() != persistent_namespace["sha256"]):
                    raise AssertionError("SDK synchronization/restart changed the unrelated namespace owner")
                journey.setdefault("persistentNamespace",[]).append({"stage":stage,"state":response,"descriptionSha256":persistent_namespace["sha256"]})
                receipt["checks"].append(label + ": " + stage + " preserves the live unrelated namespace owner")
            if arguments.provider_fresh_module_states_only:
                namespace_checkpoint("initial state")
            def module_state_request(open_state=False):
                closed = [(primary / "library").as_uri()]
                if arguments.provider_fresh_module_states_only:
                    closed.append((primary / "included").as_uri())
                return {"rootUri": primary.as_uri(), "offline": True,
                    "closedProjectUris": [] if open_state else closed,
                    "restoredProjectUris": [(primary / "library").as_uri()] if open_state else []}
            def state(operation, expected_open):
                if arguments.provider_module_states_only and operation != "state":
                    request = module_state_request(operation == "open")
                    response = client.request("workspace/executeCommand", {"command": "groovy.gradle.applyModuleStates", "arguments": [request]}, timeout=90)
                    if (set(response) != {"version", "rootUri", "closedProjectUris", "restoredProjectUris", "moduleStatePhaseComplete", "fullSynchronizationKnown"}
                            or response["version"] != 1 or response["rootUri"].rstrip("/") != primary.as_uri()
                            or response["moduleStatePhaseComplete"] is not True or response["fullSynchronizationKnown"] is not False
                            or response["closedProjectUris"] != [uri + "/" for uri in request["closedProjectUris"]]
                            or response["restoredProjectUris"] != [uri + "/" for uri in request["restoredProjectUris"]]):
                        raise AssertionError("Production Gradle module-state receipt differs or overclaims")
                    journey["commands"].append({"command":"groovy.gradle.applyModuleStates","request":request,"reply":response})
                    operation = "state"
                request = {"rootUri": primary.as_uri(), "projectUri": (primary / "library").as_uri(), "operation": operation}
                response = client.request("workspace/executeCommand", {"command": "groovy.fixture.unlink.projectState", "arguments": [request]}, timeout=30)
                journey.setdefault("workspaceStateReplies", []).append({"request": request, "reply": response})
                if (set(response) != {"version", "rootUri", "projectUri", "name", "exists", "open", "accessible"}
                        or response["version"] != 1 or response["rootUri"].rstrip("/") != primary.as_uri()
                        or Path(unquote(urlparse(response["projectUri"]).path)).resolve() != primary / "library"
                        or response["name"] != project_name or response["exists"] is not True
                        or response["open"] is not expected_open or response["accessible"] is not expected_open):
                    raise AssertionError("Actual SDK closed-resource identity/state differs")
                journey["commands"].append({"request": request, "reply": response})
                if arguments.module_state_snapshots_only:
                    module_snapshot(client,journey,label + ": " + operation + " snapshot",idle=not expected_open)
                return response
            def synchronize(stage, expected_members):
                if arguments.provider_fresh_module_states_only:
                    before = producer_counters()
                request = {"rootUri": primary.as_uri(), "offline": True}
                response = client.request("workspace/executeCommand", {"command": "groovy.gradle.importSelected", "arguments": [request]}, timeout=180)
                if (set(response) != {"version", "rootUri", "selectedImportPhaseComplete", "fullSynchronizationKnown"}
                        or response["version"] != 1 or response["rootUri"].rstrip("/") != primary.as_uri()
                        or response["selectedImportPhaseComplete"] is not True or response["fullSynchronizationKnown"] is not False):
                    raise AssertionError("Existing Gradle synchronization phase receipt differs or overclaims")
                journey["commands"].append({"label": stage, "request": request, "reply": response})
                await_membership(client, expected_members, journey, label + ": " + stage)
                if arguments.provider_fresh_module_states_only:
                    namespace_checkpoint(stage)
                    verify_producer_counters(before,journey,label + ": " + stage,"primary/library" not in expected_members)
            def restored_classpath():
                values = client.request("workspace/executeCommand", {"command": "java.project.getSettings", "arguments": [(primary / "application").as_uri(), ["org.eclipse.jdt.ls.core.classpathEntries"]]}, timeout=30)
                entries = values.get("org.eclipse.jdt.ls.core.classpathEntries", [])
                if not any(entry.get("kind") == 2 and entry.get("path") == "/" + project_name for entry in entries):
                    raise AssertionError("Restoring the module did not recover the required-project classpath")
                journey.setdefault("restoredClasspaths", []).append(entries)
            if restart:
                request = {"rootUri":primary.as_uri(),"projectUri":(primary / "library").as_uri(),"operation":"state"}
                response = client.request("workspace/executeCommand", {"command":"groovy.fixture.unlink.projectState","arguments":[request]}, timeout=30)
                journey.setdefault("workspaceStateReplies", []).append({"label":"bare closed metadata absent after startup", "request":request,"reply":response})
                if arguments.module_state_snapshots_only:
                    module_snapshot(client,journey,label + ": cold absent description snapshot",idle=True)
                if response.get("missing") is not True:
                    raise AssertionError("Pinned cold-start cleanup control changed; requalify the recovery contract")
                preserve_sources(label + ": cold startup preserves source/build/report bytes")
                state("recreateClosed", False)
                receipt["checks"].append(label + ": explicit SDK-description recovery recreates a closed resource without Java import")
            elif arguments.fresh_closed_modules_only:
                state("state", False)
            else:
                state("state", True)
            if not restart and not arguments.fresh_closed_modules_only:
                restored_classpath()
                before = membership(client)
                for request in [
                    {"rootUri":primary.as_uri(),"projectUri":unrelated.as_uri(),"operation":"close"},
                    {"rootUri":primary.as_uri(),"projectUri":(primary / "library").as_uri(),"operation":"invalid"},
                    {"rootUri":primary.as_uri(),"projectUri":(primary / "library").as_uri(),"operation":"close","extra":True},
                ]:
                    try:
                        client.request("workspace/executeCommand", {"command":"groovy.fixture.unlink.projectState","arguments":[request]}, timeout=30)
                    except RuntimeError:
                        state("state", True)
                        if membership(client) != before:
                            raise AssertionError("Rejected fixture state request changed membership")
                    else:
                        raise AssertionError("Out-of-scope or malformed state fixture request was accepted")
                receipt["checks"].append(label + ": malformed/foreign test-state requests refuse without closing a project")
                if arguments.provider_module_states_only and cached:
                    valid = module_state_request()
                    selected = (primary / "library").as_uri()
                    alias = primary / "library-alias"
                    alias.symlink_to(primary / "library", target_is_directory=True)
                    root_alias = root / "primary-alias"
                    root_alias.symlink_to(primary, target_is_directory=True)
                    invalid = [
                        {**valid,"rootUri":root_alias.as_uri()},
                        {**valid,"offline":False}, {**valid,"extra":True},
                        {**valid,"closedProjectUris":[]}, {**valid,"closedProjectUris":[selected]*129},
                        {**valid,"closedProjectUris":[selected,selected]},
                        {**valid,"restoredProjectUris":[selected]},
                        {**valid,"closedProjectUris":[primary.as_uri()]},
                        {**valid,"closedProjectUris":[unrelated.as_uri()]},
                        {**valid,"closedProjectUris":[alias.as_uri()]},
                        {**valid,"closedProjectUris":[(primary / "application/src").as_uri()]},
                        {**valid,"closedProjectUris":[selected+"?private-canary=1"]},
                        {**valid,"closedProjectUris":["https://private-canary.invalid/module"]},
                        {**valid,"closedProjectUris":["file:///"+"x"*4097]},
                        {**valid,"closedProjectUris":[selected,(primary / "private-canary").as_uri()]},
                    ]
                    def refuse(request, case):
                        try:
                            client.request("workspace/executeCommand", {"command":"groovy.gradle.applyModuleStates","arguments":[request]},timeout=90)
                        except RuntimeError as error:
                            if "Gradle module state change failed or may be partial" not in str(error) or str(root) in str(error) or "private-canary" in str(error):
                                raise AssertionError("Module-state refusal is not actionable and sanitized") from error
                            journey["commands"].append({"label":case,"error":str(error)})
                        else:
                            raise AssertionError("Invalid module-state request succeeded")
                        state("state", True)
                        if membership(client) != all_members:
                            raise AssertionError("Invalid module-state request changed membership")
                        receipt["checks"].append(case+": sanitized before mutation")
                    try:
                        for index, request in enumerate(invalid):
                            refuse(request, "module-state invalid request " + str(index))
                    finally:
                        alias.unlink()
                        root_alias.unlink()
                    descriptor = primary / "library/.project"
                    original = descriptor.read_bytes()
                    from xml.etree import ElementTree
                    application_name = ElementTree.parse(primary / "application/.project").getroot().findtext("name")
                    tree = ElementTree.fromstring(original)
                    original_name = tree.find("name").text
                    variants = [
                        ("dual Gradle/Maven description", original.replace(b"</natures>",b"<nature>org.eclipse.m2e.core.maven2Nature</nature></natures>")),
                        ("missing Gradle nature", original.replace(b"org.eclipse.buildship.core.gradleprojectnature",b"evidence.not.gradle")),
                        ("SDK name collision", original.replace(original_name.encode(),application_name.encode(),1)),
                        ("oversized SDK description",b"x"*(1024*1024+1)),
                    ]
                    try:
                        for case, content in variants:
                            if content == original: raise AssertionError("Invalid description fixture did not change")
                            descriptor.write_bytes(content)
                            refuse(valid, case)
                    finally:
                        descriptor.write_bytes(original)
                    backup = descriptor.with_name(".project.fixture-backup")
                    descriptor.rename(backup)
                    try:
                        refuse(valid, "missing SDK description")
                        descriptor.symlink_to(backup)
                        try: refuse(valid, "aliased SDK description")
                        finally: descriptor.unlink()
                    finally:
                        backup.rename(descriptor)
                    build_input = primary / "build.gradle"
                    original_build = build_input.read_bytes()
                    foreign_build = unrelated / "build.gradle"
                    foreign_original = foreign_build.read_bytes()
                    marker = primary / "forbidden-module-state-script"
                    try:
                        foreign_build.write_text("new File(rootProject.projectDir, 'forbidden-module-state-script').text = 'executed'\n" + foreign_original.decode())
                        build_input.unlink()
                        build_input.symlink_to(foreign_build)
                        refuse(valid, "escaped Gradle build input")
                        if marker.exists():
                            raise AssertionError("Escaped build script executed before provider refusal")
                    finally:
                        build_input.unlink()
                        build_input.write_bytes(original_build)
                        foreign_build.write_bytes(foreign_original)
                        if marker.exists(): marker.unlink()
                    receipt["checks"].append("escaped build script canary never executes before module-state refusal")
                    preserve_sources(label + ": malformed module-state requests preserve build/source/report bytes")
                    for held_case in ["cancel", "descriptor", "ownership", "root"]:
                        parameters = {}
                        if held_case == "descriptor": parameters["descriptor_change"] = descriptor
                        if held_case == "ownership": parameters["ownership_change"] = primary / "library"
                        if held_case == "root": parameters["replace_root"] = True
                        try:
                            qualifiers.qualify_unlink_wait(client, primary, valid, "groovy.gradle.applyModuleStates", receipt, journey, **parameters)
                        finally:
                            if held_case == "descriptor": descriptor.write_bytes(original)
                            if held_case == "ownership":
                                client.request("workspace/executeCommand", {"command":"groovy.fixture.unlink.natures","arguments":[{"rootUri":primary.as_uri(),"projectUri":(primary / "library").as_uri(),"dual":False}]}, timeout=30)
                        state("state", True)
                        if membership(client) != all_members:
                            raise AssertionError("Rejected held module-state phase changed membership")
                        receipt["checks"].append("held module-state " + held_case + ": no module closes; fixture restored")
                state("close", False)
            await_membership(client, all_members - {"primary/library"}, journey, label + ": closed module has no Java import")
            for repeat in range(2):
                synchronize("closed synchronization " + str(repeat), all_members - {"primary/library"})
                state("state", False)
            required_library_observation(client, journey, label + ": closed required-library substitution", cached, project_name, closed=True)
            if not journey["requiredLibrary"][-1]["replacementQualified"]:
                raise AssertionError("Closed-module binary substitution/readiness/diagnostics are not qualified")
            if arguments.provider_fresh_module_states_only:
                before_restore = producer_counters()
            state("open", True)
            if arguments.provider_fresh_module_states_only:
                deadline = time.monotonic() + 30
                stable_since = time.monotonic()
                previous = producer_counters()
                observations = []
                while time.monotonic() < deadline:
                    jobs = client.request("workspace/executeCommand", {"command":"groovy.fixture.unlink.sdkJobs","arguments":[{"rootUri":primary.as_uri()}]},timeout=10)
                    current = producer_counters()
                    observations.append({"jobs":jobs,"counters":current})
                    if jobs.get("jobs") or current != previous:
                        stable_since = time.monotonic()
                    if not jobs.get("jobs") and time.monotonic() - stable_since >= 1:
                        break
                    previous = current
                    time.sleep(0.1)
                else:
                    raise AssertionError("SDK restore jobs did not settle before explicit synchronization")
                journey.setdefault("restoreLifecycle",[]).append({"before":before_restore,"after":current,"observations":observations})
                if current["affected"] != before_restore["affected"] or current["sync"] - before_restore["sync"] not in (0,1) or current["ordinary"] - before_restore["ordinary"] != current["sync"] - before_restore["sync"]:
                    raise AssertionError("SDK restore lifecycle ran unexpected or duplicate producers")
                receipt["checks"].append(label + ": bounded SDK restore lifecycle settles separately from explicit sync")
            synchronize("restored module", all_members)
            state("state", True)
            restored_classpath()
            request = {"rootUri": primary.as_uri(), "projectUris": [(primary / "library").as_uri()]}
            response = client.request("workspace/executeCommand", {"command": "groovy.gradle.unlinkSelected", "arguments": [request]}, timeout=30)
            if response.get("selectedUnlinkPhaseComplete") is not True or response.get("fullSynchronizationKnown") is not False:
                raise AssertionError("Unlink control overclaims synchronization")
            await_membership(client, all_members - {"primary/library"}, journey, label + ": unlink control")
            synchronize("absent module reimports unlike closed module", all_members)
            state("state", True)
            restored_classpath()
            state("close", False)
            synchronize("closed state retained for cold restart", all_members - {"primary/library"})
            state("state", False)
            if list(root.rglob("TEST-*.xml")):
                raise AssertionError("Closed-module synchronization unexpectedly executed test reports")
            receipt["checks"].append(label + ": closed sync/repeat/restore/unlink control; required binaries ready; no selected-test claim")

        def run(label, data, expected, configurations=None, exclusions=None, actions=False, failure_actions=False, guarded_actions=False, required_cached=None, required_restart=False):
            journey = {"label": label, "states": [], "commands": [], "passed": False}
            receipt["journeys"].append(journey)
            configuration = json.loads(json.dumps(settings))
            if exclusions is not None:
                configuration["java"]["import"]["exclusions"] = exclusions
            if required_restart or arguments.fresh_closed_modules_only:
                configuration["java"]["import"]["gradle"]["enabled"] = False
            if arguments.closed_modules_only:
                configuration["java"]["import"]["gradle"]["version"] = None
            if arguments.guarded_phase_only:
                configuration["java"]["import"]["gradle"]["enabled"] = False
                configuration["java"]["import"]["gradle"]["version"] = None
            options = {"bundles": [str(provider)] + ([str(fixture_bundle)] if fixture_bundle else []), "settings": configuration,
                       "workspaceFolders": [path.as_uri() for path in [primary, unrelated]]}
            if configurations is not None:
                options["projectConfigurations"] = [path.as_uri() for path in configurations]
            temporary = Path(directory) / (label + "-host-temp")
            previous_options = os.environ.get("JAVA_TOOL_OPTIONS")
            if arguments.source_outputs_only or arguments.resolution_graph_only:
                temporary.mkdir()
                os.environ["JAVA_TOOL_OPTIONS"] = (previous_options or "") + " -Djava.io.tmpdir=" + str(temporary)
            try:
                client = client_class()(arguments.launcher.resolve(), arguments.java_home.resolve(), root, data,
                                        deny_external_network=True, gradle_user_home=home)
            finally:
                if previous_options is None:
                    os.environ.pop("JAVA_TOOL_OPTIONS", None)
                else:
                    os.environ["JAVA_TOOL_OPTIONS"] = previous_options
            try:
                initialized = client.request("initialize", {
                    "processId": os.getpid(), "rootUri": root.as_uri(),
                    "workspaceFolders": [{"uri": path.as_uri(), "name": path.name} for path in [primary, unrelated]],
                    "capabilities": {"workspace": {"workspaceFolders": True, "configuration": True}},
                    "initializationOptions": options,
                }, timeout=90)
                commands = initialized["capabilities"]["executeCommandProvider"]["commands"]
                if "java.project.changeImportedProjects" not in commands:
                    raise AssertionError("Pinned provider does not advertise project mutation")
                receipt["checks"].append(f"{label}: advertised project mutation")
                if arguments.guarded_unlink_only:
                    if not {"groovy.gradle.unlinkSelected", "groovy.maven.unlinkSelected"}.issubset(commands):
                        raise AssertionError("Guarded unlink commands are not advertised")
                    receipt["checks"].append(f"{label}: advertised guarded unlink")
                if arguments.guarded_phase_only:
                    if "groovy.gradle.importSelected" not in commands:
                        raise AssertionError("Guarded Gradle import command is not advertised")
                    receipt["checks"].append(f"{label}: advertised guarded phase")
                client.notify("initialized", {})
                if arguments.fresh_closed_modules_only and not required_restart:
                    fresh_closed_startup(client, journey, label, required_cached)
                    expected = all_members - {"primary/library"}
                await_membership(client, expected, journey, f"{label}: initial module identities")
                if arguments.resolution_graph_only:
                    if "groovy.gradle.dependencyGraph" not in commands:
                        raise AssertionError("Explicit graph delegate unavailable")
                    resolution_graph_checks(client, journey)
                    if list(temporary.glob("groovy-gradle-model-*.gradle")):
                        raise AssertionError("Graph query retained temporary init scripts")
                    receipt["checks"].append("graph success/failure/cancellation temporary-script cleanup")
                if arguments.source_outputs_only:
                    if "groovy.gradle.sourceOutputs" not in commands:
                        raise AssertionError("Output-only delegate is unavailable")
                    source_output_checks(client, journey)
                    if list(temporary.glob("groovy-gradle-model-*.gradle")):
                        raise AssertionError("Output query retained a temporary init script")
                    receipt["checks"].append("output success/failure/cancellation temporary-script cleanup")
                if required_cached is not None:
                    if "groovy.gradle.unlinkSelected" not in commands:
                        raise AssertionError("Guarded Gradle unlink capability is unavailable")
                    import xml.etree.ElementTree as ElementTree
                    project_file = primary / "library/.project"
                    if not project_file.is_file() or project_file.stat().st_size > 1024 * 1024:
                        raise AssertionError("Required-library fixture project identity is unavailable")
                    project_name = ElementTree.parse(project_file).getroot().findtext("name")
                    if not project_name or len(project_name) > 256 or "/" in project_name:
                        raise AssertionError("Required-library fixture project identity is invalid")
                    if arguments.closed_modules_only:
                        if not {"groovy.fixture.unlink.projectState", "groovy.gradle.importSelected"}.issubset(commands):
                            raise AssertionError("Closed-state fixture or existing sync command is unavailable")
                        closed_module_lifecycle(client, journey, label, required_cached, project_name, required_restart)
                    elif not required_restart:
                        result = client.request("workspace/executeCommand", {"command": "java.project.getSettings", "arguments": [(primary / "application").as_uri(), ["org.eclipse.jdt.ls.core.classpathEntries"]]}, timeout=30)
                        entries = result.get("org.eclipse.jdt.ls.core.classpathEntries", [])
                        if not any(entry.get("kind") == 2 and entry.get("path") == "/" + project_name for entry in entries):
                            raise AssertionError("Baseline application has no exact required-project classpath reference")
                        request = {"command": "groovy.gradle.unlinkSelected", "arguments": [{"rootUri": primary.as_uri(), "projectUris": [(primary / "library").as_uri()]}]}
                        reply = client.request("workspace/executeCommand", request, timeout=30)
                        if (set(reply) != {"version", "rootUri", "unlinkedProjectUris", "selectedUnlinkPhaseComplete", "fullSynchronizationKnown"}
                                or reply["version"] != 1 or reply["selectedUnlinkPhaseComplete"] is not True or reply["fullSynchronizationKnown"] is not False
                                or reply["rootUri"].rstrip("/") != primary.as_uri().rstrip("/")
                                or {uri.rstrip("/") for uri in reply["unlinkedProjectUris"]} != {(primary / "library").as_uri()}):
                            raise AssertionError("Required-library unlink receipt differs or overclaims synchronization")
                        journey["commands"].append({"request": request, "reply": reply})
                        await_membership(client, all_members - {"primary/library"}, journey, label + ": library-only unlink preserves composite and unrelated identities")
                    if not arguments.closed_modules_only:
                        required_library_observation(client, journey, label, required_cached, project_name)
                if arguments.resource_membership_only:
                    if not {"groovy.gradle.unlinkSelected", "groovy.gradle.importSelected"}.issubset(commands):
                        raise AssertionError("Guarded resource mutation commands unavailable")
                    required = {"groovy.gradle.importedBuildRoots", "groovy.gradle.importedProjects", "groovy.gradle.projectDetails.v4"}
                    if not required.issubset(commands):
                        raise AssertionError("Composite cached ownership and fenced model commands unavailable")
                    composite_cached_ownership(client, journey, label + ": before resource mutation")
                    qualifiers.qualify_imported_resources(client, root, primary, "gradle", expected,
                        {"primary", "primary/included"}, membership, journey, receipt)
                    composite_cached_ownership(client, journey, label + ": after resource repair")
                if guarded_actions:
                    def phase(root_uri=primary.as_uri(), offline=True):
                        return {"command": "groovy.gradle.importSelected",
                                "arguments": [{"rootUri": root_uri, "offline": offline}]}

                    for case_label, request in [
                        ("malformed phase", {"command": "groovy.gradle.importSelected", "arguments": [17]}),
                        ("online phase denied", phase(offline=False)),
                        ("non-file phase denied", phase("https://invalid.example/private-canary")),
                        ("noninitialized module phase denied", phase((primary / "application").as_uri())),
                        ("missing root phase denied", phase((root / "missing-private-canary").as_uri())),
                    ]:
                        error = expect_failure(client, request, "Selected Gradle import failed or may be partial", str(root))
                        journey["commands"].append({"label": case_label, "error": error})
                        receipt["checks"].append(case_label + ": sanitized")
                    for case_label, field, value in [
                        ("missing configured Gradle installation", "home", None),
                        ("wrapper bootstrap denied", "wrapper", {"enabled": True}),
                        ("version bootstrap denied", "version", "8.13"),
                        ("host online setting denied", "offline", {"enabled": False}),
                    ]:
                        unavailable_configuration = json.loads(json.dumps(configuration))
                        unavailable_configuration["java"]["import"]["gradle"][field] = value
                        client.notify("workspace/didChangeConfiguration", {"settings": unavailable_configuration})
                        try:
                            error = expect_failure(client, phase(), "Selected Gradle import failed or may be partial", str(root))
                            journey["commands"].append({"label": case_label, "error": error})
                            receipt["checks"].append(case_label + ": sanitized")
                        finally:
                            client.notify("workspace/didChangeConfiguration", {"settings": configuration})
                    build_file = primary / "build.gradle"
                    original_build = build_file.read_bytes()
                    try:
                        build_file.unlink()
                        build_file.symlink_to(unrelated / "build.gradle")
                        expect_failure(client, phase(), "Selected Gradle import failed or may be partial", str(root))
                        receipt["checks"].append("foreign build-input symlink denied despite valid settings")
                    finally:
                        build_file.unlink()
                        build_file.write_bytes(original_build)
                    reply = client.request("workspace/executeCommand", phase(), timeout=90)
                    from urllib.parse import urlparse, unquote
                    if (reply.get("version") != 1 or reply.get("selectedImportPhaseComplete") is not True
                            or reply.get("fullSynchronizationKnown") is not False
                            or Path(unquote(urlparse(reply.get("rootUri", "")).path)).resolve() != primary):
                        raise AssertionError("Guarded Gradle phase receipt differs")
                    journey["commands"].append({"label": "guarded chosen build import", "reply": reply})
                    receipt["checks"].append("guarded phase acknowledges Buildship completion only")
                    await_membership(client, primary_members, journey, "guarded phase imports chosen composite only")
                    reply = client.request("workspace/executeCommand", phase(unrelated.as_uri()), timeout=90)
                    journey["commands"].append({"label": "guarded unrelated root import", "reply": reply})
                    await_membership(client, all_members, journey, "guarded unrelated import preserves composite")

                    settings_file = primary / "settings.gradle"
                    original_settings = settings_file.read_bytes()
                    try:
                        settings_file.write_bytes(original_settings.replace(b"include 'application', 'library'", b"include 'application'"))
                        error = expect_failure(client, phase(), "Selected Gradle import failed or may be partial", str(root))
                        journey["commands"].append({"label": "guarded missing required module", "error": error})
                        receipt["checks"].append("guarded missing required module fails sanitized")
                    finally:
                        settings_file.write_bytes(original_settings)
                    client.request("workspace/executeCommand", phase(), timeout=90)
                    await_membership(client, all_members, journey, "guarded phase recovers after repaired declaration")

                    build_file = primary / "build.gradle"
                    original_build = build_file.read_bytes()
                    try:
                        build_file.write_text("plugins { id 'evidence.missing.privatecanary' version '1.0' }\n")
                        error = expect_failure(client, phase(), "Selected Gradle import failed or may be partial", "privatecanary")
                        if str(root) in error:
                            raise AssertionError("Missing cached plugin failure leaked the fixture path")
                        journey["commands"].append({"label": "missing cached plugin under denied network", "error": error})
                        receipt["checks"].append("missing cached plugin under denied network fails sanitized")
                    finally:
                        build_file.write_bytes(original_build)
                    client.request("workspace/executeCommand", phase(), timeout=90)
                    await_membership(client, all_members, journey, "guarded phase recovers after missing cached plugin")
                    started, release = primary / "phase-started", primary / "phase-release"
                    result = {}
                    worker = None
                    try:
                        build_file.write_bytes(original_build + b'''\nfile('phase-started').text = 'started'\ndef deadline = System.nanoTime() + 45000000000L\nwhile (!file('phase-release').exists() && System.nanoTime() < deadline) { Thread.sleep(50) }\nif (!file('phase-release').exists()) { throw new GradleException('isolated phase gate expired') }\n''')
                        request_id = client.next_id
                        def held_phase():
                            try:
                                result["reply"] = client.request("workspace/executeCommand", phase(), timeout=90)
                            except Exception as failure:
                                result["error"] = str(failure)
                        worker = threading.Thread(target=held_phase)
                        worker.start()
                        deadline = time.monotonic() + 60
                        while not started.is_file() and not result and time.monotonic() < deadline:
                            time.sleep(0.1)
                        if not started.is_file() or result:
                            raise AssertionError("Guarded phase did not remain held until Buildship configuration finished")
                        receipt["checks"].append("guarded phase does not acknowledge held configuration early")
                        client.notify("$/cancelRequest", {"id": request_id})
                        time.sleep(0.5)
                        release.write_text("release")
                        worker.join(timeout=90)
                        if worker.is_alive():
                            raise AssertionError("Guarded cancellation did not settle after disposable gate release")
                        if "cancel" not in result.get("error", "").lower() or str(root) in result.get("error", ""):
                            raise AssertionError("Guarded phase cancellation was not an actionable sanitized error")
                        journey["cancellation"] = {"requestId": request_id, "requestHeldUntilCancel": True,
                                                   "result": result, "gateExplicitlyReleased": True,
                                                   "cooperativeCancellationOnly": True, "rollbackKnown": False}
                        receipt["checks"].append("guarded held request cancellation rejects phase success")
                    finally:
                        release.write_text("release")
                        if worker is not None and worker.is_alive():
                            worker.join(timeout=90)
                            if worker.is_alive():
                                raise AssertionError("Isolated held phase worker did not stop")
                        build_file.write_bytes(original_build)
                        for path in [started, release]:
                            if path.is_file():
                                path.unlink()
                    client.request("workspace/executeCommand", phase(), timeout=90)
                    await_membership(client, all_members, journey, "guarded phase recovers after cancellation")
                    preserve_sources("guarded phase temporary settings restored")
                if actions:
                    request = {"command": "java.project.changeImportedProjects", "arguments": [[], [], [(root / member).as_uri() for member in sorted(primary_members)]]}
                    if arguments.guarded_unlink_only:
                        unlink_request = {"rootUri": primary.as_uri(), "projectUris": [(root / member).as_uri() for member in sorted(primary_members)]}
                        selected_uri = (primary / "application").as_uri()
                        escaped_module = primary / "escaped-unlink"
                        escaped_module.symlink_to(unrelated, target_is_directory=True)
                        invalid_requests = [
                            ("empty unlink selection", {**unlink_request, "projectUris": []}),
                            ("duplicate unlink identity", {**unlink_request, "projectUris": [selected_uri] * 2}),
                            ("unlink count bound", {**unlink_request, "projectUris": [selected_uri] * 129}),
                            ("foreign unlink identity", {**unlink_request, "projectUris": [unrelated.as_uri()]}),
                            ("escaping symlink unlink identity", {**unlink_request, "projectUris": [escaped_module.as_uri()]}),
                            ("uninitialized unlink root", {**unlink_request, "rootUri": root.as_uri()}),
                            ("missing unlink identity", {**unlink_request, "projectUris": [(primary / "private-canary").as_uri()]}),
                            ("nested source is not module identity", {**unlink_request, "projectUris": [(primary / "application/src").as_uri()]}),
                            ("nonfile unlink identity", {**unlink_request, "projectUris": ["https://private-canary.invalid/project"]}),
                            ("query unlink identity", {**unlink_request, "projectUris": [selected_uri + "?private-canary=1"]}),
                            ("oversized unlink identity", {**unlink_request, "projectUris": ["file:///" + "x" * 4097]}),
                            ("unknown unlink override", {**unlink_request, "deleteContent": True}),
                            ("later missing identity prevents earlier removal", {**unlink_request, "projectUris": [selected_uri, (primary / "private-canary").as_uri()]}),
                        ]
                        for rejection, invalid in invalid_requests:
                            try:
                                client.request("workspace/executeCommand", {"command": "groovy.gradle.unlinkSelected", "arguments": [invalid]}, timeout=30)
                            except RuntimeError as error:
                                if "Gradle unlink" not in str(error) or "private-canary" in str(error) or str(root) in str(error):
                                    raise AssertionError("Unlink failure is not sanitized") from error
                            else:
                                raise AssertionError("Invalid unlink selection succeeded")
                            if membership(client) != all_members:
                                raise AssertionError("Rejected unlink mutated another project")
                            receipt["checks"].append(rejection + ": sanitized and membership unchanged")
                        escaped_module.unlink()
                        try:
                            client.request("workspace/executeCommand", {"command": "groovy.maven.unlinkSelected", "arguments": [unlink_request]}, timeout=30)
                        except RuntimeError as error:
                            if "Maven unlink" not in str(error) or str(root) in str(error):
                                raise AssertionError("Wrong-provider rejection is not sanitized") from error
                        else:
                            raise AssertionError("Maven unlink accepted Gradle identities")
                        receipt["checks"].append("wrong-provider unlink rejected without mutation")
                        diagnostic_uri = qualifiers.qualify_unlink_wait(client, primary, unlink_request, "groovy.gradle.unlinkSelected", receipt, journey)
                        if membership(client) != all_members:
                            raise AssertionError("Cancelled held unlink changed membership")
                        receipt["checks"].append("held cancellation preserves composite membership before repaired retry")
                        qualifiers.qualify_unlink_wait(client, primary, unlink_request, "groovy.gradle.unlinkSelected", receipt, journey, replace_root=True)
                        if membership(client) != all_members:
                            raise AssertionError("Replaced-root rejection changed membership")
                        preserve_sources("replaced-root failure restores original Gradle source identities")
                        request = {"command": "groovy.gradle.unlinkSelected", "arguments": [unlink_request]}
                    reply = client.request("workspace/executeCommand", request, timeout=30)
                    if arguments.guarded_unlink_only:
                        expected_keys = {"version", "rootUri", "unlinkedProjectUris", "selectedUnlinkPhaseComplete", "fullSynchronizationKnown"}
                        if set(reply) != expected_keys or reply["version"] != 1 or reply["selectedUnlinkPhaseComplete"] is not True or reply["fullSynchronizationKnown"] is not False:
                            raise AssertionError("Unlink phase receipt differs or overclaims synchronization")
                        from urllib.parse import urlparse, unquote
                        if Path(unquote(urlparse(reply["rootUri"]).path)).resolve() != primary or {Path(unquote(urlparse(uri).path)).resolve() for uri in reply["unlinkedProjectUris"]} != {root / member for member in primary_members}:
                            raise AssertionError("Unlink phase receipt identifies another project")
                        receipt["checks"].append("exact selected composite unlink phase receipt keeps full synchronization unknown")
                        if membership(client) != {"unrelated"}:
                            raise AssertionError("Unlink replied before membership removal")
                        with client.notifications_lock:
                            deletions = [message for message in client.notifications if message.get("method") == "language/eventNotification" and message.get("params", {}).get("eventType") == 210]
                        if not any({Path(unquote(urlparse(uri).path)).resolve() for uri in event.get("params", {}).get("data", [])} == {root / member for member in primary_members} for event in deletions):
                            raise AssertionError("Unlink did not emit exact ProjectsDeleted identities")
                        receipt["checks"].append("request-bound composite removal and exact ProjectsDeleted event")
                        with client.notifications_lock:
                            cleared = [message for message in client.notifications if message.get("method") == "textDocument/publishDiagnostics" and message.get("params", {}).get("diagnostics") == []]
                        expected_diagnostic = Path(unquote(urlparse(diagnostic_uri).path)).resolve()
                        if not any(Path(unquote(urlparse(message.get("params", {}).get("uri", "")).path)).resolve() == expected_diagnostic for message in cleared):
                            raise AssertionError("Unlink did not clear its imported resource marker diagnostics")
                        receipt["checks"].append("imported resource marker diagnostics cleared on exact selected source URI")
                    journey["commands"].append({"request": request, "reply": reply})
                    await_membership(client, {"unrelated"}, journey, "selected build unlink preserves unrelated root")
                    preserve_sources("selected unlink preserves all source/build and generated sentinel hashes")
                    request = {"command": "java.project.changeImportedProjects", "arguments": [[(primary / "settings.gradle").as_uri()], [], []]}
                    reply = client.request("workspace/executeCommand", request, timeout=20)
                    journey["commands"].append({"request": request, "reply": reply})
                    await_membership(client, all_members, journey, "selected Gradle settings link restores composite without dropping unrelated root")
                if failure_actions:
                    def mutate(arguments, name):
                        request_id = client.next_id
                        request = {"command": "java.project.changeImportedProjects", "arguments": arguments}
                        try:
                            reply = client.request("workspace/executeCommand", request, timeout=20)
                        except RuntimeError as error:
                            journey["commands"].append({"label": name, "request": request, "error": str(error), "privateCanaryInError": "foreign-private-canary" in str(error)})
                            return request_id, "error"
                        journey["commands"].append({"label": name, "request": request, "reply": reply})
                        return request_id, reply

                    invalid_requests = [
                        ("malformed outer argument", [17, [], []]),
                        ("missing mutation arguments", [[]]),
                        ("non-file import identity", [["https://invalid.example/foreign-private-canary.gradle"], [], []]),
                        ("missing selected build input", [[(primary / "missing/build.gradle").as_uri()], [], []]),
                        ("non-build input", [[(primary / "src/main/java/evidence/Main.java").as_uri()], [], []]),
                    ]
                    for name, mutation_arguments in invalid_requests:
                        mutate(mutation_arguments, name)
                        await_membership(client, all_members, journey, name + ": existing membership preserved")
                        preserve_sources(name + ": source/build hashes preserved")
                    foreign = root / "foreign-private-canary"
                    foreign.mkdir()
                    (foreign / "settings.gradle").write_text("rootProject.name = 'uninitialized-foreign'\n")
                    (foreign / "build.gradle").write_text("plugins { id 'java' }\nfile('foreign-executed').text = 'executed'\n")
                    try:
                        mutate([[(foreign / "settings.gradle").as_uri()], [], []], "uninitialized foreign build")
                        await_membership(client, all_members, journey, "foreign mutation: initialized membership preserved")
                        preserve_sources("foreign mutation: original sources preserved")
                        journey["foreignBuildExecuted"] = (foreign / "foreign-executed").exists()
                        if journey["foreignBuildExecuted"]:
                            raise AssertionError("Uninitialized foreign build script executed despite preserved membership")
                        receipt["checks"].append("foreign mutation did not execute the uninitialized build script")
                    finally:
                        for path in sorted(foreign.rglob("*"), reverse=True):
                            if path.is_file():
                                path.unlink()
                            elif path.is_dir():
                                path.rmdir()
                        foreign.rmdir()

                    removed = [(root / member).as_uri() for member in sorted(primary_members)]
                    mutate([[], [], removed], "unlink before held import")
                    await_membership(client, {"unrelated"}, journey, "held import starts from unlinked chosen build")
                    build_file = primary / "build.gradle"
                    original_build = build_file.read_bytes()
                    release = primary / "cancel-release"
                    started = primary / "cancel-started"
                    finished = primary / "cancel-finished"
                    try:
                        build_file.write_bytes(original_build + b'''\nfile('cancel-started').text = 'started'\ndef deadline = System.nanoTime() + 45000000000L\nwhile (!file('cancel-release').exists() && System.nanoTime() < deadline) { Thread.sleep(50) }\nif (!file('cancel-release').exists()) { throw new GradleException('isolated cancellation gate expired') }\nfile('cancel-finished').text = 'finished'\n''')
                        request_id, reply = mutate([[(primary / "settings.gradle").as_uri()], [], []], "held selected build import")
                        if reply is not None:
                            raise AssertionError("Upstream selected Gradle command no longer returns the expected null acknowledgement")
                        deadline = time.monotonic() + 60
                        while not started.is_file() and time.monotonic() < deadline:
                            try:
                                membership(client)
                            except TimeoutError:
                                continue
                            time.sleep(0.1)
                        if not started.is_file():
                            raise AssertionError("Held import did not execute the disposable build gate")
                        if finished.exists():
                            raise AssertionError("Held import completed before explicit release")
                        client.notify("$/cancelRequest", {"id": request_id})
                        journey["cancellation"] = {"requestId": request_id, "requestAcknowledgedBeforeBuildGate": True,
                            "lateRequestCancellationSent": True, "fullSynchronizationKnown": False}
                        release.write_text("release")
                        await_membership(client, all_members, journey, "late request cancellation does not cancel the acknowledged workspace job")
                        if not finished.is_file():
                            raise AssertionError("Import membership was admitted without completing the held build gate")
                        receipt["checks"].append("workspace job completed after its acknowledged request was cancelled")
                        journey["cancellation"]["workspaceJobContinued"] = True
                    finally:
                        release.write_text("release")
                        build_file.write_bytes(original_build)
                        for path in [release, started, finished]:
                            if path.is_file():
                                path.unlink()
                    preserve_sources("held cancellation fixture restored")

                    settings_file = primary / "settings.gradle"
                    original_settings = settings_file.read_bytes()
                    try:
                        settings_file.write_bytes(original_settings.replace(b"include 'application', 'library'", b"include 'application'"))
                        failure = expect_failure(client, {
                            "command": "groovy.gradle.projects",
                            "arguments": [{"rootUri": primary.as_uri(), "offline": True}],
                        }, "Gradle model discovery failed", str(root))
                        journey["missingRequiredModuleFailure"] = failure
                        receipt["checks"].append("missing required Gradle project reports sanitized model failure")
                    finally:
                        settings_file.write_bytes(original_settings)
                    recovered = discover(client, primary)
                    (evidence / "repaired-required-project-model.json").write_text(json.dumps(recovered, indent=2) + "\n")
                    from urllib.parse import urlparse, unquote
                    paths = {project["projectPath"] for project in recovered["projects"]
                             if Path(unquote(urlparse(project["buildRootUri"]).path)).resolve() == primary}
                    if paths != {":", ":application", ":library"}:
                        raise AssertionError("Repaired Gradle model did not restore the required project identities")
                    receipt["checks"].append("same provider recovers required-project metadata after repair")
                    preserve_sources("required-project failure fixture restored")

                preserve_sources(f"{label}: sources unchanged and no test reports")
                journey["passed"] = True
            finally:
                logs = evidence / f"gradle-selection-{label}-{time.time_ns()}"
                logs.mkdir()
                with client.notifications_lock:
                    journey["eventNotifications"] = [message for message in client.notifications if message.get("method") == "language/eventNotification"]
                try:
                    client.stderr.seek(0)
                    (logs / "stderr-before-shutdown.log").write_bytes(client.stderr.read())
                finally:
                    client.close()
                provider_log = data / ".metadata/.log"
                if provider_log.is_file():
                    shutil.copy2(provider_log, logs / "jdtls.log")
                journey["evidence"] = str(logs)
                arguments.receipt.write_text(json.dumps(receipt, indent=2) + "\n")

        try:
            all_data = Path(directory) / "all-profile"
            if arguments.resolution_graph_only:
                receipt["limits"].append("Explicit offline component metadata resolution only; no artifact readiness/origin/product graph/remote/native qualification")
                run("graph", all_data, all_members)
                run("graph-restart", all_data, all_members, required_restart=True)
                return
            if arguments.source_outputs_only:
                run("outputs", all_data, all_members)
                run("outputs-restart", all_data, all_members, required_restart=True)
                return
            if arguments.required_library_only:
                specification = importlib.util.spec_from_file_location("groovy_required_library_network", REPOSITORY / "script/test-groovy-jdtls-provider.py")
                if specification is None or specification.loader is None:
                    raise RuntimeError("Cannot load existing denied-network fixture owner")
                boundary = importlib.util.module_from_spec(specification)
                specification.loader.exec_module(boundary)
                boundary.verify_gradle_network_boundary()
                environment = dict(os.environ, JAVA_HOME=str(arguments.gradle_java_home.resolve()), GRADLE_USER_HOME=str(home))
                environment["JAVA_TOOL_OPTIONS"] = (environment.get("JAVA_TOOL_OPTIONS", "") + " -Djava.net.preferIPv4Stack=true").strip()
                command = ["/usr/bin/sandbox-exec", "-p", boundary.GRADLE_DENIED_NETWORK_PROFILE,
                    str(arguments.gradle_home.resolve() / "bin/gradle"), "--offline", "--no-daemon", "--gradle-user-home", str(home), ":library:jar"]
                built = subprocess.run(command, cwd=primary, env=environment, capture_output=True, text=True, timeout=120)
                (evidence / "required-library-preparation.log").write_text(built.stdout + built.stderr)
                receipt["preparation"] = {"command": command, "exitCode": built.returncode, "productTaskEvidence": False}
                if built.returncode or not artifact.is_file():
                    raise AssertionError("Owned offline required-library output preparation failed")
                paths = [artifact, *sorted((primary / "library/build/classes/java/main").rglob("*.class"))]
                if len(paths) > 32 or not any(path.name == "Required.class" for path in paths):
                    raise AssertionError("Prepared library lacks the required bounded class output")
                import zipfile
                with zipfile.ZipFile(artifact) as archive:
                    if archive.read("library/Required.class") != (primary / "library/build/classes/java/main/library/Required.class").read_bytes():
                        raise AssertionError("Prepared library JAR differs from its actual compiled required class")
                shutil.copy2(artifact, evidence / "prepared-library.jar")
                prepared_outputs.update({path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
                receipt["preparedOutputSha256"] = {str(path.relative_to(root)): digest for path, digest in prepared_outputs.items()}
                backups = {path: path.read_bytes() for path in prepared_outputs}
                try:
                    for cached in [True, False]:
                        if not cached:
                            for path, digest in prepared_outputs.items():
                                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                                    raise AssertionError("Owned output changed before missing-output cell")
                                path.unlink()
                        if arguments.provider_fresh_module_states_only:
                            namespace = primary / "groovy-eclipse-name-reservation"
                            if namespace.exists():
                                if namespace.is_symlink() or (namespace / "evidence-preserve.txt").read_bytes() != b"unrelated namespace source\n":
                                    raise AssertionError("Owned namespace cleanup found changed source or an alias")
                                entries = list(namespace.rglob("*"))
                                if len(entries) > 128:
                                    raise AssertionError("Owned namespace metadata exceeds the cleanup audit bound")
                                audit = evidence / ("namespace-owned-cleanup-" + ("cached" if cached else "missing"))
                                audit.mkdir()
                                records = []
                                total_bytes = 0
                                for entry in entries:
                                    if entry.is_symlink() or entry.relative_to(root).as_posix() in originals:
                                        raise AssertionError("Owned namespace cleanup would cross a source/alias boundary")
                                    if entry.is_dir(): continue
                                    if not entry.is_file() or entry.stat().st_size > 1024 * 1024:
                                        raise AssertionError("Owned namespace cleanup found an unsupported file")
                                    total_bytes += entry.stat().st_size
                                    if total_bytes > 4 * 1024 * 1024:
                                        raise AssertionError("Owned namespace cleanup exceeds its byte bound")
                                    destination = audit / entry.relative_to(namespace)
                                    destination.parent.mkdir(parents=True,exist_ok=True)
                                    shutil.copy2(entry,destination)
                                    records.append({"path":entry.relative_to(namespace).as_posix(),"sha256":hashlib.sha256(entry.read_bytes()).hexdigest()})
                                receipt.setdefault("ownedNamespaceCleanup",[]).append({"afterCell":"cached" if not cached else "initial","files":records,"originalFixtureSourceDeleted":False})
                                shutil.rmtree(namespace)
                            persistent_namespace.clear()
                        if arguments.fresh_closed_modules_only:
                            for module in [primary,primary / "application",primary / "library",primary / "included",unrelated]:
                                for metadata in [module / ".project",module / ".classpath",*list((module / ".settings").glob("org.eclipse.*"))]:
                                    if metadata.exists():
                                        if metadata.relative_to(root).as_posix() in originals or not metadata.is_file() or metadata.is_symlink():
                                            raise AssertionError("Fresh-cell cleanup found non-disposable source metadata")
                                        metadata.unlink()
                        cell = "cached" if cached else "missing"
                        data = Path(directory) / ("required-" + cell + "-profile")
                        run("required-" + cell, data, all_members, required_cached=cached)
                        run("required-" + cell + "-restart", data, all_members - {"primary/library"} if arguments.closed_modules_only else (all_members - {"primary/library"}, all_members), required_cached=cached, required_restart=True)
                finally:
                    for path, content in backups.items():
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(content)
                return
            if arguments.resource_membership_only:
                run("resources", all_data, all_members)
                run("resources-restart", all_data, all_members)
                return
            if arguments.guarded_phase_only:
                run("guarded-phase", all_data, set(), guarded_actions=True)
                run("guarded-phase-restart", all_data, all_members)
                return
            run("all", all_data, all_members, actions=True, failure_actions=arguments.mutation_failures_only)
            run("all-restart", all_data, all_members)
            if arguments.guarded_unlink_only:
                return
            if arguments.mutation_failures_only:
                settings_file = primary / "settings.gradle"
                application_build = primary / "application/build.gradle"
                original_settings = settings_file.read_bytes()
                original_application_build = application_build.read_bytes()
                baseline_originals = dict(originals)
                try:
                    settings_file.write_bytes(original_settings.replace(b"include 'application', 'library'", b"include 'application'"))
                    application_build.write_bytes(original_application_build.replace(b"dependencies { implementation project(':library') }\n", b""))
                    changed = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in [settings_file, application_build]}
                    originals.update(changed)
                    without_library = all_members - {"primary/library"}
                    run("orphan-existing-profile", all_data, (all_members, without_library))
                    run("orphan-fresh-profile", Path(directory) / "orphan-fresh-profile", without_library)
                finally:
                    settings_file.write_bytes(original_settings)
                    application_build.write_bytes(original_application_build)
                    originals.update(baseline_originals)
                run("orphan-declaration-restored", all_data, all_members)
            selected_data = Path(directory) / "selected-profile"
            run("selected", selected_data, {"unrelated"}, configurations=[unrelated / "build.gradle"])
            run("selected-restart", selected_data, {"unrelated"}, configurations=[unrelated / "build.gradle"])
            excluded_data = Path(directory) / "excluded-profile"
            run("excluded-root", excluded_data, {"unrelated"}, exclusions=[str(primary)])
            run("excluded-root-restart", excluded_data, {"unrelated"}, exclusions=[str(primary)])
            run("excluded-required-module", Path(directory) / "excluded-required-profile", all_members,
                exclusions=[str(primary / "library")])
        finally:
            environment = dict(os.environ, JAVA_HOME=str(arguments.gradle_java_home.resolve()), GRADLE_USER_HOME=str(home))
            stopped = subprocess.run([str(arguments.gradle_home.resolve() / "bin/gradle"), "--offline", "--stop", "--gradle-user-home", str(home)],
                                     cwd=root, env=environment, capture_output=True, text=True)
            (evidence / "owned-gradle-stop.log").write_text(stopped.stdout + stopped.stderr)
            receipt["owned_worker_stop_exit"] = stopped.returncode
            arguments.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
            if stopped.returncode:
                raise RuntimeError("Isolated owned Gradle worker shutdown failed")
    print(json.dumps(receipt, indent=2))


def probe_startup_controls(arguments):
    provider = arguments.provider.resolve()
    if hashlib.sha256(provider.read_bytes()).hexdigest() != arguments.sha256:
        raise AssertionError("Provider binary differs from the supplied source build")
    evidence = arguments.receipt.parent
    evidence.mkdir(parents=True, exist_ok=True)
    receipt = {"provider_sha256": arguments.sha256, "checks": [], "journeys": [], "limits": [
        "Headless pinned startup import controls only; not product/native/remote/platform qualification",
        "Suppression is not unlinking or persisted selection reconciliation",
        "Phase completion and observed membership do not establish full synchronization",
        "Required modules, ignore patterns, orphan handling, Tasks/Tests/Debug remain unqualified",
    ]}
    modules = {}
    for name, script in [("inputs", "build-groovy-gradle-provider.py"),
                         ("resources", "test-groovy-maven-model-provider.py")]:
        specification = importlib.util.spec_from_file_location(name, REPOSITORY / "script" / script)
        if specification is None or specification.loader is None:
            raise RuntimeError("Cannot load existing startup qualification owners")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        modules[name] = module
    receipt["pinned_input_sha256"] = {}
    artifacts = [(arguments.launcher.resolve().parent.parent / "plugins" / name, digest)
                 for name, digest in modules["inputs"].PINNED_ARTIFACTS.items()]
    artifacts.extend((arguments.gradle_home.resolve() / name, digest)
                     for name, digest in modules["inputs"].PINNED_GRADLE_ARTIFACTS.items())
    for artifact, expected in artifacts:
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if digest != expected:
            raise AssertionError("Pinned startup provider artifact differs")
        receipt["pinned_input_sha256"][artifact.name] = digest
    with tempfile.TemporaryDirectory(prefix="groovy-mixed-startup-") as directory:
        workspace = Path(directory).resolve()
        fixture_bundle = None
        if arguments.provider_identity_only or arguments.unlink_ownership_only or arguments.root_state_only or arguments.workspace_scoped_import_only:
            fixture_bundle, fixture_source = modules["resources"].unlink_fixture_bundle(arguments, directory, identity_modes=arguments.provider_identity_only or arguments.unlink_ownership_only, ownership_modes=arguments.unlink_ownership_only, workspace_states=arguments.root_state_only or arguments.workspace_scoped_import_only, scoped_import_modes=arguments.workspace_scoped_import_only)
            receipt["testOnlyFixtureSha256"] = hashlib.sha256(fixture_bundle.read_bytes()).hexdigest()
            receipt["testOnlyFixtureSourceSha256"] = hashlib.sha256(fixture_source.read_bytes()).hexdigest()
            shutil.copy2(fixture_source, evidence / fixture_source.name)
        root = workspace / "projects"
        maven, gradle = root / "maven", root / "gradle"
        maven.mkdir(parents=True)
        gradle.mkdir()
        (maven / "pom.xml").write_text('<project xmlns="http://maven.apache.org/POM/4.0.0"><modelVersion>4.0.0</modelVersion><groupId>evidence</groupId><artifactId>mixed-reactor</artifactId><version>1.0</version><packaging>pom</packaging><modules><module>application</module><module>library</module></modules></project>\n')
        (gradle / "settings.gradle").write_text("rootProject.name = 'mixed-gradle'\ninclude 'application', 'library'\n")
        (gradle / "build.gradle").write_text("plugins { id 'base' }\n")
        for provider_name, build in [("maven", maven), ("gradle", gradle)]:
            for name in ["application", "library"]:
                module = build / name
                source = module / "src/main/java/evidence/Main.java"
                source.parent.mkdir(parents=True)
                source.write_text("package evidence; public class Main {}\n")
                if provider_name == "maven":
                    (module / "pom.xml").write_text(f'<project xmlns="http://maven.apache.org/POM/4.0.0"><modelVersion>4.0.0</modelVersion><parent><groupId>evidence</groupId><artifactId>mixed-reactor</artifactId><version>1.0</version><relativePath>../pom.xml</relativePath></parent><artifactId>mixed-{name}</artifactId><properties><maven.compiler.release>17</maven.compiler.release></properties><build><plugins><plugin><groupId>org.apache.maven.plugins</groupId><artifactId>maven-compiler-plugin</artifactId><version>3.14.1</version></plugin></plugins></build></project>\n')
                else:
                    (module / "build.gradle").write_text("plugins { id 'java' }\n")
                sentinel = module / ("target" if provider_name == "maven" else "build") / "generated-sources/startup/Sentinel.java"
                sentinel.parent.mkdir(parents=True)
                sentinel.write_text("package evidence; public class Sentinel {}\n")
        if arguments.provider_root_state_only:
            (gradle / "build.gradle").write_text("plugins { id 'java' }\ntasks.register('unrequestedRootDefault') { doLast { file('unrequested-root-task.marker').text = 'ran' } }\ndefaultTasks 'unrequestedRootDefault'\n")
            source = gradle / "src/main/java/evidence/Main.java"
            source.parent.mkdir(parents=True)
            source.write_text("package evidence; public class Main {}\n")
        if arguments.startup_configurations_only or arguments.workspace_scoped_import_only:
            build_file = gradle / "build.gradle"
            build_file.write_text(build_file.read_text() + "tasks.register('unrequestedRootDefault') { doLast { file('unrequested-root-task.marker').text = 'ran' } }\ndefaultTasks 'unrequestedRootDefault'\n")
        if arguments.single_provider_configurations_only or arguments.workspace_scoped_root_only:
            plain = root / "plain"
            (plain / "src/evidence").mkdir(parents=True)
            (plain / "src/evidence/Plain.java").write_text("package evidence; public class Plain {}\n")
            (plain / ".project").write_text(
                '<projectDescription><name>startup-plain-java</name><comment></comment>'
                '<projects></projects><buildSpec></buildSpec><natures>'
                '<nature>org.eclipse.jdt.core.javanature</nature></natures></projectDescription>\n'
            )
            (plain / ".classpath").write_text(
                '<classpath><classpathentry kind="src" path="src"/>'
                '<classpathentry kind="con" path="org.eclipse.jdt.launching.JRE_CONTAINER"/>'
                '<classpathentry kind="output" path="bin"/></classpath>\n'
            )
        if arguments.workspace_scoped_details_only:
            with (gradle / "build.gradle").open("a") as build:
                build.write("\nclass ScopedOptionProbe extends DefaultTask { @org.gradle.api.tasks.options.Option(option='scope-mode', description='Mode') void setMode(String value) {} }\ntasks.register('scopedOptionProbe', ScopedOptionProbe)\n")
        originals = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in root.rglob("*") if path.is_file()}
        receipt["fixture_sha256"] = originals
        template = workspace / "pristine-fixtures"
        shutil.copytree(root, template)
        fixtures_by_profile = {}
        repository = workspace / "repository"
        shutil.copytree(arguments.maven_repository.resolve(), repository, copy_function=modules["resources"].cache_copy,
                        ignore=shutil.ignore_patterns("*.lock"))
        from xml.sax.saxutils import escape
        settings_file = workspace / "settings.xml"
        settings_file.write_text(f'<settings><localRepository>{escape(str(repository))}</localRepository><offline>true</offline></settings>\n')
        home = workspace / "owned-gradle-home"
        home.mkdir()
        configured_gradle_home = arguments.gradle_home.resolve()
        configured_gradle_java_home = arguments.gradle_java_home.resolve()
        if arguments.workspace_scoped_details_only:
            # Settings notifications also schedule unrelated SDK synchronization jobs.
            # Remove only owned aliases to isolate backend missing-tool failures.
            configured_gradle_home = workspace / "scoped-model-gradle-home"
            configured_gradle_java_home = workspace / "scoped-model-java-home"
            configured_gradle_home.symlink_to(arguments.gradle_home.resolve(), target_is_directory=True)
            configured_gradle_java_home.symlink_to(arguments.gradle_java_home.resolve(), target_is_directory=True)
            receipt["scopedDetailsToolFaultIsolation"] = "owned aliases outside initialized fixtures; real SDK targets unchanged"
        settings = {"java": {
            "autobuild": {"enabled": False},
            "configuration": {"maven": {"userSettings": str(settings_file)}, "updateBuildConfiguration": "disabled"},
            "import": {"maven": {"offline": {"enabled": True}}, "gradle": {
                "offline": {"enabled": True}, "wrapper": {"enabled": False}, "version": None,
                "home": str(configured_gradle_home), "java": {"home": str(configured_gradle_java_home)},
                "user": {"home": str(home)}, "arguments": ["--offline"],
            }},
        }}
        maven_java = {"maven/application", "maven/library"}
        gradle_java = {"gradle/application", "gradle/library"}
        if arguments.provider_root_state_only:
            gradle_java.add("gradle")
        maven_resources, gradle_resources = maven_java | {"maven"}, gradle_java | {"gradle"}

        def preserve_sources(label):
            for relative, digest in originals.items():
                path = root / relative
                if (arguments.single_provider_configurations_only or arguments.workspace_scoped_root_only) and relative in {"plain/.project", "plain/.classpath"}:
                    import re
                    import xml.etree.ElementTree as xml

                    expected = (template / relative).read_bytes()
                    actual = path.read_bytes() if path.is_file() else b""
                    receipt.setdefault("plainMetadata", []).append({
                        "label": label, "relativePath": relative,
                        "expectedSha256": digest, "actualSha256": hashlib.sha256(actual).hexdigest(),
                        "actualBytes": len(actual),
                        "actualText": actual.decode("utf-8", errors="replace") if len(actual) <= 65536 else None,
                    })
                    if len(actual) > 65536:
                        raise AssertionError("Plain Java SDK metadata exceeds the fixture bound")

                    def semantic(element):
                        if element.tail and element.tail.strip():
                            raise AssertionError("Plain Java SDK metadata has unexpected text")
                        return (element.tag, sorted(element.attrib.items()), (element.text or "").strip(),
                                tuple(semantic(child) for child in element))

                    expected_xml, actual_xml = xml.fromstring(expected), xml.fromstring(actual)
                    if relative == "plain/.project":
                        filters = actual_xml.findall("filteredResources")
                        if len(filters) > 1:
                            raise AssertionError("Plain Java SDK metadata has duplicate resource filters")
                        for observed in filters:
                            identifier = observed.findtext("filter/id", "")
                            if not re.fullmatch(r"[0-9]{1,20}", identifier):
                                raise AssertionError("Plain Java SDK resource filter has an invalid identity")
                            # JDTLS adds its resource filter when importing an Eclipse project.
                            permitted = xml.fromstring(
                                '<filteredResources><filter><id>' + identifier + '</id><name></name><type>30</type>'
                                '<matcher><id>org.eclipse.core.resources.regexFilterMatcher</id>'
                                r'<arguments>node_modules|\.git|__CREATED_BY_JAVA_LANGUAGE_SERVER__</arguments>'
                                '</matcher></filter></filteredResources>'
                            )
                            if semantic(observed) != semantic(permitted):
                                raise AssertionError("Plain Java SDK resource filter differs from the observed SDK contract")
                            actual_xml.remove(observed)
                    if semantic(actual_xml) != semantic(expected_xml):
                        raise AssertionError("Plain Java SDK metadata changed beyond formatting and its managed filter")
                    continue
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    if arguments.single_provider_configurations_only:
                        expected = (template / relative).read_bytes()
                        actual = path.read_bytes() if path.is_file() else None
                        (evidence / "source-mismatch.json").write_text(json.dumps({
                            "label": label, "relativePath": relative,
                            "expectedSha256": digest,
                            "actualSha256": hashlib.sha256(actual).hexdigest() if actual is not None else None,
                            "expectedBytes": len(expected),
                            "actualBytes": len(actual) if actual is not None else None,
                            "expectedText": expected.decode("utf-8", errors="replace") if len(expected) <= 65536 else None,
                            "actualText": actual.decode("utf-8", errors="replace") if actual is not None and len(actual) <= 65536 else None,
                        }, indent=2) + "\n")
                    raise AssertionError("Startup controls changed or removed fixture source/build/generated files")
            if list(root.rglob("TEST-*.xml")):
                raise AssertionError("Startup controls emitted test reports")
            if (arguments.provider_root_state_only or arguments.startup_configurations_only or arguments.workspace_scoped_import_only) and (gradle / "unrequested-root-task.marker").exists():
                raise AssertionError("Root lifecycle executed an unrequested default Gradle task")
            receipt["checks"].append(label)

        def await_membership(client, expected_java, expected_resources, journey, label):
            from urllib.parse import urlparse, unquote
            deadline = time.monotonic() + 100
            stable_since = None
            previous = None
            samples = []
            while time.monotonic() < deadline:
                try:
                    resources = set(modules["resources"].imported_resource_membership(client, root))
                    java_uris = client.request("workspace/executeCommand", {"command": "java.project.getAll", "arguments": []}, timeout=30)
                    java = set()
                    for uri in java_uris or []:
                        parsed = urlparse(uri)
                        if parsed.scheme != "file" or parsed.netloc not in ("", "localhost") or parsed.query or parsed.fragment:
                            raise AssertionError("Java membership returned a nonlocal URI")
                        path = Path(unquote(parsed.path)).resolve(strict=True)
                        if not path.is_dir() or not path.is_relative_to(root):
                            raise AssertionError("Java membership escaped disposable fixtures")
                        java.add(path.relative_to(root).as_posix())
                except TimeoutError:
                    if client.process.poll() is not None:
                        raise AssertionError("Startup provider exited before membership was available")
                    continue
                samples.append({"java": sorted(java), "resources": sorted(resources)})
                current = (java, resources)
                matches = (java == expected_java and resources == expected_resources) if expected_java is not None else current == previous
                previous = current
                if matches:
                    if stable_since is None:
                        stable_since = time.monotonic()
                    if time.monotonic() - stable_since >= 5:
                        journey["states"].append({"label": label, "samples": samples, "minimumStableSeconds": 5})
                        receipt["checks"].append(label)
                        print(label, flush=True)
                        return
                else:
                    stable_since = None
                time.sleep(0.25)
            journey["states"].append({"label": label, "expectedJava": sorted(expected_java) if expected_java is not None else None, "expectedResources": sorted(expected_resources) if expected_resources is not None else None, "samples": samples})
            raise AssertionError("Mixed startup membership differs: " + label)

        def run(label, data, enabled, expected_java, expected_resources, repair=False, reuse_fixture=None, folders=None, linked_actions=False, identity_reads=False, ownership_actions=False, metadata_exclusion=False, metadata_actions=False, root_actions=False, configurations=None):
            nonlocal root, maven, gradle
            scoped_policy_cold = arguments.workspace_scoped_policy_only and label.endswith("restart")
            if arguments.workspace_scoped_root_only and label.endswith("restart"):
                expected_java = (maven_java if scoped_policy_cold else expected_java) | {"plain"}
                expected_resources = (maven_resources if scoped_policy_cold else expected_resources) | {"plain"}
            if data not in fixtures_by_profile:
                root = reuse_fixture or workspace / (label + "-fixtures")
                if reuse_fixture is None:
                    shutil.copytree(template, root)
                fixtures_by_profile[data] = root
            root = fixtures_by_profile[data]
            maven, gradle = root / "maven", root / "gradle"
            journey = {"label": label, "states": [], "commands": [], "passed": False,
                       "explicitEnabled": enabled, "globalProjectConfigurationsAbsent": configurations is None,
                       "fixtureRoot": str(root), "reusesImportedFiles": reuse_fixture is not None}
            receipt["journeys"].append(journey)
            configuration = json.loads(json.dumps(settings))
            for name, value in enabled.items():
                configuration["java"]["import"][name]["enabled"] = value
            if metadata_exclusion:
                configuration["java"]["import"]["exclusions"] = [str(gradle)]
                journey["sdkImportExclusions"] = configuration["java"]["import"]["exclusions"]
            initialized_folders = [root / relative for relative in (folders if folders is not None else ["maven", "gradle"])]
            journey["initializedFolderUris"] = [path.as_uri() for path in initialized_folders]
            options = {"bundles": [str(provider)] + ([str(fixture_bundle)] if fixture_bundle else []), "settings": configuration,
                       "workspaceFolders": [path.as_uri() for path in initialized_folders]}
            if configurations is not None:
                options["projectConfigurations"] = [path.as_uri() for path in configurations]
                journey["explicitProjectConfigurations"] = options["projectConfigurations"]
            client = client_class()(arguments.launcher.resolve(), arguments.java_home.resolve(), root, data,
                                    deny_external_network=True, gradle_user_home=home)
            try:
                initialized = client.request("initialize", {
                    "processId": os.getpid(), "rootUri": root.as_uri(),
                    "workspaceFolders": [{"uri": path.as_uri(), "name": path.name} for path in initialized_folders],
                    "capabilities": {"workspace": {"workspaceFolders": True, "configuration": True}},
                    "initializationOptions": options,
                }, timeout=90)
                journey["serverInfo"] = initialized.get("serverInfo")
                commands = initialized["capabilities"]["executeCommandProvider"]["commands"]
                if not {"java.project.getAll", "groovy.maven.importSelected", "groovy.gradle.importSelected"}.issubset(commands):
                    raise AssertionError("Pinned mixed provider lacks guarded import commands")
                if identity_reads and not {"groovy.maven.importedProjects", "groovy.gradle.importedProjects"}.issubset(commands):
                    raise AssertionError("Pinned mixed provider lacks imported build identity commands")
                receipt["checks"].append(label + ": advertised guarded provider commands")
                client.notify("initialized", {})
                await_membership(client, expected_java, expected_resources, journey, label + ": stable startup membership")
                if arguments.workspace_scoped_import_only and label.startswith("scoped-import-audit"):
                    if "groovy.gradle.importSelected.v2" not in commands:
                        raise AssertionError("Scoped Gradle import command is unavailable")

                    def host_state():
                        value = client.request("workspace/executeCommand", {
                            "command": "groovy.fixture.unlink.importHostState",
                            "arguments": [{"rootUri": root.as_uri()}],
                        }, timeout=30)
                        if set(value) != {"workspaceRootUris", "gradleArguments", "gradleJvmArguments", "offline",
                                          "wrapperEnabled", "gradleHome", "gradleJavaHome", "gradleVersion"}:
                            raise AssertionError("Scoped import host oracle schema differs")
                        if sorted(uri.rstrip("/") for uri in value["workspaceRootUris"]) != sorted([root.as_uri(), maven.as_uri()]):
                            raise AssertionError("Nested build became an initialized root or lost workspace authority")
                        return value

                    original_host = host_state()
                    scoped_maven_java = maven_java
                    scoped_maven_resources = maven_resources
                    if arguments.workspace_scoped_root_only:
                        if not {"groovy.gradle.rootState.v2", "groovy.gradle.applyRootState.v2"}.issubset(commands):
                            raise AssertionError("Scoped root commands are unavailable")
                        def resource_state(path, operation):
                            request = {"rootUri": root.as_uri(), "projectUri": path.as_uri(), "operation": operation}
                            reply = client.request("workspace/executeCommand", {"command": "groovy.fixture.unlink.projectState", "arguments": [request]}, timeout=30)
                            journey.setdefault("scopedRootResourceStates", []).append({"request": request, "reply": reply})
                            return reply
                        plain = root / "plain"
                        observed_plain = resource_state(plain, "recreateClosed")
                        if observed_plain.get("open") is False:
                            observed_plain = resource_state(plain, "open")
                        if observed_plain.get("open") is not True or observed_plain.get("exists") is not True:
                            raise AssertionError("Test-only plain Java resource setup failed")
                        scoped_maven_java = maven_java | {"plain"}
                        scoped_maven_resources = maven_resources | {"plain"}
                        await_membership(client, (expected_java - gradle_java) | {"plain"} | (gradle_java if label.endswith("restart") and not scoped_policy_cold else set()),
                            expected_resources | {"plain"}, journey, label + ": actual plain Java setup without workspace-folder changes")
                        receipt["checks"].append(label + ": plain Java is prepared through the existing test-only resource fixture")
                    maven_request = {"command": "groovy.maven.importSelected", "arguments": [{
                        "rootUri": maven.as_uri(), "offline": True,
                        "pomUris": [(maven / path).as_uri() for path in ["pom.xml", "application/pom.xml", "library/pom.xml"]],
                    }]}
                    maven_reply = client.request("workspace/executeCommand", maven_request, timeout=100)
                    if maven_reply.get("selectedImportPhaseComplete") is not True or maven_reply.get("fullSynchronizationKnown") is not False:
                        raise AssertionError("Scoped fixture Maven preservation setup lacks a phase receipt")
                    journey["commands"].append({"request": maven_request, "reply": maven_reply})
                    current_java = scoped_maven_java | (gradle_java if label.endswith("restart") and not scoped_policy_cold else set())
                    current_resources = scoped_maven_resources | (gradle_resources if label.endswith("restart") and not scoped_policy_cold else set())
                    await_membership(client, current_java, current_resources, journey, label + ": Maven sibling admitted")
                    if label.endswith("pristine"):
                        descriptions = [gradle / ".project", gradle / "application/.project", gradle / "library/.project"]
                        if any(path.exists() for path in descriptions):
                            raise AssertionError("Pristine scoped build unexpectedly has retained SDK descriptions")
                        journey["pristineGradleDescriptionsAbsent"] = True
                        receipt["checks"].append("pristine nested root and modules have no retained SDK descriptions before scoped import")
                    expect_failure(client, {"command": "groovy.gradle.importSelected", "arguments": [{
                        "rootUri": gradle.as_uri(), "offline": True,
                    }]}, "Selected Gradle import failed or may be partial", str(root))
                    receipt["checks"].append(label + ": legacy nested root remains unauthorized")
                    selection = {"workspaceRootUri": root.as_uri(), "buildRootUri": gradle.as_uri(), "offline": True}

                    def discover_scoped_defaults(stage):
                        if "groovy.gradle.defaultImportRoots" not in commands:
                            raise AssertionError("Cached Gradle default-root command is unavailable")
                        candidates = sorted(path for path in gradle.rglob("*") if path.is_file()
                            and path.name in {"settings.gradle", "settings.gradle.kts", "build.gradle", "build.gradle.kts"})
                        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in candidates}
                        request = {"command": "groovy.gradle.defaultImportRoots", "arguments": [{
                            "rootUri": root.as_uri(), "candidateBuildFileUris": [path.as_uri() for path in candidates], "cachedOnly": True}]}
                        reply = client.request("workspace/executeCommand", request, timeout=120)
                        journey["commands"].append({"stage": stage, "request": request, "reply": reply})
                        if (set(reply) != {"version", "rootUri", "buildRootUris", "cachedOnly", "readOnly", "partial", "fullSynchronizationKnown"}
                                or type(reply["version"]) is not int or reply["version"] != 1
                                or reply["rootUri"] != root.as_uri() + "/" or reply["buildRootUris"] != [gradle.as_uri() + "/"]
                                or reply["cachedOnly"] is not True or reply["readOnly"] is not True
                                or reply["partial"] is not False or reply["fullSynchronizationKnown"] is not False):
                            raise AssertionError("Authoritative default roots differ from the real nested build or overclaim synchronization")
                        if host_state() != original_host or any(hashlib.sha256(path.read_bytes()).hexdigest() != digest for path,digest in before.items()):
                            raise AssertionError("Cached default-root discovery changes host authority or build inputs")
                        receipt["checks"].append(label + ": " + stage + " authoritative cached default root and unchanged inputs/host")

                    if arguments.workspace_scoped_default_policy_only:
                        discover_scoped_defaults("before initial scoped policy")

                    def read_cached_associations(stage, extra_build=None):
                        command = "groovy.gradle.importedBuildRoots"
                        if command not in commands:
                            raise AssertionError("Cached Gradle build-root association command is unavailable")
                        before = set(modules["resources"].imported_resource_membership(client, root))
                        metadata = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*")
                            if path.is_file() and (path.name in {".project", ".classpath"} or ".settings" in path.parts)}
                        request = {"command": command, "arguments": [{"rootUri": root.as_uri(), "cachedOnly": True}]}
                        reply = client.request("workspace/executeCommand", request, timeout=60)
                        legacy = client.request("workspace/executeCommand", {"command": "groovy.gradle.importedProjects",
                            "arguments": request["arguments"]}, timeout=60)
                        records = reply.get("projects", [])
                        if (set(reply) != {"version", "rootUri", "projects", "cachedOnly", "readOnly", "partial", "fullSynchronizationKnown"}
                                or type(reply["version"]) is not int or reply["version"] != 1
                                or reply["rootUri"] != root.as_uri() + "/" or reply["cachedOnly"] is not True
                                or reply["readOnly"] is not True or reply["partial"] is not False
                                or reply["fullSynchronizationKnown"] is not False
                                or legacy["partial"] is not False or legacy["fullSynchronizationKnown"] is not False
                                or [{key: record[key] for key in ["projectUri", "javaNature"]} for record in records] != legacy["projects"]
                                or any(set(record) != {"projectUri", "buildRootUri", "javaNature"}
                                    or type(record["javaNature"]) is not bool
                                    or record["buildRootUri"] not in {path.as_uri() + "/" for path in ([gradle, extra_build] if extra_build else [gradle])}
                                    or not record["projectUri"].startswith(record["buildRootUri"]) for record in records)):
                            raise AssertionError("Cached Buildship associations differ from actual imported Gradle roots/modules")
                        if (before != set(modules["resources"].imported_resource_membership(client, root))
                                or any(hashlib.sha256(path.read_bytes()).hexdigest() != digest for path,digest in metadata.items())
                                or host_state() != original_host):
                            raise AssertionError("Cached association read changes membership, metadata or host authority")
                        journey["commands"].append({"stage": stage, "request": request, "reply": reply, "legacy": legacy})
                        receipt["checks"].append(label + ": " + stage + " current cached Buildship roots, v1 identities and metadata preserved")
                        return reply

                    if arguments.cached_build_associations_only:
                        read_cached_associations("before initial import")

                    def apply_configuration(value, expected_arguments):
                        client.notify("workspace/didChangeConfiguration", {"settings": value})
                        deadline = time.monotonic() + 30
                        while time.monotonic() < deadline:
                            if host_state()["gradleArguments"] == expected_arguments:
                                return
                            time.sleep(0.05)
                        raise AssertionError("Scoped host-settings oracle did not observe configuration replacement")

                    detached_build = gradle.with_name("scoped-detached-gradle")

                    def replace_selected_build():
                        workspace_before = root.stat()
                        original = gradle.stat()
                        gradle.rename(detached_build)
                        shutil.copytree(detached_build, gradle, symlinks=True)
                        replacement = gradle.stat()
                        workspace_after = root.stat()
                        if ((original.st_dev, original.st_ino) == (replacement.st_dev, replacement.st_ino)
                                or (workspace_before.st_dev, workspace_before.st_ino) != (workspace_after.st_dev, workspace_after.st_ino)):
                            raise AssertionError("Selected build replacement did not isolate its directory identity")
                        journey["selectedBuildDirectoryReplacement"] = {
                            "workspaceDevice": workspace_before.st_dev, "workspaceInode": workspace_before.st_ino,
                            "originalBuildDevice": original.st_dev, "originalBuildInode": original.st_ino,
                            "replacementBuildDevice": replacement.st_dev, "replacementBuildInode": replacement.st_ino,
                        }


                    def import_scoped(stage, expected_java=None, expected_resources=None):
                        request = {"command": "groovy.gradle.importSelected.v2", "arguments": [selection]}
                        reply = client.request("workspace/executeCommand", request, timeout=120)
                        if (set(reply) != {"version", "workspaceRootUri", "buildRootUri", "selectedImportPhaseComplete", "fullSynchronizationKnown"}
                                or reply["version"] != 2 or reply["workspaceRootUri"] != root.as_uri() + "/"
                                or reply["buildRootUri"] != gradle.as_uri() + "/"
                                or reply["selectedImportPhaseComplete"] is not True or reply["fullSynchronizationKnown"] is not False):
                            raise AssertionError("Scoped import receipt differs or overclaims synchronization")
                        journey["commands"].append({"stage": stage, "request": request, "reply": reply})
                        await_membership(client,
                                         scoped_maven_java | gradle_java if expected_java is None else expected_java,
                                         scoped_maven_resources | gradle_resources if expected_resources is None else expected_resources,
                                         journey, label + ": " + stage + " selected nested build preserves Maven")
                        if host_state() != original_host:
                            raise AssertionError("Scoped import changed initialized roots or host settings")
                        receipt["checks"].append(label + ": " + stage + " exact scoped receipt and unchanged host authority")
                        if arguments.cached_build_associations_only:
                            read_cached_associations("after " + stage)


                    if arguments.workspace_scoped_root_only:
                        def read_scoped_root(stage):
                            oracle = resource_state(gradle, "state")
                            descriptor = gradle / ".project"
                            before = hashlib.sha256(descriptor.read_bytes()).hexdigest() if descriptor.is_file() else None
                            request = {"command": "groovy.gradle.rootState.v2", "arguments": [selection]}
                            reply = client.request("workspace/executeCommand", request, timeout=120)
                            expected = "absent" if oracle.get("missing") is True else "open" if oracle["open"] else "closed"
                            if (set(reply) != {"version", "workspaceRootUri", "buildRootUri", "workspaceState", "descriptionAvailable", "fullSynchronizationKnown"}
                                    or reply["version"] != 2 or type(reply["version"]) is not int
                                    or reply["workspaceRootUri"] != root.as_uri() + "/" or reply["buildRootUri"] != gradle.as_uri() + "/"
                                    or reply["workspaceState"] != expected or reply["descriptionAvailable"] != descriptor.is_file()
                                    or reply["fullSynchronizationKnown"] is not False):
                                raise AssertionError("Scoped root facts differ from actual SDK resource state")
                            after = hashlib.sha256(descriptor.read_bytes()).hexdigest() if descriptor.is_file() else None
                            if before != after or host_state() != original_host:
                                raise AssertionError("Scoped root read changes description or initialized host settings")
                            journey["commands"].append({"stage": stage, "request": request, "reply": reply, "oracle": oracle})
                            receipt["checks"].append(label + ": " + stage + " exact scoped root facts and unchanged host authority")
                            return reply

                        def mutate_scoped_root(opening, stage):
                            request = {"command": "groovy.gradle.applyRootState.v2", "arguments": [{**selection, "open": opening}]}
                            reply = client.request("workspace/executeCommand", request, timeout=120)
                            if (set(reply) != {"version", "workspaceRootUri", "buildRootUri", "open", "rootStatePhaseComplete", "fullSynchronizationKnown"}
                                    or reply["version"] != 2 or type(reply["version"]) is not int
                                    or reply["workspaceRootUri"] != root.as_uri() + "/" or reply["buildRootUri"] != gradle.as_uri() + "/"
                                    or reply["open"] is not opening or reply["rootStatePhaseComplete"] is not True
                                    or reply["fullSynchronizationKnown"] is not False):
                                raise AssertionError("Scoped root mutation receipt differs or overclaims synchronization")
                            oracle = resource_state(gradle, "state")
                            if oracle.get("open") is not opening or oracle.get("exists") is not True or host_state() != original_host:
                                raise AssertionError("Scoped root phase differs from actual SDK state or changes host authority")
                            journey["commands"].append({"stage": stage, "request": request, "reply": reply, "oracle": oracle})
                            receipt["checks"].append(label + ": " + stage + " actual retained scoped root phase")
                            return reply

                        if arguments.workspace_scoped_module_only:
                            if not {"groovy.gradle.moduleStates.v2", "groovy.gradle.applyModuleStates.v2"}.issubset(commands):
                                raise AssertionError("Scoped module commands are unavailable")
                            scoped_modules = [gradle / name for name in ["application", "library"]]

                            def read_scoped_modules(stage):
                                before = {path: (path / ".project").read_bytes() if (path / ".project").is_file() else None
                                          for path in scoped_modules}
                                request = {"command": "groovy.gradle.moduleStates.v2", "arguments": [selection]}
                                reply = client.request("workspace/executeCommand", request, timeout=120)
                                expected = []
                                for path in scoped_modules:
                                    state = resource_state(path, "state")
                                    expected.append({"moduleUri": path.as_uri() + "/", "buildRootUri": gradle.as_uri() + "/",
                                        "workspaceState": "absent" if state.get("missing") is True else "open" if state["open"] else "closed",
                                        "descriptionAvailable": (path / ".project").is_file()})
                                if (set(reply) != {"version", "workspaceRootUri", "buildRootUri", "modules", "fullSynchronizationKnown"}
                                        or type(reply["version"]) is not int or reply["version"] != 2
                                        or reply["workspaceRootUri"] != root.as_uri() + "/" or reply["buildRootUri"] != gradle.as_uri() + "/"
                                        or sorted(reply["modules"], key=lambda value: value["moduleUri"]) != expected
                                        or reply["fullSynchronizationKnown"] is not False):
                                    raise AssertionError("Scoped module facts differ from actual current SDK resources")
                                if (host_state() != original_host or any(
                                        ((path / ".project").read_bytes() if (path / ".project").is_file() else None) != contents
                                        for path, contents in before.items())):
                                    raise AssertionError("Scoped module read changes metadata or host authority")
                                journey["commands"].append({"stage": stage, "request": request, "reply": reply, "expected": expected})
                                receipt["checks"].append(label + ": " + stage + " exact scoped module facts")
                                return reply

                            def mutate_scoped_modules(closed, restored, stage):
                                request = {"command": "groovy.gradle.applyModuleStates.v2", "arguments": [{**selection,
                                    "closedProjectUris": [path.as_uri() for path in closed],
                                    "restoredProjectUris": [path.as_uri() for path in restored]}]}
                                reply = client.request("workspace/executeCommand", request, timeout=120)
                                if (set(reply) != {"version", "workspaceRootUri", "buildRootUri", "closedProjectUris", "restoredProjectUris",
                                                 "moduleStatePhaseComplete", "fullSynchronizationKnown"}
                                        or type(reply["version"]) is not int or reply["version"] != 2
                                        or reply["workspaceRootUri"] != root.as_uri() + "/" or reply["buildRootUri"] != gradle.as_uri() + "/"
                                        or reply["closedProjectUris"] != [path.as_uri() + "/" for path in closed]
                                        or reply["restoredProjectUris"] != [path.as_uri() + "/" for path in restored]
                                        or reply["moduleStatePhaseComplete"] is not True or reply["fullSynchronizationKnown"] is not False):
                                    raise AssertionError("Scoped module phase receipt differs or overclaims synchronization")
                                for path, opening in [(path, False) for path in closed] + [(path, True) for path in restored]:
                                    state = resource_state(path, "state")
                                    if state.get("exists") is not True or state.get("open") is not opening:
                                        raise AssertionError("Scoped module phase does not match actual SDK state")
                                if host_state() != original_host:
                                    raise AssertionError("Scoped module mutation changes initialized roots or host settings")
                                journey["commands"].append({"stage": stage, "request": request, "reply": reply})
                                receipt["checks"].append(label + ": " + stage + " actual scoped module phase")

                            before_modules = read_scoped_modules("before selected import")
                            if label.endswith("pristine"):
                                expect_failure(client, {"command": "groovy.gradle.applyModuleStates.v2", "arguments": [{**selection,
                                    "closedProjectUris": [], "restoredProjectUris": [scoped_modules[0].as_uri()]}]},
                                    "Scoped Gradle module state change failed or may be partial", str(root))
                                if any((path / ".project").exists() for path in scoped_modules):
                                    raise AssertionError("Fresh module restore fabricated retained metadata")
                                mutate_scoped_modules(scoped_modules, [], "fresh model-derived closed descriptions")
                                read_scoped_modules("fresh closed modules")
                                if arguments.workspace_scoped_policy_only:
                                    state = read_scoped_root("fresh ignored root remains absent without fabricated metadata")
                                    if state["workspaceState"] != "absent" or state["descriptionAvailable"] is not False:
                                        raise AssertionError("Fresh whole-root ignore creates root metadata")
                                    await_membership(client, scoped_maven_java, scoped_maven_resources | {"gradle/application", "gradle/library"}, journey,
                                        label + ": fresh ignored root and closed children preserve Maven and plain Java")
                                restored = scoped_modules[:1] if arguments.workspace_scoped_policy_only else scoped_modules
                                mutate_scoped_modules([], restored, "restore model-derived retained descriptions")
                                current_resources = current_resources | {"gradle/application", "gradle/library"}
                                receipt["checks"].append(label + ": missing-description restore refuses; fresh close and retained restore use current model")

                        before_import = read_scoped_root("before selected import")
                        if not (gradle / ".project").is_file():
                            for opening in [False, True]:
                                expect_failure(client, {"command": "groovy.gradle.applyRootState.v2", "arguments": [{**selection, "open": opening}]},
                                    "Scoped Gradle root state change failed or may be partial", str(root))
                                if (gradle / ".project").exists() or resource_state(gradle, "state").get("missing") is not True:
                                    raise AssertionError("Fresh scoped root refusal fabricated a description or resource")
                            receipt["checks"].append(label + ": fresh root close/restore refuse without synthesizing descriptions")
                        elif scoped_policy_cold:
                            if (before_import["workspaceState"] not in {"closed", "absent"}
                                    or before_import["descriptionAvailable"] is not True):
                                raise AssertionError("Cold startup activates the ignored root or loses its retained description")
                            journey["coldInactiveRootState"] = before_import["workspaceState"]
                            mutate_scoped_root(True, "cold policy restores retained root before children")
                            mutate_scoped_modules([], scoped_modules, "cold policy restores retained children")
                            current_java = current_java | gradle_java
                            current_resources = current_resources | gradle_resources
                        elif before_import["workspaceState"] == "absent":
                            mutate_scoped_root(True, "cold retained root restore before import")
                            read_scoped_root("cold retained root restored")
                            current_resources = current_resources | {"gradle"}

                    alias = root / "scoped-build-alias"
                    alias.symlink_to(gradle, target_is_directory=True)
                    try:
                        invalid = [{**selection, "offline": False}, {**selection, "private-canary": "hidden"},
                                   {**selection, "workspaceRootUri": gradle.as_uri()},
                                   {**selection, "buildRootUri": root.parent.as_uri()},
                                   {**selection, "buildRootUri": alias.as_uri()},
                                   {**selection, "buildRootUri": (root / "missing-private-canary").as_uri()},
                                   {**selection, "buildRootUri": (gradle / ".." / "gradle").as_uri()}]
                        for value in invalid:
                            message = expect_failure(client, {"command": "groovy.gradle.importSelected.v2", "arguments": [value]},
                                                     "Selected Gradle import failed or may be partial", str(root))
                            if "private-canary" in message:
                                raise AssertionError("Scoped import exposes private request fields")
                            receipt["checks"].append(label + ": malformed/foreign/offline/alias scoped import refuses")
                    finally:
                        alias.unlink()
                    await_membership(client, current_java, current_resources, journey, label + ": refused requests preserve membership")
                    if arguments.workspace_scoped_policy_only and label.endswith("pristine"):
                        import_scoped("initial import preserves fresh module ignore",
                            scoped_maven_java | {"gradle/application"}, scoped_maven_resources | gradle_resources)
                        if resource_state(gradle / "library", "state").get("open") is not False:
                            raise AssertionError("Scoped import reopens a freshly ignored module")
                        read_scoped_modules("fresh module ignore persists through scoped import")
                        mutate_scoped_modules([], [gradle / "library"], "remove fresh module ignore")
                        import_scoped("recover fresh module selection")
                    else:
                        import_scoped("initial import")
                    if arguments.cached_build_associations_only and label.endswith("fresh"):
                        cached_request = {"rootUri": root.as_uri(), "cachedOnly": True}
                        for invalid in [{**cached_request, "cachedOnly": False},
                                {**cached_request, "extra": "private-canary"},
                                {"rootUri": root.as_uri()}, {**cached_request, "rootUri": gradle.as_uri()}]:
                            expect_failure(client, {"command": "groovy.gradle.importedBuildRoots", "arguments": [invalid]},
                                "Gradle imported project identities unavailable", str(root))
                            receipt["checks"].append("cached build-root read refuses malformed or uninitialized authority")
                        for control in ["cancel", "workspace-identity", "missing-module"]:
                            detached_module = gradle / "detached-application-fixture"
                            def remove_module():
                                (gradle / "application").rename(detached_module)
                            try:
                                modules["resources"].qualify_unlink_wait(client, root, cached_request,
                                    "groovy.gradle.importedBuildRoots", receipt, journey,
                                    replace_root_directory=control == "workspace-identity",
                                    during_wait=remove_module if control == "missing-module" else None)
                            finally:
                                if detached_module.exists():
                                    detached_module.rename(gradle / "application")
                            read_cached_associations("recovery after held " + control)
                    if arguments.workspace_scoped_root_only:
                        read_scoped_root("after selected import")
                        for command, request in [("groovy.gradle.rootState", {"rootUri": gradle.as_uri(), "offline": True}),
                                                 ("groovy.gradle.applyRootState", {"rootUri": gradle.as_uri(), "offline": True, "open": False})]:
                            expect_failure(client, {"command": command, "arguments": [request]}, "Gradle root state", str(root))
                        receipt["checks"].append(label + ": legacy nested root remains unauthorized")
                        expect_failure(client, {"command": "groovy.gradle.applyRootState.v2", "arguments": [{**selection, "open": False}]},
                            "Scoped Gradle root state change failed or may be partial", str(root))
                        if resource_state(gradle, "state").get("open") is not True:
                            raise AssertionError("Open-child refusal closes the root")
                        receipt["checks"].append(label + ": open Gradle children block scoped root close")
                        for name in ["application", "library"]:
                            state = resource_state(gradle / name, "state")
                            if state.get("open") is True:
                                resource_state(gradle / name, "close")
                        # These test-only child operations establish the SDK phase precondition.
                        mutate_scoped_root(False, "close with closed children")
                        read_scoped_root("closed retained root")
                        mutate_scoped_root(False, "already-closed noop")
                        mutate_scoped_root(True, "retained root restore")
                        read_scoped_root("restored retained root")
                        mutate_scoped_root(True, "already-open noop")
                        for name in ["application", "library"]:
                            state = resource_state(gradle / name, "state")
                            if state.get("open") is False:
                                resource_state(gradle / name, "open")
                        await_membership(client, scoped_maven_java | gradle_java, scoped_maven_resources | gradle_resources,
                            journey, label + ": scoped root restore preserves Maven plain Java and child identities")
                        import_scoped("repair after scoped root policy")
                        if label.endswith("fresh"):
                            alias = root / "scoped-root-alias"
                            alias.symlink_to(gradle, target_is_directory=True)
                            try:
                                invalid = [{**selection, "offline": False}, {**selection, "private-canary": "hidden"},
                                    {**selection, "workspaceRootUri": gradle.as_uri()},
                                    {**selection, "buildRootUri": root.parent.as_uri()},
                                    {**selection, "buildRootUri": alias.as_uri()},
                                    {**selection, "buildRootUri": (root / "missing-private-canary").as_uri()},
                                    {**selection, "buildRootUri": (gradle / ".." / "gradle").as_uri()},
                                    {**selection, "buildRootUri": "https://private-canary.invalid/"},
                                    {**selection, "buildRootUri": "x" * 4097}]
                                for command in ["groovy.gradle.rootState.v2", "groovy.gradle.applyRootState.v2"]:
                                    for value in invalid:
                                        request = {**value, "open": True} if command.endswith("applyRootState.v2") else value
                                        message = expect_failure(client, {"command": command, "arguments": [request]}, "Scoped Gradle root state", str(root))
                                        if "private-canary" in message:
                                            raise AssertionError("Scoped root error exposes private request fields")
                                        receipt["checks"].append(label + ": malformed foreign offline or aliased scoped root request refuses")
                                expect_failure(client, {"command": "groovy.gradle.applyRootState.v2", "arguments": [{**selection, "open": "false"}]},
                                    "Scoped Gradle root state", str(root))
                            finally:
                                alias.unlink()
                            descriptor = gradle / ".project"
                            original_descriptor = descriptor.read_bytes()
                            try:
                                dual = original_descriptor.replace(b"</natures>", b"<nature>org.eclipse.m2e.core.maven2Nature</nature></natures>")
                                if dual == original_descriptor:
                                    raise AssertionError("Scoped root fixture lacks retained nature insertion point")
                                descriptor.write_bytes(dual)
                                for command in ["groovy.gradle.rootState.v2", "groovy.gradle.applyRootState.v2"]:
                                    value = {**selection, "open": True} if command.endswith("applyRootState.v2") else selection
                                    expect_failure(client, {"command": command, "arguments": [value]}, "Scoped Gradle root state", str(root))
                            finally:
                                descriptor.write_bytes(original_descriptor)
                            read_scoped_root("recovery after retained dual-nature refusal")
                            # Invalid inputs must also fail a normally successful root-open no-op.
                            build_file = gradle / "build.gradle"
                            original_build = build_file.read_bytes()
                            for stage, contents in [("oversized", b"x" * (4 * 1024 * 1024 + 1)),
                                                    ("broken", original_build + b"\nthrow new GradleException('private-canary')\n")]:
                                try:
                                    build_file.write_bytes(contents)
                                    for command in ["groovy.gradle.rootState.v2", "groovy.gradle.applyRootState.v2"]:
                                        value = {**selection, "open": True} if command.endswith("applyRootState.v2") else selection
                                        message = expect_failure(client, {"command": command, "arguments": [value]}, "Scoped Gradle root state", str(root))
                                        if "private-canary" in message:
                                            raise AssertionError("Scoped root failure exposes private build text")
                                    receipt["checks"].append(stage + " scoped root input refuses with sanitized errors")
                                finally:
                                    build_file.write_bytes(original_build)
                                read_scoped_root("recovery after " + stage + " root input")
                            try:
                                descriptor.unlink()
                                for command in ["groovy.gradle.rootState.v2", "groovy.gradle.applyRootState.v2"]:
                                    value = {**selection, "open": True} if command.endswith("applyRootState.v2") else selection
                                    expect_failure(client, {"command": command, "arguments": [value]}, "Scoped Gradle root state", str(root))
                                receipt["checks"].append("missing retained root description refuses without source fabrication")
                            finally:
                                descriptor.write_bytes(original_descriptor)
                            read_scoped_root("recovery after missing root description")

                            settings_input = gradle / "settings.gradle"
                            original_input = settings_input.read_bytes()
                            try:
                                orphaned = original_input.replace(b"include 'application', 'library'", b"include 'application'")
                                if orphaned == original_input:
                                    raise AssertionError("Scoped root fixture lacks orphan declaration insertion point")
                                resource_state(gradle / "application", "close")
                                settings_input.write_bytes(orphaned)
                                environment = os.environ.copy()
                                environment["JAVA_HOME"] = str((arguments.gradle_java_home or arguments.java_home).resolve())
                                environment["GRADLE_USER_HOME"] = str(home)
                                discovery_command = [str(arguments.gradle_home.resolve() / "bin/gradle"), "--offline", "--console=plain", "-q", "-g", str(home), "-p", str(gradle), "projects"]
                                discovery = subprocess.run(discovery_command, env=environment, capture_output=True, text=True, timeout=120)
                                (evidence / "scoped-root-orphan-projects.log").write_text(discovery.stdout + discovery.stderr)
                                if discovery.returncode or "Project ':application'" not in discovery.stdout or "Project ':library'" in discovery.stdout:
                                    raise AssertionError("Actual Gradle discovery did not establish the removed library declaration")
                                journey["scopedRootOrphanDiscovery"] = {"command": discovery_command, "exitCode": discovery.returncode,
                                    "stdoutSha256": hashlib.sha256(discovery.stdout.encode()).hexdigest(), "libraryStillOpen": resource_state(gradle / "library", "state").get("open")}
                                expect_failure(client, {"command": "groovy.gradle.applyRootState.v2", "arguments": [{**selection, "open": False}]},
                                    "Scoped Gradle root state change failed or may be partial", str(root))
                                if resource_state(gradle, "state").get("open") is not True or resource_state(gradle / "library", "state").get("open") is not True:
                                    raise AssertionError("Orphan refusal closed a retained root or SDK orphan")
                                receipt["checks"].append("open SDK orphan absent from actual Gradle declarations blocks scoped root close")
                            finally:
                                settings_input.write_bytes(original_input)
                                if resource_state(gradle / "application", "state").get("open") is False:
                                    resource_state(gradle / "application", "open")
                            read_scoped_root("recovery after SDK orphan refusal")

                            for name in ["application", "library"]:
                                if resource_state(gradle / name, "state").get("open") is True:
                                    resource_state(gradle / name, "close")
                            try:
                                for command in ["groovy.gradle.rootState.v2", "groovy.gradle.applyRootState.v2"]:
                                    value = {**selection, "open": False} if command.endswith("applyRootState.v2") else selection
                                    modules["resources"].qualify_unlink_wait(client, root, value, command, receipt, journey)
                                    read_scoped_root("recovery after held " + command + " cancellation")
                                    try:
                                        modules["resources"].qualify_unlink_wait(client, root, value, command, receipt, journey,
                                            descriptor_change=settings_input, descriptor_contents=original_input + b"\n")
                                    finally:
                                        settings_input.write_bytes(original_input)
                                    read_scoped_root("recovery after held " + command + " input change")
                                    modules["resources"].qualify_unlink_wait(client, root, value, command, receipt, journey,
                                        replace_root_directory=True)
                                    read_scoped_root("recovery after held " + command + " workspace replacement")
                                    try:
                                        modules["resources"].qualify_unlink_wait(client, root, value, command, receipt, journey,
                                            during_wait=replace_selected_build)
                                    finally:
                                        if detached_build.exists():
                                            if gradle.exists():
                                                shutil.rmtree(gradle)
                                            detached_build.rename(gradle)
                                    read_scoped_root("recovery after held " + command + " build identity replacement")
                                    changed = json.loads(json.dumps(configuration))
                                    changed_arguments = [*original_host["gradleArguments"], "-Pscope_fixture_changed=true"]
                                    changed["java"]["import"]["gradle"]["arguments"] = changed_arguments
                                    try:
                                        modules["resources"].qualify_unlink_wait(client, root, value, command, receipt, journey,
                                            during_wait=lambda: apply_configuration(changed, changed_arguments))
                                    finally:
                                        apply_configuration(configuration, original_host["gradleArguments"])
                                    read_scoped_root("recovery after held " + command + " host settings change")
                                    changed_description = original_descriptor + b"\n" if command.endswith("applyRootState.v2") else original_descriptor.replace(
                                        b"</natures>", b"<nature>org.eclipse.m2e.core.maven2Nature</nature></natures>")
                                    try:
                                        modules["resources"].qualify_unlink_wait(client, root, value, command, receipt, journey,
                                            descriptor_change=descriptor, descriptor_contents=changed_description)
                                    finally:
                                        descriptor.write_bytes(original_descriptor)
                                    read_scoped_root("recovery after held " + command + " description change")
                                    if resource_state(gradle, "state").get("open") is not True:
                                        raise AssertionError("Held failed scoped root phase changed the root without a valid receipt")
                            finally:
                                for name in ["application", "library"]:
                                    if resource_state(gradle / name, "state").get("open") is False:
                                        resource_state(gradle / name, "open")
                            await_membership(client, scoped_maven_java | gradle_java, scoped_maven_resources | gradle_resources,
                                journey, label + ": all held scoped root failures preserve Maven plain Java and module identities")
                            import_scoped("recovery after all held root controls")
                        if arguments.workspace_scoped_module_only:
                            read_scoped_modules("after scoped root recovery")
                            mutate_scoped_modules(scoped_modules, [], "close both retained modules")
                            read_scoped_modules("closed retained modules")
                            mutate_scoped_modules(scoped_modules, [], "already-closed module noop")
                            mutate_scoped_modules([], scoped_modules, "restore both retained modules")
                            mutate_scoped_modules([], scoped_modules, "already-open module noop")
                            read_scoped_modules("restored retained modules")
                            await_membership(client, scoped_maven_java | gradle_java, scoped_maven_resources | gradle_resources,
                                journey, label + ": module phases preserve root Maven plain Java and current Java identities")
                            if label.endswith("fresh"):
                                module = scoped_modules[0]
                                mutation = {**selection, "closedProjectUris": [module.as_uri()], "restoredProjectUris": []}
                                alias = root / "scoped-module-alias"
                                alias.symlink_to(module, target_is_directory=True)
                                try:
                                    common_invalid = [{**selection, "offline": False}, {**selection, "private-canary": "hidden"},
                                        {**selection, "workspaceRootUri": gradle.as_uri()}, {**selection, "buildRootUri": root.parent.as_uri()},
                                        {**selection, "buildRootUri": alias.as_uri()}, {**selection, "buildRootUri": "x" * 4097}]
                                    for command in ["groovy.gradle.moduleStates.v2", "groovy.gradle.applyModuleStates.v2"]:
                                        for value in common_invalid:
                                            request = {**value, "closedProjectUris": [module.as_uri()], "restoredProjectUris": []} if command.endswith("applyModuleStates.v2") else value
                                            expect_failure(client, {"command": command, "arguments": [request]}, "Scoped Gradle module state", str(root))
                                            receipt["checks"].append(label + ": invalid scoped module authority refuses")
                                    invalid_lists = [([], []), ([gradle.as_uri()], []), ([maven.as_uri()], []),
                                        ([alias.as_uri()], []), ([module.as_uri(), module.as_uri()], []),
                                        ([module.as_uri()], [module.as_uri()]), ([module.as_uri()] * 129, []),
                                        ([17], []), ([module.as_uri()], "private-canary"),
                                        ([(gradle / "missing-private-canary").as_uri()], [])]
                                    for closed, restored in invalid_lists:
                                        expect_failure(client, {"command": "groovy.gradle.applyModuleStates.v2", "arguments": [{**selection,
                                            "closedProjectUris": closed, "restoredProjectUris": restored}]}, "Scoped Gradle module state", str(root))
                                        receipt["checks"].append(label + ": invalid root foreign duplicate alias budget or malformed module list refuses")
                                finally:
                                    alias.unlink()
                                descriptor = module / ".project"
                                original_descriptor = descriptor.read_bytes()
                                try:
                                    descriptor.write_bytes(original_descriptor.replace(b"</natures>", b"<nature>org.eclipse.m2e.core.maven2Nature</nature></natures>"))
                                    for command, value in [("groovy.gradle.moduleStates.v2", selection), ("groovy.gradle.applyModuleStates.v2", mutation)]:
                                        expect_failure(client, {"command": command, "arguments": [value]}, "Scoped Gradle module state", str(root))
                                        receipt["checks"].append(label + ": dual-nature module metadata refuses")
                                finally:
                                    descriptor.write_bytes(original_descriptor)
                                read_scoped_modules("recovery after dual-owned module refusal")
                                settings_input = gradle / "settings.gradle"
                                original_input = settings_input.read_bytes()
                                try:
                                    settings_input.write_bytes(original_input.replace(b"include 'application', 'library'", b"include 'application'"))
                                    for opening in [False, True]:
                                        expect_failure(client, {"command": "groovy.gradle.applyModuleStates.v2", "arguments": [{**selection,
                                            "closedProjectUris": [] if opening else [scoped_modules[1].as_uri()],
                                            "restoredProjectUris": [scoped_modules[1].as_uri()] if opening else []}]},
                                            "Scoped Gradle module state", str(root))
                                    if resource_state(scoped_modules[1], "state").get("open") is not True:
                                        raise AssertionError("Orphan module rejection changes unrelated retained resource")
                                    receipt["checks"].append(label + ": removed module cannot be closed or restored by current-model policy")
                                finally:
                                    settings_input.write_bytes(original_input)
                                read_scoped_modules("recovery after orphan module refusal")
                                module_build = module / "build.gradle"
                                module_build_bytes = module_build.read_bytes()
                                try:
                                    module_build.unlink()
                                    read_scoped_modules("current module without its own build file")
                                    mutate_scoped_modules([], [module], "retained module without its own build file")
                                finally:
                                    module_build.write_bytes(module_build_bytes)
                                read_scoped_modules("recovery after optional module build file removal")
                                for command, value in [("groovy.gradle.moduleStates.v2", selection), ("groovy.gradle.applyModuleStates.v2", mutation)]:
                                    modules["resources"].qualify_unlink_wait(client, root, value, command, receipt, journey)
                                    read_scoped_modules("recovery after held " + command + " cancellation")
                                    try:
                                        modules["resources"].qualify_unlink_wait(client, root, value, command, receipt, journey,
                                            descriptor_change=settings_input, descriptor_contents=original_input + b"\n")
                                    finally:
                                        settings_input.write_bytes(original_input)
                                    read_scoped_modules("recovery after held " + command + " input change")
                                    modules["resources"].qualify_unlink_wait(client, root, value, command, receipt, journey,
                                        replace_root_directory=True)
                                    read_scoped_modules("recovery after held " + command + " workspace replacement")
                                    try:
                                        modules["resources"].qualify_unlink_wait(client, root, value, command, receipt, journey,
                                            during_wait=replace_selected_build)
                                    finally:
                                        if detached_build.exists():
                                            if gradle.exists():
                                                shutil.rmtree(gradle)
                                            detached_build.rename(gradle)
                                    read_scoped_modules("recovery after held " + command + " build replacement")
                                    detached_module = module.with_name("scoped-detached-module")

                                    def replace_selected_module():
                                        original_identity = module.stat()
                                        module.rename(detached_module)
                                        shutil.copytree(detached_module, module, symlinks=True)
                                        replacement_identity = module.stat()
                                        if (original_identity.st_dev, original_identity.st_ino) == (replacement_identity.st_dev, replacement_identity.st_ino):
                                            raise AssertionError("Selected module replacement did not change directory identity")
                                        journey.setdefault("selectedModuleReplacements", []).append({
                                            "originalDevice": original_identity.st_dev, "originalInode": original_identity.st_ino,
                                            "replacementDevice": replacement_identity.st_dev, "replacementInode": replacement_identity.st_ino})

                                    try:
                                        modules["resources"].qualify_unlink_wait(client, root, value, command, receipt, journey,
                                            during_wait=replace_selected_module)
                                    finally:
                                        if detached_module.exists():
                                            if module.exists():
                                                shutil.rmtree(module)
                                            detached_module.rename(module)
                                    read_scoped_modules("recovery after held " + command + " module directory replacement")
                                    module_build = module / "build.gradle"
                                    module_build_bytes = module_build.read_bytes()
                                    try:
                                        modules["resources"].qualify_unlink_wait(client, root, value, command, receipt, journey,
                                            descriptor_change=module_build, descriptor_contents=module_build_bytes + b"\n")
                                    finally:
                                        module_build.write_bytes(module_build_bytes)
                                    read_scoped_modules("recovery after held " + command + " module input change")
                                    changed = json.loads(json.dumps(configuration))
                                    changed_arguments = [*original_host["gradleArguments"], "-Pscope_module_changed=true"]
                                    changed["java"]["import"]["gradle"]["arguments"] = changed_arguments
                                    try:
                                        modules["resources"].qualify_unlink_wait(client, root, value, command, receipt, journey,
                                            during_wait=lambda: apply_configuration(changed, changed_arguments))
                                    finally:
                                        apply_configuration(configuration, original_host["gradleArguments"])
                                    read_scoped_modules("recovery after held " + command + " settings change")
                                    try:
                                        changed_description = original_descriptor + b"\n" if command.endswith("applyModuleStates.v2") else original_descriptor.replace(
                                            b"</natures>", b"<nature>org.eclipse.m2e.core.maven2Nature</nature></natures>")
                                        modules["resources"].qualify_unlink_wait(client, root, value, command, receipt, journey,
                                            descriptor_change=descriptor, descriptor_contents=changed_description)
                                    finally:
                                        descriptor.write_bytes(original_descriptor)
                                    read_scoped_modules("recovery after held " + command + " description change")
                                    if resource_state(module, "state").get("open") is not True:
                                        raise AssertionError("Held failed module phase changes resource without a valid receipt")
                                import_scoped("recovery after held module phases")
                            if arguments.workspace_scoped_policy_only:
                                if arguments.workspace_scoped_default_policy_only:
                                    discover_scoped_defaults("before module ignore")
                                library = gradle / "library"
                                mutate_scoped_modules([library], [], "apply current module ignore")
                                import_scoped("import preserves current module ignore", scoped_maven_java | {"gradle/application"},
                                    scoped_maven_resources | gradle_resources)
                                state = read_scoped_modules("verify ignored module remains closed after import")
                                if next(record for record in state["modules"] if record["moduleUri"] == library.as_uri() + "/")["workspaceState"] != "closed":
                                    raise AssertionError("Provider fails to preserve explicit module ignore")
                                mutate_scoped_modules([], [library], "remove module ignore")
                                import_scoped("restore full selected build")
                                mutate_scoped_modules(scoped_modules, [], "whole-root ignore closes children first")
                                mutate_scoped_root(False, "whole-root ignore closes retained root")
                                read_scoped_root("verify ignored root")
                                read_scoped_modules("verify ignored children")
                                if arguments.workspace_scoped_default_policy_only:
                                    discover_scoped_defaults("after ignored root remains inactive")
                                if arguments.cached_build_associations_only:
                                    read_cached_associations("after whole-root close")

                                await_membership(client, scoped_maven_java, scoped_maven_resources | gradle_resources, journey,
                                    label + ": whole-root ignore preserves unrelated Maven and plain Java")
                                if not label.endswith("fresh"):
                                    mutate_scoped_root(True, "restore root before restoring children")
                                    mutate_scoped_modules([], scoped_modules, "restore children after root")
                                    import_scoped("recover whole-root selection")
                                else:
                                    journey["retainsClosedRootAndChildrenForRestart"] = True
                                receipt["checks"].append(label + ": explicit module/root policy ordered and unrelated resources preserved")
                            preserve_sources(label + ": scoped module policy preserves source generated and plain metadata")
                        preserve_sources(label + ": scoped root policy preserves sources generated files and plain metadata")
                    if label.endswith("fresh") and not arguments.workspace_scoped_root_only:
                        descriptor = gradle / ".project"
                        original_descriptor = descriptor.read_bytes()
                        try:
                            dual = original_descriptor.replace(b"</natures>", b"<nature>org.eclipse.m2e.core.maven2Nature</nature></natures>")
                            if dual == original_descriptor:
                                raise AssertionError("Scoped SDK fixture lacks nature insertion point")
                            descriptor.write_bytes(dual)
                            expect_failure(client, {"command": "groovy.gradle.importSelected.v2", "arguments": [selection]},
                                           "Selected Gradle import failed or may be partial", str(root))
                            receipt["checks"].append("scoped import refuses dual-owned retained metadata")
                        finally:
                            descriptor.write_bytes(original_descriptor)
                        build_file = gradle / "build.gradle"
                        original_build = build_file.read_bytes()
                        for stage, content in [("oversized", b"x" * (4 * 1024 * 1024 + 1)),
                                               ("broken", original_build + b"\nthrow new GradleException('private-canary')\n")]:
                            try:
                                build_file.write_bytes(content)
                                message = expect_failure(client, {"command": "groovy.gradle.importSelected.v2", "arguments": [selection]},
                                                         "Selected Gradle import failed or may be partial", str(root))
                                if "private-canary" in message:
                                    raise AssertionError("Scoped build failure exposes private build text")
                                receipt["checks"].append(stage + " scoped build refuses with sanitized error")
                            finally:
                                build_file.write_bytes(original_build)
                            import_scoped("recovery after " + stage)
                        settings_input = gradle / "settings.gradle"
                        original_input = settings_input.read_bytes()
                        try:
                            orphaned = original_input.replace(b"include 'application', 'library'", b"include 'application'")
                            if orphaned == original_input:
                                raise AssertionError("Scoped fixture lacks orphan-selection insertion point")
                            settings_input.write_bytes(orphaned)
                            expect_failure(client, {"command": "groovy.gradle.importSelected.v2", "arguments": [selection]},
                                           "Selected Gradle import failed or may be partial", str(root))
                            await_membership(client, scoped_maven_java | gradle_java, scoped_maven_resources | gradle_resources,
                                             journey, "scoped orphan refusal preserves the retained library and Maven")
                            receipt["checks"].append("scoped import refuses an unreviewed retained orphan rather than removing it")
                        finally:
                            settings_input.write_bytes(original_input)
                        import_scoped("recovery after orphan refusal")
                        modules["resources"].qualify_unlink_wait(client, root, selection, "groovy.gradle.importSelected.v2", receipt, journey)
                        import_scoped("recovery after held cancellation")
                        settings_input = gradle / "settings.gradle"
                        original_input = settings_input.read_bytes()
                        try:
                            modules["resources"].qualify_unlink_wait(client, root, selection, "groovy.gradle.importSelected.v2",
                                receipt, journey, descriptor_change=settings_input, descriptor_contents=original_input + b"\n")
                        finally:
                            settings_input.write_bytes(original_input)
                        import_scoped("recovery after held input change")
                        modules["resources"].qualify_unlink_wait(client, root, selection, "groovy.gradle.importSelected.v2",
                            receipt, journey, replace_root_directory=True)
                        import_scoped("recovery after workspace identity replacement")
                        try:
                            modules["resources"].qualify_unlink_wait(client, root, selection, "groovy.gradle.importSelected.v2",
                                receipt, journey, during_wait=replace_selected_build)
                        finally:
                            if detached_build.exists():
                                if gradle.exists():
                                    shutil.rmtree(gradle)
                                detached_build.rename(gradle)
                        import_scoped("recovery after selected build identity replacement")

                        changed = json.loads(json.dumps(configuration))
                        changed_arguments = [*original_host["gradleArguments"], "-Pscope_fixture_changed=true"]
                        changed["java"]["import"]["gradle"]["arguments"] = changed_arguments
                        try:
                            modules["resources"].qualify_unlink_wait(client, root, selection, "groovy.gradle.importSelected.v2",
                                receipt, journey, during_wait=lambda: apply_configuration(changed, changed_arguments))
                        finally:
                            apply_configuration(configuration, original_host["gradleArguments"])
                        import_scoped("recovery after held host setting change")
                    if arguments.workspace_scoped_reselection_only:
                        if resource_state(gradle, "state").get("open") is not True:
                            mutate_scoped_root(True, "prepare selection transition root")
                        mutate_scoped_modules([], scoped_modules, "prepare selection transition modules")
                        import_scoped("prepare selection transition namespace")
                        independent = root / "independent"
                        if independent.exists():
                            raise AssertionError("Independent disposable build unexpectedly exists")
                        independent.mkdir()
                        (independent / "settings.gradle").write_text("rootProject.name = 'selection-independent'\n")
                        (independent / "build.gradle").write_text("plugins { id 'java' }\n")
                        source = independent / "src/main/java/evidence/Independent.java"
                        source.parent.mkdir(parents=True)
                        source.write_text("package evidence; public class Independent {}\n")
                        generated = independent / "build/generated-sources/Sentinel.java"
                        generated.parent.mkdir(parents=True)
                        generated.write_text("package evidence; public class Sentinel {}\n")
                        preserved = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in [independent / "settings.gradle", independent / "build.gradle", source, generated]}
                        def import_independent(stage, includes_original):
                            request = {"command": "groovy.gradle.importSelected.v2", "arguments": [{**selection, "buildRootUri": independent.as_uri()}]}
                            reply = client.request("workspace/executeCommand", request, timeout=120)
                            if (set(reply) != {"version", "workspaceRootUri", "buildRootUri", "selectedImportPhaseComplete", "fullSynchronizationKnown"}
                                    or type(reply["version"]) is not int or reply["version"] != 2
                                    or reply["workspaceRootUri"] != root.as_uri() + "/" or reply["buildRootUri"] != independent.as_uri() + "/"
                                    or reply["selectedImportPhaseComplete"] is not True or reply["fullSynchronizationKnown"] is not False):
                                raise AssertionError("Independent selected import receipt is invalid")
                            journey["commands"].append({"stage": stage, "request": request, "reply": reply})
                            await_membership(client, scoped_maven_java | {"independent"} | (gradle_java if includes_original else set()),
                                scoped_maven_resources | {"independent"} | (gradle_resources if includes_original else set()), journey,
                                label + ": " + stage + " current independent build and Maven/plain Java preserved")
                        def unlink_transition_targets(targets, stage):
                            request = {"command": "groovy.gradle.unlinkSelected", "arguments": [{"rootUri": root.as_uri(), "projectUris": targets}]}
                            reply = client.request("workspace/executeCommand", request, timeout=60)
                            if (set(reply) != {"version", "rootUri", "unlinkedProjectUris", "selectedUnlinkPhaseComplete", "fullSynchronizationKnown"}
                                    or type(reply["version"]) is not int or reply["version"] != 1 or reply["rootUri"] != root.as_uri() + "/"
                                    or reply["selectedUnlinkPhaseComplete"] is not True or reply["fullSynchronizationKnown"] is not False
                                    or set(reply["unlinkedProjectUris"]) != set(targets) or len(reply["unlinkedProjectUris"]) != len(targets)):
                                raise AssertionError("Selection transition unlink receipt is invalid")
                            journey["commands"].append({"stage": stage, "request": request, "reply": reply})
                        import_independent("prepare independent selected build", True)
                        cached = read_cached_associations("both independent build associations", independent)
                        targets = [record["projectUri"] for record in cached["projects"] if record["buildRootUri"] == gradle.as_uri() + "/"]
                        expected_targets = {path.as_uri() + "/" for path in [gradle, *scoped_modules]}
                        if set(targets) != expected_targets or len(cached["projects"]) != len(expected_targets) + 1:
                            raise AssertionError("Independent transition lacks complete current build associations")
                        build_input = gradle / "build.gradle"
                        original = build_input.read_bytes()
                        marker = gradle / "forbidden-reselection-model-run"
                        if marker.exists():
                            raise AssertionError("Transition no-model marker already exists")
                        probe = {"omittedBuildUri": gradle.as_uri() + "/", "selectedBuildUri": independent.as_uri() + "/", "targetUris": targets}
                        try:
                            build_input.write_bytes(original + b"\nnew File(rootDir, 'forbidden-reselection-model-run').text = 'executed'\nthrow new GradleException('deselected-build-must-not-evaluate')\n")
                            read_cached_associations("poisoned omitted build cached ownership", independent)
                            unlink_transition_targets(targets, "unlink omitted build before selected import")
                            await_membership(client, scoped_maven_java | {"independent"}, scoped_maven_resources | {"independent"}, journey,
                                label + ": omitted root/modules unlinked while selected build remains active")
                            current = read_cached_associations("only selected independent build remains", independent)
                            if len(current["projects"]) != 1 or current["projects"][0]["buildRootUri"] != independent.as_uri() + "/":
                                raise AssertionError("Deselected Gradle ownership remains after subset unlink")
                            import_independent("selected reimport never evaluates omitted build", False)
                            probe["modelMarkerObserved"] = marker.exists()
                            if marker.exists() or any(hashlib.sha256(path.read_bytes()).hexdigest() != digest for path, digest in preserved.items()):
                                raise AssertionError("Transition evaluates omitted build or changes independent source/generated files")
                            receipt["checks"].append(label + ": exact cached subset unlink then selected import never evaluates poisoned omitted build")
                        finally:
                            probe["modelMarkerObserved"] = marker.exists()
                            if marker.exists():
                                probe["modelMarkerSha256"] = hashlib.sha256(marker.read_bytes()).hexdigest()
                                marker.unlink()
                            build_input.write_bytes(original)
                            journey["selectionTransitionProbe"] = probe
                        descriptor = independent / ".project"
                        descriptor_hash = hashlib.sha256(descriptor.read_bytes()).hexdigest()
                        unlink_transition_targets([independent.as_uri() + "/"], "finish disposable independent transition fixture")
                        if hashlib.sha256(descriptor.read_bytes()).hexdigest() != descriptor_hash or any(hashlib.sha256(path.read_bytes()).hexdigest() != digest for path,digest in preserved.items()):
                            raise AssertionError("Independent unlink changes retained description or source bytes")
                        probe["sourceAndDescriptionPreserved"] = True
                        preserve_sources(label + ": selection transition preserves original and generated sources")
                        # This extra fixture is complete; it is not part of the following startup/restart oracle.
                        shutil.rmtree(independent)
                        probe["disposableIndependentFixtureRemovedAfterObservation"] = True
                        import_scoped("restore original selection after transition fixture")

                    if arguments.workspace_scoped_details_only:
                        details_version = 4 if arguments.workspace_scoped_module_details_only else 3
                        details_command = f"groovy.gradle.projectDetails.v{details_version}"
                        if details_command not in commands:
                            raise AssertionError("Scoped details command is unavailable")
                        request = {"command": details_command, "arguments": [dict(selection)]}
                        members = set(modules["resources"].imported_resource_membership(client, root))
                        metadata = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*")
                            if path.is_file() and (path.name in {".project", ".classpath"} or ".settings" in path.parts)}
                        from urllib.parse import unquote, urlparse
                        def model_path(uri):
                            parsed = urlparse(uri)
                            if parsed.scheme != "file" or parsed.netloc or parsed.query or parsed.fragment:
                                raise AssertionError("Scoped model contains a nonlocal identity")
                            return Path(unquote(parsed.path)).resolve()
                        def details_witness():
                            return {
                                "membership": sorted(modules["resources"].imported_resource_membership(client, root)),
                                "metadata": {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in root.rglob("*") if path.is_file()
                                    and (path.name in {".project", ".classpath"} or ".settings" in path.parts)},
                                "host": host_state(),
                                "defaultTaskMarker": (gradle / "unrequested-root-task.marker").exists(),
                            }
                        baseline_witness = {"membership": sorted(members),
                            "metadata": {path.relative_to(root).as_posix(): digest for path,digest in metadata.items()},
                            "host": original_host, "defaultTaskMarker": False}
                        def query_scoped_details(stage):
                            before = details_witness()
                            value = client.request("workspace/executeCommand", request, timeout=150)
                            after = details_witness()
                            witness = {"stage": stage, "before": before, "after": after,
                                "changedDuringQuery": [key for key in baseline_witness if before[key] != after[key]],
                                "differsFromBaseline": [key for key in baseline_witness if after[key] != baseline_witness[key]]}
                            journey.setdefault("scopedDetailsWitnesses", []).append(witness)
                            if (set(value) != {"version", "workspaceRootUri", "buildRootUri", "model", "modelQueryComplete", "fullSynchronizationKnown"}
                                    or type(value["version"]) is not int or value["version"] != details_version
                                    or value["workspaceRootUri"] != root.as_uri() + "/" or value["buildRootUri"] != gradle.as_uri() + "/"
                                    or value["modelQueryComplete"] is not True or value["fullSynchronizationKnown"] is not False):
                                raise AssertionError("Scoped details envelope differs or overclaims synchronization")
                            model = value["model"]
                            if (set(model) != {"version", "rootUri", "projects", "sourceSetsKnown"}
                                    or type(model["version"]) is not int or model["version"] != 2
                                    or model["rootUri"] != gradle.as_uri() + "/" or model["sourceSetsKnown"] is not False
                                    or len(model["projects"]) != 3
                                    or {model_path(record["moduleUri"]) for record in model["projects"]} != {gradle, *scoped_modules}
                                    or any(model_path(record["buildRootUri"]) != gradle for record in model["projects"])):
                                raise AssertionError("Scoped inner model differs from actual nested root/module identities")
                            root_model = next(record for record in model["projects"] if record["projectPath"] == ":")
                            details = json.loads(root_model["detailsJson"])
                            task = next(task for task in details["tasks"] if task["path"] == ":scopedOptionProbe")
                            if not any(option["name"] == "scope-mode" for option in task["options"]):
                                raise AssertionError("Nested task option declaration missing")
                            for name in ["application", "library"]:
                                record = next(record for record in model["projects"] if record["projectPath"] == ":" + name)
                                source = next(source for source in json.loads(record["detailsJson"])["sourceSets"] if source["name"] == "main")
                                if not any(model_path(uri) == gradle / name / "src/main/java" for uri in source["roots"]) or source["generatednessKnown"]:
                                    raise AssertionError("Nested source roots missing or generatedness invented")
                            if after != baseline_witness:
                                raise AssertionError("Scoped model changes membership/Eclipse metadata/host authority or runs default tasks")
                            journey["commands"].append({"stage":stage,"request":request,"reply":value})
                            receipt["checks"].append(label + ": " + stage + " nested tasks/options/sources, stable membership and unknown full synchronization")
                            return value
                        baseline = query_scoped_details("explicit scoped nested details")
                        if query_scoped_details("stable scoped nested details") != baseline:
                            raise AssertionError("Repeated nested details changed stable model identities")
                        expect_failure(client, {"command":"groovy.gradle.projectDetails","arguments":[{"rootUri":gradle.as_uri(),"offline":True}]}, "root must match an initialized project folder")
                        receipt["checks"].append(label + ": legacy nested details authority remains exact")
                        if label.endswith("fresh"):
                            legacy_build = root / "build.gradle"
                            legacy_settings = root / "settings.gradle"
                            if legacy_build.exists() or legacy_settings.exists():
                                raise AssertionError("Same-root model oracle would overwrite fixture input")
                            try:
                                legacy_build.write_text("plugins { id 'base' }\n")
                                legacy_settings.write_text("rootProject.name='legacy-scope-oracle'\n")
                                legacy = client.request("workspace/executeCommand", {"command":"groovy.gradle.projectDetails",
                                    "arguments":[{"rootUri":root.as_uri(),"offline":True}]}, timeout=150)
                                equal_root = client.request("workspace/executeCommand", {"command":request["command"],
                                    "arguments":[{"workspaceRootUri":root.as_uri(),"buildRootUri":root.as_uri(),"offline":True}]}, timeout=150)
                                if (set(equal_root) != {"version", "workspaceRootUri", "buildRootUri", "model", "modelQueryComplete", "fullSynchronizationKnown"}
                                        or equal_root["version"] != details_version or equal_root["workspaceRootUri"] != root.as_uri()+"/"
                                        or equal_root["buildRootUri"] != root.as_uri()+"/" or equal_root["modelQueryComplete"] is not True
                                        or equal_root["fullSynchronizationKnown"] is not False or equal_root["model"] != legacy
                                        or set(modules["resources"].imported_resource_membership(client, root)) != members or host_state()!=original_host):
                                    raise AssertionError("Current same-root scoped/legacy model or membership differs")
                                if arguments.workspace_scoped_module_details_only:
                                    root_v3 = client.request("workspace/executeCommand", {"command":"groovy.gradle.projectDetails.v3",
                                        "arguments":[{"workspaceRootUri":root.as_uri(),"buildRootUri":root.as_uri(),"offline":True}]}, timeout=150)
                                    if (set(root_v3) != set(equal_root) or root_v3["version"] != 3
                                            or {key:value for key,value in root_v3.items() if key != "version"}
                                            != {key:value for key,value in equal_root.items() if key != "version"}):
                                        raise AssertionError("Module-fenced details changes the existing v3 model or envelope")
                                    journey["moduleFencedLegacyOracle"] = {"v3":root_v3,"v4":equal_root}
                                journey["sameRootLegacyDetailsOracle"] = {"legacy":legacy,"scoped":equal_root}
                                receipt["checks"].append(label+": current same-root scoped inner model exactly equals legacy without import")
                            finally:
                                if legacy_build.exists():legacy_build.unlink()
                                if legacy_settings.exists():legacy_settings.unlink()
                        alias = root / "scoped-details-alias"
                        alias.symlink_to(gradle, target_is_directory=True)
                        workspace_alias = root.parent / "scoped-details-workspace-alias"
                        workspace_alias.symlink_to(root, target_is_directory=True)
                        try:
                            invalid = [dict(selection, offline=False),dict(selection, unknown=True),
                                {key:value for key,value in selection.items() if key!="workspaceRootUri"},
                                dict(selection, workspaceRootUri=gradle.as_uri()),dict(selection, buildRootUri=alias.as_uri()),
                                dict(selection, workspaceRootUri=workspace_alias.as_uri()),dict(selection, buildRootUri="file:///private-canary/"),
                                dict(selection, buildRootUri="https://private-canary/"),dict(selection, buildRootUri="x"*4097),
                                dict(selection, buildRootUri=gradle.as_uri()+"/../gradle")]
                            for value in invalid:
                                message = expect_failure(client,{"command":request["command"],"arguments":[value]},"Scoped Gradle model discovery failed","private-canary")
                                if str(root) in message:
                                    raise AssertionError("Scoped model refusal exposes host paths")
                                receipt["checks"].append(label + ": malformed/aliased/noncanonical/foreign scoped details rejected before model")
                        finally:
                            alias.unlink()
                            workspace_alias.unlink()
                        if label.endswith("fresh"):
                            build_input = gradle / "build.gradle"
                            original = build_input.read_bytes()
                            categories = ["input","settings","build","workspace","cancel","timeout"]
                            if arguments.workspace_scoped_module_details_only:
                                categories.extend(["module-edit","module-add","module-remove","module-directory"])
                            for category in categories:
                                module_input = gradle / "library/build.gradle"
                                module_original = module_input.read_bytes()
                                module_alternate = gradle / "library/build.gradle.kts"
                                if module_alternate.exists() or module_alternate.is_symlink():
                                    raise AssertionError("Module descriptor addition would overwrite fixture input")
                                import queue
                                waiting_file = gradle / "scoped-model-started"
                                release_file = gradle / "scoped-model-release"
                                detached = None
                                destination = None
                                try:
                                    if waiting_file.exists() or release_file.exists():
                                        raise AssertionError("Scoped model gate is not fresh")
                                    delay = "Thread.sleep(65000)" if category == "timeout" else "while (!file('scoped-model-release').exists()) { Thread.sleep(25) }"
                                    build_input.write_bytes(original + ("\nfile('scoped-model-started').text = 'started'\n" + delay + "\n").encode())
                                    request_id = client.next_id
                                    client.next_id += 1
                                    client.send({"jsonrpc":"2.0","id":request_id,"method":"workspace/executeCommand","params":request})
                                    response = None
                                    def receive_model_message(timeout):
                                        try:
                                            message = client.messages.get(timeout=timeout)
                                        except queue.Empty:
                                            return None
                                        if isinstance(message,Exception):raise message
                                        if message.get("id")==request_id and "method" not in message:
                                            return message
                                        if "id" in message and "method" in message:
                                            if message["method"] != "workspace/configuration":
                                                raise AssertionError("Unexpected client request during scoped model gate")
                                            client.send({"jsonrpc":"2.0","id":message["id"],"result":[{} for _ in message.get("params",{}).get("items",[])]})
                                        elif "method" in message:
                                            with client.notifications_lock:client.notifications.append(message)
                                        else:
                                            raise AssertionError("Unmatched reply during sole scoped model request")
                                        return None
                                    deadline = time.monotonic()+50
                                    while not waiting_file.exists() and time.monotonic()<deadline:
                                        response = receive_model_message(0.05)
                                        if response is not None:break
                                    if not waiting_file.exists() or response is not None:
                                        journey.setdefault("earlyScopedDetails",[]).append(response)
                                        raise AssertionError("Held scoped model did not enter configuration before reply")
                                    if category == "input":
                                        build_input.write_bytes(build_input.read_bytes()+b"\n// changed during model\n")
                                    elif category == "settings":
                                        changed = json.loads(json.dumps(configuration))
                                        changed_args = [*original_host["gradleArguments"],"-Pscoped_details_authority_changed=true"]
                                        changed["java"]["import"]["gradle"]["arguments"] = changed_args
                                        # The configuration gate prevents a model reply while the existing serial client observes settings.
                                        apply_configuration(changed, changed_args)
                                    elif category == "module-edit":
                                        module_input.write_bytes(module_original + b"\n// changed during model\n")
                                    elif category == "module-add":
                                        module_alternate.write_text("// added during model\n")
                                    elif category == "module-remove":
                                        module_input.unlink()
                                    elif category in {"workspace","build","module-directory"}:
                                        destination = root if category == "workspace" else gradle / "library" if category == "module-directory" else gradle
                                        detached = destination.with_name(destination.name + "-model-retired")
                                        destination.rename(detached)
                                        shutil.copytree(detached,destination,symlinks=True)
                                    elif category == "cancel":
                                        client.notify("$/cancelRequest",{"id":request_id})
                                    if category != "timeout":release_file.write_text("release\n")
                                    deadline = time.monotonic()+150
                                    while response is None and time.monotonic()<deadline:
                                        response = receive_model_message(max(0.01,deadline-time.monotonic()))
                                    if response is None or "error" not in response:
                                        journey.setdefault("unexpectedScopedDetails",[]).append({"category":category,"response":response})
                                        raise AssertionError("Changed/cancelled/timed out scoped model publishes success or no reply")
                                    journey.setdefault("heldScopedDetailsResponses",[]).append({"category":category,"response":response})
                                    error = response["error"]
                                    message = str(error)
                                    protocol_cancel = (category == "cancel" and isinstance(error,dict)
                                                       and set(error) <= {"code","message","data"}
                                                       and error.get("code") == -32800 and error.get("data") is None
                                                       and error.get("message") == f"The request (id: {request_id}, method: 'workspace/executeCommand') has been cancelled")
                                    if (("Scoped Gradle model discovery" not in message and not protocol_cancel)
                                            or str(root) in message or "private-canary" in message
                                            or category == "timeout" and "timed out" not in message
                                            or category == "cancel" and "cancel" not in message):
                                        raise AssertionError("Scoped model error loses category or exposes private cause")
                                    journey.setdefault("heldScopedDetails",[]).append({"category":category,"error":message,"response":response})
                                    receipt["checks"].append(label + ": held scoped details " + category + " refuses stale success and sanitizes failure")
                                finally:
                                    if detached is not None and detached.exists():
                                        shutil.rmtree(destination)
                                        detached.rename(destination)
                                    build_input.write_bytes(original)
                                    if category in {"module-edit","module-remove"}:
                                        module_input.write_bytes(module_original)
                                    if category == "module-add" and module_alternate.exists():
                                        module_alternate.unlink()
                                    # Protocol cancellation can reply before the owned model observes its release gate.
                                    if category != "cancel":
                                        for path in [waiting_file,release_file]:
                                            if path.exists():path.unlink()
                                    apply_configuration(configuration, original_host["gradleArguments"])
                                if query_scoped_details("recovery after held " + category) != baseline:
                                    raise AssertionError("Scoped model did not recover after " + category)
                                if category == "cancel":
                                    for path in [waiting_file,release_file]:
                                        if path.exists():path.unlink()
                            for setting,alias,target in [
                                ("home", configured_gradle_home, arguments.gradle_home.resolve()),
                                ("java", configured_gradle_java_home, arguments.gradle_java_home.resolve()),
                            ]:
                                if not alias.is_symlink() or alias.readlink() != target or alias.parent != workspace:
                                    raise AssertionError("Missing-tool control does not own the configured alias")
                                try:
                                    alias.unlink()
                                    if alias.exists() or alias.is_symlink():
                                        raise AssertionError("Configured tool alias remains available")
                                    expect_failure(client,request,"Scoped Gradle model discovery failed")
                                    if host_state() != original_host:
                                        raise AssertionError("Missing-tool control changes host settings")
                                    receipt["checks"].append(label+": scoped details missing owned " + setting + " alias fails without bootstrap claim")
                                finally:
                                    alias.symlink_to(target, target_is_directory=True)
                                if query_scoped_details("recovery after missing " + setting) != baseline:
                                    raise AssertionError("Scoped details changed after missing-tool recovery")
                            try:
                                build_input.write_bytes(original+b"\nthrow new GradleException('SCOPED_MODEL_PRIVATE_CANARY')\n")
                                expect_failure(client,request,"Scoped Gradle model discovery failed","SCOPED_MODEL_PRIVATE_CANARY")
                                receipt["checks"].append(label+": scoped build failure strips private script cause")
                            finally:
                                build_input.write_bytes(original)
                            query_scoped_details("recovery after private script failure")
                        preserve_sources(label + ": scoped model preserves source and generated files")
                        journey["scopedDetailsQualified"] = True
                        if arguments.workspace_scoped_module_details_only:
                            journey["moduleFencedDetailsQualified"] = True

                    if arguments.workspace_scoped_empty_selection_only:
                        if resource_state(gradle, "state").get("open") is not True:
                            mutate_scoped_root(True, "prepare empty selection root")
                        mutate_scoped_modules([], scoped_modules, "prepare empty selection modules")
                        import_scoped("prepare empty selection namespace")
                        cached = read_cached_associations("prepare empty unlink targets")
                        targets = [record["projectUri"] for record in cached["projects"]]
                        expected_targets = {path.as_uri() + "/" for path in [gradle, *scoped_modules]}
                        if set(targets) != expected_targets or len(targets) != len(expected_targets):
                            raise AssertionError("Empty selection lacks complete current cached Gradle identities")
                        build_input = gradle / "build.gradle"
                        original = build_input.read_bytes()
                        marker = gradle / "forbidden-empty-model-run"
                        if marker.exists():
                            raise AssertionError("Empty no-model marker already exists")
                        probe = {"buildInputSha256": hashlib.sha256(original).hexdigest(), "targetUris": targets}
                        try:
                            build_input.write_bytes(original + b"\nnew File(rootDir, 'forbidden-empty-model-run').text = 'executed'\nthrow new GradleException('empty-selection-must-not-evaluate')\n")
                            poisoned = read_cached_associations("poisoned build uses only cached ownership")
                            if {record["projectUri"] for record in poisoned["projects"]} != expected_targets:
                                raise AssertionError("Poisoned input changes cached imported ownership")
                            request = {"command": "groovy.gradle.unlinkSelected", "arguments": [{"rootUri": root.as_uri(), "projectUris": targets}]}
                            reply = client.request("workspace/executeCommand", request, timeout=60)
                            if (set(reply) != {"version", "rootUri", "unlinkedProjectUris", "selectedUnlinkPhaseComplete", "fullSynchronizationKnown"}
                                    or type(reply["version"]) is not int or reply["version"] != 1
                                    or reply["rootUri"] != root.as_uri() + "/" or reply["selectedUnlinkPhaseComplete"] is not True
                                    or reply["fullSynchronizationKnown"] is not False
                                    or set(reply["unlinkedProjectUris"]) != expected_targets or len(reply["unlinkedProjectUris"]) != len(expected_targets)):
                                raise AssertionError("Empty unlink receipt is incomplete or overclaims synchronization")
                            journey["commands"].append({"stage": "explicit empty selection", "request": request, "reply": reply})
                            await_membership(client, scoped_maven_java, scoped_maven_resources, journey,
                                label + ": empty selection removes only Gradle SDK membership")
                            empty = read_cached_associations("verify empty selection and idempotent targets")
                            probe["emptyTargets"] = empty["projects"]
                            probe["modelMarkerObserved"] = marker.exists()
                            if empty["projects"] or marker.exists():
                                raise AssertionError("Empty cached/unlink path evaluates a deselected build or retains active Gradle")
                            receipt["checks"].append(label + ": poisoned build is never evaluated by cached empty unlink or no-op")
                        finally:
                            probe["modelMarkerObserved"] = marker.exists()
                            if marker.exists():
                                probe["modelMarkerSha256"] = hashlib.sha256(marker.read_bytes()).hexdigest()
                                marker.unlink()
                            build_input.write_bytes(original)
                            journey["emptySelectionProbe"] = probe
                        preserve_sources(label + ": empty unlink preserves original and generated sources")
                        if label.endswith("fresh"):
                            journey["retainsClosedRootAndChildrenForRestart"] = False
                            journey["retainsUnlinkedDescriptionsForRestart"] = True
                        else:
                            import_scoped("restore after explicit empty selection")
                            receipt["checks"].append(label + ": explicit import restores stable Gradle identities after empty unlink")

                    journey["scopedHostStateAfter"] = host_state()
                def verify_identities(name, java, resources, action):
                    from urllib.parse import urlparse, unquote
                    build = root / name
                    request = {"command": f"groovy.{name}.importedProjects", "arguments": [
                        {"rootUri": build.as_uri(), "cachedOnly": True}
                    ]}
                    reply = client.request("workspace/executeCommand", request, timeout=30)
                    if not isinstance(reply, dict) or set(reply) != {"version", "rootUri", "projects", "partial", "fullSynchronizationKnown"}:
                        raise AssertionError("Imported build identity response has unexpected fields")
                    if reply["version"] != 1 or reply["partial"] is not False or reply["fullSynchronizationKnown"] is not False:
                        raise AssertionError("Imported build identity response overclaims coverage/synchronization")
                    if Path(unquote(urlparse(reply["rootUri"]).path)).resolve(strict=True) != build:
                        raise AssertionError("Imported build identity root differs")
                    observed = {}
                    for record in reply["projects"]:
                        if set(record) != {"projectUri", "javaNature"} or not isinstance(record["javaNature"], bool):
                            raise AssertionError("Imported build identity record differs")
                        uri = urlparse(record["projectUri"])
                        if uri.scheme != "file" or uri.netloc not in ("", "localhost") or uri.query or uri.fragment:
                            raise AssertionError("Imported build identity is not a confined file URI")
                        path = Path(unquote(uri.path)).resolve(strict=True)
                        if not path.is_dir() or not path.is_relative_to(build):
                            raise AssertionError("Imported build identity escaped its provider root")
                        relative = path.relative_to(root).as_posix()
                        if relative in observed:
                            raise AssertionError("Imported build identity is duplicated")
                        observed[relative] = record["javaNature"]
                    if observed != {relative: relative in java for relative in resources}:
                        raise AssertionError("Imported build kind/Java nature differs from observed fixture membership")
                    journey["commands"].append({"label": action, "request": request, "reply": reply})
                    receipt["checks"].append(action + ": exact confined provider and Java/non-Java identities")

                if arguments.provider_root_read_only:
                    def inspect_root(stage):
                        oracle_request = {"rootUri": gradle.as_uri(), "projectUri": gradle.as_uri(), "operation": "state"}
                        oracle = client.request("workspace/executeCommand", {"command": "groovy.fixture.unlink.projectState", "arguments": [oracle_request]}, timeout=30)
                        descriptor = gradle / ".project"
                        descriptor_hash = hashlib.sha256(descriptor.read_bytes()).hexdigest() if descriptor.is_file() else None
                        def output_hashes():
                            outputs = {}
                            for directory in [gradle / "build", gradle / "application/build", gradle / "library/build"]:
                                for file in directory.rglob("*"):
                                    if file.is_file():
                                        outputs[file.relative_to(gradle).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
                            return outputs
                        outputs = output_hashes()
                        request = {"rootUri": gradle.as_uri() + "/", "offline": True}
                        reply = client.request("workspace/executeCommand", {"command": "groovy.gradle.rootState", "arguments": [request]}, timeout=120)
                        expected_state = "absent" if oracle.get("missing") is True else "open" if oracle["open"] else "closed"
                        if (set(reply) != {"version", "rootUri", "workspaceState", "descriptionAvailable", "fullSynchronizationKnown"}
                                or type(reply["version"]) is not int or reply["version"] != 1
                                or reply["rootUri"] != gradle.as_uri() + "/" or reply["workspaceState"] != expected_state
                                or reply["descriptionAvailable"] is not (descriptor_hash is not None)
                                or reply["fullSynchronizationKnown"] is not False):
                            raise AssertionError("Production root state differs from exact SDK/descriptor facts")
                        after = client.request("workspace/executeCommand", {"command": "groovy.fixture.unlink.projectState", "arguments": [oracle_request]}, timeout=30)
                        if after != oracle or output_hashes() != outputs or (hashlib.sha256(descriptor.read_bytes()).hexdigest() if descriptor.is_file() else None) != descriptor_hash:
                            raise AssertionError("Root read changed SDK resources, descriptors or build outputs")
                        journey.setdefault("productionRootSnapshots", []).append({"stage": stage, "request": request, "reply": reply, "oracle": oracle, "descriptorSha256": descriptor_hash, "outputHashes": outputs})
                        receipt["checks"].append(label + ": " + stage + " exact read-only production root state")
                        return reply

                    inspect_root("startup")
                    if label == "root-state-seed":
                        alias = gradle.parent / "root-read-alias"
                        alias.symlink_to(gradle, target_is_directory=True)
                        valid = {"rootUri": gradle.as_uri() + "/", "offline": True}
                        try:
                            for request in [{**valid, "offline": False}, {**valid, "extra": "private-canary"},
                                            {"rootUri": valid["rootUri"]}, {**valid, "rootUri": root.as_uri()},
                                            {**valid, "rootUri": alias.as_uri()}, {**valid, "rootUri": "https://private-canary.invalid/"}]:
                                message = expect_failure(client, {"command": "groovy.gradle.rootState", "arguments": [request]}, "Gradle root state unavailable", str(root))
                                if "private-canary" in message:
                                    raise AssertionError("Root read failure exposes private input")
                                receipt["checks"].append("invalid/foreign/offline/alias root read refuses with sanitized error")
                        finally:
                            alias.unlink()
                        inspect_root("recovery after invalid inputs")
                    if label == "root-state-retained":
                        descriptor = gradle / ".project"
                        original = descriptor.read_bytes()
                        dual = original.replace(b"</natures>", b"<nature>org.eclipse.m2e.core.maven2Nature</nature></natures>")
                        if dual == original:
                            raise AssertionError("SDK root description lacks fixture nature insertion point")
                        for name, content in [("oversized", b"x" * (1024 * 1024 + 1)), ("dual", dual)]:
                            try:
                                descriptor.write_bytes(content)
                                expect_failure(client, {"command": "groovy.gradle.rootState", "arguments": [{"rootUri": gradle.as_uri(), "offline": True}]}, "Gradle root state unavailable", str(root))
                                receipt["checks"].append(name + " root description refuses without SDK mutation")
                            finally:
                                descriptor.write_bytes(original)
                            inspect_root("recovery after " + name)
                        held_request = {"rootUri": gradle.as_uri() + "/", "offline": True}
                        for name, parameters in [("cancel", {}), ("descriptor", {"descriptor_change": descriptor, "descriptor_contents": dual}), ("root", {"replace_root": True})]:
                            try:
                                modules["resources"].qualify_unlink_wait(client, gradle, held_request, "groovy.gradle.rootState", receipt, journey, **parameters)
                            finally:
                                descriptor.write_bytes(original)
                            inspect_root("recovery after held " + name)
                if arguments.provider_root_state_only:
                    root_faults_pending = [True]
                    def mutate_root(open_state, stage):
                        request = {"rootUri": gradle.as_uri() + "/", "offline": True, "open": open_state}
                        reply = client.request("workspace/executeCommand", {"command": "groovy.gradle.applyRootState", "arguments": [request]}, timeout=120)
                        journey.setdefault("productionRootPhases", []).append({"stage": stage, "request": request, "reply": reply})
                        if (set(reply) != {"version", "rootUri", "open", "rootStatePhaseComplete", "fullSynchronizationKnown"}
                                or type(reply["version"]) is not int or reply["version"] != 1
                                or reply["rootUri"] != request["rootUri"] or reply["open"] is not open_state
                                or reply["rootStatePhaseComplete"] is not True or reply["fullSynchronizationKnown"] is not False):
                            raise AssertionError("Production root phase receipt differs or overclaims")
                        receipt["checks"].append(label + ": " + stage + " exact production root phase")

                    valid = {"rootUri": gradle.as_uri() + "/", "offline": True, "open": True}
                    if label == "root-state-seed":
                        for open_state in [False, True]:
                            expect_failure(client, {"command": "groovy.gradle.applyRootState", "arguments": [{**valid, "open": open_state}]}, "Gradle root state change failed or may be partial", str(root))
                            receipt["checks"].append("fresh missing-description root mutation refuses without first import")
                        inspect_root("fresh root mutation refusal preserved absence")
                    if label == "root-state-retained":
                        alias = gradle.parent / "root-mutation-alias"
                        alias.symlink_to(gradle, target_is_directory=True)
                        try:
                            for request in [{**valid, "offline": False}, {**valid, "extra": "private-canary"},
                                            {"rootUri": valid["rootUri"], "offline": True}, {**valid, "open": "private-canary"},
                                            {**valid, "rootUri": root.as_uri()}, {**valid, "rootUri": alias.as_uri()},
                                            {**valid, "rootUri": "https://private-canary.invalid/"}]:
                                message = expect_failure(client, {"command": "groovy.gradle.applyRootState", "arguments": [request]}, "Gradle root state change failed or may be partial", str(root))
                                if "private-canary" in message:
                                    raise AssertionError("Root mutation error exposes private input")
                                receipt["checks"].append("malformed/foreign/offline/alias root mutation refuses with sanitized error")
                        finally:
                            alias.unlink()
                        mutate_root(True, "valid already-open control")
                        expect_failure(client, {"command": "groovy.gradle.applyRootState", "arguments": [{**valid, "open": False}]}, "Gradle root state change failed or may be partial", str(root))
                        receipt["checks"].append("open Gradle children block root close before mutation")
                        if inspect_root("recovery after active-child refusal")["workspaceState"] != "open":
                            raise AssertionError("Refused root mutation changed root state")

                    def root_faults():
                        descriptor = gradle / ".project"
                        original = descriptor.read_bytes()
                        request = {"rootUri": gradle.as_uri() + "/", "offline": True, "open": False}
                        for name, parameters in [("cancel", {}), ("descriptor", {"descriptor_change": descriptor}),
                                                 ("root", {"replace_root": True}), ("same-path directory", {"replace_root_directory": True})]:
                            try:
                                modules["resources"].qualify_unlink_wait(client, gradle, request, "groovy.gradle.applyRootState", receipt, journey, **parameters)
                            finally:
                                descriptor.write_bytes(original)
                            if inspect_root("root phase recovery after held " + name)["workspaceState"] != "open":
                                raise AssertionError("Held refusal changed the root")
                        library = (gradle / "library").as_uri() + "/"
                        restored = {"rootUri": gradle.as_uri() + "/", "offline": True, "closedProjectUris": [], "restoredProjectUris": [library]}
                        client.request("workspace/executeCommand", {"command": "groovy.gradle.applyModuleStates", "arguments": [restored]}, timeout=120)
                        await_membership(client, maven_java | {"gradle", "gradle/library"}, maven_resources | gradle_resources, journey, "orphan fixture retains only reopened library and Java root")
                        settings_file = gradle / "settings.gradle"
                        settings_bytes = settings_file.read_bytes()
                        changed = settings_bytes.replace(b"include 'application', 'library'", b"include 'application'")
                        if changed == settings_bytes:
                            raise AssertionError("Orphan fixture did not remove its model declaration")
                        try:
                            settings_file.write_bytes(changed)
                            model = client.request("workspace/executeCommand", {"command": "groovy.gradle.projects", "arguments": [{"rootUri": gradle.as_uri(), "offline": True}]}, timeout=120)
                            if any(item["moduleUri"].rstrip("/") == library.rstrip("/") for item in model["projects"]):
                                raise AssertionError("Orphan remains in current Gradle model")
                            expect_failure(client, {"command": "groovy.gradle.applyRootState", "arguments": [request]}, "Gradle root state change failed or may be partial", str(root))
                            journey["orphanRootRefusal"] = {"currentModel": model, "stillOpenSdkLibrary": True}
                            receipt["checks"].append("open SDK orphan absent from current model blocks root close")
                        finally:
                            settings_file.write_bytes(settings_bytes)
                        closed = {"rootUri": gradle.as_uri() + "/", "offline": True, "closedProjectUris": [library], "restoredProjectUris": []}
                        client.request("workspace/executeCommand", {"command": "groovy.gradle.applyModuleStates", "arguments": [closed]}, timeout=120)
                        inspect_root("recovery after orphan refusal")
                if root_actions:
                    def root_resource(operation):
                        request = {"rootUri": gradle.as_uri(), "projectUri": gradle.as_uri(), "operation": operation}
                        if arguments.provider_root_state_only and operation != "state":
                            if label == "root-state-retained" and operation == "close" and root_faults_pending[0]:
                                root_faults_pending[0] = False
                                root_faults()
                            mutate_root(operation == "open", operation)
                            request["operation"] = "state"
                        reply = client.request("workspace/executeCommand", {"command": "groovy.fixture.unlink.projectState", "arguments": [request]}, timeout=30)
                        journey.setdefault("rootResourceStates", []).append({"operation": operation, "reply": reply})
                        if reply.get("missing") is True:
                            if operation != "state":
                                raise AssertionError("Root SDK mutation unexpectedly returned absence")
                        elif (reply.get("exists") is not True or not isinstance(reply.get("open"), bool)
                                or reply.get("rootUri", "").rstrip("/") != gradle.as_uri()
                                or reply.get("projectUri", "").rstrip("/") != gradle.as_uri()):
                            raise AssertionError("Root SDK resource state escaped the selected fixture")
                        if arguments.provider_root_read_only and operation == "state":
                            inspect_root("SDK root state observation")
                        receipt["checks"].append(label + ": root SDK " + operation + " response confined")
                        return reply

                    def children(closing, stage):
                        directories = [(gradle / name).as_uri() + "/" for name in ["application", "library"]]
                        request = {"rootUri": gradle.as_uri() + "/", "offline": True,
                                   "closedProjectUris": directories if closing else [], "restoredProjectUris": [] if closing else directories}
                        reply = client.request("workspace/executeCommand", {"command": "groovy.gradle.applyModuleStates", "arguments": [request]}, timeout=120)
                        journey["commands"].append({"stage": stage, "command": "groovy.gradle.applyModuleStates", "request": request, "reply": reply})
                        if (reply.get("moduleStatePhaseComplete") is not True or reply.get("fullSynchronizationKnown") is not False
                                or any(reply.get(field) != request[field] for field in ["rootUri", "closedProjectUris", "restoredProjectUris"])):
                            raise AssertionError("Root SDK experiment child phase differs or overclaims")
                        query = {"rootUri": gradle.as_uri() + "/", "offline": True}
                        states = client.request("workspace/executeCommand", {"command": "groovy.gradle.moduleStates", "arguments": [query]}, timeout=120)
                        journey.setdefault("childModuleStates", []).append({"stage": stage, "reply": states})
                        records = {item["moduleUri"]: item for item in states.get("modules", [])}
                        if (states.get("fullSynchronizationKnown") is not False or set(records) != set(directories)
                                or any(item["workspaceState"] != ("closed" if closing else "open") or item["descriptionAvailable"] is not True for item in records.values())):
                            raise AssertionError("Child SDK states differ from the production phase")
                        receipt["checks"].append(label + ": " + stage + " exact production child policy")

                    def synchronize(stage, closing):
                        request = {"rootUri": gradle.as_uri() + "/", "offline": True}
                        reply = client.request("workspace/executeCommand", {"command": "groovy.gradle.importSelected", "arguments": [request]}, timeout=120)
                        journey["commands"].append({"stage": stage, "command": "groovy.gradle.importSelected", "request": request, "reply": reply})
                        if reply.get("selectedImportPhaseComplete") is not True or reply.get("fullSynchronizationKnown") is not False:
                            raise AssertionError("Root SDK synchronization overclaims completion")
                        java = maven_java if closing else maven_java | gradle_java
                        await_membership(client, java, maven_resources | gradle_resources, journey, label + ": " + stage + " preserves Maven and child membership")
                        observed = root_resource("state")
                        if observed.get("missing") is True:
                            raise AssertionError("Root resource disappeared during explicit synchronization")
                        journey.setdefault("synchronizedRootStates", []).append({"stage": stage, "open": observed["open"], "childrenClosed": closing})
                        preserve_sources(label + ": " + stage + " source/generated/report integrity")

                    initial_root = root_resource("state")
                    if initial_root.get("missing") is True:
                        if not (gradle / ".project").is_file():
                            raise AssertionError("Cold closed root lost its retained description")
                        initial_root = root_resource("recreateClosed")
                        if initial_root["open"] is not False:
                            raise AssertionError("Root description recovery imported the root")
                    children(True, "close/recover children")
                    if root_resource("state")["open"]:
                        root_resource("close")
                    if root_resource("state")["open"] is not False:
                        raise AssertionError("Root SDK close did not close the selected build resource")
                    synchronize("explicit sync with closed root", True)
                    if root_resource("state")["open"] is False:
                        root_resource("open")
                    children(False, "restore children")
                    synchronize("explicit root and child restore", False)
                    if root_actions == "retain":
                        children(True, "retain closed children")
                        if root_resource("state")["open"]:
                            root_resource("close")
                        await_membership(client, maven_java, maven_resources | gradle_resources, journey, label + ": retained closed build preserves Maven")
                        if root_resource("state")["open"] is not False:
                            raise AssertionError("Retained root did not stay closed before shutdown")
                    preserve_sources(label + ": root SDK lifecycle preserves sources and reports")
                if metadata_actions:
                    def module_states(stage):
                        request = {"rootUri": gradle.as_uri() + "/", "offline": True}
                        reply = client.request("workspace/executeCommand", {"command": "groovy.gradle.moduleStates", "arguments": [request]}, timeout=120)
                        if (set(reply) != {"version", "rootUri", "modules", "fullSynchronizationKnown"}
                                or reply["version"] != 1 or reply["rootUri"].rstrip("/") != gradle.as_uri()
                                or reply["fullSynchronizationKnown"] is not False):
                            raise AssertionError("Startup module snapshot schema/phase coverage differs")
                        records = {item["moduleUri"].rstrip("/"): item for item in reply["modules"]}
                        if set(records) != {(gradle / name).as_uri() for name in ["application", "library"]}:
                            raise AssertionError("Startup module snapshot escapes exact fixture scope")
                        journey.setdefault("moduleSnapshots", []).append({"stage": stage, "request": request, "reply": reply})
                        receipt["checks"].append(label + ": " + stage + " exact production module snapshot")
                        return records[(gradle / "library").as_uri()]

                    def policy(closing, stage):
                        before = module_states(stage + " before")
                        if before["descriptionAvailable"] is not True:
                            raise AssertionError("Retained startup fixture lost its module description")
                        request = {"rootUri": gradle.as_uri() + "/", "offline": True,
                                   "closedProjectUris": [(gradle / "library").as_uri() + "/"] if closing else [],
                                   "restoredProjectUris": [] if closing else [(gradle / "library").as_uri() + "/"]}
                        reply = client.request("workspace/executeCommand", {"command": "groovy.gradle.applyModuleStates", "arguments": [request]}, timeout=120)
                        journey["commands"].append({"stage": stage, "command": "groovy.gradle.applyModuleStates", "request": request, "reply": reply})
                        if (set(reply) != {"version", "rootUri", "closedProjectUris", "restoredProjectUris", "moduleStatePhaseComplete", "fullSynchronizationKnown"}
                                or reply["version"] != 1 or reply["moduleStatePhaseComplete"] is not True
                                or reply["fullSynchronizationKnown"] is not False
                                or any(reply[field] != request[field] for field in ["rootUri", "closedProjectUris", "restoredProjectUris"])):
                            raise AssertionError("Startup module policy receipt differs or overclaims")
                        after = module_states(stage + " after")
                        if after["workspaceState"] != ("closed" if closing else "open"):
                            raise AssertionError("Startup production policy did not reach the requested SDK state")
                        request = {"rootUri": gradle.as_uri() + "/", "offline": True}
                        reply = client.request("workspace/executeCommand", {"command": "groovy.gradle.importSelected", "arguments": [request]}, timeout=120)
                        journey["commands"].append({"stage": stage, "command": "groovy.gradle.importSelected", "request": request, "reply": reply})
                        if (set(reply) != {"version", "rootUri", "selectedImportPhaseComplete", "fullSynchronizationKnown"}
                                or reply["version"] != 1 or reply["rootUri"].rstrip("/") != gradle.as_uri()
                                or reply["selectedImportPhaseComplete"] is not True or reply["fullSynchronizationKnown"] is not False):
                            raise AssertionError("Startup root import differs or overclaims synchronization")
                        java = maven_java | (gradle_java - {"gradle/library"} if closing else gradle_java)
                        resources = (expected_resources - gradle_resources) | gradle_resources
                        await_membership(client, java, resources, journey, label + ": " + stage + " preserves Maven and exact Gradle membership")
                        current = module_states(stage + " synchronized")
                        if current["workspaceState"] != ("closed" if closing else "open"):
                            raise AssertionError("SDK root synchronization bypassed module policy")
                        preserve_sources(label + ": " + stage + " source/generated/report integrity")

                    policy(True, "close/recover")
                    policy(False, "explicit restore")
                    policy(True, "retain closed for restart")
                if identity_reads:
                    verify_identities("maven", expected_java & maven_java, expected_resources & maven_resources, label + ": Maven startup read")
                    verify_identities("gradle", expected_java & gradle_java, expected_resources & gradle_resources, label + ": Gradle startup read")
                if repair:
                    for name, build, selected, java, resources in [
                        ("maven", maven, [maven / "pom.xml", maven / "application/pom.xml", maven / "library/pom.xml"], maven_java, maven_resources),
                        ("gradle", gradle, None, maven_java | gradle_java, maven_resources | gradle_resources),
                    ]:
                        params = {"rootUri": build.as_uri(), "offline": True}
                        if selected is not None:
                            params["pomUris"] = [path.as_uri() for path in selected]
                        request = {"command": f"groovy.{name}.importSelected", "arguments": [params]}
                        reply = client.request("workspace/executeCommand", request, timeout=100)
                        if reply.get("version") != 1 or reply.get("selectedImportPhaseComplete") is not True or reply.get("fullSynchronizationKnown") is not False:
                            raise AssertionError("Guarded offline startup repair overclaims completion")
                        journey["commands"].append({"request": request, "reply": reply})
                        receipt["checks"].append(label + f": guarded offline {name} phase only")
                        await_membership(client, java, resources, journey, label + f": {name} repair preserves other roots")
                        if identity_reads:
                            verify_identities("maven", java & maven_java, resources & maven_resources, label + f": {name} repair Maven read")
                            verify_identities("gradle", java & gradle_java, resources & gradle_resources, label + f": {name} repair Gradle read")
                if identity_reads:
                    from urllib.parse import urlparse, unquote
                    if repair:
                        nature_request = {"command": "groovy.fixture.unlink.natures", "arguments": [
                            {"rootUri": maven.as_uri(), "projectUri": maven.as_uri(), "dual": True}
                        ]}
                        changed = client.request("workspace/executeCommand", nature_request, timeout=30)
                        if changed.get("dualNature") is not True:
                            raise AssertionError("Disposable fixture did not establish actual ambiguous build natures")
                        try:
                            for name in ["maven", "gradle"]:
                                request = {"command": f"groovy.{name}.importedProjects", "arguments": [
                                    {"rootUri": maven.as_uri(), "cachedOnly": True}
                                ]}
                                reply = client.request("workspace/executeCommand", request, timeout=30)
                                if reply.get("partial") is not True or reply.get("fullSynchronizationKnown") is not False:
                                    raise AssertionError("Ambiguous imported project claimed complete build identity")
                                if any(Path(unquote(urlparse(record["projectUri"]).path)).resolve() == maven for record in reply["projects"]):
                                    raise AssertionError("Ambiguous imported project was assigned a provider kind")
                                journey["commands"].append({"label": "ambiguous nature read", "request": request, "reply": reply})
                                receipt["checks"].append(label + f": {name} ambiguous nature omitted with incomplete coverage")
                        finally:
                            nature_request["arguments"][0]["dual"] = False
                            restored = client.request("workspace/executeCommand", nature_request, timeout=30)
                            if restored.get("dualNature") is not False:
                                raise AssertionError("Disposable ambiguous nature was not restored")
                        await_membership(client, maven_java | gradle_java, maven_resources | gradle_resources, journey, label + ": restored nature preserves resources")
                        verify_identities("maven", maven_java, maven_resources, label + ": restored Maven provider kind")
                        verify_identities("gradle", gradle_java, gradle_resources, label + ": restored Gradle provider kind")
                    for name in ["maven", "gradle"]:
                        for invalid in [
                            {"rootUri": root.as_uri(), "cachedOnly": True},
                            {"rootUri": (root / name).as_uri(), "cachedOnly": False},
                            {"rootUri": (root / name).as_uri(), "cachedOnly": True, "private-canary": "do-not-export"},
                            {"rootUri": "https://private.invalid/do-not-export", "cachedOnly": True},
                        ]:
                            request = {"command": f"groovy.{name}.importedProjects", "arguments": [invalid]}
                            error = expect_failure(client, request, "imported project identities unavailable", str(root))
                            if "do-not-export" in error:
                                raise AssertionError("Imported build identity failure exposes private input")
                            journey["commands"].append({"label": "invalid identity read", "request": request, "error": error})
                            receipt["checks"].append(label + f": {name} invalid/foreign identity read rejected and sanitized")
                if ownership_actions:
                    for name, build in [("maven", maven), ("gradle", gradle)]:
                        for module in [build, build / "application"]:
                            nature = {"command": "groovy.fixture.unlink.natures", "arguments": [
                                {"rootUri": build.as_uri(), "projectUri": module.as_uri(), "dual": True}
                            ]}
                            changed = client.request("workspace/executeCommand", nature, timeout=30)
                            journey["commands"].append({"label": "establish dual nature", "request": json.loads(json.dumps(nature)), "reply": changed})
                            if changed.get("dualNature") is not True:
                                raise AssertionError("Ownership fixture did not establish both actual build natures")
                            try:
                                for chosen in ["maven", "gradle"]:
                                    request = {"command": f"groovy.{chosen}.unlinkSelected", "arguments": [
                                        {"rootUri": build.as_uri(), "projectUris": [module.as_uri()]}
                                    ]}
                                    error = expect_failure(client, request, "unlink failed or may be partial", str(root))
                                    journey["commands"].append({"label": "dual ownership unlink rejection", "request": request, "error": error})
                                    receipt["checks"].append(f"{name} {module.name}: ambiguous {chosen} unlink rejected and sanitized")
                                await_membership(client, maven_java | gradle_java, maven_resources | gradle_resources, journey,
                                                 f"{name} {module.name}: both rejected unlinks preserve all imported identities")
                                preserve_sources(f"{name} {module.name}: rejected unlink preserves source/build/generated hashes")
                            finally:
                                nature["arguments"][0]["dual"] = False
                                restored = client.request("workspace/executeCommand", nature, timeout=30)
                                journey["commands"].append({"label": "restore single nature", "request": nature, "reply": restored})
                                if restored.get("dualNature") is not False:
                                    raise AssertionError("Ownership fixture did not restore its original nature")
                            selection = {"rootUri": build.as_uri(), "projectUris": [module.as_uri()]}
                            try:
                                modules["resources"].qualify_unlink_wait(client, build, selection, f"groovy.{name}.unlinkSelected",
                                                                       receipt, journey, ownership_change=module)
                                await_membership(client, maven_java | gradle_java, maven_resources | gradle_resources, journey,
                                                 f"{name} {module.name}: held ownership change preserves all imported identities")
                                preserve_sources(f"{name} {module.name}: held rejection preserves source/build/generated hashes")
                            finally:
                                restored = client.request("workspace/executeCommand", nature, timeout=30)
                                journey["commands"].append({"label": "restore held ownership change", "request": nature, "reply": restored})
                                if restored.get("dualNature") is not False:
                                    raise AssertionError("Held ownership change was not restored")
                        verify_identities(name, maven_java if name == "maven" else gradle_java,
                                          maven_resources if name == "maven" else gradle_resources, f"{name}: ownership recovery")
                        application = build / "application"
                        request = {"command": f"groovy.{name}.unlinkSelected", "arguments": [
                            {"rootUri": build.as_uri(), "projectUris": [application.as_uri()]}
                        ]}
                        reply = client.request("workspace/executeCommand", request, timeout=30)
                        journey["commands"].append({"label": "restored single-provider unlink", "request": request, "reply": reply})
                        if (reply.get("version") != 1 or reply.get("selectedUnlinkPhaseComplete") is not True
                                or reply.get("fullSynchronizationKnown") is not False
                                or {Path(unquote(urlparse(uri).path)).resolve() for uri in reply.get("unlinkedProjectUris", [])} != {application}):
                            raise AssertionError("Restored unlink did not remove exactly the selected membership")
                        identity = application.relative_to(root).as_posix()
                        await_membership(client, (maven_java | gradle_java) - {identity},
                                         (maven_resources | gradle_resources) - {identity}, journey,
                                         f"{name}: restored single-provider unlink removes only selection")
                        preserve_sources(f"{name}: successful unlink preserves source/build/generated hashes")
                        parameters = {"rootUri": build.as_uri(), "offline": True}
                        if name == "maven":
                            parameters["pomUris"] = [(application / "pom.xml").as_uri()]
                        request = {"command": f"groovy.{name}.importSelected", "arguments": [parameters]}
                        reply = client.request("workspace/executeCommand", request, timeout=100)
                        journey["commands"].append({"label": "guarded offline repair after unlink", "request": request, "reply": reply})
                        if reply.get("selectedImportPhaseComplete") is not True or reply.get("fullSynchronizationKnown") is not False:
                            raise AssertionError("Ownership recovery import overclaims completion")
                        await_membership(client, maven_java | gradle_java, maven_resources | gradle_resources, journey,
                                         f"{name}: guarded offline repair restores selected membership")
                if linked_actions:
                    def phase(name, build, poms=None):
                        params = {"rootUri": build.as_uri(), "offline": True}
                        if poms is not None:
                            params["pomUris"] = [pom.as_uri() for pom in poms]
                        return {"command": f"groovy.{name}.importSelected", "arguments": [params]}

                    def admitted_phase(request, action):
                        reply = client.request("workspace/executeCommand", request, timeout=100)
                        if reply.get("version") != 1 or reply.get("selectedImportPhaseComplete") is not True or reply.get("fullSynchronizationKnown") is not False:
                            raise AssertionError("Linked root receipt overclaims completion")
                        journey["commands"].append({"label": action, "request": request, "reply": reply})
                        receipt["checks"].append(action + ": guarded offline phase only")

                    request = phase("gradle", gradle)
                    error = expect_failure(client, request, "Selected Gradle import failed or may be partial", str(root))
                    journey["commands"].append({"label": "nested uninitialized root", "request": request, "error": error})
                    receipt["checks"].append("nested uninitialized root rejected without host path leak")
                    admitted_phase(phase("maven", root, [maven / "pom.xml", maven / "application/pom.xml", maven / "library/pom.xml"]), "Maven under initialized container")
                    await_membership(client, maven_java, maven_resources, journey, "Maven container import preserves excluded Gradle")
                    folder = {"uri": gradle.as_uri(), "name": gradle.name}
                    added = {"event": {"added": [folder], "removed": []}}
                    client.notify("workspace/didChangeWorkspaceFolders", added)
                    journey.setdefault("folderEvents", []).append(added)
                    await_membership(client, None, None, journey, "after nested folder addition: observed membership")
                    admitted_phase(request, "selected nested Gradle import")
                    await_membership(client, maven_java | gradle_java, maven_resources | gradle_resources, journey, "nested Gradle import preserves Maven roots")
                    removed = {"event": {"added": [], "removed": [folder]}}
                    client.notify("workspace/didChangeWorkspaceFolders", removed)
                    journey["folderEvents"].append(removed)
                    await_membership(client, None, None, journey, "after nested folder removal: observed membership")
                    observed = journey["states"][-1]["samples"][-1]
                    if not maven_java.issubset(observed["java"]) or not maven_resources.issubset(observed["resources"]):
                        raise AssertionError("Removing selected Gradle folder dropped unrelated Maven resources")
                    receipt["checks"].append("folder removal preserves unrelated Maven membership")
                    error = expect_failure(client, request, "Selected Gradle import failed or may be partial", str(root))
                    journey["commands"].append({"label": "removed selected root", "request": request, "error": error})
                    receipt["checks"].append("removed nested root rejected without host path leak")
                    preserve_sources("folder removal preserves sources/build/generated files")
                    client.notify("workspace/didChangeWorkspaceFolders", added)
                    journey["folderEvents"].append(added)
                    await_membership(client, None, None, journey, "after nested folder re-addition: observed membership")
                    admitted_phase(request, "re-added nested Gradle offline repair")
                    await_membership(client, maven_java | gradle_java, maven_resources | gradle_resources, journey, "re-added selected Gradle repair preserves Maven")
                preserve_sources(label + ": source/build/generated hashes preserved; no tests run")
                journey["passed"] = True
            finally:
                logs = evidence / label
                logs.mkdir(exist_ok=True)
                client.stderr.seek(0)
                (logs / "stderr-before-shutdown.log").write_bytes(client.stderr.read())
                client.close()
                if (data / ".metadata/.log").is_file():
                    shutil.copy2(data / ".metadata/.log", logs / "jdtls.log")
                journey["evidence"] = str(logs)
                arguments.receipt.write_text(json.dumps(receipt, indent=2) + "\n")

        try:
            if arguments.workspace_scoped_import_only:
                seed = workspace / "scoped-import-seed-profile"
                run("scoped-import-seed", seed, {"maven": False, "gradle": False}, set(), set(),
                    configurations=[], repair=True)
                retained = fixtures_by_profile[seed]
                data = workspace / "scoped-import-profile"
                run("scoped-import-audit-fresh", data, {"maven": False, "gradle": False}, set(), set(),
                    configurations=[], reuse_fixture=retained, folders=["", "maven"])
                run("scoped-import-audit-restart", data, {"maven": False, "gradle": False},
                    maven_java | gradle_java, maven_resources | gradle_resources,
                    configurations=[], reuse_fixture=retained, folders=["", "maven"])
                run("scoped-import-audit-pristine", workspace / "scoped-import-pristine-profile",
                    {"maven": False, "gradle": False}, set(), set(), configurations=[], folders=["", "maven"])
                receipt["limits"].append("Cached provider-default roots joined to scoped module/root/import sequencing only; no typed default consumer, complete startup/default/empty/link/required/orphan/composite, installed/native/remote/platform or release qualification" if arguments.workspace_scoped_default_policy_only else "Direct scoped module/root/import sequencing only, including retained cold and pristine suppression; no installed editor pair, full default/link/required/orphan/composite policy, native/remote/platform or release qualification" if arguments.workspace_scoped_policy_only else "Provider scoped import/root/module SDK phases only; no typed module/root client sequence, complete saved policy, composite/default replay, native/remote/platform or release qualification" if arguments.workspace_scoped_module_only else "Provider scoped import and root-state phases only; test-only children prepare root close, not production nested module policy; no client saved-policy, composite/default replay, native/remote/platform or release qualification" if arguments.workspace_scoped_root_only else "Provider scoped-import phase only; no client saved-policy, nested root/module close/restore, composite/default replay, native/remote/platform or release qualification")
                return
            if arguments.root_state_only:
                data = workspace / "root-state-profile"
                run("root-state-seed", data, {"maven": False, "gradle": False}, set(), set(), repair=True)
                run("root-state-retained", data, {"maven": False, "gradle": False}, maven_java | gradle_java,
                    maven_resources | gradle_resources, root_actions="retain")
                run("root-state-cold", data, {"maven": False, "gradle": False}, maven_java, maven_resources,
                    metadata_exclusion=True, root_actions="restore")
                receipt["limits"].append("Root mutation uses existing test-only SDK resource commands; unchanged production0.35 still refuses root policy. Sync root state is observed, not promised. Fresh roots without descriptions/product startup/root-ignore/native/parity remain unqualified")
                return
            if arguments.single_provider_configurations_only:
                folders = ["maven", "gradle", "plain"]
                seed = workspace / "single-configurations-seed-profile"
                run("single-configurations-seed", seed, {"maven": False, "gradle": False}, set(), set(),
                    configurations=[], repair=True, folders=folders)
                retained = fixtures_by_profile[seed]
                plain_java = {"plain"}
                for name, expected_java, expected_resources, file in [
                        ("maven", maven_java, maven_resources, retained / "maven/pom.xml"),
                        ("gradle", gradle_java, gradle_resources, retained / "gradle/settings.gradle")]:
                    data = workspace / ("single-configurations-" + name + "-profile")
                    enabled = {"maven": name == "maven", "gradle": name == "gradle"}
                    configurations = [file, retained / "plain/.project"]
                    for suffix in ["fresh", "restart"]:
                        run("single-configurations-" + name + "-" + suffix, data, enabled,
                            expected_java | plain_java, expected_resources | plain_java,
                            configurations=configurations, reuse_fixture=retained, folders=folders)
                run("single-configurations-disabled", workspace / "single-configurations-disabled-profile",
                    {"maven": False, "gradle": False}, plain_java, plain_java,
                    configurations=[retained / "maven/pom.xml", retained / "gradle/settings.gradle",
                                    retained / "plain/.project"], reuse_fixture=retained, folders=folders)
                run("single-configurations-empty", workspace / "single-configurations-empty-profile",
                    {"maven": True, "gradle": True}, set(), set(),
                    configurations=[], reuse_fixture=retained, folders=folders)
                run("single-configurations-removed", workspace / "single-configurations-removed-profile",
                    {"maven": False, "gradle": False}, None, None, reuse_fixture=retained, folders=folders)
                baseline = receipt["journeys"][-1]["states"][-1]["samples"][-1]
                if not (maven_java | gradle_java | plain_java).issubset(baseline["java"]):
                    raise AssertionError("Removed configuration list did not witness retained metadata import")
                receipt["checks"].append("removing explicit configurations admits retained Maven/Gradle and plain Java metadata")
                receipt["limits"].append("Explicit known file lists are not qualified as default scanner/exclusion/nesting/depth equivalents; existing SDK resources are not closed by initialization filtering. No production startup correction is claimed")
                return
            if arguments.startup_configurations_only:
                seed = workspace / "configurations-seed-profile"
                run("configurations-seed", seed, {"maven": False, "gradle": False}, set(), set(), repair=True)
                retained = fixtures_by_profile[seed]
                run("configurations-bypass-witness", workspace / "configurations-witness-profile", {"maven": False, "gradle": False}, None, None, reuse_fixture=retained)
                baseline = receipt["journeys"][-1]["states"][-1]["samples"][-1]
                if not (maven_java | gradle_java).issubset(baseline["java"]):
                    raise AssertionError("Retained metadata did not witness ordinary Eclipse import")
                receipt["checks"].append("absent explicit configurations admit retained metadata with both build importers disabled")
                run("configurations-empty-enabled", workspace / "configurations-enabled-profile", {"maven": True, "gradle": True}, set(), set(), reuse_fixture=retained, configurations=[])
                selected = workspace / "configurations-empty-profile"
                run("configurations-empty-repair", selected, {"maven": False, "gradle": False}, set(), set(), reuse_fixture=retained, configurations=[], repair=True)
                run("configurations-retained-sdk", selected, {"maven": False, "gradle": False}, maven_java | gradle_java,
                    maven_resources | gradle_resources, configurations=[], metadata_actions=True)
                closed_java = maven_java | {"gradle/application"}
                closed_resources = maven_resources | {"gradle", "gradle/application"}
                run("configurations-empty-cold", selected, {"maven": False, "gradle": False}, closed_java, closed_resources,
                    configurations=[], metadata_actions=True)
                run("configurations-removed", workspace / "configurations-removed-profile", {"maven": False, "gradle": False},
                    maven_java | gradle_java, set(baseline["resources"]), reuse_fixture=retained)
                receipt["limits"].append("Explicit empty configuration list is an SDK initialization observation, not provider-specific production startup/default replay, nested policy or resource closing")
                return
            if arguments.startup_metadata_only:
                seed = workspace / "metadata-seed-profile"
                run("metadata-seed", seed, {"maven": False, "gradle": False}, set(), set(), repair=True)
                retained = fixtures_by_profile[seed]
                run("metadata-bypass-witness", workspace / "metadata-witness-profile", {"maven": False, "gradle": False}, None, None, reuse_fixture=retained)
                baseline = receipt["journeys"][-1]["states"][-1]["samples"][-1]
                if not (maven_java | gradle_java).issubset(baseline["java"]):
                    raise AssertionError("Retained metadata did not witness the pinned Eclipse-importer bypass")
                receipt["checks"].append("retained Eclipse metadata admits Java modules with both build importers disabled")
                maven_retained = {name for name in baseline["resources"] if name == "maven" or name.startswith("maven/")}
                selected = workspace / "metadata-excluded-profile"
                run("metadata-excluded-fresh", selected, {"maven": False, "gradle": False}, maven_java, maven_retained,
                    reuse_fixture=retained, metadata_exclusion=True, metadata_actions=True)
                closed_java = maven_java | {"gradle/application"}
                closed_resources = maven_retained | {"gradle", "gradle/application"}
                run("metadata-excluded-cold", selected, {"maven": False, "gradle": False}, closed_java, closed_resources,
                    metadata_exclusion=True, metadata_actions=True)
                run("metadata-exclusion-removed", workspace / "metadata-removed-profile", {"maven": False, "gradle": False},
                    maven_java | gradle_java, set(baseline["resources"]), reuse_fixture=retained)
                receipt["checks"].append("removing SDK exclusion restores retained Java module discovery without a source deletion")
                receipt["limits"].append("Exact ASCII macOS fixture root exclusion only; no product settings derivation, mixed-root ownership, literal glob escaping, physical platform/remote/default delivery or full parity qualification")
                return
            if arguments.unlink_ownership_only:
                data = workspace / "unlink-ownership-profile"
                run("unlink-ownership-fresh", data, {"maven": False, "gradle": False}, set(), set(), repair=True, identity_reads=True, ownership_actions=True)
                run("unlink-ownership-restart", data, {"maven": False, "gradle": False},
                    maven_java | gradle_java, maven_resources | gradle_resources, identity_reads=True)
                receipt["limits"].append("Guarded provider nature checks do not qualify product saved policy, native/remote mutation or causal full synchronization")
                return
            if arguments.provider_identity_only:
                data = workspace / "identity-profile"
                run("provider-identities-fresh", data, {"maven": False, "gradle": False}, set(), set(), repair=True, identity_reads=True)
                run("provider-identities-restart", data, {"maven": False, "gradle": False},
                    maven_java | gradle_java, maven_resources | gradle_resources, identity_reads=True)
                receipt["limits"].append("Cached provider kind/Java nature reads do not qualify product snapshot admission, remote, ignore policy or full synchronization")
                return
            if arguments.linked_root_phase_only:
                data = workspace / "linked-profile"
                run("nested-linked-root", data, {"maven": False, "gradle": False}, set(), set(), folders=[""], linked_actions=True)
                run("nested-linked-root-restart", data, {"maven": False, "gradle": False},
                    maven_java | gradle_java, maven_resources | gradle_resources, folders=["", "gradle"])
                receipt["limits"].append("Workspace-folder events and stable membership are not causal full synchronization or product trust/remote certification")
                return
            run("maven-default-gradle-controlled", workspace / "maven-profile", {"gradle": False}, maven_java, maven_resources)
            run("gradle-default-maven-controlled", workspace / "gradle-profile", {"maven": False}, gradle_java, gradle_resources)
            data = workspace / "repaired-profile"
            run("both-controlled-fresh-repair", data, {"maven": False, "gradle": False}, set(), set(), repair=True)
            run("both-controlled-retained-restart", data, {"maven": False, "gradle": False}, maven_java | gradle_java, maven_resources | gradle_resources)
            run("one-controlled-retained-restart", data, {"gradle": False}, maven_java | gradle_java, maven_resources | gradle_resources)
            run("both-controlled-fresh-profile-existing-files", workspace / "existing-files-profile",
                {"maven": False, "gradle": False}, None, None, reuse_fixture=root)
            receipt["limits"].append("Existing Eclipse metadata may be admitted by the Eclipse importer even with both build importers disabled; last journey is observation, not exclusion acceptance")
        finally:
            environment = dict(os.environ, JAVA_HOME=str(arguments.gradle_java_home.resolve()), GRADLE_USER_HOME=str(home))
            stopped = subprocess.run([str(arguments.gradle_home.resolve() / "bin/gradle"), "--offline", "--stop", "--gradle-user-home", str(home)],
                                     cwd=root, env=environment, capture_output=True, text=True)
            (evidence / "owned-gradle-stop.log").write_text(stopped.stdout + stopped.stderr)
            receipt["owned_worker_stop_exit"] = stopped.returncode
            arguments.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
            if stopped.returncode:
                raise RuntimeError("Isolated owned Gradle worker shutdown failed")
    print(json.dumps(receipt, indent=2))


def probe_default_roots(arguments):
    specification = importlib.util.spec_from_file_location(
        "groovy_maven_fixture", REPOSITORY / "script/test-groovy-maven-model-provider.py"
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Cannot load existing SDK fixture helpers")
    fixture = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(fixture)
    provider = arguments.provider.resolve()
    if hashlib.sha256(provider.read_bytes()).hexdigest() != arguments.sha256:
        raise AssertionError("Provider binary differs from the source build")
    evidence = arguments.receipt.parent
    evidence.mkdir(parents=True, exist_ok=True)
    workspace = evidence / "fixture"
    workspace.mkdir()
    root = workspace / "projects"
    root.mkdir()
    home = workspace / "empty-gradle-home"
    home.mkdir()
    receipt = {"provider_sha256": arguments.sha256, "checks": [], "journeys": [], "status": "running", "limits": [
        "Pinned cached headless SDK fixture, not a product consumer, app, physical remote or release qualification",
        "Only the independent SDK scanner oracle walks this disposable fixture; production evaluates explicit candidates",
        "Existing Gradle resources do not imply independent Gradle build roots; membership remains a separate owner",
    ]}
    names = [("aggregate", "settings.gradle"), ("aggregate", "build.gradle.kts"),
             ("aggregate/child", "build.gradle"), ("standalone", "settings.gradle.kts"),
             ("opaque", "build.gradle"), ("blocked", "build.gradle"),
             ("blocked/included", "settings.gradle.kts"), ("build/generated", "build.gradle"),
             ("bin/generated", "settings.gradle"), ("node_modules/package", "build.gradle"),
             (".metadata/hidden", "settings.gradle.kts"), ("deep/a/b/c/d", "build.gradle"),
             ("deep/a/b/c/e/f", "settings.gradle")]
    files = []
    for directory, name in names:
        path = root / directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("invalid Gradle deliberately unread by discovery" if directory == "opaque" else
                        ('java.io.File(rootDir, "forbidden-import").writeText("executed")\nerror("must not run")\n'
                         if name.endswith(".kts") else
                         "new File(rootDir, 'forbidden-import').text='executed'\nthrow new GradleException('must not run')\n"))
        files.append(path)
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    bundle_directory = workspace / "bundle"
    bundle_directory.mkdir()
    try:
        bundle, source = fixture.unlink_fixture_bundle(arguments, bundle_directory, default_root_modes=True, workspace_states=True)
    except subprocess.CalledProcessError as failure:
        (evidence / "fixture-compilation.log").write_bytes((failure.stdout or b"") + (failure.stderr or b""))
        raise
    (evidence / "UnlinkFixtureHandler.java").write_bytes(source.read_bytes())
    receipt["fixture_bundle_sha256"] = hashlib.sha256(bundle.read_bytes()).hexdigest()
    boundary = jdtls_probe()
    command = "groovy.gradle.defaultImportRoots"
    try:
        for configuration, exclusions in [("baseline", None),
                ("negation", ["**/blocked", "!**/blocked/included", "**/standalone"])]:
            settings = {"java": {"import": {"maven": {"enabled": False}, "gradle": {"enabled": False}}}}
            if exclusions is not None:
                settings["java"]["import"]["exclusions"] = exclusions
            data = workspace / ("data-" + configuration)
            for restart in [False, True]:
                journey = {"configuration": configuration, "restart": restart, "responses": []}
                receipt["journeys"].append(journey)
                client = boundary.JdtlsClient(arguments.launcher.resolve(), arguments.java_home.resolve(), root, data,
                                        deny_external_network=True, gradle_user_home=home)
                def execute(name, request):
                    return client.request("workspace/executeCommand", {"command": name, "arguments": [request]}, timeout=60)
                def membership():
                    return fixture.imported_resource_membership(client, root)
                def check_sources():
                    if source_hashes != {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}:
                        raise AssertionError("Discovery changed build-script bytes")
                    if any(root.rglob("forbidden-import")) or any(home.iterdir()):
                        raise AssertionError("Read-only discovery executed scripts or used Gradle worker/cache infrastructure")
                def project_metadata():
                    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in sorted(root.rglob("*")) if path.is_file()
                            and (path.name in {".project", ".classpath"}
                                 or ".settings" in path.parts and path.suffix == ".prefs")}
                def discover(label, candidates=files):
                    before = membership()
                    metadata_before = project_metadata()
                    oracle = execute("groovy.fixture.unlink.gradleDefaultRootsOracle", {"rootUri": root.as_uri()})
                    if oracle["automaticGradleImportEnabled"] is not False:
                        raise AssertionError("Discovery enabled automatic Gradle import")
                    if exclusions is not None and oracle["exclusions"] != exclusions:
                        raise AssertionError("SDK oracle uses different host exclusions")
                    request = {"rootUri": root.as_uri(), "candidateBuildFileUris": [p.as_uri() for p in candidates], "cachedOnly": True}
                    response = execute(command, request)
                    normalize = lambda values: {value.rstrip("/") for value in values}
                    supplied = {p.parent.as_uri().rstrip("/") for p in candidates}
                    expected = normalize(oracle["directories"]) & supplied
                    if (set(response) != {"version", "rootUri", "buildRootUris", "cachedOnly", "readOnly", "partial", "fullSynchronizationKnown"}
                            or response["version"] != 1 or response["rootUri"].rstrip("/") != root.as_uri()
                            or response["cachedOnly"] is not True or response["readOnly"] is not True
                            or response["partial"] is not False or response["fullSynchronizationKnown"] is not False
                            or normalize(response["buildRootUris"]) != expected
                            or len(response["buildRootUris"]) != len(expected)):
                        raise AssertionError("Candidate discovery differs from actual SDK scanner roots")
                    if membership() != before or project_metadata() != metadata_before:
                        raise AssertionError("Read-only query changed project membership or metadata bytes")
                    check_sources()
                    journey["responses"].append({"label": label, "request": request, "oracle": oracle,
                        "result": response, "membershipBefore": before, "membershipAfter": membership(),
                        "projectMetadataBefore": metadata_before, "projectMetadataAfter": project_metadata()})
                    receipt["checks"].append(configuration + (" restart: " if restart else " fresh: ") + label)
                    return response
                def reject(request, label):
                    before = membership()
                    metadata_before = project_metadata()
                    try:
                        execute(command, request)
                    except RuntimeError as failure:
                        message = str(failure)
                        if "Gradle default-root discovery failed" not in message or str(workspace) in message or "private-canary" in message:
                            raise AssertionError("Root discovery error is not sanitized") from failure
                        journey["responses"].append({"label": label, "error": message})
                    else:
                        raise AssertionError("Malformed default-root request was accepted")
                    if membership() != before or project_metadata() != metadata_before:
                        raise AssertionError("Rejected discovery changed imported projects or metadata bytes")
                    check_sources()
                    receipt["checks"].append(label + ": sanitized refusal without import")
                def cached_associations(stage, partial):
                    before = membership()
                    metadata = project_metadata()
                    request = {"rootUri": root.as_uri(), "cachedOnly": True}
                    response = execute("groovy.gradle.importedBuildRoots", request)
                    if (set(response) != {"version", "rootUri", "projects", "cachedOnly", "readOnly", "partial", "fullSynchronizationKnown"}
                            or type(response["version"]) is not int or response["version"] != 1
                            or response["rootUri"] != root.as_uri() + "/" or response["cachedOnly"] is not True
                            or response["readOnly"] is not True or response["partial"] is not partial
                            or response["fullSynchronizationKnown"] is not False or response["projects"]):
                        raise AssertionError("Unknown Buildship ownership became authoritative roots or false empty success")
                    if membership() != before or project_metadata() != metadata:
                        raise AssertionError("Pure cached association read changes resources or metadata")
                    check_sources()
                    journey["responses"].append({"label": stage, "request": request, "result": response})
                    receipt["checks"].append(stage + ": cached-only associations preserve empty worker home and data")
                def reject_cached_association(request, label):
                    before = membership()
                    metadata = project_metadata()
                    try:
                        response = execute("groovy.gradle.importedBuildRoots", request)
                    except RuntimeError as failure:
                        message = str(failure)
                        if ("Gradle imported project identities unavailable" not in message
                                or str(workspace) in message or "private-canary" in message):
                            raise AssertionError("Cached association error is not sanitized") from failure
                        journey["responses"].append({"label": label, "request": request, "error": message})
                    else:
                        journey["responses"].append({"label": label, "request": request, "unexpectedResult": response})
                        raise AssertionError("Malformed or aliased cached association authority was accepted: " + label)
                    if membership() != before or project_metadata() != metadata:
                        raise AssertionError("Rejected association query changes project data")
                    check_sources()
                    receipt["checks"].append(label + ": sanitized cached association refusal preserves metadata and empty worker home")
                def set_resource_state(directory, kind):
                    request = {"rootUri": root.as_uri(), "projectUri": (root / directory).as_uri(), "kind": kind}
                    result = execute("groovy.fixture.unlink.gradleDefaultRootsState", request)
                    if result.get("kind") != kind or result.get("gradleNature") is not (kind == "gradle") or result.get("open") is not True:
                        raise AssertionError("SDK fixture did not establish requested provider ownership")
                    journey.setdefault("resourceTransitions", []).append({"request": request, "result": result})
                def cancel_held_read(request, remove_input=False):
                    import queue
                    before = membership()
                    metadata_before = project_metadata()
                    for name in ["unlink-rule-started", "unlink-rule-release", "unlink-rule-finished"]:
                        marker = root.parent / name
                        if marker.exists():
                            marker.unlink()
                    paths = execute("groovy.fixture.unlink.hold", {"rootUri": root.as_uri()})
                    started, release, finished = (Path(paths[name]) for name in ["started", "release", "finished"])
                    detached = files[0].with_name(files[0].name + ".held-fixture")
                    if detached.exists():
                        raise AssertionError("Held fixture destination already exists")
                    try:
                        deadline = time.monotonic() + 5
                        while not started.exists() and time.monotonic() < deadline:
                            time.sleep(0.01)
                        if not started.exists():
                            raise AssertionError("Workspace fixture did not hold the SDK root rule")
                        request_id = client.next_id
                        client.next_id += 1
                        client.send({"jsonrpc": "2.0", "id": request_id, "method": "workspace/executeCommand",
                            "params": {"command": command, "arguments": [request]}})
                        deadline = time.monotonic() + 5
                        waiting = False
                        while time.monotonic() < deadline:
                            waiting = execute("groovy.fixture.unlink.stack", {"rootUri": root.as_uri()})["unlinkRequestWaiting"]
                            if waiting:
                                break
                            time.sleep(0.02)
                        if not waiting:
                            raise AssertionError("Default-root query did not enter its owned workspace read")
                        if remove_input:
                            files[0].rename(detached)
                        else:
                            client.notify("$/cancelRequest", {"id": request_id})
                        release.write_text("release\n")
                        deadline = time.monotonic() + 10
                        response = None
                        while time.monotonic() < deadline:
                            try:
                                message = client.messages.get(timeout=max(0.01, deadline - time.monotonic()))
                            except queue.Empty:
                                break
                            if isinstance(message, Exception):
                                raise message
                            if message.get("id") == request_id and "method" not in message:
                                response = message
                                break
                            if "id" in message and "method" in message:
                                if message["method"] != "workspace/configuration":
                                    raise AssertionError("Unexpected server request during held discovery")
                                client.send({"jsonrpc": "2.0", "id": message["id"], "result": [{} for _ in message.get("params", {}).get("items", [])]})
                            elif "method" in message:
                                with client.notifications_lock:
                                    client.notifications.append(message)
                        error = response.get("error", {}) if response else {}
                        expected_error = ("Gradle default-root discovery failed" in error.get("message", "") if remove_input else
                            error.get("code") == -32800 or "cancelled" in error.get("message", "").lower())
                        if not error or not expected_error:
                            raise AssertionError("Held discovery returned success or no expected failure")
                        if str(workspace) in json.dumps(error) or "private-canary" in json.dumps(error):
                            raise AssertionError("Cancellation exposed a host path")
                        journey["heldInputFailure" if remove_input else "heldReadCancellation"] = response
                    finally:
                        if detached.exists():
                            detached.rename(files[0])
                        release.write_text("release\n")
                        deadline = time.monotonic() + 5
                        while not finished.exists() and time.monotonic() < deadline:
                            time.sleep(0.01)
                        if not finished.exists():
                            raise AssertionError("Owned workspace fixture did not release the rule")
                    if membership() != before or project_metadata() != metadata_before:
                        raise AssertionError("Held read changed workspace resources or metadata bytes")
                    check_sources()
                    receipt["checks"].append("actual held workspace read " + ("missing-input refusal" if remove_input else "cancellation") + " preserves membership and build bytes")
                try:
                    initialized = client.request("initialize", {"processId": os.getpid(), "rootUri": root.as_uri(),
                        "workspaceFolders": [{"uri": root.as_uri(), "name": root.name}],
                        "capabilities": {"workspace": {"workspaceFolders": True, "configuration": True}},
                        "initializationOptions": {"bundles": [str(provider), str(bundle)], "settings": settings,
                                                  "projectConfigurations": [], "workspaceFolders": [root.as_uri()]}}, timeout=90)
                    if command not in initialized["capabilities"]["executeCommandProvider"]["commands"]:
                        raise AssertionError("Provider does not advertise default-root discovery")
                    client.notify("initialized", {})
                    if arguments.cached_build_associations_only:
                        cached_associations("before fixture ownership", restart)
                        cached_alias = root.parent / "cached-authority-alias"
                        cached_alias.symlink_to(root, target_is_directory=True)
                        cached_request = {"rootUri": root.as_uri(), "cachedOnly": True}
                        try:
                            for request, case in [({**cached_request, "rootUri": cached_alias.as_uri()}, "aliased initialized root"),
                                    ({**cached_request, "rootUri": root.as_uri() + "/../" + root.name}, "unnormalized initialized root"),
                                    ({**cached_request, "rootUri": root.parent.as_uri()}, "foreign uninitialized root"),
                                    ({**cached_request, "rootUri": "https://example.invalid/private-canary"}, "nonfile root"),
                                    ({**cached_request, "rootUri": "x" * 4097}, "root URI budget"),
                                    ({**cached_request, "cachedOnly": False}, "noncached association read"),
                                    ({**cached_request, "extra": True}, "unknown association key"),
                                    ({"rootUri": root.as_uri()}, "missing cached-only authority")]:
                                reject_cached_association(request, case)
                        finally:
                            cached_alias.unlink()
                    response = discover("four descriptor names, nested/default exclusions, depth and unread script text")
                    if not restart and configuration == "baseline":
                        expected = {(root / name).as_uri() for name in ["aggregate", "standalone", "opaque", "blocked", "deep/a/b/c/d"]}
                        if {p.rstrip("/") for p in response["buildRootUris"]} != expected:
                            raise AssertionError("Baseline does not exercise actual SDK default roots and exact depth")
                    discover("explicit empty candidate set", [])
                    for directory, kind in [("aggregate/child", "gradle"), ("standalone", "plain")]:
                        set_resource_state(directory, kind)
                    if arguments.cached_build_associations_only:
                        cached_associations("unknown retained Gradle configuration", True)
                    foreign_owned = discover("retained child is not a new build root; foreign provider excludes its root")
                    if any(value.rstrip("/") in {(root / "aggregate/child").as_uri(), (root / "standalone").as_uri()}
                           for value in foreign_owned["buildRootUris"]):
                        raise AssertionError("A retained module or foreign project was promoted to an independent Gradle build")
                    set_resource_state("standalone", "gradle")
                    restored = discover("restored Gradle ownership respects current host exclusion policy")
                    if configuration == "baseline" and (root / "standalone").as_uri() not in {value.rstrip("/") for value in restored["buildRootUris"]}:
                        raise AssertionError("Restored Gradle resource did not recover default-root discovery")
                    request = {"rootUri": root.as_uri(), "candidateBuildFileUris": [p.as_uri() for p in files], "cachedOnly": True}
                    if configuration == "baseline" and not restart:
                        cancel_held_read(request)
                        discover("recovery after cancelled SDK read")
                        cancel_held_read(request, remove_input=True)
                        discover("recovery after held candidate removal and restoration")
                    foreign = workspace / "private-canary/build.gradle"
                    foreign.parent.mkdir(exist_ok=True); foreign.write_text("private-canary")
                    alias = root / "alias"
                    alias.symlink_to(root / "standalone", target_is_directory=True)
                    linked = root / "linked/build.gradle"
                    linked.parent.mkdir(exist_ok=True); linked.symlink_to(root / "aggregate/build.gradle.kts")
                    try:
                        invalid = [({**request, "cachedOnly": False}, "noncached query"),
                            ({**request, "extra": True}, "unknown key"),
                            ({k: v for k, v in request.items() if k != "cachedOnly"}, "missing key"),
                            ({**request, "rootUri": (root / "aggregate").as_uri()}, "uninitialized nested authority"),
                            ({**request, "candidateBuildFileUris": [files[0].as_uri()] * 2}, "duplicate candidate"),
                            ({**request, "candidateBuildFileUris": [foreign.as_uri()]}, "foreign candidate"),
                            ({**request, "candidateBuildFileUris": [(alias / "settings.gradle.kts").as_uri()]}, "aliased candidate"),
                            ({**request, "candidateBuildFileUris": [linked.as_uri()]}, "symlink descriptor"),
                            ({**request, "candidateBuildFileUris": [(root / "missing/build.gradle").as_uri()]}, "missing descriptor"),
                            ({**request, "candidateBuildFileUris": [(root / "aggregate").as_uri()]}, "directory candidate"),
                            ({**request, "candidateBuildFileUris": [files[0].as_uri()] * 129}, "candidate count budget"),
                            ({**request, "candidateBuildFileUris": "not-a-list"}, "candidate type"),
                            ({**request, "candidateBuildFileUris": ["file:///" + "x" * 4097]}, "URI size budget"),
                            ({**request, "candidateBuildFileUris": ["https://example.invalid/private-canary/build.gradle"]}, "nonfile URI"),
                            ({**request, "candidateBuildFileUris": [(root / "aggregate/child/build.gradle").as_uri()]}, "incomplete ancestor coverage")]
                        for value, label in invalid:
                            reject(value, label)
                    finally:
                        alias.unlink(); linked.unlink()
                    check_sources()
                    receipt["checks"].append("build bytes, disabled import and empty worker home preserved")
                finally:
                    logs = evidence / (configuration + ("-restart" if restart else "-fresh"))
                    logs.mkdir(exist_ok=True)
                    try:
                        client.stderr.seek(0)
                        (logs / "stderr-before-shutdown.log").write_bytes(client.stderr.read())
                    finally:
                        client.close()
                    log = data / ".metadata/.log"
                    if log.exists():
                        shutil.copy2(log, logs / "jdtls.log")
                    (logs / "notifications.json").write_text(json.dumps(list(client.notifications), indent=2) + "\n")
                    arguments.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        direct = workspace / "direct"
        direct.mkdir()
        source = direct / "DefaultRootsCancellation.java"
        source.write_text('''import java.util.List;
import org.eclipse.core.runtime.NullProgressMonitor;
public final class DefaultRootsCancellation {
 public static void main(String[] arguments) throws Exception {
  var monitor = new NullProgressMonitor(); monitor.setCanceled(true);
  try { new dev.made.groovy.GradleImportHandler().executeCommand("groovy.gradle.defaultImportRoots", List.of(), monitor);
   throw new AssertionError("Cancelled discovery succeeded");
  } catch (IllegalStateException failure) {
   if (!failure.getMessage().equals("Gradle default-root discovery cancelled; no projects were imported. Refresh before retrying.") || failure.getCause() != null) throw new AssertionError("Cancellation not sanitized", failure);
  }
  System.out.println("Pre-cancellation refused before SDK service access with no cause chain");
 }
}
''')
        direct_receipt = {"checks": []}
        direct_evidence = evidence / "cancellation"
        direct_evidence.mkdir()
        fixture.run_direct_maven_check(argparse.Namespace(**{**vars(arguments), "receipt": direct_evidence / "result.json"}),
                                       direct, source, direct_receipt, "DefaultRootsCancellation")
        receipt["directCancellation"] = direct_receipt
        receipt["checks"].extend(direct_receipt["checks"])
        receipt["status"] = "passed"
    except Exception:
        receipt["status"] = "failed"
        raise
    finally:
        receipt["sourceHashes"] = source_hashes
        if any(home.iterdir()):
            environment = os.environ.copy()
            environment["JAVA_HOME"] = str(arguments.gradle_java_home.resolve())
            environment["GRADLE_USER_HOME"] = str(home)
            stopped = subprocess.run(["/usr/bin/sandbox-exec", "-p", boundary.GRADLE_DENIED_NETWORK_PROFILE,
                str(arguments.gradle_home.resolve() / "bin/gradle"), "--offline", "--stop",
                "--gradle-user-home", str(home)], cwd=root, env=environment,
                capture_output=True, timeout=90)
            (evidence / "owned-worker-stop.log").write_bytes(stopped.stdout + stopped.stderr)
            receipt["unexpectedOwnedWorkerCleanupExit"] = stopped.returncode
            receipt["status"] = "failed"
            arguments.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
            if stopped.returncode:
                raise RuntimeError("Unexpected isolated Gradle worker cleanup failed; inspect owned-worker-stop.log")
            raise AssertionError("Discovery unexpectedly used isolated Gradle worker infrastructure")
        arguments.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"checks": len(receipt["checks"]), "journeys": len(receipt["journeys"]), "status": receipt["status"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Opt-in real JDTLS Gradle delegate qualification; cached fixture only")
    for name in ["launcher", "java-home", "gradle-java-home", "gradle-home", "provider", "gradle-user-home", "receipt"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--default-roots-only", action="store_true",
                        help="Qualify read-only default-root candidates against the pinned SDK detector without Gradle builds")
    parser.add_argument("--generated-source-declarations-only", action="store_true",
                        help="Qualify the actual Gradle compiler output declaration projection without JDTLS or build tasks")
    parser.add_argument("--enum-declarations-only", action="store_true",
                        help="Qualify the actual provider reflection boundary without launching Gradle/JDTLS")
    parser.add_argument("--import-selection-only", action="store_true",
                        help="Qualify pinned Gradle project link/unlink/exclusion and restart semantics")
    parser.add_argument("--resolution-graph-only", action="store_true",
                        help="Qualify an explicit offline configuration resolution graph without artifact/output Tasks")
    parser.add_argument("--source-outputs-only", action="store_true",
                        help="Qualify authoritative source-set output declarations without producing outputs")
    parser.add_argument("--module-state-snapshots-only", action="store_true",
                        help="Qualify read-only SDK module states during fresh closed-module journeys")
    parser.add_argument("--provider-fresh-module-states-only", action="store_true",
                        help="Use production creation with a live namespace conflict through restart")
    parser.add_argument("--fresh-closed-modules-only", action="store_true",
                        help="Qualify SDK model names and a closed resource before its first Java import")
    parser.add_argument("--provider-module-states-only", action="store_true",
                        help="Use production module-state phase in the opt-in closed-module journeys")
    parser.add_argument("--closed-modules-only", action="store_true",
                        help="Qualify closed module synchronization; requires --import-selection-only --required-library-only")
    parser.add_argument("--required-library-only", action="store_true",
                        help="Observe required-library classpaths after exact unlink and restart in cached/missing-output cells")
    parser.add_argument("--mutation-failures-only", action="store_true",
                        help="Add malformed/foreign requests, late cancellation, repaired failure and orphan qualification")
    parser.add_argument("--guarded-phase-only", action="store_true",
                        help="Qualify explicit guarded Buildship phase without automatic imports")
    parser.add_argument("--guarded-unlink-only", action="store_true",
                        help="Qualify guarded source-preserving composite unlink and restart")
    parser.add_argument("--resource-membership-only", action="store_true",
                        help="Qualify existing non-Java resource enumeration, unlink and repair")
    parser.add_argument("--startup-selection-only", action="store_true",
                        help="Qualify independent import-enabled settings on mixed cached builds")
    parser.add_argument("--maven-repository", type=Path, help="Existing cached Maven repository for mixed startup mode")
    parser.add_argument("--linked-root-phase-only", action="store_true",
                        help="Qualify nested guarded imports through existing workspace folder events")
    parser.add_argument("--provider-identity-only", action="store_true",
                        help="Qualify cached imported Maven/Gradle Java and non-Java identities")
    parser.add_argument("--unlink-ownership-only", action="store_true",
                        help="Qualify actual dual build natures before unlink and at a held workspace rule")
    parser.add_argument("--startup-configurations-only", action="store_true",
                        help="Qualify explicit empty startup configurations against retained SDK metadata and cached repair")
    parser.add_argument("--single-provider-configurations-only", action="store_true",
                        help="Qualify explicit uncontrolled-provider and plain Java configuration lists; requires startup configuration mode")
    parser.add_argument("--workspace-scoped-module-details-only", action="store_true",
                        help="Qualify v4 registered-module descriptor/directory freshness; requires scoped details mode")
    parser.add_argument("--workspace-scoped-details-only", action="store_true",
                        help="Qualify scoped nested detail model through existing cached policy fixtures")
    parser.add_argument("--workspace-scoped-reselection-only", action="store_true",
                        help="Qualify independent nonempty Gradle selection transitions; requires cached empty-selection mode")
    parser.add_argument("--workspace-scoped-empty-selection-only", action="store_true",
                        help="Qualify cached metadata-only empty Gradle selection; requires scoped-default cached-association mode")
    parser.add_argument("--cached-build-associations-only", action="store_true",
                        help="Qualify cached provider build-root associations; requires default-roots or scoped-default mode")
    parser.add_argument("--workspace-scoped-default-policy-only", action="store_true",
                        help="Qualify cached default-root discovery joined to existing scoped policy journeys")
    parser.add_argument("--workspace-scoped-policy-only", action="store_true",
                        help="Qualify scoped module/root policy sequencing through the existing cached SDK fixture")
    parser.add_argument("--workspace-scoped-module-only", action="store_true",
                        help="Qualify scoped module read/close/restore through the existing scoped root fixture")
    parser.add_argument("--workspace-scoped-root-only", action="store_true",
                        help="Qualify scoped root read and retained mutation; requires scoped import startup mode")
    parser.add_argument("--workspace-scoped-import-only", action="store_true",
                        help="Qualify nested Gradle import under an explicit initialized workspace without adding folders")
    parser.add_argument("--startup-metadata-only", action="store_true",
                        help="Qualify retained Eclipse metadata startup exclusion and explicit Gradle policy")
    parser.add_argument("--root-state-only", action="store_true",
                        help="Qualify closed Gradle root resources and cold recovery through existing SDK fixtures")
    parser.add_argument("--provider-root-read-only", action="store_true",
                        help="Qualify production root-state reads during the SDK root lifecycle")
    parser.add_argument("--provider-root-state-only", action="store_true",
                        help="Qualify retained production root mutation with a Java Gradle root")
    arguments = parser.parse_args()
    if arguments.default_roots_only and any(value is True for name, value in vars(arguments).items() if name not in {"default_roots_only", "cached_build_associations_only"}):
        parser.error("--default-roots-only requires no other specialized mode")

    if arguments.workspace_scoped_module_details_only and not arguments.workspace_scoped_details_only:
        parser.error("--workspace-scoped-module-details-only requires --workspace-scoped-details-only")

    if arguments.workspace_scoped_details_only and not (arguments.workspace_scoped_policy_only and arguments.cached_build_associations_only):
        parser.error("--workspace-scoped-details-only requires --workspace-scoped-policy-only --cached-build-associations-only")

    if arguments.workspace_scoped_reselection_only and not arguments.workspace_scoped_empty_selection_only:
        parser.error("--workspace-scoped-reselection-only requires --workspace-scoped-empty-selection-only")

    if arguments.workspace_scoped_empty_selection_only and not (arguments.workspace_scoped_default_policy_only and arguments.cached_build_associations_only):
        parser.error("--workspace-scoped-empty-selection-only requires --workspace-scoped-default-policy-only --cached-build-associations-only")

    if arguments.cached_build_associations_only and not (arguments.default_roots_only or arguments.workspace_scoped_default_policy_only):
        parser.error("--cached-build-associations-only requires --default-roots-only or --workspace-scoped-default-policy-only")

    if arguments.workspace_scoped_default_policy_only and not arguments.workspace_scoped_policy_only:
        parser.error("--workspace-scoped-default-policy-only requires --workspace-scoped-policy-only")

    if arguments.workspace_scoped_policy_only and not arguments.workspace_scoped_module_only:
        parser.error("--workspace-scoped-policy-only requires --workspace-scoped-module-only")

    if arguments.workspace_scoped_module_only and not arguments.workspace_scoped_root_only:
        parser.error("--workspace-scoped-module-only requires --workspace-scoped-root-only")

    if arguments.workspace_scoped_root_only and not arguments.workspace_scoped_import_only:
        parser.error("--workspace-scoped-root-only requires --workspace-scoped-import-only")

    if arguments.workspace_scoped_import_only and (not arguments.startup_selection_only or any(
            value is True for name, value in vars(arguments).items()
            if name not in {"workspace_scoped_import_only", "workspace_scoped_root_only", "workspace_scoped_module_only", "workspace_scoped_policy_only", "workspace_scoped_default_policy_only", "cached_build_associations_only", "workspace_scoped_empty_selection_only", "workspace_scoped_reselection_only", "workspace_scoped_details_only", "workspace_scoped_module_details_only", "startup_selection_only"})):
        parser.error("--workspace-scoped-import-only requires --startup-selection-only without other specialized modes")

    if arguments.single_provider_configurations_only and not arguments.startup_configurations_only:
        parser.error("--single-provider-configurations-only requires --startup-configurations-only")

    if arguments.startup_configurations_only and (not arguments.startup_selection_only or any([
            arguments.root_state_only, arguments.startup_metadata_only, arguments.linked_root_phase_only,
            arguments.provider_identity_only, arguments.unlink_ownership_only])):
        parser.error("--startup-configurations-only requires --startup-selection-only without other specialized modes")

    if arguments.provider_root_state_only and not arguments.provider_root_read_only:
        parser.error("--provider-root-state-only requires --provider-root-read-only")

    if arguments.provider_root_read_only and not arguments.root_state_only:
        parser.error("--provider-root-read-only requires --root-state-only")

    if arguments.root_state_only and (not arguments.startup_selection_only or any([
            arguments.startup_metadata_only, arguments.linked_root_phase_only, arguments.provider_identity_only, arguments.unlink_ownership_only])):
        parser.error("--root-state-only requires --startup-selection-only without other specialized modes")

    if arguments.startup_metadata_only and (not arguments.startup_selection_only or any([
            arguments.linked_root_phase_only, arguments.provider_identity_only, arguments.unlink_ownership_only])):
        parser.error("--startup-metadata-only requires --startup-selection-only without other specialized modes")

    if arguments.module_state_snapshots_only and not arguments.provider_fresh_module_states_only:
        parser.error("--module-state-snapshots-only requires --provider-fresh-module-states-only")
    if arguments.provider_fresh_module_states_only and not arguments.fresh_closed_modules_only:
        parser.error("--provider-fresh-module-states-only requires --fresh-closed-modules-only")
    if arguments.fresh_closed_modules_only and not arguments.provider_module_states_only:
        parser.error("--fresh-closed-modules-only requires --provider-module-states-only")
    if arguments.provider_module_states_only and not arguments.closed_modules_only:
        parser.error("--provider-module-states-only requires --closed-modules-only")
    if arguments.closed_modules_only and not (arguments.import_selection_only and arguments.required_library_only):
        parser.error("--closed-modules-only requires --import-selection-only --required-library-only")
    if arguments.generated_source_declarations_only and any([
            arguments.enum_declarations_only, arguments.import_selection_only, arguments.resolution_graph_only,
            arguments.source_outputs_only, arguments.required_library_only, arguments.mutation_failures_only,
            arguments.guarded_phase_only, arguments.guarded_unlink_only, arguments.resource_membership_only,
            arguments.startup_selection_only, arguments.linked_root_phase_only, arguments.provider_identity_only,
            arguments.unlink_ownership_only]):
        parser.error("--generated-source-declarations-only must be used without other specialized modes")
    if arguments.enum_declarations_only and any([arguments.import_selection_only, arguments.resolution_graph_only,
            arguments.source_outputs_only, arguments.required_library_only, arguments.mutation_failures_only,
            arguments.guarded_phase_only, arguments.guarded_unlink_only, arguments.resource_membership_only,
            arguments.startup_selection_only, arguments.linked_root_phase_only, arguments.provider_identity_only,
            arguments.unlink_ownership_only]):
        parser.error("--enum-declarations-only must be used without other specialized modes")
    if arguments.resolution_graph_only and (not arguments.import_selection_only or any([
            arguments.source_outputs_only, arguments.required_library_only, arguments.mutation_failures_only,
            arguments.guarded_phase_only, arguments.guarded_unlink_only, arguments.resource_membership_only,
            arguments.startup_selection_only, arguments.linked_root_phase_only, arguments.provider_identity_only,
            arguments.unlink_ownership_only])):
        parser.error("--resolution-graph-only requires --import-selection-only without other specialized modes")
    if arguments.source_outputs_only and (not arguments.import_selection_only or any([
            arguments.required_library_only, arguments.mutation_failures_only, arguments.guarded_phase_only,
            arguments.guarded_unlink_only, arguments.resource_membership_only, arguments.startup_selection_only,
            arguments.linked_root_phase_only, arguments.provider_identity_only, arguments.unlink_ownership_only])):
        parser.error("--source-outputs-only requires --import-selection-only without other specialized modes")
    if arguments.required_library_only and (not arguments.import_selection_only or any([
            arguments.mutation_failures_only, arguments.guarded_phase_only, arguments.guarded_unlink_only,
            arguments.resource_membership_only, arguments.startup_selection_only, arguments.linked_root_phase_only,
            arguments.provider_identity_only, arguments.unlink_ownership_only])):
        parser.error("--required-library-only requires --import-selection-only without other specialized modes")
    if arguments.unlink_ownership_only and (not arguments.startup_selection_only or arguments.linked_root_phase_only or arguments.provider_identity_only):
        parser.error("--unlink-ownership-only requires --startup-selection-only without other specialized modes")
    if arguments.provider_identity_only and (not arguments.startup_selection_only or arguments.linked_root_phase_only):
        parser.error("--provider-identity-only requires --startup-selection-only without --linked-root-phase-only")
    if arguments.linked_root_phase_only and not arguments.startup_selection_only:
        parser.error("--linked-root-phase-only requires --startup-selection-only")
    if arguments.startup_selection_only and (arguments.maven_repository is None or arguments.import_selection_only
            or arguments.resource_membership_only or arguments.guarded_unlink_only
            or arguments.guarded_phase_only or arguments.mutation_failures_only):
        parser.error("--startup-selection-only requires --maven-repository without other specialized modes")
    if arguments.resource_membership_only and (not arguments.import_selection_only or arguments.guarded_unlink_only
            or arguments.guarded_phase_only or arguments.mutation_failures_only):
        parser.error("--resource-membership-only requires --import-selection-only without other specialized modes")
    if arguments.guarded_unlink_only and (not arguments.import_selection_only or arguments.guarded_phase_only or arguments.mutation_failures_only):
        parser.error("--guarded-unlink-only requires --import-selection-only without other specialized modes")
    if arguments.guarded_phase_only and (not arguments.import_selection_only or arguments.mutation_failures_only):
        parser.error("--guarded-phase-only requires --import-selection-only without --mutation-failures-only")
    if arguments.mutation_failures_only and not arguments.import_selection_only:
        parser.error("--mutation-failures-only requires --import-selection-only")
    if arguments.default_roots_only:
        probe_default_roots(arguments)
    elif arguments.generated_source_declarations_only:
        probe_generated_source_declarations(arguments)
    elif arguments.enum_declarations_only:
        probe_enum_declarations(arguments)
    elif arguments.startup_selection_only:
        probe_startup_controls(arguments)
    elif arguments.import_selection_only:
        probe_import_selection(arguments)
    else:
        probe(arguments)
