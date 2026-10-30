#!/usr/bin/env python3

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import zipfile
from xml.sax.saxutils import escape
from urllib.parse import unquote, urlparse


REPOSITORY = Path(__file__).resolve().parent.parent


def client_class():
    specification = importlib.util.spec_from_file_location(
        "groovy_jdtls_probe", REPOSITORY / "script/test-groovy-jdtls-provider.py"
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Cannot load existing JDTLS fixture client")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.JdtlsClient


def discover(client, root, details=False, parameters=False):
    return client.request("workspace/executeCommand", {
        "command": "groovy.maven.goalDetails" if parameters else "groovy.maven.projectDetails" if details else "groovy.maven.projects",
        "arguments": [{"rootUri": root.as_uri(), "cachedOnly": True}],
    }, timeout=60)


def expect_failure(client, root, cached=True, forbidden=None, parameters=False):
    try:
        client.request("workspace/executeCommand", {
            "command": "groovy.maven.goalDetails" if parameters else "groovy.maven.projects",
            "arguments": [{"rootUri": root.as_uri(), "cachedOnly": cached}],
        }, timeout=60)
    except RuntimeError as failure:
        message = str(failure)
        if "Maven" not in message or str(root) in message or forbidden and forbidden in message:
            raise AssertionError("Maven failure category is not sanitized") from failure
        return
    raise AssertionError("Invalid Maven request succeeded")


def cache_copy(source, destination):
    if Path(source).suffix == ".jar":
        os.link(source, destination)
        return destination
    return shutil.copy2(source, destination)


def run_direct_maven_check(arguments, workspace, source, receipt, entrypoint):
    plugins = arguments.launcher.resolve().parent.parent / "plugins"
    runtime = workspace / "runtime"
    runtime.mkdir()
    with zipfile.ZipFile(plugins / "org.eclipse.m2e.maven.runtime_3.9.1600.20260522-0510.jar") as archive:
        for name in archive.namelist():
            if name.startswith("jars/") and name.endswith(".jar"):
                (runtime / Path(name).name).write_bytes(archive.read(name))
    classpath = os.pathsep.join(str(path) for path in [arguments.provider.resolve(), *sorted(plugins.glob("*.jar")), *sorted(runtime.glob("*.jar"))])
    (arguments.receipt.parent / source.name).write_bytes(source.read_bytes())
    compilation = subprocess.run([str(arguments.java_home / "bin/javac"), "-J-Xmx128m", "--release", "17",
                                  "-Xlint:all", "-Werror", "-classpath", classpath, "-d", str(workspace), str(source)],
                                 capture_output=True, timeout=60)
    (arguments.receipt.parent / "compilation.log").write_bytes(compilation.stdout + compilation.stderr)
    receipt["compilation_exit"] = compilation.returncode
    arguments.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    compilation.check_returncode()
    execution = subprocess.run([str(arguments.java_home / "bin/java"), "-Xmx64m", "-classpath",
                                str(workspace) + os.pathsep + classpath, entrypoint],
                               capture_output=True, timeout=60)
    (arguments.receipt.parent / "checks.log").write_bytes(execution.stdout + execution.stderr)
    receipt["execution_exit"] = execution.returncode
    receipt["checks"] = execution.stdout.decode().splitlines()
    arguments.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    execution.check_returncode()


def probe_pom_masks(arguments):
    provider = arguments.provider.resolve()
    if hashlib.sha256(provider.read_bytes()).hexdigest() != arguments.sha256:
        raise AssertionError("Provider binary differs from the source build")
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"provider_sha256": arguments.sha256, "checks": [], "limits": [
        "Direct production matcher checks; no M2E import or physical Windows qualification",
        "Fixed reference cases use the archived IDEA 2f55a79617872c5774aa993087f353b8a610ec5e oracle",
    ]}
    cases = [
        [['*'], '/fixture/module/pom.xml', True],
        [['*'], '/fixture/pom.xml', True],
        [[], '/fixture/pom.xml', False],
        [['pom.xml'], '/fixture/pom.xml', False],
        [['/fixture/pom.xml'], '/fixture/pom.xml', True],
        [['/fixture/POM.xml'], '/fixture/pom.xml', False],
        [['/fixture/pom.xml'], '/fixture/module/pom.xml', False],
        [['*/pom.xml'], '/fixture/deep/module/pom.xml', True],
        [['/fixture/m?dule/pom.xml'], '/fixture/module/pom.xml', True],
        [['/fixture/m?dule/pom.xml'], '/fixture/mdule/pom.xml', False],
        [['/fixture/m?dule/pom.xml'], '/fixture/moddule/pom.xml', False],
        [['*/[ab]/pom.xml'], '/fixture/[ab]/pom.xml', True],
        [['*/[ab]/pom.xml'], '/fixture/a/pom.xml', False],
        [['*/{api,app}/pom.xml'], '/fixture/{api,app}/pom.xml', True],
        [['*/{api,app}/pom.xml'], '/fixture/api/pom.xml', False],
        [['*/a+b/pom.xml'], '/fixture/a+b/pom.xml', True],
        [['*/pom.xml'], '/fixture/pomXxml', False],
        [['*/a(b)/pom.xml'], '/fixture/a(b)/pom.xml', True],
        [['*/a|b/pom.xml'], '/fixture/a|b/pom.xml', True],
        [['*/a$/pom.xml'], '/fixture/a$/pom.xml', True],
        [['*/^a/pom.xml'], '/fixture/^a/pom.xml', True],
        [['!*/pom.xml'], '/fixture/pom.xml', False],
        [['*/other.xml', '*/pom.xml'], '/fixture/pom.xml', True],
        [['/fixture/?/pom.xml'], '/fixture/λ/pom.xml', True],
        [['\\*/pom.xml\\'], '/fixture/pom.xml', True],
        [['*/x\\\\y/pom.xml'], '/fixture/x\\y/pom.xml', True],
        [['*/x\\y/pom.xml'], '/fixture/x\\y/pom.xml', True],
        [['/fixture/?/pom.xml'], '/fixture/😀/pom.xml', True],
        [['/fixture/é/pom.xml'], '/fixture/é/pom.xml', False],
        [['\\'], '/fixture/pom.xml', False],
        [['\\\\'], '/fixture/pom.xml', False],
        [['*/[ab]/pom.xml'], '/fixture/[ab]/pom.xml', True],
        [['*/x|y/pom.xml'], '/fixture/x|y/pom.xml', True],
    ]
    with tempfile.TemporaryDirectory(prefix="groovy-maven-mask-contract-") as directory:
        workspace = Path(directory)
        cases_file = workspace / "cases.json"
        cases_file.write_text(json.dumps(cases, ensure_ascii=False))
        (arguments.receipt.parent / "cases.json").write_bytes(cases_file.read_bytes())
        source = workspace / "MavenPomMaskCheck.java"
        source.write_text(r'''package dev.made.groovy;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import com.google.gson.JsonArray;
import com.google.gson.JsonParser;
import org.eclipse.core.runtime.NullProgressMonitor;
import org.eclipse.core.runtime.OperationCanceledException;
public final class MavenPomMaskCheck {
    private interface Check { void run(); }
    private static Map<String, Object> contract(List<String> patterns) {
        return Map.of("version", 1, "syntax", "idea-mask-v1", "pathFormat", "portable-absolute-v1", "patterns", patterns);
    }
    private static void rejected(Check check) {
        try { check.run(); throw new AssertionError("Invalid mask accepted"); }
        catch (IllegalArgumentException expected) {
            if (expected.getMessage() != null) { throw new AssertionError("Private mask error leaked"); }
        }
    }
    private static final class CancelDuringLoop extends NullProgressMonitor {
        private int polls;
        @Override public boolean isCanceled() { return ++polls >= 3; }
    }
    public static void main(String[] arguments) throws Exception {
        JsonArray cases = JsonParser.parseString(Files.readString(Path.of(CASE_PATH))).getAsJsonArray();
        int count = 0;
        for (var value : cases) {
            var row = value.getAsJsonArray();
            List<String> patterns = new ArrayList<>();
            row.get(0).getAsJsonArray().forEach(pattern -> patterns.add(pattern.getAsString()));
            if (MavenImportHandler.PomMasks.fromRequest(contract(patterns)).matches(row.get(1).getAsString(), new NullProgressMonitor())
                    != row.get(2).getAsBoolean()) { throw new AssertionError("Pinned reference case " + count + " differs"); }
            count++;
        }
        System.out.println(count + " pinned whole-path Unicode and literal-punctuation reference cases");
        for (String pattern : List.of("\\", "\\\\")) {
            if (MavenImportHandler.PomMasks.fromRequest(contract(List.of(pattern))).matches("/fixture/pom.xml", new NullProgressMonitor())) {
                throw new AssertionError("Backslash-only mask became match-all");
            }
        }
        System.out.println("both empty backslash translations cannot become match-all");
        List<String> exact = new ArrayList<>();
        for (int index = 0; index < 128; index++) { exact.add("*" + index); }
        MavenImportHandler.PomMasks.fromRequest(contract(exact)); exact.add("overflow");
        rejected(() -> MavenImportHandler.PomMasks.fromRequest(contract(exact)));
        for (String text : List.of("", "x\n", "x\u0085", "x\u2028", "x\u2029", "\ud800", "\udc00", "x".repeat(4097), "λ".repeat(2049))) {
            rejected(() -> MavenImportHandler.PomMasks.fromRequest(contract(List.of(text))));
        }
        rejected(() -> MavenImportHandler.PomMasks.fromRequest(contract(List.of("*", "*"))));
        List<String> total = new ArrayList<>();
        for (int index = 0; index < 16; index++) { total.add(String.format("%02d", index) + "x".repeat(4094)); }
        MavenImportHandler.PomMasks.fromRequest(contract(total)); total.add("y".repeat(4096));
        rejected(() -> MavenImportHandler.PomMasks.fromRequest(contract(total)));
        System.out.println("raw uniqueness, UTF-8, count, aggregate and surrogate/control bounds");
        for (String path : List.of("/fixture/pom.xml", "/fixture/é/pom.xml", "/fixture/x\\y/pom.xml")) {
            MavenImportHandler.PomMasks.validatePath(path, false);
        }
        for (String path : List.of("C:/fixture/pom.xml", "//server/share/pom.xml")) {
            MavenImportHandler.PomMasks.validatePath(path, true);
        }
        for (String path : List.of("fixture/pom.xml", "//fixture/pom.xml", "/fixture//pom.xml", "/fixture/../pom.xml", "/fixture/pom.xml/", "/fixture/pom\n.xml")) {
            rejected(() -> MavenImportHandler.PomMasks.validatePath(path, false));
        }
        for (String path : List.of("C:\\fixture\\pom.xml", "/fixture/pom.xml", "//?/C:/pom.xml", "//server/pom.xml", "C:/fixture./pom.xml", "C:/fixture /pom.xml", "C:/fixture/pom.xml:stream")) {
            rejected(() -> MavenImportHandler.PomMasks.validatePath(path, true));
        }
        System.out.println("explicit portable POSIX/Windows path forms without physical Windows claim");
        var cancelled = new NullProgressMonitor(); cancelled.setCanceled(true);
        try {
            MavenImportHandler.PomMasks.fromRequest(contract(List.of("*"))).matches("/fixture/pom.xml", cancelled);
            throw new AssertionError("Pre-cancellation ignored");
        } catch (OperationCanceledException expected) { }
        var during = new CancelDuringLoop();
        try {
            MavenImportHandler.PomMasks.fromRequest(contract(List.of("*" + "x".repeat(4095)))).matches("/fixture/pom.xml", during);
            throw new AssertionError("In-flight cancellation ignored");
        } catch (OperationCanceledException expected) { }
        if (during.polls != 3) { throw new AssertionError("Cancellation polling differs"); }
        System.out.println("pre- and in-flight monitor cancellation");
        var bounded = MavenImportHandler.PomMasks.fromRequest(contract(List.of("*a".repeat(2048))));
        String longPath = "/" + "b".repeat(4095);
        rejected(() -> { for (int index = 0; index < 5; index++) { bounded.matches(longPath, new NullProgressMonitor()); } });
        System.out.println("adversarial aggregate matching work stops at the request cell budget");
    }
}
''')
        text = source.read_text().replace("CASE_PATH", json.dumps(str(cases_file)))
        source.write_text(text)
        run_direct_maven_check(arguments, workspace, source, receipt, "dev.made.groovy.MavenPomMaskCheck")
    print(f"Passed {len(receipt['checks'])} direct Maven mask contract checks")


def probe_cached_ancestry(arguments):
    provider = arguments.provider.resolve()
    if hashlib.sha256(provider.read_bytes()).hexdigest() != arguments.sha256:
        raise AssertionError("Provider binary differs from the source build")
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"provider_sha256": arguments.sha256, "checks": [], "limits": [
        "Direct production cached-queue method only; no M2E import, discovery or native workflow",
        "Does not qualify full reactor order, model freshness, remote delivery or release",
    ]}
    with tempfile.TemporaryDirectory(prefix="groovy-maven-cached-ancestry-") as directory:
        workspace = Path(directory)
        source = workspace / "MavenCachedAncestryCheck.java"
        source.write_text(r'''import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.core.runtime.NullProgressMonitor;
public final class MavenCachedAncestryCheck {
    private static final Class<?> READER;
    private static final Method APPEND;
    static {
        try {
            READER = Class.forName("dev.made.groovy.MavenModelHandler$Reader");
            APPEND = READER.getDeclaredMethod("appendCachedDescendants", Path.class, Set.class,
                    ArrayDeque.class, Set.class, IProgressMonitor.class);
            APPEND.setAccessible(true);
        } catch (ReflectiveOperationException failure) { throw new ExceptionInInitializerError(failure); }
    }
    private static Object reader() throws Exception {
        var constructor = READER.getDeclaredConstructor(); constructor.setAccessible(true);
        return constructor.newInstance();
    }
    private static void append(Path pom, Set<Path> cached, ArrayDeque<Path> pending, Set<Path> visited,
            IProgressMonitor monitor) throws Exception {
        APPEND.invoke(reader(), pom, cached, pending, visited, monitor);
    }
    private interface Check { void run() throws Exception; }
    private static void rejected(Check check) throws Exception {
        try { check.run(); throw new AssertionError("Invalid or cancelled cached queue accepted"); }
        catch (InvocationTargetException failure) {
            if (!(failure.getCause() instanceof IllegalStateException) || failure.getCause().getMessage() != null) {
                throw new AssertionError("Unexpected or unsanitized queue failure", failure);
            }
        }
    }
    private static final class CancelDuringLoop extends NullProgressMonitor {
        private int polls;
        @Override public boolean isCanceled() { return ++polls >= 3; }
    }
    public static void main(String[] arguments) throws Exception {
        Path root = Path.of(System.getProperty("java.io.tmpdir")).toAbsolutePath().normalize().resolve("ancestry-fixture");
        Path pom = root.resolve("nested/pom.xml");
        Path leaf = root.resolve("nested/leaf/pom.xml");
        Path second = root.resolve("nested/second/pom.xml");
        Path existing = root.resolve("sibling/pom.xml");
        Path visitedChild = root.resolve("nested/visited/pom.xml");
        Set<Path> cached = Set.of(pom, leaf, second, existing, visitedChild, root.resolve("nested-alias/pom.xml"));
        ArrayDeque<Path> pending = new ArrayDeque<>(List.of(existing));
        Set<Path> visited = new HashSet<>(Set.of(pom,visitedChild));
        append(pom,cached,pending,visited,new NullProgressMonitor());
        if (!new ArrayList<>(pending).equals(List.of(existing,leaf,second))) {
            throw new AssertionError("Existing order, canonical prefix, visited dedup or sorted fallback differs");
        }
        append(pom,cached,pending,visited,new NullProgressMonitor());
        if (!new ArrayList<>(pending).equals(List.of(existing,leaf,second))) { throw new AssertionError("Queued descendants duplicated"); }
        System.out.println("only cached descendants append in stable order; existing queue, visited paths and unrelated prefixes stay intact");
        ArrayDeque<Path> empty = new ArrayDeque<>();
        append(pom,Set.of(),empty,Set.of(pom),new NullProgressMonitor());
        if (!empty.isEmpty()) { throw new AssertionError("Empty cache discovered a model"); }
        System.out.println("empty cache stays empty without filesystem discovery");
        List<Path> scheduled = new ArrayList<>();
        for (int index=0;index<126;index++) { scheduled.add(root.resolve("scheduled-"+index+"/pom.xml")); }
        ArrayDeque<Path> boundary = new ArrayDeque<>(scheduled);
        append(pom,Set.of(leaf),boundary,Set.of(pom),new NullProgressMonitor());
        if (boundary.size()!=127 || !boundary.getLast().equals(leaf)) { throw new AssertionError("Exact distinct budget differs"); }
        ArrayDeque<Path> overflow = new ArrayDeque<>(scheduled);
        rejected(() -> append(pom,Set.of(leaf,second),overflow,Set.of(pom),new NullProgressMonitor()));
        if (!new ArrayList<>(overflow).equals(scheduled)) { throw new AssertionError("Overflow partially mutated queue"); }
        System.out.println("128 distinct identities succeed; 129th refuses before committing any fallback additions");
        ArrayDeque<Path> duplicateQueue = new ArrayDeque<>();
        for (int index=0;index<128;index++) { duplicateQueue.add(existing); }
        rejected(() -> append(pom,Set.of(leaf),duplicateQueue,Set.of(pom),new NullProgressMonitor()));
        if (duplicateQueue.size()!=128) { throw new AssertionError("Queued entry budget changed"); }
        Set<Path> tooMany = new HashSet<>();
        for (int index=0;index<129;index++) { tooMany.add(root.resolve("cached-"+index+"/pom.xml")); }
        rejected(() -> append(pom,tooMany,new ArrayDeque<>(),Set.of(pom),new NullProgressMonitor()));
        System.out.println("queued-entry and cached-inventory budgets refuse independently of distinct traversal");
        NullProgressMonitor cancelled = new NullProgressMonitor(); cancelled.setCanceled(true);
        rejected(() -> append(pom,cached,empty,Set.of(pom),cancelled));
        if (!empty.isEmpty()) { throw new AssertionError("Pre-cancellation changed queue"); }
        CancelDuringLoop during = new CancelDuringLoop();
        rejected(() -> append(pom,Set.of(leaf,second),empty,Set.of(pom),during));
        if (during.polls!=3 || !empty.isEmpty()) { throw new AssertionError("Cancellation committed partial fallback"); }
        System.out.println("pre- and in-flight cancellation preserve the pending queue atomically");
    }
}
''')
        run_direct_maven_check(arguments, workspace, source, receipt, "MavenCachedAncestryCheck")
    print(f"Passed {len(receipt['checks'])} direct Maven cached ancestry checks")


def probe_profile_projection(arguments):
    provider = arguments.provider.resolve()
    if hashlib.sha256(provider.read_bytes()).hexdigest() != arguments.sha256:
        raise AssertionError("Provider binary differs from the source build")
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"provider_sha256": arguments.sha256, "checks": [], "limits": [
        "Direct production-method reflection only; no JDTLS process, build worker or native app",
        "Does not qualify provider command lifecycle, installed delivery, remote, Copper or resource budgets",
    ]}
    with tempfile.TemporaryDirectory(prefix="groovy-maven-profile-projection-") as directory:
        workspace = Path(directory)
        source = workspace / "MavenProfileProjectionCheck.java"
        source.write_text('''import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.maven.model.Model;
import org.apache.maven.model.Profile;
import org.apache.maven.project.MavenProject;
import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.core.runtime.NullProgressMonitor;

public class MavenProfileProjectionCheck {
    private static final Class<?> READER;
    private static final Method STATUSES;
    private static final Method ADD;
    static {
        try {
            READER = Class.forName("dev.made.groovy.MavenModelHandler$Reader");
            STATUSES = READER.getDeclaredMethod("profileStatuses", MavenProject.class, IProgressMonitor.class);
            ADD = READER.getDeclaredMethod("addProfileStatus", Map.class, String.class, boolean.class, IProgressMonitor.class);
            STATUSES.setAccessible(true); ADD.setAccessible(true);
        } catch (ReflectiveOperationException failure) { throw new ExceptionInInitializerError(failure); }
    }
    private static Object reader() throws Exception {
        var constructor = READER.getDeclaredConstructor(); constructor.setAccessible(true);
        return constructor.newInstance();
    }
    private static Profile profile(String identifier) {
        Profile profile = new Profile(); profile.setId(identifier); return profile;
    }
    private static MavenProject project(int count) {
        Model model = new Model();
        for (int index = 0; index < count; index++) { model.addProfile(profile("profile-" + index)); }
        MavenProject project = new MavenProject(model); project.setOriginalModel(model);
        return project;
    }
    private static Map<?, ?> statuses(MavenProject project, IProgressMonitor monitor) throws Exception {
        return (Map<?, ?>) STATUSES.invoke(reader(), project, monitor);
    }
    private interface CheckedCall { void run() throws Exception; }
    private static void rejected(CheckedCall call) throws Exception {
        try { call.run(); throw new AssertionError("Invalid or cancelled profile projection accepted"); }
        catch (InvocationTargetException failure) {
            if (!(failure.getCause() instanceof IllegalStateException) || failure.getCause().getMessage() != null) {
                throw new AssertionError("Unsanitized profile failure", failure);
            }
        }
    }
    private static final class CancelDuringLoop extends NullProgressMonitor {
        private int polls;
        @Override public boolean isCanceled() { return ++polls >= 3; }
    }
    public static void main(String[] arguments) throws Exception {
        List<String> checks = new ArrayList<>();
        if (!statuses(project(0), new NullProgressMonitor()).isEmpty()) { throw new AssertionError("Empty model differs"); }
        checks.add("empty original profile model stays empty");
        MavenProject missingOriginal = new MavenProject(new Model()); missingOriginal.setOriginalModel(null);
        missingOriginal.setActiveProfiles(List.of(profile("active")));
        if (!Boolean.TRUE.equals(statuses(missingOriginal, new NullProgressMonitor()).get("active"))) {
            throw new AssertionError("Active facts lost without original model");
        }
        checks.add("missing original model retains cached active facts");
        MavenProject exact = project(128); exact.setActiveProfiles(List.of(profile("profile-0"), profile("profile-0")));
        Map<?, ?> facts = statuses(exact, new NullProgressMonitor());
        if (facts.size() != 128 || !Boolean.TRUE.equals(facts.get("profile-0")) || !Boolean.FALSE.equals(facts.get("profile-127"))) {
            throw new AssertionError("Exact budget or active precedence differs");
        }
        checks.add("128 distinct IDs admit repeated active override and preserve declared inactive facts");
        rejected(() -> statuses(project(129), new NullProgressMonitor()));
        checks.add("129th declared ID rejects");
        exact.setActiveProfiles(List.of(profile("profile-128")));
        rejected(() -> statuses(exact, new NullProgressMonitor()));
        checks.add("129th active ID rejects");
        Map<String, Boolean> admitted = new LinkedHashMap<>();
        for (int index = 0; index < 128; index++) { admitted.put("profile-" + index, false); }
        Object reader = reader();
        ADD.invoke(reader, admitted, "profile-0", true, new NullProgressMonitor());
        Map<String, Boolean> before = new LinkedHashMap<>(admitted);
        rejected(() -> ADD.invoke(reader, admitted, "profile-128", true, new NullProgressMonitor()));
        if (!admitted.equals(before)) { throw new AssertionError("Overflow mutated admitted map"); }
        checks.add("new ID refuses before mutation while exact-limit override succeeds");
        Map<String, Boolean> malformed = new LinkedHashMap<>();
        rejected(() -> ADD.invoke(reader(), malformed, "private\\u0000identifier", true, new NullProgressMonitor()));
        rejected(() -> ADD.invoke(reader(), malformed, "x".repeat(4097), true, new NullProgressMonitor()));
        if (!malformed.isEmpty()) { throw new AssertionError("Malformed identifier mutated map"); }
        checks.add("control and oversized identifiers reject without map mutation or private error text");
        NullProgressMonitor cancelled = new NullProgressMonitor(); cancelled.setCanceled(true);
        rejected(() -> statuses(project(0), cancelled));
        checks.add("pre-cancelled empty projection rejects");
        CancelDuringLoop declaredCancellation = new CancelDuringLoop();
        rejected(() -> statuses(project(5), declaredCancellation));
        if (declaredCancellation.polls != 3) { throw new AssertionError("Declared loop cancellation differs"); }
        checks.add("declared profile loop checks cancellation in flight");
        MavenProject activeOnly = project(0);
        activeOnly.setActiveProfiles(List.of(profile("one"), profile("two"), profile("three")));
        CancelDuringLoop activeCancellation = new CancelDuringLoop();
        rejected(() -> statuses(activeOnly, activeCancellation));
        if (activeCancellation.polls != 3) { throw new AssertionError("Active loop cancellation differs"); }
        checks.add("active profile loop checks cancellation in flight");
        for (String check : checks) { System.out.println(check); }
    }
}
''')
        run_direct_maven_check(arguments, workspace, source, receipt, "MavenProfileProjectionCheck")
    print(f"Passed {len(receipt['checks'])} direct cached Maven profile projection checks")


def probe(arguments):
    provider = arguments.provider.resolve()
    if hashlib.sha256(provider.read_bytes()).hexdigest() != arguments.sha256:
        raise AssertionError("Provider binary differs from the source build")
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"provider_sha256": arguments.sha256, "checks": [], "limits": [
        "Headless cached-model probe; no installed/default, mounted UI, native, remote or platform qualification",
        "Cached declarations are not effective values, CLI property mappings, import freshness or native qualification; resolver/reimport and client consumption remain incomplete",
    ]}
    with tempfile.TemporaryDirectory(prefix="groovy-maven-model-") as directory:
        root = (Path(directory) / "project").resolve()
        shutil.copytree(REPOSITORY / "crates/project/test_data/groovy_jvm_maven", root,
                        ignore=shutil.ignore_patterns("target", ".git"))
        fixture_reports = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in root.rglob("TEST-*.xml")}
        repository = (Path(directory) / "repository").resolve()
        shutil.copytree(arguments.maven_repository.resolve(), repository, copy_function=cache_copy)
        settings_path = Path(directory) / "settings.xml"
        settings_path.write_text("<settings><offline>true</offline><localRepository>"
                                 + escape(str(repository)) + "</localRepository></settings>\n")
        harness = Path(directory) / "CancelledMavenQuery.java"
        harness.write_text('''import java.util.List;
import java.util.Map;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.lang.reflect.Method;
import java.nio.file.Path;
import org.apache.maven.model.Model;
import org.apache.maven.model.Dependency;
import org.apache.maven.model.Parent;
import org.apache.maven.model.Profile;
import org.apache.maven.project.MavenProject;
import dev.made.groovy.MavenModelHandler;
import org.eclipse.core.runtime.NullProgressMonitor;
public class CancelledMavenQuery {
    public static void main(String[] arguments) throws Exception {
        NullProgressMonitor monitor = new NullProgressMonitor();
        monitor.setCanceled(true);
        for (String command : List.of("groovy.maven.projects", "groovy.maven.projectDetails", "groovy.maven.goalDetails")) {
            try {
                new MavenModelHandler().executeCommand(command,
                    List.of(Map.of("rootUri", "file:///private/unused", "cachedOnly", true)), monitor);
                throw new AssertionError("Cancelled query succeeded");
            } catch (IllegalStateException cancelled) {
                if (!cancelled.getMessage().contains("cancelled") || cancelled.getMessage().contains("unused")) {
                    throw new AssertionError("Cancellation category differs");
                }
            }
        }
        Class<?> readerClass = Class.forName("dev.made.groovy.MavenModelHandler$Reader");
        var constructor = readerClass.getDeclaredConstructor();
        constructor.setAccessible(true);
        Object reader = constructor.newInstance();
        Method profiles = readerClass.getDeclaredMethod("profileDefinitions", Map.class, MavenProject.class, Path.class,
                                                       org.eclipse.core.runtime.IProgressMonitor.class);
        profiles.setAccessible(true);
        Model model = new Model();
        model.setGroupId("fixture"); model.setArtifactId("child"); model.setVersion("1");
        model.setParent(new Parent());
        Profile profile = new Profile(); profile.setId("shared"); model.addProfile(profile);
        MavenProject child = new MavenProject(model); child.setOriginalModel(model);
        Map<String, Object> record = new LinkedHashMap<>();
        profiles.invoke(reader, record, child, Path.of(arguments[0]), new NullProgressMonitor());
        if (!Boolean.FALSE.equals(record.get("inheritedProfileDefinitionsKnown"))) {
            throw new AssertionError("Missing cached parent was claimed known");
        }
        Model parentModel = model.clone(); parentModel.setArtifactId("parent");
        MavenProject parent = new MavenProject(parentModel); parent.setOriginalModel(parentModel);
        parent.setFile(Path.of(arguments[0]).getParent().resolve("external-parent.xml").toFile());
        child.setParent(parent); parent.setParent(child);
        record.clear();
        profiles.invoke(reader, record, child, Path.of(arguments[0]), new NullProgressMonitor());
        if (!Boolean.FALSE.equals(record.get("inheritedProfileDefinitionsKnown"))) {
            throw new AssertionError("Cyclic cached parents were claimed known");
        }
        List<?> definitions = (List<?>) record.get("profileDefinitions");
        if (definitions.size() != 2 || ((Map<?, ?>) definitions.get(1)).get("sourcePomUri") != null) {
            throw new AssertionError("Duplicate origins lost or external cache path exposed");
        }
        Method order = readerClass.getDeclaredMethod("reactorOrder", Map.class, List.class, List.class, Path.class,
                                                    org.eclipse.core.runtime.IProgressMonitor.class);
        order.setAccessible(true);
        record.clear();
        order.invoke(reader, record, new ArrayList<>(), List.of("missing"), Path.of(arguments[0]), new NullProgressMonitor());
        if (!"missing-cached-model".equals(record.get("reactorOrderProblem")) || !Boolean.FALSE.equals(record.get("reactorOrderKnown"))) {
            throw new AssertionError("Missing reactor models claimed an order");
        }
        record.clear();
        order.invoke(reader, record, List.of(child, child), List.of(), Path.of(arguments[0]), new NullProgressMonitor());
        if (!"duplicate-coordinates".equals(record.get("reactorOrderProblem")) || !Boolean.FALSE.equals(record.get("reactorOrderKnown"))) {
            throw new AssertionError("Duplicate coordinates claimed an order");
        }
        child.setParent(null); parent.setParent(null);
        model.setParent(null); parentModel.setParent(null);
        child.setFile(Path.of(arguments[0]).resolve("pom.xml").toFile());
        parent.setFile(Path.of(arguments[0]).resolve("library/pom.xml").toFile());
        Dependency dependency = new Dependency();
        dependency.setGroupId("fixture"); dependency.setArtifactId("parent"); dependency.setVersion("1");
        model.addDependency(dependency);
        record.clear();
        order.invoke(reader, record, List.of(child, parent), List.of(), Path.of(arguments[0]), new NullProgressMonitor());
        if (!Boolean.TRUE.equals(record.get("reactorOrderKnown")) || !child.getProjectReferences().isEmpty()) {
            throw new AssertionError("Sorter failed or mutated cached project references");
        }
        Dependency backEdge = dependency.clone(); backEdge.setArtifactId("child"); parentModel.addDependency(backEdge);
        record.clear();
        order.invoke(reader, record, List.of(child, parent), List.of(), Path.of(arguments[0]), new NullProgressMonitor());
        if (!"invalid-cached-reactor".equals(record.get("reactorOrderProblem")) || !Boolean.FALSE.equals(record.get("reactorOrderKnown"))) {
            throw new AssertionError("Cyclic reactor claimed an order");
        }
        model.setDependencies(java.util.Collections.nCopies(4097, dependency));
        try {
            order.invoke(reader, new LinkedHashMap<>(), List.of(child, parent), List.of(), Path.of(arguments[0]), new NullProgressMonitor());
            throw new AssertionError("Oversized cached graph accepted");
        } catch (java.lang.reflect.InvocationTargetException rejected) {
            if (!(rejected.getCause() instanceof IllegalStateException)) {
                throw rejected;
            }
        }
    }
}
''')
        if arguments.use_groovy_import:
            harness.write_text(harness.read_text().replace(
                '        Class<?> readerClass =',
                '\n'.join([
                    '        try {',
                    '            new dev.made.groovy.MavenImportHandler().executeCommand("groovy.maven.importSelected", null, monitor);',
                    '            throw new AssertionError("Cancelled selected import succeeded");',
                    '        } catch (IllegalStateException cancelled) {',
                    '            if (!cancelled.getMessage().contains("cancelled")) {',
                    '                throw new AssertionError("Selected import cancellation category differs");',
                    '            }',
                    '        }',
                    '        Class<?> readerClass =',
                ]), 1))
        plugins = arguments.launcher.resolve().parent.parent / "plugins"
        maven_runtime = Path(directory) / "runtime"
        maven_runtime.mkdir()
        with zipfile.ZipFile(plugins / "org.eclipse.m2e.maven.runtime_3.9.1600.20260522-0510.jar") as archive:
            for name in archive.namelist():
                if name.startswith("jars/") and name.endswith(".jar"):
                    (maven_runtime / Path(name).name).write_bytes(archive.read(name))
        (Path(directory) / "external-parent.xml").write_text("<project/>\n")
        classpath = os.pathsep.join(str(path) for path in [provider, *plugins.glob("*.jar"), *maven_runtime.glob("*.jar")])
        subprocess.run([str(arguments.java_home / "bin/javac"), "--release", "17", "-classpath", classpath,
                        "-d", directory, str(harness)], check=True, capture_output=True)
        subprocess.run([str(arguments.java_home / "bin/java"), "-classpath", directory + os.pathsep + classpath,
                        "CancelledMavenQuery", str(root)], check=True, capture_output=True)
        receipt["checks"].extend(["pre-cancelled command rejected before host/model access",
                                  "synthetic cached missing parent stays unknown", "synthetic parent cycle is bounded",
                                  "external parent URI redacted and duplicate origins retained", "missing reactor models have explicit unknown category",
                                  "duplicate reactor coordinates have explicit unknown category", "runtime sorting preserves cached project references",
                                  "runtime cyclic reactor has explicit sanitized unknown category", "cached reactor edge budget enforced"])
        if arguments.use_groovy_import:
            receipt["checks"].append("pre-cancelled selected import rejected before argument/host/model access")
        client = client_class()(
            arguments.launcher.resolve(), arguments.java_home.resolve(), root,
            Path(directory) / "jdtls-data", deny_external_network=True,
        )
        settings = {"java": {
            "import": {"maven": {"enabled": True, "offline": {"enabled": True}}, "gradle": {"enabled": False}},
            "configuration": {"maven": {"userSettings": str(settings_path)}},
        }}
        try:
            initialized = client.request("initialize", {
                "processId": os.getpid(), "rootUri": root.as_uri(),
                "workspaceFolders": [{"uri": root.as_uri(), "name": root.name}],
                "capabilities": {"workspace": {"workspaceFolders": True, "configuration": True}},
                "initializationOptions": {"bundles": [str(provider)], "settings": settings},
            }, timeout=120)
            if not {"groovy.maven.projects", "groovy.maven.projectDetails", "groovy.maven.goalDetails"}.issubset(
                    initialized["capabilities"]["executeCommandProvider"]["commands"]):
                raise AssertionError("JDTLS did not register the Maven command")
            client.notify("initialized", {})
            deadline = time.monotonic() + 120
            response = None
            while time.monotonic() < deadline:
                response = discover(client, root)
                if len(response["projects"]) >= 3:
                    break
                time.sleep(0.2)
            (arguments.receipt.parent / "maven-model-response.json").write_text(json.dumps(response, indent=2) + "\n")
            if len(response["projects"]) != 3 or response["version"] != 1 or not response["cachedOnly"]:
                raise AssertionError("Cached Maven model or module count differs")
            projects = response["projects"]
            if len({project["pomUri"] for project in projects}) != len(projects):
                raise AssertionError("Maven module identities collide")
            application = next(project for project in projects if project["artifactId"] == "application")
            parent = next(project for project in projects if project["artifactId"] == "groovy-jvm-maven")
            if len(parent["moduleUris"]) != 2 or parent["profiles"].get("parity") is not False:
                raise AssertionError("Effective modules or declared inactive profile differ")
            if not any(application["profiles"].values()):
                raise AssertionError("Effective active profile facts missing")
            if any(project["inheritedProfileDefinitionsKnown"] for project in projects) or response["reactorOrderKnown"]:
                raise AssertionError("Unqualified profile/order facts claimed known")
            lifecycles = {lifecycle["id"]: lifecycle["phases"] for lifecycle in response["lifecycles"]}
            if set(lifecycles) != {"clean", "default", "site"} or "test" not in lifecycles["default"]:
                raise AssertionError("Runtime Maven lifecycles missing")
            surefire = next(plugin for plugin in application["plugins"] if plugin["artifactId"] == "maven-surefire-plugin")
            if surefire["version"] != "3.5.2" or not surefire["goalsKnown"] or "test" not in {goal["name"] for goal in surefire["goals"]}:
                raise AssertionError("Cached effective plugin goal differs")
            if surefire["managedOnly"] or not any(plugin["managedOnly"] for plugin in application["plugins"]):
                raise AssertionError("Build plugins and plugin-management-only entries were conflated")
            if projects != discover(client, root)["projects"]:
                raise AssertionError("Repeat cached metadata changed identities")
            receipt["checks"].extend(["cached three-module metadata with external network denied", "stable module identities",
                                      "effective module links", "declared inactive profile", "effective active profile facts", "runtime lifecycle phases",
                                      "cached surefire goal and managed version", "build versus managed-only plugins distinguished",
                                      "unqualified inherited profiles/order remain unknown"])
            details = discover(client, root, details=True)
            (arguments.receipt.parent / "maven-details-response.json").write_text(json.dumps(details, indent=2) + "\n")
            if details["version"] != 2 or not details["cachedOnly"]:
                raise AssertionError("Extended cached Maven protocol differs")
            expected_order = [parent["pomUri"], next(project["pomUri"] for project in projects
                                                   if project["artifactId"] == "library"), application["pomUri"]]
            if not details["reactorOrderKnown"] or details["reactorPomUris"] != expected_order or details.get("reactorOrderProblem"):
                raise AssertionError("Runtime reactor dependency order differs")
            extended_application = next(project for project in details["projects"] if project["artifactId"] == "application")
            definitions = extended_application["profileDefinitions"]
            local_parity = next(definition for definition in definitions if definition["id"] == "parity" and not definition["inherited"])
            inherited_parity = next(definition for definition in definitions if definition["id"] == "parity" and definition["inherited"])
            if not extended_application["inheritedProfileDefinitionsKnown"] or local_parity["sourcePomUri"] != application["pomUri"] or inherited_parity["sourcePomUri"] != parent["pomUri"]:
                raise AssertionError("Local and inherited profile origins were conflated")
            if local_parity["activeAtSource"] or inherited_parity["activeAtSource"]:
                raise AssertionError("Inactive declaration claimed active")
            if not any(definition["id"] == "parity-cycle" and definition["inherited"] for definition in definitions):
                raise AssertionError("Inherited inactive definition missing")
            if details != discover(client, root, details=True):
                raise AssertionError("Extended cached response changed without import")
            receipt["checks"].extend(["v1 basic contract preserved", "runtime dependency reactor order",
                                      "inherited and local duplicate profile ID origins retained", "source activation facts remain distinct",
                                      "inherited inactive definition available", "stable extended cached response"])
            parameter_details = discover(client, root, parameters=True)
            (arguments.receipt.parent / "maven-parameters-response.json").write_text(json.dumps(parameter_details, indent=2) + "\n")
            if parameter_details["version"] != 3 or not parameter_details["cachedOnly"]:
                raise AssertionError("Parameter metadata protocol differs")
            parameter_application = next(project for project in parameter_details["projects"] if project["artifactId"] == "application")
            parameter_surefire = next(plugin for plugin in parameter_application["plugins"] if plugin["artifactId"] == "maven-surefire-plugin")
            test_goal = next(goal for goal in parameter_surefire["goals"] if goal["name"] == "test")
            declarations = {parameter["name"]: parameter for parameter in test_goal["parameters"]}
            if not test_goal["parametersKnown"] or not {"test", "skipTests", "forkCount"}.issubset(declarations):
                raise AssertionError("Cached Surefire parameter declarations missing")
            if declarations["skipTests"]["declaredType"] != "boolean" or declarations["skipTests"]["required"] or not declarations["skipTests"]["editable"]:
                raise AssertionError("Cached Surefire parameter facts differ")
            legacy_parameters = [goal for project in details["projects"] for plugin in project["plugins"] for goal in plugin["goals"]]
            if any("parameters" in goal or "parametersKnown" in goal for goal in legacy_parameters):
                raise AssertionError("v2 output changed to include v3 fields")
            if parameter_details != discover(client, root, parameters=True):
                raise AssertionError("Cached parameter response changed without import")
            receipt["checks"].extend(["v3 cached Surefire parameter names and types", "required/editable descriptor declarations",
                                      "v2 shape excludes v3 fields", "stable cached parameter identities"])
            expect_failure(client, Path(directory).resolve())
            expect_failure(client, root, cached=False)
            receipt["checks"].extend(["foreign root rejected", "resolving mode rejected", "sanitized request errors"])
            plugin = repository / "org/apache/maven/plugins/maven-surefire-plugin/3.5.2/maven-surefire-plugin-3.5.2.jar"
            original_plugin = plugin.read_bytes()
            plugin.unlink()
            try:
                missing = discover(client, root)
                missing_application = next(project for project in missing["projects"] if project["artifactId"] == "application")
                missing_plugin = next(plugin for plugin in missing_application["plugins"] if plugin["artifactId"] == "maven-surefire-plugin")
                if missing_plugin["goalsKnown"] or missing_plugin["goals"]:
                    raise AssertionError("Missing cached descriptor claimed available goals")
                missing_parameters = discover(client, root, parameters=True)
                missing_parameter_application = next(project for project in missing_parameters["projects"] if project["artifactId"] == "application")
                missing_parameter_plugin = next(plugin for plugin in missing_parameter_application["plugins"] if plugin["artifactId"] == "maven-surefire-plugin")
                if missing_parameter_plugin["goalsKnown"] or missing_parameter_plugin["goals"]:
                    raise AssertionError("Missing descriptor claimed available parameter declarations")
                receipt["checks"].append("missing cached plugin remains explicitly unknown")
                canary = "GROOVY_PRIVATE_MAVEN_DESCRIPTOR_CANARY"
                with zipfile.ZipFile(plugin, "w") as archive:
                    archive.writestr("META-INF/maven/plugin.xml", f'<!DOCTYPE plugin [<!ENTITY secret SYSTEM "file:///private/{canary}">]><plugin>&secret;</plugin>')
                expect_failure(client, root, forbidden=canary)
                expect_failure(client, root, forbidden=canary, parameters=True)
                receipt["checks"].append("external XML entity rejected with sanitized failure")
                with zipfile.ZipFile(plugin, "w") as archive:
                    archive.writestr("META-INF/maven/plugin.xml", b"x" * (1024 * 1024 + 1))
                expect_failure(client, root)
                expect_failure(client, root, parameters=True)
                receipt["checks"].append("oversized cached descriptor rejected")
                prefix = ('<plugin><groupId>org.apache.maven.plugins</groupId><artifactId>maven-surefire-plugin</artifactId>'
                          '<version>3.5.2</version><goalPrefix>surefire</goalPrefix><mojos><mojo><goal>test</goal><parameters>')
                suffix = ('</parameters><configuration><example default-value="/private/' + canary + '">${env.SECRET}</example>'
                          '</configuration></mojo></mojos></plugin>')
                def replace_parameters(contents):
                    with zipfile.ZipFile(plugin, "w") as archive:
                        archive.writestr("META-INF/maven/plugin.xml", prefix + contents + suffix)
                replace_parameters('<parameter><name>example</name><alias>friendly</alias><type>java.lang.String</type></parameter>')
                defaults = discover(client, root, parameters=True)
                default_application = next(project for project in defaults["projects"] if project["artifactId"] == "application")
                default_plugin = next(plugin for plugin in default_application["plugins"] if plugin["artifactId"] == "maven-surefire-plugin")
                declaration = default_plugin["goals"][0]["parameters"][0]
                if declaration["required"] or not declaration["editable"] or declaration["alias"] != "friendly":
                    raise AssertionError("Absent descriptor booleans or alias differ")
                if canary in json.dumps(defaults) or "env.SECRET" in json.dumps(defaults):
                    raise AssertionError("Configuration expressions or default values leaked")
                receipt["checks"].extend(["absent descriptor booleans use documented defaults", "configuration/default expressions excluded"])
                replace_parameters(''.join(f'<parameter><name>parameter{index}</name><description>' + 'x' * 4000
                                          + '</description></parameter>' for index in range(240)))
                if discover(client, root, details=True)["version"] != 2:
                    raise AssertionError("Legacy metadata changed under added declaration volume")
                expect_failure(client, root, forbidden=canary, parameters=True)
                receipt["checks"].append("aggregate declaration budget enforced while v2 remains compatible")
                failures = [
                    ("duplicate parameter names rejected", '<parameter><name>same</name></parameter>' * 2),
                    ("malformed required declaration rejected", '<parameter><name>example</name><required>maybe</required></parameter>'),
                    ("malformed editable declaration rejected", '<parameter><name>example</name><editable>maybe</editable></parameter>'),
                    ("invalid parameter name rejected", '<parameter><name>../private</name></parameter>'),
                    ("parameter count bounded", ''.join(f'<parameter><name>parameter{index}</name></parameter>' for index in range(257))),
                    ("parameter text bounded", '<parameter><name>example</name><description>' + 'x' * 4097 + '</description></parameter>'),
                ]
                for label, contents in failures:
                    replace_parameters(contents)
                    expect_failure(client, root, forbidden=canary, parameters=True)
                    receipt["checks"].append(label)
            finally:
                plugin.write_bytes(original_plugin)
            recovered = discover(client, root)
            if recovered["projects"] != projects:
                raise AssertionError("Repaired cache did not restore metadata")
            if discover(client, root, details=True) != details:
                raise AssertionError("Extended metadata did not recover after descriptor repair")
            if discover(client, root, parameters=True) != parameter_details:
                raise AssertionError("Parameter metadata did not recover after descriptor repair")
            receipt["checks"].append("descriptor repair recovery for all three commands")
            reports = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in root.rglob("TEST-*.xml")}
            if reports != fixture_reports:
                raise AssertionError("Metadata journey unexpectedly emitted selected test reports")
            receipt["checks"].append("fixture reports unchanged; no selected test report emitted")
            receipt["classes_present_after_import"] = any(root.rglob("*.class"))
            receipt["limits"].append("Existing M2E/Eclipse import can generate classes; absence of class files is not a metadata-read invariant")
        except Exception:
            evidence = arguments.receipt.parent / f"failed-maven-provider-{time.time_ns()}"
            evidence.mkdir(parents=True, exist_ok=True)
            log = Path(directory) / "jdtls-data/.metadata/.log"
            if log.exists():
                shutil.copy2(log, evidence / "jdtls.log")
            client.stderr.seek(0)
            (evidence / "stderr.log").write_bytes(client.stderr.read())
            raise
        finally:
            client.close()
    arguments.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def unlink_fixture_bundle(arguments, directory, identity_modes=False, ownership_modes=False, hydration_modes=False, default_root_modes=False, workspace_states=False, scoped_import_modes=False):
    directory = Path(directory)
    source = directory / "UnlinkFixtureHandler.java"
    source.write_text('''package dev.made.groovy.fixture;
import java.nio.file.Files;
import java.nio.file.Path;
import java.net.URI;
import java.util.List;
import java.util.Map;
import org.eclipse.core.resources.ResourcesPlugin;
import org.eclipse.core.resources.WorkspaceJob;
import org.eclipse.core.resources.IMarker;
import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.core.runtime.IStatus;
import org.eclipse.core.runtime.Status;
import org.eclipse.jdt.ls.core.internal.IDelegateCommandHandler;
import org.eclipse.jdt.ls.core.internal.JavaLanguageServerPlugin;
import org.eclipse.jdt.ls.core.internal.ProjectUtils;
import org.eclipse.lsp4j.Diagnostic;
import org.eclipse.lsp4j.Position;
import org.eclipse.lsp4j.Range;
import org.eclipse.lsp4j.PublishDiagnosticsParams;
public final class UnlinkFixtureHandler implements IDelegateCommandHandler {
 public Object executeCommand(String command, List<Object> arguments, IProgressMonitor monitor) throws Exception {
  if (command.equals("groovy.fixture.unlink.stack")) {
   boolean waiting = Thread.getAllStackTraces().values().stream().anyMatch(stack ->
    java.util.Arrays.stream(stack).anyMatch(frame -> frame.getClassName().equals("dev.made.groovy.ProjectUnlinkHandler")) &&
    java.util.Arrays.stream(stack).anyMatch(frame -> frame.getClassName().startsWith("org.eclipse.core.internal.jobs.") && frame.getMethodName().equals("begin")));
   return Map.of("unlinkRequestWaiting", waiting);
  }
  if (arguments == null || arguments.size() != 1 || !(arguments.get(0) instanceof Map<?, ?> request)) throw new IllegalArgumentException();
  Path root = Path.of(new URI((String) request.get("rootUri"))).toRealPath();
  boolean initialized = false;
  for (var path : JavaLanguageServerPlugin.getPreferencesManager().getPreferences().getRootPaths()) {
   if (path.toFile().toPath().toRealPath().equals(root)) initialized = true;
  }
  if (!initialized) throw new IllegalArgumentException();
  if (command.equals("groovy.fixture.unlink.prepare")) {
   Path module = Path.of(new URI((String) request.get("projectUri"))).toRealPath();
   if (!module.startsWith(root)) throw new IllegalArgumentException();
   for (var project : ProjectUtils.getAllProjects()) {
    if (project.getLocation() != null && project.getLocation().toFile().toPath().toRealPath().equals(module)) {
     var file = project.getFile("src/main/java/evidence/Main.java");
     if (!file.exists()) throw new IllegalArgumentException();
     var marker = file.createMarker(IMarker.PROBLEM);
     marker.setAttribute(IMarker.MESSAGE, "isolated unlink diagnostic");
     marker.setAttribute(IMarker.SEVERITY, IMarker.SEVERITY_ERROR);
     String uri = file.getLocationURI().toString();
     JavaLanguageServerPlugin.getProjectsManager().getConnection().publishDiagnostics(new PublishDiagnosticsParams(uri,
      List.of(new Diagnostic(new Range(new Position(0,0),new Position(0,1)),"isolated unlink diagnostic"))));
     return Map.of("diagnosticUri", uri);
    }
   }
   throw new IllegalArgumentException();
  }
  if (!command.equals("groovy.fixture.unlink.hold")) throw new IllegalArgumentException();
  Path started = root.getParent().resolve("unlink-rule-started");
  Path release = root.getParent().resolve("unlink-rule-release");
  Path finished = root.getParent().resolve("unlink-rule-finished");
  WorkspaceJob job = new WorkspaceJob("Isolated unlink qualification gate") {
   public IStatus runInWorkspace(IProgressMonitor progress) {
    try {
     Files.writeString(started,"started");
     long deadline = System.nanoTime() + java.util.concurrent.TimeUnit.SECONDS.toNanos(30);
     while (!Files.exists(release) && !progress.isCanceled() && System.nanoTime() < deadline) Thread.sleep(10);
     return Files.exists(release) ? Status.OK_STATUS : Status.CANCEL_STATUS;
    } catch (Exception failure) { return Status.CANCEL_STATUS; }
    finally {
     try { Files.writeString(finished,"finished"); }
     catch (java.io.IOException failure) { JavaLanguageServerPlugin.logException("Isolated unlink gate cleanup failed",failure); }
    }
   }
  };
  job.setRule(ResourcesPlugin.getWorkspace().getRoot()); job.schedule();
  return Map.of("started",started.toString(),"release",release.toString(),"finished",finished.toString());
 }
}
''')
    if identity_modes or ownership_modes or hydration_modes or default_root_modes:
        fixture = source.read_text().replace(
            'public final class UnlinkFixtureHandler implements IDelegateCommandHandler {',
            'public final class UnlinkFixtureHandler implements IDelegateCommandHandler {\n'
            ' private final java.util.Map<Path, String[]> originalNatures = new java.util.HashMap<>();')
        fixture = fixture.replace('  if (command.equals("groovy.fixture.unlink.prepare")) {', '''
  if (command.equals("groovy.fixture.unlink.natures")) {
   Path module = Path.of(new URI((String) request.get("projectUri"))).toRealPath();
   if (!module.equals(root) || monitor.isCanceled()) throw new IllegalArgumentException();
   for (var project : ProjectUtils.getAllProjects()) {
    if (project.getLocation() != null && project.getLocation().toFile().toPath().toRealPath().equals(module)) {
     var description = project.getDescription();
     String[] natures;
     if (Boolean.TRUE.equals(request.get("dual"))) {
      if (originalNatures.putIfAbsent(module, description.getNatureIds()) != null) throw new IllegalArgumentException();
      var selected = new java.util.LinkedHashSet<String>(java.util.Arrays.asList(description.getNatureIds()));
      selected.add("org.eclipse.buildship.core.gradleprojectnature");
      natures = selected.toArray(String[]::new);
     } else {
      natures = originalNatures.remove(module);
      if (natures == null) throw new IllegalArgumentException();
     }
     description.setNatureIds(natures); project.setDescription(description, monitor);
     return Map.of("dualNature", ProjectUtils.isMavenProject(project) && ProjectUtils.isGradleProject(project));
    }
   }
   throw new IllegalArgumentException();
  }
  if (command.equals("groovy.fixture.unlink.prepare")) {''')
        if ownership_modes or hydration_modes:
            fixture = fixture.replace('!module.equals(root)', '!module.startsWith(root)')
            fixture = fixture.replace('      selected.add("org.eclipse.buildship.core.gradleprojectnature");',
                                      '      selected.add("org.eclipse.buildship.core.gradleprojectnature");\n'
                                      '      selected.add("org.eclipse.m2e.core.maven2Nature");')
            fixture = fixture.replace('     return Files.exists(release) ? Status.OK_STATUS : Status.CANCEL_STATUS;',
                                      '     if (Files.exists(release) && Boolean.TRUE.equals(request.get("changeOwnership"))) {\n'
                                      '      executeCommand("groovy.fixture.unlink.natures", List.of(Map.of("rootUri", request.get("rootUri"), "projectUri", request.get("projectUri"), "dual", true)), progress);\n'
                                      '     }\n'
                                      '     return Files.exists(release) ? Status.OK_STATUS : Status.CANCEL_STATUS;')
        if hydration_modes:
            fixture = fixture.replace('frame.getClassName().equals("dev.made.groovy.ProjectUnlinkHandler")',
                                      '(frame.getClassName().equals("dev.made.groovy.ProjectUnlinkHandler") || frame.getClassName().equals("dev.made.groovy.MavenImportHandler"))')
            fixture = fixture.replace('     return Files.exists(release) ? Status.OK_STATUS : Status.CANCEL_STATUS;',
                                      '     if (Files.exists(release) && Boolean.TRUE.equals(request.get("changeInputs"))) {\n'
                                      '      Path module = Path.of(new URI((String) request.get("projectUri"))).toRealPath();\n'
                                      '      if (!module.startsWith(root)) throw new IllegalArgumentException();\n'
                                      '      Path pom = module.resolve("pom.xml");\n'
                                      '      Files.writeString(pom, Files.readString(pom) + "\\n");\n'
                                      '     }\n'
                                      '     return Files.exists(release) ? Status.OK_STATUS : Status.CANCEL_STATUS;')
        if default_root_modes:
            fixture = fixture.replace('!module.equals(root)', '!module.startsWith(root)')
            fixture = fixture.replace('if (Boolean.TRUE.equals(request.get("dual"))) {',
                                      'if (Boolean.TRUE.equals(request.get("plain"))) {\n'
                                      '      if (originalNatures.putIfAbsent(module, description.getNatureIds()) != null) throw new IllegalArgumentException();\n'
                                      '      natures = new String[0];\n'
                                      '     } else if (Boolean.TRUE.equals(request.get("dual"))) {')
            fixture = fixture.replace('  if (command.equals("groovy.fixture.unlink.prepare")) {', '''
  if (command.equals("groovy.fixture.unlink.defaultRootsOracle")) {
   var detector = new org.eclipse.jdt.ls.core.internal.managers.BasicFileDetector(root, "pom.xml").includeNested(false).addExclusions("**/target");
   for (var project : ProjectUtils.getAllProjects()) {
    if (!ProjectUtils.isMavenProject(project) && project.getLocation() != null) {
     detector.addExclusions(project.getLocation().toFile().toPath().toString().replace("\\\\", "\\\\\\\\"));
    }
   }
   var preferences = JavaLanguageServerPlugin.getPreferencesManager().getPreferences();
   return Map.of("directories", detector.scan(monitor).stream().map(path -> path.toUri().toString()).toList(),
    "automaticMavenImportEnabled", preferences.isImportMavenEnabled(),
    "exclusions", preferences.getJavaImportExclusions());
  }
  if (command.equals("groovy.fixture.unlink.prepare")) {''')
        if default_root_modes:
            fixture = fixture.replace('  if (command.equals("groovy.fixture.unlink.prepare")) {', '''
  if (command.equals("groovy.fixture.unlink.gradleDefaultRootsOracle")) {
   var detector = new org.eclipse.jdt.ls.core.internal.managers.BasicFileDetector(root,
    "build.gradle", "settings.gradle", "build.gradle.kts", "settings.gradle.kts")
    .includeNested(false).addExclusions("**/build", "**/bin");
   for (var project : ProjectUtils.getAllProjects()) {
    if (!ProjectUtils.isGradleProject(project)
      && !ProjectUtils.hasNature(project, org.eclipse.jdt.ls.core.internal.managers.GradleProjectImporter.GRADLE_BUILD_SERVER_NATURE)
      && project.getLocation() != null) {
     detector.addExclusions(project.getLocation().toOSString().replace("\\\\", "\\\\\\\\"));
    }
   }
   var preferences = JavaLanguageServerPlugin.getPreferencesManager().getPreferences();
   return Map.of("directories", detector.scan(monitor).stream().map(path -> path.toUri().toString()).toList(),
    "automaticGradleImportEnabled", preferences.isImportGradleEnabled(),
    "exclusions", preferences.getJavaImportExclusions());
  }
  if (command.equals("groovy.fixture.unlink.gradleDefaultRootsState")) {
   Path module = Path.of(new URI((String) request.get("projectUri"))).toRealPath();
   if (!module.startsWith(root) || !Files.isDirectory(module) || monitor.isCanceled()) throw new IllegalArgumentException();
   String kind = (String) request.get("kind");
   if (!List.of("plain", "gradle").contains(kind)) throw new IllegalArgumentException();
   var workspace = ResourcesPlugin.getWorkspace();
   var project = workspace.getRoot().getProject("groovy-default-fixture-" + Integer.toUnsignedString(module.toString().hashCode()));
   if (!project.exists()) {
    var description = workspace.newProjectDescription(project.getName());
    description.setLocationURI(module.toUri());
    project.create(description, monitor); project.open(monitor);
   }
   if (!project.getLocation().toFile().toPath().toRealPath().equals(module)) throw new IllegalArgumentException();
   var description = project.getDescription();
   description.setNatureIds("gradle".equals(kind) ? new String[]{"org.eclipse.buildship.core.gradleprojectnature"} : new String[0]);
   project.setDescription(description, monitor);
   return Map.of("kind", kind, "gradleNature", ProjectUtils.isGradleProject(project), "open", project.isOpen());
  }
  if (command.equals("groovy.fixture.unlink.prepare")) {''')
        source.write_text(fixture)
    if workspace_states:
        fixture = source.read_text().replace('  WorkspaceJob job =', '  String[] gateOutcome = {"no ownership change requested"};\n  WorkspaceJob job =')
        fixture = fixture.replace('      executeCommand("groovy.fixture.unlink.natures", List.of(Map.of("rootUri", request.get("rootUri"), "projectUri", request.get("projectUri"), "dual", true)), progress);',
            '      gateOutcome[0] = String.valueOf(executeCommand("groovy.fixture.unlink.natures", List.of(Map.of("rootUri", request.get("rootUri"), "projectUri", request.get("projectUri"), "dual", true)), progress));')
        fixture = fixture.replace('    } catch (Exception failure) { return Status.CANCEL_STATUS; }',
            '    } catch (Exception failure) { gateOutcome[0] = failure.getClass().getName() + ": " + failure.getMessage(); return Status.CANCEL_STATUS; }')
        fixture = fixture.replace('Files.writeString(finished,"finished")', 'Files.writeString(finished,"finished|" + gateOutcome[0])')
        fixture = fixture.replace('     var description = project.getDescription();',
            '     project.getFile(".project").refreshLocal(org.eclipse.core.resources.IResource.DEPTH_ZERO, monitor);\n     var description = project.getDescription();')
        source.write_text(fixture)
        fixture = source.read_text().replace('frame.getClassName().equals("dev.made.groovy.ProjectUnlinkHandler")',
            '(frame.getClassName().equals("dev.made.groovy.ProjectUnlinkHandler") || frame.getClassName().equals("dev.made.groovy.GradleImportHandler"))')
        source.write_text(fixture)
        fixture = source.read_text().replace('  if (arguments == null',
            '  if (command.equals("groovy.fixture.unlink.projectState") && monitor.isCanceled()) throw new org.eclipse.core.runtime.OperationCanceledException();\n  if (arguments == null', 1)
        fixture = fixture.replace('  if (command.equals("groovy.fixture.unlink.prepare")) {', '''
  if (command.equals("groovy.fixture.unlink.projectState")) {
   if (!request.keySet().equals(java.util.Set.of("rootUri", "projectUri", "operation")) || monitor.isCanceled()) throw new IllegalArgumentException();
   Path module = Path.of(new URI((String) request.get("projectUri"))).toRealPath();
   if (!module.startsWith(root) || !Files.isDirectory(module)) throw new IllegalArgumentException();
   org.eclipse.core.resources.IProject selected = null;
   for (var project : ResourcesPlugin.getWorkspace().getRoot().getProjects()) {
    if (project.getLocation() != null && project.getLocation().toFile().toPath().toAbsolutePath().normalize().equals(module)) {
     if (selected != null) throw new IllegalArgumentException();
     selected = project;
    }
   }
   if (selected == null && "recreateClosed".equals(request.get("operation"))) {
    Path file = module.resolve(".project");
    if (!Files.isRegularFile(file, java.nio.file.LinkOption.NOFOLLOW_LINKS) || Files.size(file) > 1024 * 1024) throw new IllegalArgumentException();
    var workspace = ResourcesPlugin.getWorkspace();
    var description = workspace.loadProjectDescription(org.eclipse.core.runtime.IPath.fromOSString(file.toString()));
    if (description.getName() == null || description.getName().length() > 256 || !workspace.validateName(description.getName(), org.eclipse.core.resources.IResource.PROJECT).isOK()) throw new IllegalArgumentException();
    selected = workspace.getRoot().getProject(description.getName());
    if (selected.exists()) throw new IllegalArgumentException();
    description.setLocationURI(module.toUri());
    selected.create(description, monitor);
    if (selected.isOpen()) throw new IllegalStateException("Recovered fixture project must remain closed");
   }
   if (selected == null && "state".equals(request.get("operation"))) {
    var projects = ResourcesPlugin.getWorkspace().getRoot().getProjects();
    if (projects.length > 128) throw new IllegalArgumentException();
    return Map.of("missing", true, "catalog", java.util.Arrays.stream(projects).map(project -> Map.of(
      "name", project.getName(), "exists", project.exists(), "open", project.isOpen(),
      "location", project.getLocation() == null ? "unavailable" : project.getLocation().toString())).toList());
   }
   if (selected == null || !selected.exists()) throw new IllegalArgumentException();
   if ("close".equals(request.get("operation"))) {
    if (!selected.isOpen()) throw new IllegalArgumentException();
    selected.close(monitor);
   } else if ("open".equals(request.get("operation"))) {
    if (selected.isOpen()) throw new IllegalArgumentException();
    selected.open(monitor);
   } else if (!"state".equals(request.get("operation")) && !"recreateClosed".equals(request.get("operation"))) throw new IllegalArgumentException();
   return Map.of("version", 1, "rootUri", root.toUri().toString(), "projectUri", module.toUri().toString(),
    "name", selected.getName(), "exists", selected.exists(), "open", selected.isOpen(), "accessible", selected.isAccessible());
  }
  if (command.equals("groovy.fixture.unlink.prepare")) {''', 1)
        source.write_text(fixture)
    if scoped_import_modes:
        fixture = source.read_text().replace('  if (command.equals("groovy.fixture.unlink.prepare")) {', '''
  if (command.equals("groovy.fixture.unlink.importHostState")) {
   var preferences = JavaLanguageServerPlugin.getPreferencesManager().getPreferences();
   return Map.of("workspaceRootUris", preferences.getRootPaths().stream()
     .map(path -> path.toFile().toPath().toAbsolutePath().normalize().toUri().toString()).sorted().toList(),
    "gradleArguments", List.copyOf(preferences.getGradleArguments()),
    "gradleJvmArguments", List.copyOf(preferences.getGradleJvmArguments()),
    "offline", preferences.isImportGradleOfflineEnabled(),
    "wrapperEnabled", preferences.isGradleWrapperEnabled(),
    "gradleHome", String.valueOf(preferences.getGradleHome()),
    "gradleJavaHome", String.valueOf(preferences.getGradleJavaHome()),
    "gradleVersion", String.valueOf(preferences.getGradleVersion()));
  }
  if (command.equals("groovy.fixture.unlink.prepare")) {''', 1)
        source.write_text(fixture)
    classes = directory / "unlink-fixture-classes"
    classes.mkdir()
    plugins = arguments.launcher.resolve().parent.parent / "plugins"
    classpath = os.pathsep.join(str(path) for path in plugins.glob("*.jar"))
    try:
        subprocess.run([str(arguments.java_home.resolve() / "bin/javac"), "--release", "17", "-Xlint:all", "-Werror",
                        "-classpath", classpath, "-d", str(classes), str(source)], check=True, capture_output=True)
    except subprocess.CalledProcessError as failure:
        if scoped_import_modes:
            evidence = arguments.receipt.parent
            evidence.mkdir(parents=True, exist_ok=True)
            (evidence / "scoped-import-fixture-compile.java").write_bytes(source.read_bytes())
            (evidence / "scoped-import-fixture-compile.log").write_bytes((failure.stdout or b"")[:1048576] + (failure.stderr or b"")[:1048576])
            (evidence / "scoped-import-fixture-compile.json").write_text(json.dumps({
                "exitCode": failure.returncode, "stdoutBytes": len(failure.stdout or b""),
                "stderrBytes": len(failure.stderr or b""), "streamLimitBytes": 1048576,
            }, indent=2) + "\n")
        raise
    bundle = directory / "unlink-fixture.jar"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("META-INF/MANIFEST.MF", "Manifest-Version: 1.0\nBundle-ManifestVersion: 2\nBundle-SymbolicName: dev.made.groovy.unlink.fixture;singleton:=true\nBundle-Version: 1.0.0\nBundle-RequiredExecutionEnvironment: JavaSE-17\nRequire-Bundle: org.eclipse.jdt.ls.core,org.eclipse.core.resources,\n org.eclipse.core.runtime,org.eclipse.lsp4j\n\n")
        commands = "".join(f'<command id="groovy.fixture.unlink.{name}"/>' for name in
                           ["prepare", "hold", "stack"] + (["natures"] if identity_modes or ownership_modes or hydration_modes or default_root_modes else []) + (["defaultRootsOracle", "gradleDefaultRootsOracle", "gradleDefaultRootsState"] if default_root_modes else []) + (["projectState"] if workspace_states else []) + (["importHostState"] if scoped_import_modes else []))
        archive.writestr("plugin.xml", '<plugin><extension point="org.eclipse.jdt.ls.core.delegateCommandHandler"><delegateCommandHandler class="dev.made.groovy.fixture.UnlinkFixtureHandler">' + commands + '</delegateCommandHandler></extension></plugin>')
        for path in classes.rglob("*.class"):
            archive.write(path, path.relative_to(classes).as_posix())
    return bundle, source


def qualify_unlink_wait(client, root, selection, command, receipt, journey, replace_root=False, ownership_change=None, input_change=None, descriptor_change=None, descriptor_contents=None, replace_root_directory=False, during_wait=None):
    import queue
    association_read = command == "groovy.gradle.importedBuildRoots"
    scoped_import = command == "groovy.gradle.importSelected.v2"
    importing = scoped_import or command.endswith(("importSelected", "importReactor", "importReactorWithMasks"))
    module_state = command in {"groovy.gradle.applyModuleStates", "groovy.gradle.applyModuleStates.v2"}
    module_read = command == "groovy.gradle.moduleStates.v2"
    root_read = command in {"groovy.gradle.rootState", "groovy.gradle.rootState.v2"}
    root_mutation = command in {"groovy.gradle.applyRootState", "groovy.gradle.applyRootState.v2"}
    phase_name = "cached Gradle build-root read" if association_read else "Gradle root state change" if root_mutation else "Gradle root state read" if root_read else "Gradle module state read" if module_read else "Gradle module state" if module_state else "selected import" if importing else "unlink"
    prepared = {} if association_read or root_mutation or root_read or module_state or module_read or ownership_change is not None or input_change is not None or importing else client.request("workspace/executeCommand", {"command": "groovy.fixture.unlink.prepare", "arguments": [{"rootUri": root.as_uri(), "projectUri": (root / "application").as_uri()}]}, timeout=30)
    hold_parameters = {"rootUri": root.as_uri()}
    if ownership_change is not None:
        hold_parameters.update(projectUri=ownership_change.as_uri(), changeOwnership=True)
    if input_change is not None:
        hold_parameters.update(projectUri=input_change.as_uri(), changeInputs=True)
    held = client.request("workspace/executeCommand", {"command": "groovy.fixture.unlink.hold", "arguments": [hold_parameters]}, timeout=30)
    paths = {name: Path(held[name]) for name in ["started", "release", "finished"]}
    if any(path.parent != root.parent for path in paths.values()):
        raise AssertionError("Qualification gate escaped its disposable fixture")
    request_id = None
    detached = None
    try:
        deadline = time.monotonic() + 10
        while not paths["started"].exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not paths["started"].exists():
            raise AssertionError("Isolated workspace rule was not held")
        request_id = client.next_id
        client.next_id += 1
        client.send({"jsonrpc": "2.0", "id": request_id, "method": "workspace/executeCommand", "params": {"command": command, "arguments": [selection]}})
        deadline = time.monotonic() + 10
        waiting = False
        while time.monotonic() < deadline:
            waiting = client.request("workspace/executeCommand", {"command": "groovy.fixture.unlink.stack", "arguments": []}, timeout=10).get("unlinkRequestWaiting") is True
            if waiting:
                break
            time.sleep(0.02)
        if not waiting:
            raise AssertionError("Unlink request did not wait inside the existing workspace rule owner")
        receipt["checks"].append(f"real {phase_name} request waits on the held Eclipse workspace rule without early receipt")
        if during_wait is not None:
            during_wait()
            paths["release"].write_text("release\n")
        elif descriptor_change is not None:
            descriptor_change.write_bytes(descriptor_contents if descriptor_contents is not None else descriptor_change.read_bytes() + b"\n")
            paths["release"].write_text("release\n")
        elif ownership_change is not None or input_change is not None:
            paths["release"].write_text("release\n")
        elif replace_root or replace_root_directory:
            detached = root.with_name(root.name + "-detached")
            root.rename(detached)
            if replace_root_directory:
                shutil.copytree(detached, root, symlinks=True)
                original = detached.stat()
                replacement = root.stat()
                if (original.st_dev, original.st_ino) == (replacement.st_dev, replacement.st_ino):
                    raise AssertionError("Fixture did not replace directory identity")
                journey["samePathDirectoryReplacement"] = {"originalDevice": original.st_dev, "originalInode": original.st_ino,
                    "replacementDevice": replacement.st_dev, "replacementInode": replacement.st_ino}
            else:
                root.symlink_to(root.parent / "unrelated", target_is_directory=True)
            paths["release"].write_text("release\n")
        else:
            client.notify("$/cancelRequest", {"id": request_id})
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
                    raise AssertionError("Unexpected provider client request during cancellation")
                client.send({"jsonrpc": "2.0", "id": message["id"], "result": [{} for _ in message.get("params", {}).get("items", [])]})
            elif "method" in message:
                with client.notifications_lock:
                    client.notifications.append(message)
        journey.setdefault("heldPhaseResponses", []).append({"command": command, "response": response})
        error = response.get("error", {}) if response is not None else {}
        cancelled = error.get("code") == -32800 or "cancelled" in error.get("message", "").lower()
        ownership_rejected = (module_state and "Gradle module state change failed or may be partial" in error.get("message", "")
            or ownership_change is not None and importing and "Selected Maven import failed or may be partial" in error.get("message", ""))
        expected_failure = cancelled or ownership_rejected
        if scoped_import and (during_wait is not None or replace_root or replace_root_directory or ownership_change is not None
                              or input_change is not None or descriptor_change is not None):
            expected_failure = cancelled or "Selected Gradle import failed or may be partial" in error.get("message", "")
        if root_mutation:
            expected_failure = cancelled or "Gradle root state change failed or may be partial" in error.get("message", "")

        if root_read:
            expected_failure = cancelled or "Gradle root state unavailable" in error.get("message", "")

        if module_read:
            expected_failure = cancelled or "Gradle module states unavailable" in error.get("message", "")

        if module_state:
            if replace_root:
                expected_failure = "Gradle module state change cancelled and may be partial." in error.get("message", "")
            else:
                expected_failure = ownership_rejected if descriptor_change is not None or ownership_change is not None else cancelled
        if command == "groovy.gradle.applyModuleStates.v2":
            expected_failure = cancelled or ownership_rejected
        association_partial = (association_read and response is not None
            and isinstance(response.get("result"), dict) and response["result"].get("partial") is True
            and response["result"].get("cachedOnly") is True and response["result"].get("readOnly") is True
            and response["result"].get("fullSynchronizationKnown") is False)
        if association_read:
            expected_failure = (association_partial or cancelled
                or "Gradle imported project identities unavailable" in error.get("message", ""))
        if response is None or ("error" not in response and not association_partial) or not expected_failure:
            raise AssertionError("Held selected phase returned success or no bounded expected failure")
        if association_read:
            if not association_partial and str(root) in json.dumps(response):
                raise AssertionError("Cached association failure exposes a host path")
            journey.setdefault("heldAssociationReads", []).append(response)
            receipt["checks"].append("held cached build-root read cancels, rejects replaced authority or reports incomplete resource data without mutation")
        elif root_mutation:
            if str(root) in json.dumps(response):
                raise AssertionError("Root mutation failure is not sanitized")
            receipt["checks"].append("held root mutation rechecks identity/description or cancels before resource mutation")
        elif root_read:
            if str(root) in json.dumps(response):
                raise AssertionError("Root read failure is not sanitized")
            receipt["checks"].append("held root state read rechecks descriptor/root or cancels without resource mutation")
        elif module_read:
            if str(root) in json.dumps(response):
                raise AssertionError("Scoped module read failure is not sanitized")
            receipt["checks"].append("held scoped module read rechecks identity/description or cancels without resource mutation")
        elif module_state:
            if str(root) in json.dumps(response):
                raise AssertionError("Gradle module-state failure is not sanitized")
            receipt["checks"].append("held Gradle module-state phase rechecks current scope/descriptor/natures or cancels before mutation")
        elif input_change is not None:
            if str(root) in json.dumps(response) or "Selected" not in json.dumps(response):
                raise AssertionError("Changed-input phase error is not sanitized")
            journey.setdefault("changedInputResponses", []).append(response)
            receipt["checks"].append("held selected phase rechecks changed POM bytes before configuration")
        elif ownership_change is not None:
            if str(root) in json.dumps(response) or "Selected" not in json.dumps(response):
                raise AssertionError("Changed-ownership unlink error is not sanitized")
            journey.setdefault("changedOwnershipResponses", []).append(response)
            receipt["checks"].append(f"held {phase_name} rechecks changed build natures and rejects before mutation")
        elif replace_root or (scoped_import and replace_root_directory):
            if str(root) in json.dumps(response) or "Selected" not in json.dumps(response):
                raise AssertionError("Changed-root unlink error is not sanitized")
            journey["changedRootUnlinkResponse"] = response
            receipt["checks"].append("held scoped import rejects replaced workspace without a success receipt; phase may be partial"
                                     if scoped_import else f"held {phase_name} rejects replaced root before mutation with sanitized failure")
        elif scoped_import and descriptor_change is not None:
            if str(root) in json.dumps(response):
                raise AssertionError("Scoped import input failure is not sanitized")
            receipt["checks"].append("held scoped import rejects changed build inputs without a success receipt; phase may be partial")
        elif scoped_import and during_wait is not None:
            if str(root) in json.dumps(response):
                raise AssertionError("Scoped import scope/host failure is not sanitized")
            receipt["checks"].append("held scoped import rejects changed scope/host state without a success receipt")
        else:
            journey["cancelledUnlinkResponse"] = response
            receipt["checks"].append(f"in-flight held {phase_name} cancellation rejects a success receipt")
    finally:
        if detached is not None:
            if root.is_symlink():
                root.unlink()
            elif replace_root_directory and root.is_dir():
                shutil.rmtree(root)
            detached.rename(root)
        paths["release"].write_text("release\n")
        deadline = time.monotonic() + 10
        while not paths["finished"].exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not paths["finished"].exists():
            raise AssertionError("Isolated owned workspace gate did not finish")
        if module_state:
            outcome = paths["finished"].read_text()
            journey.setdefault("heldGateOutcomes", []).append(outcome)
            if ownership_change is not None and "{dualNature=true}" not in outcome:
                raise AssertionError("Isolated gate did not establish actual dual ownership before rejection")
        for path in paths.values():
            if path.is_file():
                path.unlink()
    return prepared.get("diagnosticUri")


def imported_resource_membership(client, root):
    values = client.request("workspace/executeCommand", {
        "command": "java.project.getAll", "arguments": [json.dumps({"includeNonJava": True})],
    }, timeout=30)
    if not isinstance(values, list) or len(values) > 128:
        raise AssertionError("Imported resource membership is missing or exceeds the bound")
    result = {}
    for uri in values:
        if not isinstance(uri, str) or len(uri) > 4096 or any(ord(character) < 32 for character in uri):
            raise AssertionError("Imported resource identity is malformed or exceeds the bound")
        parsed = urlparse(uri)
        if parsed.scheme != "file" or parsed.netloc not in ("", "localhost") or parsed.query or parsed.fragment:
            raise AssertionError("Imported resource identity is not a local file URI")
        path = Path(unquote(parsed.path)).resolve(strict=True)
        if not path.is_dir() or not path.is_relative_to(root):
            raise AssertionError("Imported resource identity escapes the disposable fixtures")
        identity = path.relative_to(root).as_posix()
        if identity in result:
            raise AssertionError("Imported resource membership contains aliased identities")
        result[identity] = uri
    return result


def qualify_imported_resources(client, root, build_root, provider, java_members,
                               non_java_members, java_membership, journey, receipt):
    before = imported_resource_membership(client, root)
    if set(before) != java_members | non_java_members or java_membership(client) != java_members:
        raise AssertionError("Java and non-Java imported resource identities differ from the fixture")
    journey["importedResourcesBefore"] = before
    journey["javaSemanticProjects"] = sorted(java_members)
    journey["nonJavaResources"] = sorted(non_java_members)
    journey["resourceProviderKindKnownFromEnumeration"] = False
    receipt["checks"].append("existing includeNonJava query retains exact aggregate/base resources separately from Java semantics")
    for identity in sorted(non_java_members):
        try:
            settings = client.request("workspace/executeCommand", {"command": "java.project.getSettings",
                "arguments": [before[identity], ["org.eclipse.jdt.ls.core.natureIds"]]}, timeout=30)
        except RuntimeError:
            journey.setdefault("resourceNatureIds", {})[identity] = None
            receipt["checks"].append("unsupported resource settings leave nature and semantic facts unknown")
        else:
            natures = settings.get("org.eclipse.jdt.ls.core.natureIds") if isinstance(settings, dict) else None
            required_nature = "org.eclipse.m2e.core.maven2Nature" if provider == "maven" else "org.eclipse.buildship.core.gradleprojectnature"
            if (not isinstance(natures, list) or len(natures) > 32
                    or not all(isinstance(nature, str) and len(nature) <= 4096 for nature in natures)
                    or required_nature not in natures or "org.eclipse.jdt.core.javanature" in natures):
                raise AssertionError("Imported resource nature differs from its actual non-Java build ownership")
            journey.setdefault("resourceNatureIds", {})[identity] = natures
            receipt["checks"].append("existing nature-only settings identify build ownership without Java nature or invented semantics")
    if imported_resource_membership(client, root) != before or java_membership(client) != java_members:
        raise AssertionError("Read-only membership/settings requests mutated provider identities")
    receipt["checks"].append("membership and unsupported semantic reads preserve imported resources")
    request = {"rootUri": build_root.as_uri(), "projectUris": [before[identity] for identity in sorted(non_java_members)]}
    reply = client.request("workspace/executeCommand", {
        "command": f"groovy.{provider}.unlinkSelected", "arguments": [request],
    }, timeout=30)
    if (set(reply) != {"version", "rootUri", "unlinkedProjectUris", "selectedUnlinkPhaseComplete", "fullSynchronizationKnown"}
            or reply["version"] != 1 or reply["selectedUnlinkPhaseComplete"] is not True
            or reply["fullSynchronizationKnown"] is not False
            or Path(unquote(urlparse(reply["rootUri"]).path)).resolve() != build_root
            or {Path(unquote(urlparse(uri).path)).resolve() for uri in reply["unlinkedProjectUris"]}
                != {root / identity for identity in non_java_members}):
        raise AssertionError("Non-Java unlink returned an invalid or overclaiming phase receipt")
    journey["nonJavaUnlinkReceipt"] = reply
    if set(imported_resource_membership(client, root)) != java_members or java_membership(client) != java_members:
        raise AssertionError("Selected non-Java unlink removed another imported resource or Java project")
    receipt["checks"].append("guarded unlink removes only exact non-Java resources and preserves Java/unrelated projects")
    import_request = {"rootUri": build_root.as_uri(), "offline": True}
    if provider == "maven":
        import_request["pomUris"] = [(build_root / "pom.xml").as_uri()]
    reply = client.request("workspace/executeCommand", {
        "command": f"groovy.{provider}.importSelected", "arguments": [import_request],
    }, timeout=90)
    if reply.get("version") != 1 or reply.get("selectedImportPhaseComplete") is not True or reply.get("fullSynchronizationKnown") is not False:
        raise AssertionError("Non-Java resource repair overclaims synchronization")
    journey["nonJavaImportReceipt"] = reply
    if imported_resource_membership(client, root) != before or java_membership(client) != java_members:
        raise AssertionError("Guarded repair did not restore the exact imported resources")
    receipt["checks"].append("guarded offline repair restores non-Java resources without claiming full synchronization")


def probe_import_selection(arguments):
    provider = arguments.provider.resolve()
    if hashlib.sha256(provider.read_bytes()).hexdigest() != arguments.sha256:
        raise AssertionError("Provider binary differs from the source build")
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"provider_sha256": arguments.sha256,
               "reactorRelationship": "aggregate-only" if arguments.parentless_reactor else "inherited",
               "importCommand": "groovy.maven.importSelected" if arguments.use_groovy_import else "java.project.changeImportedProjects",
               "unlinkCommand": "groovy.maven.unlinkSelected" if arguments.guarded_unlink_only or arguments.resource_membership_only else "java.project.changeImportedProjects",
               "checks": [], "journeys": [], "limits": [
        "Headless JDTLS command qualification only; no product link/settings UI or native/remote/platform qualification",
        "Command acknowledgement and eventual membership do not establish causal update/build completion or cancellation",
        "No Kotlin/Gradle ignore semantics, required module/orphan policy or default delivery qualification",
    ]}
    with tempfile.TemporaryDirectory(prefix="groovy-maven-import-selection-") as directory:
        fixture_bundle = None
        if arguments.guarded_unlink_only:
            fixture_bundle, fixture_source = unlink_fixture_bundle(arguments, directory)
            receipt["testOnlyFixtureSha256"] = hashlib.sha256(fixture_bundle.read_bytes()).hexdigest()
            receipt["testOnlyFixtureSourceSha256"] = hashlib.sha256(fixture_source.read_bytes()).hexdigest()
            shutil.copy2(fixture_source, arguments.receipt.parent / fixture_source.name)
            shutil.copy2(fixture_bundle, arguments.receipt.parent / fixture_bundle.name)
        root = Path(directory).resolve() / "projects"
        root.mkdir()
        first = root / "reactor"
        unrelated = root / "unrelated"
        root_pom = '<project xmlns="http://maven.apache.org/POM/4.0.0"><modelVersion>4.0.0</modelVersion><groupId>evidence</groupId><artifactId>selection-reactor</artifactId><version>1.0</version><packaging>pom</packaging><modules><module>application</module><module>library</module></modules></project>'
        first.mkdir()
        (first / "pom.xml").write_text(root_pom + "\n")
        for module, artifact in [(first / "application", "selection-application"), (first / "library", "selection-library"), (unrelated, "selection-unrelated")]:
            source = module / "src/main/java/evidence/Main.java"
            source.parent.mkdir(parents=True)
            source.write_text('package evidence; public class Main { public static String message() { return "selection"; } }\n')
            coordinates = '<groupId>evidence</groupId><version>1.0</version>'
            if module.is_relative_to(first) and not arguments.parentless_reactor:
                coordinates = '<parent><groupId>evidence</groupId><artifactId>selection-reactor</artifactId><version>1.0</version><relativePath>../pom.xml</relativePath></parent>'
            (module / "pom.xml").write_text(f'<project xmlns="http://maven.apache.org/POM/4.0.0"><modelVersion>4.0.0</modelVersion>{coordinates}<artifactId>{artifact}</artifactId><properties><maven.compiler.release>17</maven.compiler.release></properties><build><plugins><plugin><groupId>org.apache.maven.plugins</groupId><artifactId>maven-compiler-plugin</artifactId><version>3.14.1</version></plugin></plugins></build></project>\n')
        if arguments.exclusion_only or arguments.guarded_unlink_only or arguments.resource_membership_only:
            application_pom = first / "application/pom.xml"
            application_pom.write_text(application_pom.read_text().replace("</project>",
                '<dependencies><dependency><groupId>evidence</groupId><artifactId>selection-library</artifactId><version>1.0</version></dependency></dependencies></project>'))
            specification = importlib.util.spec_from_file_location("groovy_provider_build", REPOSITORY / "script/build-groovy-gradle-provider.py")
            if specification is None or specification.loader is None:
                raise RuntimeError("Cannot load pinned provider build inputs")
            build_inputs = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(build_inputs)
            receipt["pinned_input_sha256"] = {}
            for name, expected in build_inputs.PINNED_ARTIFACTS.items():
                artifact = arguments.launcher.resolve().parent.parent / "plugins" / name
                digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
                if digest != expected:
                    raise AssertionError("Pinned provider input changed: " + name)
                receipt["pinned_input_sha256"][name] = digest
            receipt["qualification"] = ("existing non-Java imported membership, guarded unlink/repair and restart"
                                        if arguments.resource_membership_only else
                                        "guarded selected unlink phase, cancellation, diagnostics, source preservation and restart"
                                        if arguments.guarded_unlink_only else
                                        "root scan exclusions, explicit selection, required dependency and orphan observations")
        if arguments.guarded_unlink_only or arguments.resource_membership_only:
            for module in [first / "application", first / "library", unrelated]:
                sentinel = module / "target/generated-sources/unlink/Sentinel.java"
                sentinel.parent.mkdir(parents=True)
                sentinel.write_text("package evidence; public class Sentinel {}\n")
        originals = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in root.rglob("*") if path.is_file()}
        receipt["fixture_sha256"] = originals
        repository = Path(directory) / "repository"
        shutil.copytree(arguments.maven_repository.resolve(), repository, copy_function=cache_copy,
                        ignore=shutil.ignore_patterns("*.lock"))
        settings_path = Path(directory) / "settings.xml"
        settings_path.write_text(f'<settings><localRepository>{escape(str(repository))}</localRepository><offline>true</offline></settings>\n')
        settings = {"java": {
            "autobuild": {"enabled": False},
            "import": {"maven": {"enabled": True, "offline": {"enabled": True}}, "gradle": {"enabled": False}},
            "configuration": {"maven": {"userSettings": str(settings_path)}, "updateBuildConfiguration": "disabled"},
        }}
        expected_all = {"reactor/application", "reactor/library", "unrelated"}
        expected_selected = {"reactor/application", "unrelated"}

        def membership(client):
            result = client.request("workspace/executeCommand", {"command": "java.project.getAll", "arguments": []}, timeout=15)
            paths = set()
            for uri in result or []:
                parsed = urlparse(uri)
                if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
                    raise AssertionError("Import returned a nonlocal URI")
                path = Path(unquote(parsed.path)).resolve()
                if not path.is_relative_to(root):
                    raise AssertionError("Import returned a project outside disposable fixtures")
                paths.add(path.relative_to(root).as_posix())
            return paths

        def await_membership(client, expected, journey, label):
            deadline = time.monotonic() + 90
            observed = set()
            while time.monotonic() < deadline:
                try:
                    observed = membership(client)
                except RuntimeError as error:
                    if not arguments.exclusion_only or "busy" not in str(error).lower():
                        raise
                    continue
                if observed == expected:
                    journey["states"].append({"label": label, "members": sorted(observed)})
                    receipt["checks"].append(label)
                    print(label, flush=True)
                    return
                time.sleep(0.2)
            journey["states"].append({"label": label, "expected": sorted(expected), "members": sorted(observed)})
            raise AssertionError(f"{label}: membership did not match selected fixtures")

        def preserve_sources(label):
            for relative, digest in originals.items():
                path = root / relative
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise AssertionError("Import selection changed or removed a fixture source/POM")
            if list(root.rglob("TEST-*.xml")):
                raise AssertionError("Import selection emitted selected test reports")
            receipt["checks"].append(label)

        def run(label, data, selected, actions=False, exclusions=(), selected_poms=None, expected_members=None):
            journey = {"label": label, "states": [], "commands": [], "passed": False}
            receipt["journeys"].append(journey)
            client = client_class()(arguments.launcher.resolve(), arguments.java_home.resolve(), root, data, deny_external_network=True)
            try:
                current_settings = json.loads(json.dumps(settings))
                if exclusions:
                    current_settings["java"]["import"]["exclusions"] = list(exclusions)
                options = {"bundles": [str(provider)] + ([str(fixture_bundle)] if fixture_bundle else []), "settings": current_settings, "workspaceFolders": [first.as_uri(), unrelated.as_uri()]}
                if selected:
                    options["projectConfigurations"] = [(first / "application/pom.xml").as_uri(), (unrelated / "pom.xml").as_uri()] if selected_poms is None else [path.as_uri() for path in selected_poms]
                journey["importExclusions"] = list(exclusions)
                journey["selectedPomUris"] = options.get("projectConfigurations", [])
                initialized = client.request("initialize", {
                    "processId": os.getpid(), "rootUri": root.as_uri(),
                    "workspaceFolders": [{"uri": path.as_uri(), "name": path.name} for path in [first, unrelated]],
                    "capabilities": {"workspace": {"workspaceFolders": True, "configuration": True}},
                    "initializationOptions": options,
                }, timeout=90)
                journey["serverInfo"] = initialized.get("serverInfo")
                commands = initialized["capabilities"]["executeCommandProvider"]["commands"]
                required = {"java.project.getAll", "java.project.changeImportedProjects"}
                if arguments.use_groovy_import:
                    required.add("groovy.maven.importSelected")
                if arguments.guarded_unlink_only:
                    required.update({"groovy.maven.unlinkSelected", "groovy.gradle.unlinkSelected"})
                if not required.issubset(commands):
                    raise AssertionError("Pinned server does not advertise selection commands")
                receipt["checks"].append(f"{label}: advertised selection commands")
                client.notify("initialized", {})
                expected = expected_members if expected_members is not None else expected_selected if selected else expected_all
                await_membership(client, expected, journey, f"{label}: initial module identities")
                if arguments.resource_membership_only:
                    if not {"groovy.maven.unlinkSelected", "groovy.maven.importSelected"}.issubset(commands):
                        raise AssertionError("Guarded resource mutation commands unavailable")
                    qualify_imported_resources(client, root, first, "maven", expected, {"reactor"}, membership, journey, receipt)
                if (arguments.exclusion_only or arguments.guarded_unlink_only) and "reactor/application" in expected:
                    details = client.request("workspace/executeCommand", {"command": "java.project.getSettings", "arguments": [(first / "application").as_uri(), ["org.eclipse.jdt.ls.core.classpathEntries"]]}, timeout=30)
                    entries = details.get("org.eclipse.jdt.ls.core.classpathEntries", [])
                    journey["applicationProjectReferences"] = [entry.get("path") for entry in entries if entry.get("kind") == 2]
                    journey["applicationExternalEntries"] = sum(entry.get("kind") == 1 for entry in entries)
                    journey["requiredLibraryImported"] = "reactor/library" in expected
                    if "reactor/library" in expected:
                        if not any(str(path).endswith("selection-library") for path in journey["applicationProjectReferences"]):
                            raise AssertionError("Imported required library has no provider project classpath reference")
                        receipt["checks"].append(f"{label}: required library provider classpath reference")
                if actions:
                    application_uri = (first / "application").as_uri()
                    params = {"command": "java.project.changeImportedProjects", "arguments": [[], [], [application_uri]]}
                    if arguments.guarded_unlink_only:
                        unlink_request = {"rootUri": first.as_uri(), "projectUris": [application_uri]}
                        escaped_module = first / "escaped-unlink"
                        escaped_module.symlink_to(unrelated, target_is_directory=True)
                        invalid_requests = [
                            ("empty unlink selection", {**unlink_request, "projectUris": []}),
                            ("duplicate unlink identity", {**unlink_request, "projectUris": [application_uri] * 2}),
                            ("unlink count bound", {**unlink_request, "projectUris": [application_uri] * 129}),
                            ("foreign unlink identity", {**unlink_request, "projectUris": [unrelated.as_uri()]}),
                            ("escaping symlink unlink identity", {**unlink_request, "projectUris": [escaped_module.as_uri()]}),
                            ("uninitialized unlink root", {**unlink_request, "rootUri": root.as_uri()}),
                            ("missing unlink identity", {**unlink_request, "projectUris": [(first / "private-canary").as_uri()]}),
                            ("nested source is not module identity", {**unlink_request, "projectUris": [(first / "application/src").as_uri()]}),
                            ("nonfile unlink identity", {**unlink_request, "projectUris": ["https://private-canary.invalid/project"]}),
                            ("query unlink identity", {**unlink_request, "projectUris": [application_uri + "?private-canary=1"]}),
                            ("oversized unlink identity", {**unlink_request, "projectUris": ["file:///" + "x" * 4097]}),
                            ("unknown unlink override", {**unlink_request, "deleteContent": True}),
                            ("later missing identity prevents earlier removal", {**unlink_request, "projectUris": [application_uri, (first / "private-canary").as_uri()]}),
                        ]
                        for rejection, invalid in invalid_requests:
                            try:
                                client.request("workspace/executeCommand", {"command": "groovy.maven.unlinkSelected", "arguments": [invalid]}, timeout=30)
                            except RuntimeError as error:
                                if "Maven unlink" not in str(error) or "private-canary" in str(error) or str(root) in str(error):
                                    raise AssertionError("Unlink failure is not sanitized") from error
                            else:
                                raise AssertionError("Invalid unlink selection succeeded")
                            if membership(client) != expected_all:
                                raise AssertionError("Rejected unlink mutated another project")
                            receipt["checks"].append(rejection + ": sanitized and membership unchanged")
                        escaped_module.unlink()
                        try:
                            client.request("workspace/executeCommand", {"command": "groovy.gradle.unlinkSelected", "arguments": [unlink_request]}, timeout=30)
                        except RuntimeError as error:
                            if "Gradle unlink" not in str(error) or str(root) in str(error):
                                raise AssertionError("Wrong-provider rejection is not sanitized") from error
                        else:
                            raise AssertionError("Gradle unlink accepted Maven identity")
                        receipt["checks"].append("wrong-provider unlink rejected without mutation")
                        diagnostic_uri = qualify_unlink_wait(client, first, unlink_request, "groovy.maven.unlinkSelected", receipt, journey)
                        if membership(client) != expected_all:
                            raise AssertionError("Cancelled held unlink changed membership")
                        receipt["checks"].append("held cancellation preserves Maven membership before repaired retry")
                        qualify_unlink_wait(client, first, unlink_request, "groovy.maven.unlinkSelected", receipt, journey, replace_root=True)
                        if membership(client) != expected_all:
                            raise AssertionError("Replaced-root rejection changed membership")
                        preserve_sources("replaced-root failure restores original Maven source identities")
                        params = {"command": "groovy.maven.unlinkSelected", "arguments": [unlink_request]}
                    reply = client.request("workspace/executeCommand", params, timeout=30)
                    if arguments.guarded_unlink_only:
                        expected_keys = {"version", "rootUri", "unlinkedProjectUris", "selectedUnlinkPhaseComplete", "fullSynchronizationKnown"}
                        if set(reply) != expected_keys or reply["version"] != 1 or reply["selectedUnlinkPhaseComplete"] is not True or reply["fullSynchronizationKnown"] is not False:
                            raise AssertionError("Unlink phase receipt differs or overclaims synchronization")
                        if Path(unquote(urlparse(reply["rootUri"]).path)).resolve() != first or [Path(unquote(urlparse(uri).path)).resolve() for uri in reply["unlinkedProjectUris"]] != [first / "application"]:
                            raise AssertionError("Unlink phase receipt identifies another project")
                        receipt["checks"].append("exact selected unlink phase receipt keeps full synchronization unknown")
                        if membership(client) != expected_all - {"reactor/application"}:
                            raise AssertionError("Unlink replied before membership removal")
                        with client.notifications_lock:
                            deletions = [message for message in client.notifications if message.get("method") == "language/eventNotification" and message.get("params", {}).get("eventType") == 210]
                        if not any([Path(unquote(urlparse(uri).path)).resolve() for uri in event.get("params", {}).get("data", [])] == [first / "application"] for event in deletions):
                            raise AssertionError("Unlink did not emit exact ProjectsDeleted identities")
                        receipt["checks"].append("request-bound removal and exact ProjectsDeleted event")
                        with client.notifications_lock:
                            cleared = [message for message in client.notifications if message.get("method") == "textDocument/publishDiagnostics" and message.get("params", {}).get("diagnostics") == []]
                        expected_diagnostic = Path(unquote(urlparse(diagnostic_uri).path)).resolve()
                        if not any(Path(unquote(urlparse(message.get("params", {}).get("uri", "")).path)).resolve() == expected_diagnostic for message in cleared):
                            raise AssertionError("Unlink did not clear its imported resource marker diagnostics")
                        receipt["checks"].append("imported resource marker diagnostics cleared on exact selected source URI")
                    journey["commands"].append({"request": params, "reply": reply})
                    await_membership(client, expected_all - {"reactor/application"}, journey, "selected unlink preserves library and unrelated root")
                    preserve_sources("selected unlink preserves all source/POM and generated sentinel hashes")
                    import_request = {"rootUri": first.as_uri(), "pomUris": [(first / "application/pom.xml").as_uri()], "offline": True}
                    if arguments.use_groovy_import:
                        invalid_directory = first / "invalid-selection"
                        invalid_directory.mkdir()
                        invalid_pom = invalid_directory / "pom.xml"
                        invalid_pom.write_text("<project><malformed>\n")
                        escaped_directory = first / "escaped-selection"
                        escaped_directory.symlink_to(unrelated, target_is_directory=True)
                        invalid_requests = [
                            ("foreign POM rejected", {**import_request, "pomUris": [(unrelated / "pom.xml").as_uri()]}),
                            ("symlink escape rejected", {**import_request, "pomUris": [(escaped_directory / "pom.xml").as_uri()]}),
                            ("duplicate POM rejected", {**import_request, "pomUris": import_request["pomUris"] * 2}),
                            ("empty selection rejected", {**import_request, "pomUris": []}),
                            ("selection count bounded", {**import_request, "pomUris": import_request["pomUris"] * 33}),
                            ("non-offline intent rejected", {**import_request, "offline": False}),
                            ("unknown override rejected", {**import_request, "settings": {"network": True}}),
                            ("uninitialized root rejected", {**import_request, "rootUri": root.as_uri()}),
                            ("non-POM input rejected", {**import_request, "pomUris": [(first / "application/src/main/java/evidence/Main.java").as_uri()]}),
                            ("missing POM rejected", {**import_request, "pomUris": [(first / "missing/pom.xml").as_uri()]}),
                            ("oversized URI rejected", {**import_request, "pomUris": ["file:///" + "x" * 4097]}),
                            ("non-string URI rejected", {**import_request, "pomUris": [17]}),
                            ("network URI rejected", {**import_request, "pomUris": ["https://invalid.example/pom.xml"]}),
                            ("URI query rejected", {**import_request, "pomUris": [import_request["pomUris"][0] + "?private=value"]}),
                            ("malformed POM rejected", {**import_request, "pomUris": [invalid_pom.as_uri()]}),
                            ("malformed later POM prevents earlier import", {**import_request, "pomUris": import_request["pomUris"] + [invalid_pom.as_uri()]}),
                        ]
                        try:
                            for rejection, invalid in invalid_requests:
                                try:
                                    client.request("workspace/executeCommand", {"command": "groovy.maven.importSelected", "arguments": [invalid]}, timeout=15)
                                except RuntimeError as failure:
                                    if str(root) in str(failure) or "Selected Maven import" not in str(failure):
                                        raise AssertionError("Selected import failure is not sanitized/actionable") from failure
                                else:
                                    raise AssertionError(f"{rejection}: invalid selected import succeeded")
                                if membership(client) != expected_all - {"reactor/application"}:
                                    raise AssertionError("Rejected selected import mutated provider project membership")
                                receipt["checks"].append(rejection)
                            invalid_pom.write_text('<project><modelVersion>4.0.0</modelVersion><groupId>evidence</groupId><artifactId>selection-unrelated</artifactId><version>1.0</version></project>')
                            try:
                                client.request("workspace/executeCommand", {"command": "groovy.maven.importSelected", "arguments": [{**import_request, "pomUris": [invalid_pom.as_uri()]}]}, timeout=15)
                            except RuntimeError as failure:
                                if str(root) in str(failure) or "may be partial" not in str(failure):
                                    raise AssertionError("Existing project-name collision is not a sanitized partial-failure category") from failure
                            else:
                                raise AssertionError("Existing unrelated project-name collision accepted")
                            if membership(client) != expected_all - {"reactor/application"}:
                                raise AssertionError("Project-name collision changed unrelated membership")
                            preserve_sources("project-name collision preserves unrelated source/POM hashes")
                            canary = "groovy-import-secret-canary"
                            secret_file = Path(directory) / "private-host-secret.txt"
                            secret_file.write_text(canary)
                            invalid_pom.write_text(f'<!DOCTYPE project [<!ENTITY private SYSTEM "{secret_file.as_uri()}">]><project><modelVersion>4.0.0</modelVersion><groupId>evidence</groupId><artifactId>&private;</artifactId><version>1.0</version></project>')
                            try:
                                client.request("workspace/executeCommand", {"command": "groovy.maven.importSelected", "arguments": [{**import_request, "pomUris": [invalid_pom.as_uri()]}]}, timeout=15)
                            except RuntimeError as failure:
                                if canary in str(failure) or str(root) in str(failure):
                                    raise AssertionError("Entity failure leaked private host data") from failure
                            else:
                                raise AssertionError("External-entity selected POM accepted")
                            if membership(client) != expected_all - {"reactor/application"}:
                                raise AssertionError("Entity POM rejection mutated provider membership")
                            receipt["checks"].append("external-entity model rejected without private data or membership mutation")
                            invalid_pom.write_text("x" * (1024 * 1024 + 1))
                            try:
                                client.request("workspace/executeCommand", {"command": "groovy.maven.importSelected", "arguments": [{**import_request, "pomUris": [invalid_pom.as_uri()]}]}, timeout=15)
                            except RuntimeError as failure:
                                if str(root) in str(failure):
                                    raise AssertionError("Oversized POM failure leaked host paths") from failure
                            else:
                                raise AssertionError("Oversized selected POM accepted")
                            if membership(client) != expected_all - {"reactor/application"}:
                                raise AssertionError("Oversized POM rejection mutated provider membership")
                            receipt["checks"].append("selected POM bytes bounded before mutation")
                        finally:
                            invalid_pom.unlink()
                            invalid_directory.rmdir()
                            escaped_directory.unlink()
                        params = {"command": "groovy.maven.importSelected", "arguments": [import_request]}
                    else:
                        params = {"command": "java.project.changeImportedProjects", "arguments": [[(first / "application/pom.xml").as_uri()], [], []]}
                    reply = client.request("workspace/executeCommand", params, timeout=30)
                    if arguments.use_groovy_import:
                        if reply.get("version") != 1 or reply.get("selectedImportPhaseComplete") is not True or reply.get("fullSynchronizationKnown") is not False:
                            raise AssertionError("Selected import receipt overclaims synchronization or differs")
                        if reply["importedPomUris"] != [(first / "application/pom.xml").as_uri()]:
                            raise AssertionError("Selected import receipt includes another module")
                        receipt["checks"].append("selected import receipt keeps full synchronization unknown")
                    journey["commands"].append({"request": params, "reply": reply})
                    await_membership(client, expected_all, journey, "selected POM import restores application without dropping unrelated root")
                preserve_sources(f"{label}: sources unchanged and no test reports")
                journey["passed"] = True
            finally:
                evidence = arguments.receipt.parent / f"import-selection-{label}-{time.time_ns()}"
                evidence.mkdir(parents=True)
                with client.notifications_lock:
                    events = [message for message in client.notifications if message.get("method") == "language/eventNotification"]
                journey["eventNotifications"] = events
                client.stderr.seek(0)
                (evidence / "stderr.log").write_bytes(client.stderr.read())
                log = data / ".metadata/.log"
                if log.exists():
                    shutil.copy2(log, evidence / "jdtls.log")
                journey["evidence"] = str(evidence)
                client.close()
                arguments.receipt.write_text(json.dumps(receipt, indent=2) + "\n")

        all_data = Path(directory) / "all-profile"
        selected_data = Path(directory) / "selected-profile"
        if arguments.resource_membership_only:
            run("resources", all_data, False)
            run("resources-restart", all_data, False)
        elif arguments.exclusion_only:
            run("required-all", all_data, False)
            excluded_data = Path(directory) / "excluded-profile"
            run("excluded-root", excluded_data, False, exclusions=[str(first)], expected_members={"unrelated"})
            run("excluded-root-restart", excluded_data, False, exclusions=[str(first)], expected_members={"unrelated"})
            run("excluded-required-module", Path(directory) / "child-excluded-profile", False, exclusions=[str(first / "library")])
            run("explicit-selection-over-root-exclusion", selected_data, True, exclusions=[str(first)])
            run("explicit-selection-restart", selected_data, True, exclusions=[str(first)])
            run("changed-selection-retains-existing-import", selected_data, True,
                selected_poms=[first / "library/pom.xml", unrelated / "pom.xml"], expected_members=expected_all)
            root_pom_path = first / "pom.xml"
            original_root_pom = root_pom_path.read_bytes()
            try:
                root_pom_path.write_bytes(original_root_pom.replace(b"<module>library</module>", b""))
                originals["reactor/pom.xml"] = hashlib.sha256(root_pom_path.read_bytes()).hexdigest()
                run("removed-module-existing-profile", all_data, False)
                run("removed-module-fresh-profile", Path(directory) / "orphan-fresh-profile", False,
                    expected_members={"reactor/application", "unrelated"})
            finally:
                root_pom_path.write_bytes(original_root_pom)
                originals["reactor/pom.xml"] = hashlib.sha256(original_root_pom).hexdigest()
                preserve_sources("deliberate module declaration edit restored")
            run("restored-module-existing-profile", all_data, False)
        else:
            run("all", all_data, False, actions=True)
            run("all-restart", all_data, False)
            run("selected", selected_data, True)
            run("selected-restart", selected_data, True)
    print(json.dumps(receipt, indent=2))


def probe_required_library(arguments, container_models=False, hydrate_models=False):
    provider = arguments.provider.resolve()
    if hashlib.sha256(provider.read_bytes()).hexdigest() != arguments.sha256:
        raise AssertionError("Provider binary differs from the source build")
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"provider_sha256": arguments.sha256, "containerModelsQualified": container_models, "hydrationQualified": hydrate_models, "checks": [], "journeys": [], "limits": [
        "Pinned headless provider observations only; unsupported replacement or restart behavior is a parity gap",
        "Fixture JAR preparation is not a product install action or a Maven build qualification",
        "No saved ignore/pattern policy, mounted/native/remote error UI, full synchronization or release qualification",
    ]}
    specification = importlib.util.spec_from_file_location("groovy_provider_build", REPOSITORY / "script/build-groovy-gradle-provider.py")
    if specification is None or specification.loader is None:
        raise RuntimeError("Cannot load pinned provider build inputs")
    build_inputs = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(build_inputs)
    receipt["pinned_input_sha256"] = {}
    for name, expected in build_inputs.PINNED_ARTIFACTS.items():
        artifact = arguments.launcher.resolve().parent.parent / "plugins" / name
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if digest != expected:
            raise AssertionError("Pinned provider input changed: " + name)
        receipt["pinned_input_sha256"][name] = digest
    with tempfile.TemporaryDirectory(prefix="groovy-maven-required-library-") as directory:
        temporary = Path(directory).resolve()
        root = temporary / "projects"
        reactor = root / "reactor"
        application = reactor / "application"
        library = reactor / "library"
        unrelated = root / "unrelated"
        coordinates = '<groupId>dev.made.groovy.fixture.ignore</groupId><version>7.40</version>'
        reactor.mkdir(parents=True)
        (reactor / "pom.xml").write_text(f'<project><modelVersion>4.0.0</modelVersion>{coordinates}<artifactId>required-reactor</artifactId><packaging>pom</packaging><modules><module>application</module><module>library</module></modules></project>\n')
        for module, name in [(application, "Application"), (library, "Library"), (unrelated, "Unrelated")]:
            source = module / f"src/main/java/evidence/{name}.java"
            source.parent.mkdir(parents=True)
            expression = "Library.message()" if module == application else '"retained-cache"'
            source.write_text(f'package evidence; public class {name} {{ public static String message() {{ return {expression}; }} }}\n')
            dependency = '<dependencies><dependency><groupId>dev.made.groovy.fixture.ignore</groupId><artifactId>required-library</artifactId><version>7.40</version></dependency></dependencies>' if module == application else ''
            (module / "pom.xml").write_text(f'<project><modelVersion>4.0.0</modelVersion>{coordinates}<artifactId>required-{module.name}</artifactId><properties><maven.compiler.release>17</maven.compiler.release></properties>{dependency}<build><plugins><plugin><groupId>org.apache.maven.plugins</groupId><artifactId>maven-compiler-plugin</artifactId><version>3.14.1</version></plugin></plugins></build></project>\n')
        originals = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in root.rglob("*") if path.is_file()}
        receipt["fixture_sha256"] = originals
        fixture_bundle = None
        if hydrate_models:
            fixture_bundle, fixture_source = unlink_fixture_bundle(arguments, temporary, hydration_modes=True)
            (arguments.receipt.parent / "HydrationFixtureHandler.java").write_bytes(fixture_source.read_bytes())
            receipt["fixtureBundleSha256"] = hashlib.sha256(fixture_bundle.read_bytes()).hexdigest()
            receipt["fixtureSourceSha256"] = hashlib.sha256(fixture_source.read_bytes()).hexdigest()
        classes = temporary / "library-classes"
        classes.mkdir()
        compiler = arguments.java_home.resolve() / "bin/javac"
        artifact_jar = temporary / "required-library-7.40.jar"
        preparation = [str(compiler), "--release", "17", "-d", str(classes), str(library / "src/main/java/evidence/Library.java")]
        subprocess.run(preparation, check=True, capture_output=True, text=True)
        class_file = classes / "evidence/Library.class"
        if class_file.read_bytes()[:8] != bytes.fromhex("cafebabe0000003d"):
            raise AssertionError("Fixture class does not target Java 17")
        with zipfile.ZipFile(artifact_jar, "w") as archive:
            archive.writestr(zipfile.ZipInfo("evidence/Library.class", date_time=(1980, 1, 1, 0, 0, 0)), class_file.read_bytes())
        receipt["fixturePreparation"] = preparation
        receipt["fixtureCompilerSha256"] = hashlib.sha256(compiler.read_bytes()).hexdigest()
        receipt["fixturePackaging"] = "Python ZipFile; one Java-17 class with a fixed fixture timestamp"
        receipt["fixtureArtifactSha256"] = hashlib.sha256(artifact_jar.read_bytes()).hexdigest()
        receipt["checks"].append("unique Java-17 fixture artifact prepared without a build engine or download")
        expected_all = {"reactor/application", "reactor/library", "unrelated"}
        expected_removed = expected_all - {"reactor/library"}
        for cached in [True, False]:
            label = "cached" if cached else "missing"
            repository = temporary / f"repository-{label}"
            shutil.copytree(arguments.maven_repository.resolve(), repository, copy_function=cache_copy,
                            ignore=shutil.ignore_patterns("*.lock"))
            artifact_directory = repository / "dev/made/groovy/fixture/ignore/required-library/7.40"
            if artifact_directory.exists():
                raise AssertionError("Fixture artifact collides with the cache seed")
            if cached:
                artifact_directory.mkdir(parents=True)
                shutil.copy2(artifact_jar, artifact_directory / artifact_jar.name)
                shutil.copy2(library / "pom.xml", artifact_directory / "required-library-7.40.pom")
            settings_path = temporary / f"settings-{label}.xml"
            settings_path.write_text(f'<settings><localRepository>{escape(str(repository))}</localRepository><offline>true</offline></settings>\n')
            settings = {"java": {"autobuild": {"enabled": False}, "import": {
                "maven": {"enabled": True, "offline": {"enabled": True}}, "gradle": {"enabled": False}},
                "configuration": {"maven": {"userSettings": str(settings_path)}, "updateBuildConfiguration": "disabled"}}}
            data = temporary / f"profile-{label}"
            journey = {"label": label, "states": [], "passed": False}
            receipt["journeys"].append(journey)

            def membership(client):
                projects = imported_resource_membership(client, root)
                java_projects = client.request("workspace/executeCommand", {"command": "java.project.getAll", "arguments": []}, timeout=30)
                java_paths = {Path(unquote(urlparse(uri).path)).resolve() for uri in java_projects or []}
                if any(not path.is_relative_to(root) for path in java_paths):
                    raise AssertionError("Java project identity escapes disposable fixtures")
                return projects, {path.relative_to(root).as_posix() for path in java_paths}

            def classpath(client):
                result = client.request("workspace/executeCommand", {"command": "java.project.getSettings", "arguments": [application.as_uri(), ["org.eclipse.jdt.ls.core.classpathEntries"]]}, timeout=30)
                entries = result.get("org.eclipse.jdt.ls.core.classpathEntries") if isinstance(result, dict) else None
                if not isinstance(entries, list) or len(entries) > 1024:
                    raise AssertionError("Application classpath is unavailable or exceeds its bound")
                return entries

            def check_cached_models(client, after_unlink, cold_cache=False):
                before = imported_resource_membership(client, root)
                removed_pom = library / "pom.xml"
                for scope in [root, reactor, unrelated]:
                    expected = {path / "pom.xml" for path in [reactor, application, library, unrelated] if path.is_relative_to(scope)}
                    unavailable = {removed_pom} if after_unlink and removed_pom.is_relative_to(scope) else set()
                    expected -= unavailable
                    if cold_cache:
                        # Startup suppression retains resources/classpaths without rebuilding Maven models.
                        unavailable = expected if scope == root else {scope / "pom.xml"}
                        expected = set()
                    for details, parameters, version in [(False, False, 1), (True, False, 2), (True, True, 3)]:
                        result = discover(client, scope, details=details, parameters=parameters)
                        journey.setdefault("cachedModels", []).append({"afterUnlink": after_unlink, "coldCache": cold_cache, "scope": scope.as_uri(), "reply": result})
                        records = result.get("projects", [])
                        observed = [Path(unquote(urlparse(record.get("pomUri", "")).path)).resolve() for record in records]
                        missing = [Path(unquote(urlparse(uri).path)).resolve() for uri in result.get("unavailablePomUris", [])]
                        if (result.get("version") != version or result.get("cachedOnly") is not True
                                or Path(unquote(urlparse(result.get("rootUri", "")).path)).resolve() != scope
                                or len(observed) != len(expected) or set(observed) != expected
                                or len(missing) != len(unavailable) or set(missing) != unavailable):
                            raise AssertionError("Cached container/rooted Maven model identities or unavailable facts differ")
                        if version > 1:
                            if result.get("reactorOrderKnown") != (not unavailable):
                                raise AssertionError("Cached reactor ordering coverage overclaims unavailable models")
                            if not unavailable:
                                order = [Path(unquote(urlparse(uri).path)).resolve() for uri in result.get("reactorPomUris", [])]
                                if len(order) != len(expected) or set(order) != expected:
                                    raise AssertionError("Cached reactor ordering invents or duplicates a POM")
                        receipt["checks"].append(f"{label}: cached-only v{version} {'container' if scope == root else 'rooted'} model identities and explicit unavailable coverage")
                foreign = temporary / "foreign-root"
                foreign.mkdir(exist_ok=True)
                expect_failure(client, foreign)
                receipt["checks"].append(f"{label}: uninitialized Maven scope rejected without private paths")
                escaped_pom = root / "pom.xml"
                foreign_pom = foreign / "pom.xml"
                foreign_pom.write_text('<project><modelVersion>4.0.0</modelVersion><groupId>foreign</groupId><artifactId>outside</artifactId><version>1</version></project>')
                escaped_pom.symlink_to(foreign_pom)
                try:
                    expect_failure(client, root, forbidden=str(foreign))
                finally:
                    escaped_pom.unlink()
                if imported_resource_membership(client, root) != before:
                    raise AssertionError("Cached model requests or foreign-root rejection mutated imported resources")
                receipt["checks"].append(f"{label}: escaping root POM rejected and read-only membership preserved")

            def hydrate_selected(client, selections, expected_resources, expected_java, action):
                request = {"command": "groovy.maven.importSelected", "arguments": [
                    {"rootUri": root.as_uri(), "pomUris": [path.as_uri() for path in selections], "offline": True}
                ]}
                result = client.request("workspace/executeCommand", request, timeout=100)
                journey.setdefault("hydration", []).append({"action": action, "request": request, "reply": result})
                if (set(result) != {"version", "rootUri", "importedPomUris", "selectedImportPhaseComplete", "fullSynchronizationKnown"}
                        or result["version"] != 1 or result["selectedImportPhaseComplete"] is not True
                        or result["fullSynchronizationKnown"] is not False
                        or {Path(unquote(urlparse(uri).path)).resolve() for uri in result["importedPomUris"]} != set(selections)):
                    raise AssertionError("Selected hydration receipt differs or overclaims synchronization")
                deadline = time.monotonic() + 20
                while True:
                    resources, java = membership(client)
                    if set(resources) == expected_resources and java == expected_java:
                        break
                    if time.monotonic() >= deadline:
                        raise AssertionError("Selected hydration changed unrelated or excluded membership")
                    time.sleep(0.2)
                receipt["checks"].append(action + ": exact selected phase, unchanged unrelated membership and full sync unknown")

            for restart in [False, True]:
                client = client_class()(arguments.launcher.resolve(), arguments.java_home.resolve(), root, data, deny_external_network=True)
                try:
                    settings["java"]["import"]["maven"]["enabled"] = not restart
                    initialized_folders = [root, reactor, unrelated] if container_models else [reactor, unrelated]
                    initialized = client.request("initialize", {"processId": os.getpid(), "rootUri": root.as_uri(),
                        "workspaceFolders": [{"uri": path.as_uri(), "name": path.name} for path in initialized_folders],
                        "capabilities": {"workspace": {"workspaceFolders": True, "configuration": True}},
                        "initializationOptions": {"bundles": [str(provider)] + ([str(fixture_bundle)] if fixture_bundle else []), "settings": settings,
                                                  "workspaceFolders": [path.as_uri() for path in initialized_folders]}}, timeout=90)
                    commands = initialized["capabilities"]["executeCommandProvider"]["commands"]
                    if not {"groovy.maven.importedProjects", "groovy.maven.unlinkSelected", "java.project.getSettings"}.issubset(commands):
                        raise AssertionError("Required guarded provider commands are unavailable")
                    client.notify("initialized", {})
                    deadline = time.monotonic() + 90
                    while True:
                        resources, java_members = membership(client)
                        if (expected_all if not restart else expected_removed).issubset(java_members):
                            break
                        if time.monotonic() >= deadline:
                            raise AssertionError("Application and unrelated Java projects did not initialize")
                        time.sleep(0.2)
                    if not restart:
                        if java_members != expected_all:
                            raise AssertionError("Initial Java module identities differ from the fixture")
                        for build_root, expected_roots in [(reactor, {reactor, application, library}), (unrelated, {unrelated})]:
                            ownership = client.request("workspace/executeCommand", {"command": "groovy.maven.importedProjects", "arguments": [{"rootUri": build_root.as_uri(), "cachedOnly": True}]}, timeout=30)
                            records = ownership.get("projects") if isinstance(ownership, dict) else None
                            if (ownership.get("version") != 1 or ownership.get("partial") is not False or ownership.get("fullSynchronizationKnown") is not False
                                    or not isinstance(records, list) or len(records) != len(expected_roots)
                                    or {Path(unquote(urlparse(record.get("projectUri", "")).path)).resolve() for record in records} != expected_roots
                                    or any(record.get("javaNature") != (Path(unquote(urlparse(record.get("projectUri", "")).path)).resolve() != reactor) for record in records)):
                                raise AssertionError("Current Maven imported ownership differs from fixture Java/non-Java resources")
                            journey.setdefault("initialImportedOwnership", []).append(ownership)
                        entries = classpath(client)
                        if not any(entry.get("kind") == 2 and str(entry.get("path", "")).endswith("required-library") for entry in entries):
                            raise AssertionError("Required library has no initial provider project reference")
                        journey["initialClasspath"] = entries
                        receipt["checks"].append(f"{label}: exact application/library/unrelated ownership and required project reference")
                        if container_models:
                            check_cached_models(client, False)
                        if hydrate_models and cached:
                            selected_poms = [path / "pom.xml" for path in [reactor, application, library, unrelated]]
                            for attempt in range(2):
                                hydrate_selected(client, selected_poms, expected_all | {"reactor"}, expected_all,
                                                 f"{label}: existing selected hydration {attempt + 1}")
                                check_cached_models(client, False)
                            selection = {"rootUri": reactor.as_uri(), "pomUris": [(application / "pom.xml").as_uri()], "offline": True}
                            baseline_membership = imported_resource_membership(client, root)
                            application_pom = application / "pom.xml"
                            original_application_pom = application_pom.read_bytes()
                            for action in ["cancel", "root", "ownership", "inputs"]:
                                try:
                                    qualify_unlink_wait(client, reactor, selection, "groovy.maven.importSelected", receipt, journey,
                                                        replace_root=action == "root", ownership_change=application if action == "ownership" else None,
                                                        input_change=application if action == "inputs" else None)
                                finally:
                                    if action == "ownership":
                                        restored = client.request("workspace/executeCommand", {"command": "groovy.fixture.unlink.natures", "arguments": [
                                            {"rootUri": reactor.as_uri(), "projectUri": application.as_uri(), "dual": False}
                                        ]}, timeout=30)
                                        if restored.get("dualNature") is not False:
                                            raise AssertionError("Hydration ownership fixture did not restore its nature")
                                    if action == "inputs":
                                        application_pom.write_bytes(original_application_pom)
                                if imported_resource_membership(client, root) != baseline_membership:
                                    raise AssertionError("Rejected selected hydration changed imported membership")
                                hydrate_selected(client, [application_pom], expected_all | {"reactor"}, expected_all,
                                                 f"{label}: recovered held hydration {action}")
                            missing_parent = b'<parent><groupId>dev.made.groovy.fixture.missing</groupId><artifactId>uncached-parent</artifactId><version>7.43</version><relativePath/></parent>'
                            try:
                                application_pom.write_bytes(original_application_pom.replace(b'<modelVersion>4.0.0</modelVersion>', b'<modelVersion>4.0.0</modelVersion>' + missing_parent))
                                request = {"command": "groovy.maven.importSelected", "arguments": [selection]}
                                error = client.request("workspace/executeCommand", request, timeout=100)
                            except RuntimeError as failure:
                                error = str(failure)
                                if "Selected Maven import failed or may be partial" not in error or str(root) in error:
                                    raise AssertionError("Missing cached-parent hydration failure is not sanitized") from failure
                                journey["missingParentFailure"] = {"request": request, "error": error}
                                receipt["checks"].append("unavailable cached-parent hydration fails without an overclaiming receipt or private paths")
                            else:
                                raise AssertionError("Unavailable cached-parent hydration returned a success receipt")
                            finally:
                                application_pom.write_bytes(original_application_pom)
                            hydrate_selected(client, [application_pom], expected_all | {"reactor"}, expected_all,
                                             f"{label}: restored POM recovers unavailable cached-parent hydration")
                            check_cached_models(client, False)
                        reply = client.request("workspace/executeCommand", {"command": "groovy.maven.unlinkSelected", "arguments": [{"rootUri": reactor.as_uri(), "projectUris": [library.as_uri()]}]}, timeout=30)
                        if (reply.get("selectedUnlinkPhaseComplete") is not True or reply.get("fullSynchronizationKnown") is not False
                                or [Path(unquote(urlparse(uri).path)).resolve() for uri in reply.get("unlinkedProjectUris", [])] != [library]):
                            raise AssertionError("Selected-library unlink receipt is invalid or overclaims synchronization")
                        journey["unlinkReceipt"] = reply
                        resources, java_members = membership(client)
                        if java_members != expected_removed or "reactor/library" in resources:
                            raise AssertionError("Guarded unlink did not remove only the selected library")
                        receipt["checks"].append(f"{label}: exact source-preserving selected-library removal with unrelated project retained")
                    samples = []
                    for _ in range(16):
                        samples.append(classpath(client))
                        time.sleep(0.2)
                    entries = samples[-1]
                    project_reference = any(entry.get("kind") == 2 and str(entry.get("path", "")).endswith("required-library") for entry in entries)
                    cached_library = any(entry.get("kind") == 1 and Path(str(entry.get("path", ""))).resolve() == artifact_directory / artifact_jar.name for entry in entries)
                    artifact_path = artifact_directory / artifact_jar.name
                    artifact_exists = artifact_path.is_file()
                    if artifact_exists != cached or cached and hashlib.sha256(artifact_path.read_bytes()).hexdigest() != receipt["fixtureArtifactSha256"]:
                        raise AssertionError("Actual repository artifact availability differs from the isolated cache cell")
                    source = application / "src/main/java/evidence/Application.java"
                    with client.notifications_lock:
                        notification_count = len(client.notifications)
                    client.notify("textDocument/didOpen", {"textDocument": {"uri": source.as_uri(), "languageId": "java", "version": 1, "text": source.read_text()}})
                    diagnostics = []
                    for _ in range(25):
                        classpath(client)
                        with client.notifications_lock:
                            publications = [message for message in list(client.notifications)[notification_count:]
                                if message.get("method") == "textDocument/publishDiagnostics"
                                and Path(unquote(urlparse(message.get("params", {}).get("uri", "")).path)).resolve() == source]
                        if publications:
                            diagnostics = publications[-1].get("params", {}).get("diagnostics", [])
                            if not isinstance(diagnostics, list) or len(diagnostics) > 128:
                                raise AssertionError("Source diagnostics exceed the fixture bound")
                            if cached or any(diagnostic.get("severity") == 1 for diagnostic in diagnostics):
                                break
                        time.sleep(0.2)
                    reduced_diagnostics = [{"code": diagnostic.get("code"), "severity": diagnostic.get("severity"), "range": diagnostic.get("range")}
                                           for diagnostic in diagnostics]
                    if any(str(root) in diagnostic.get("message", "") or str(repository) in diagnostic.get("message", "") for diagnostic in diagnostics):
                        raise AssertionError("Missing-library source diagnostic contains private fixture paths")
                    if cached and any(diagnostic.get("severity") == 1 for diagnostic in diagnostics):
                        raise AssertionError("Cached required library leaves a source error after unlink")
                    client.notify("textDocument/didClose", {"textDocument": {"uri": source.as_uri()}})
                    resources, java_members = membership(client)
                    if project_reference or not cached_library or java_members != expected_removed or "reactor/library" in resources:
                        raise AssertionError("Required-library replacement or restart differs from the qualified provider contract")
                    if not cached and not any(diagnostic.get("severity") == 1 for diagnostic in diagnostics):
                        raise AssertionError("Missing required artifact produced no source error within the bounded diagnostic window")
                    receipt["checks"].append(f"{label}: repository replacement, removed membership and {'missing-artifact source error' if not cached else 'available fixture artifact'} satisfy the qualified provider contract")
                    journey["states"].append({"restart": restart, "members": sorted(resources), "javaMembers": sorted(java_members), "classpathSamples": samples,
                        "requiredProjectReferenceRemains": project_reference, "repositoryLibraryEntryPresent": cached_library,
                        "repositoryArtifactExists": artifact_exists, "sourceDiagnostics": reduced_diagnostics,
                        "sourceErrorObserved": any(diagnostic.get("severity") == 1 for diagnostic in diagnostics),
                        "removedLibraryReimported": "reactor/library" in resources})
                    receipt["checks"].append(f"{label}: actual artifact existence/hash and bounded source diagnostics distinguish entries from available libraries")
                    receipt["checks"].append(f"{label}: bounded actual provider classpath and {'same-profile restart' if restart else 'post-unlink'} observations retained")
                    if container_models:
                        check_cached_models(client, True, cold_cache=restart)
                    if hydrate_models and cached:
                        remaining_poms = [path / "pom.xml" for path in [reactor, application, unrelated]]
                        if restart:
                            for attempt in range(2):
                                hydrate_selected(client, remaining_poms, expected_removed | {"reactor"}, expected_removed,
                                                 f"{label}: cold/current remaining hydration {attempt + 1}")
                                check_cached_models(client, True)
                        else:
                            selected_poms = remaining_poms + [library / "pom.xml"]
                            hydrate_selected(client, selected_poms, expected_all | {"reactor"}, expected_all,
                                             f"{label}: mixed existing/missing selected hydration")
                            check_cached_models(client, False)
                            reply = client.request("workspace/executeCommand", {"command": "groovy.maven.unlinkSelected", "arguments": [
                                {"rootUri": reactor.as_uri(), "projectUris": [library.as_uri()]}
                            ]}, timeout=30)
                            if reply.get("selectedUnlinkPhaseComplete") is not True or reply.get("fullSynchronizationKnown") is not False:
                                raise AssertionError("Hydration fixture cleanup overclaims unlink completion")
                            journey["hydrationCleanupUnlink"] = reply
                            resources, java = membership(client)
                            if set(resources) != expected_removed | {"reactor"} or java != expected_removed:
                                raise AssertionError("Hydration cleanup did not restore exact removed-library membership")
                            journey["hydrationCleanupSettlingSeconds"] = 2
                            time.sleep(2)
                            check_cached_models(client, True)
                    if any(not (root / relative).is_file() or hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest for relative, digest in originals.items()):
                        raise AssertionError("Qualification modified a fixture source or POM")
                    if list(root.rglob("TEST-*.xml")):
                        raise AssertionError("Qualification emitted test reports")
                    if (arguments.maven_repository.resolve() / "dev/made/groovy/fixture/ignore/required-library/7.40").exists():
                        raise AssertionError("Qualification modified the cache seed")
                    receipt["checks"].append(f"{label}: {'restart' if restart else 'unlink'} preserves source/POM hashes, seed repository and test-report absence")
                finally:
                    evidence = arguments.receipt.parent / f"required-library-{label}-{'restart' if restart else 'unlink'}"
                    evidence.mkdir(parents=True, exist_ok=True)
                    client.stderr.seek(0)
                    (evidence / "stderr.log").write_bytes(client.stderr.read())
                    log = data / ".metadata/.log"
                    if log.exists():
                        shutil.copy2(log, evidence / "jdtls.log")
                    client.close()
                    arguments.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
            journey["passed"] = True
            arguments.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def probe_reactor_root(arguments):
    provider = arguments.provider.resolve()
    if hashlib.sha256(provider.read_bytes()).hexdigest() != arguments.sha256:
        raise AssertionError("Provider binary differs from the source build")
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"provider_sha256": arguments.sha256, "checks": [], "journeys": [], "limits": [
        "Headless, explicitly requested M2E reactor phase with external network denial; no saved client policy or default delivery",
        "Exact phase receipts do not establish full causal synchronization, native/remote workflows or release readiness",
        "Root and declared module input hashes are checked; external parent/settings generations remain a synchronization gate",
    ]}
    with tempfile.TemporaryDirectory(prefix="groovy-maven-reactor-") as directory:
        temporary = Path(directory).resolve()
        projects = temporary / "projects"
        reactor = projects / "reactor"
        unrelated = projects / "unrelated"
        foreign = temporary / "foreign"
        expected_modules = {".", "application", "library", "nested", "nested/leaf", "extra"}
        coordinates = '<groupId>dev.made.groovy.fixture.reactor</groupId><version>7.44</version>'
        properties = '<properties><maven.compiler.release>17</maven.compiler.release></properties>'
        compiler = '<build><plugins><plugin><groupId>org.apache.maven.plugins</groupId><artifactId>maven-compiler-plugin</artifactId><version>3.14.1</version></plugin></plugins></build>'
        reactor.mkdir(parents=True)
        root_contents = (f'<project><modelVersion>4.0.0</modelVersion>{coordinates}<artifactId>reactor</artifactId><packaging>pom</packaging>{properties}'
                         '<modules><module>application</module><module>library</module><module>nested</module><module>ignored</module></modules>'
                         '<profiles><profile><id>active-reactor</id><modules><module>extra</module></modules></profile>'
                         '<profile><id>inactive-reactor</id><modules><module>not-present</module></modules></profile></profiles></project>\n')
        (reactor / "pom.xml").write_text(root_contents)
        nested = reactor / "nested"
        nested.mkdir()
        (nested / "pom.xml").write_text(f'<project><modelVersion>4.0.0</modelVersion>{coordinates}<artifactId>nested</artifactId><packaging>pom</packaging><modules><module>leaf</module></modules></project>\n')
        ignored = reactor / "ignored"
        (ignored / "descendant").mkdir(parents=True)
        (ignored / "pom.xml").write_text('invalid ignored Maven model: /private/isolated-error-canary\n')
        (ignored / "descendant/pom.xml").write_text('unvisited excluded descendant\n')
        if arguments.pom_masks_only:
            (ignored / "pom.xml").write_text(f'<project><modelVersion>4.0.0</modelVersion>{coordinates}<artifactId>ignored</artifactId><packaging>pom</packaging><modules><module>descendant</module></modules></project>\n')
            (ignored / "descendant/pom.xml").write_text(f'<project><modelVersion>4.0.0</modelVersion>{coordinates}<artifactId>ignored-descendant</artifactId><packaging>pom</packaging></project>\n')
        for module in [reactor / name for name in ["application", "library", "nested/leaf", "extra"]] + [unrelated, foreign]:
            module.mkdir(parents=True, exist_ok=True)
            dependency = ('<dependencies><dependency><groupId>dev.made.groovy.fixture.reactor</groupId><artifactId>library</artifactId><version>7.44</version></dependency></dependencies>'
                          if module.name == "application" else '')
            parent = '<parent><groupId>dev.made.groovy.fixture.reactor</groupId><artifactId>reactor</artifactId><version>7.44</version></parent>' if module.name == "extra" else coordinates
            (module / "pom.xml").write_text(f'<project><modelVersion>4.0.0</modelVersion>{parent}<artifactId>{module.name}</artifactId>{properties}{dependency}{compiler}</project>\n')
            name = module.name.title()
            source = module / f"src/main/java/evidence/{name}.java"
            source.parent.mkdir(parents=True)
            expression = "Library.message()" if module.name == "application" else '"cached-reactor"'
            source.write_text(f'package evidence; public class {name} {{ public static String message() {{ return {expression}; }} }}\n')
        generated = reactor / "application/target/generated-sources/fixture/evidence/Generated.java"
        generated.parent.mkdir(parents=True)
        generated.write_text('package evidence; public class Generated {}\n')
        source_hashes = {path.relative_to(projects).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in projects.rglob("*.java")}
        receipt["fixtureSourceSha256"] = source_hashes
        receipt["fixturePomSha256"] = {path.relative_to(projects).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in projects.rglob("pom.xml")}
        repository = temporary / "repository"
        shutil.copytree(arguments.maven_repository.resolve(), repository, copy_function=cache_copy,
                        ignore=shutil.ignore_patterns("*.lock"))
        classes = temporary / "library-classes"
        classes.mkdir()
        preparation = [str(arguments.java_home.resolve() / "bin/javac"), "--release", "17", "-d", str(classes),
                       str(reactor / "library/src/main/java/evidence/Library.java")]
        subprocess.run(preparation, check=True, capture_output=True)
        artifact = repository / "dev/made/groovy/fixture/reactor/library/7.44"
        if artifact.exists():
            raise AssertionError("Reactor fixture coordinates collide with the cache seed")
        artifact.mkdir(parents=True)
        shutil.copy2(reactor / "library/pom.xml", artifact / "library-7.44.pom")
        with zipfile.ZipFile(artifact / "library-7.44.jar", "w") as archive:
            archive.writestr(zipfile.ZipInfo("evidence/Library.class", date_time=(1980, 1, 1, 0, 0, 0)),
                             (classes / "evidence/Library.class").read_bytes())
        receipt["fixtureArtifactSha256"] = hashlib.sha256((artifact / "library-7.44.jar").read_bytes()).hexdigest()
        settings_path = temporary / "settings.xml"
        settings_path.write_text('<settings><offline>true</offline><localRepository>' + escape(str(repository))
                                 + '</localRepository><activeProfiles><activeProfile>active-reactor</activeProfile></activeProfiles></settings>\n')
        settings_contents = settings_path.read_text()
        bundle, fixture_source = unlink_fixture_bundle(arguments, temporary, hydration_modes=True)
        (arguments.receipt.parent / "ReactorFixtureHandler.java").write_bytes(fixture_source.read_bytes())
        receipt["fixtureBundleSha256"] = hashlib.sha256(bundle.read_bytes()).hexdigest()
        settings = {"java": {"import": {"maven": {"enabled": False, "offline": {"enabled": True}}, "gradle": {"enabled": False}},
                             "configuration": {"maven": {"userSettings": str(settings_path)}}}}
        data = temporary / "data"
        root_request = {"rootUri": reactor.as_uri(), "pomUri": (reactor / "pom.xml").as_uri(),
                        "excludedProjectUris": [ignored.as_uri()], "offline": True}
        expected_all = {"reactor" if name == "." else "reactor/" + name for name in expected_modules} | {"unrelated"}
        expected_poms = {reactor / name / "pom.xml" for name in expected_modules}
        for restart in [False, True]:
            client = client_class()(arguments.launcher.resolve(), arguments.java_home.resolve(), projects, data,
                                    deny_external_network=True)
            journey = {"restart": restart, "responses": []}
            receipt["journeys"].append(journey)
            try:
                initialized = client.request("initialize", {"processId": os.getpid(), "rootUri": projects.as_uri(),
                    "workspaceFolders": [{"uri": path.as_uri(), "name": path.name} for path in [projects, reactor, unrelated]],
                    "capabilities": {"workspace": {"workspaceFolders": True, "configuration": True}},
                    "initializationOptions": {"bundles": [str(provider), str(bundle)], "settings": settings,
                                              "workspaceFolders": [path.as_uri() for path in [projects, reactor, unrelated]]}}, timeout=90)
                commands = initialized["capabilities"]["executeCommandProvider"]["commands"]
                if not {"groovy.maven.importReactor", "groovy.maven.importSelected", "groovy.maven.unlinkSelected"}.issubset(commands):
                    raise AssertionError("Guarded reactor commands are unavailable")
                client.notify("initialized", {})

                def membership():
                    return set(imported_resource_membership(client, projects))

                def invoke(command, request, expected, label):
                    result = client.request("workspace/executeCommand", {"command": command, "arguments": [request]}, timeout=120)
                    journey["responses"].append({"label": label, "command": command, "request": request, "result": result})
                    if (result.get("version") != 1 or result.get("selectedImportPhaseComplete") is not True
                            or result.get("fullSynchronizationKnown") is not False
                            or {Path(unquote(urlparse(uri).path)).resolve() for uri in result.get("importedPomUris", [])} != expected):
                        raise AssertionError("Reactor phase receipt overclaims or differs from exact selected modules")
                    if command.endswith("importReactor") and (result.get("reactorPomUri") != request["pomUri"]
                            or {Path(unquote(urlparse(uri).path)).resolve() for uri in result.get("excludedProjectUris", [])} != {ignored}):
                        raise AssertionError("Reactor/exclusion receipt differs")
                    receipt["checks"].append(label + ": exact bounded identities, explicit phase completion, full sync unknown")
                    return result

                def reject(request, label, command="groovy.maven.importReactor"):
                    before = membership()
                    try:
                        result = client.request("workspace/executeCommand", {"command": command, "arguments": [request]}, timeout=90)
                        journey["responses"].append({"label": label, "unexpectedResult": result})
                    except RuntimeError as failure:
                        message = str(failure)
                        journey["responses"].append({"label": label, "error": message})
                        if "Selected Maven import" not in message or str(temporary) in message or "isolated-error-canary" in message:
                            raise AssertionError("Reactor failure is not sanitized") from failure
                    else:
                        raise AssertionError("Invalid reactor selection succeeded")
                    if membership() != before:
                        raise AssertionError("Rejected reactor expansion mutated resource membership")
                    receipt["checks"].append(label + ": sanitized rejection before imported resource mutation")

                if not restart:
                    deadline = time.monotonic() + 90
                    while True:
                        try:
                            before = membership()
                            break
                        except RuntimeError:
                            if time.monotonic() >= deadline:
                                raise
                            time.sleep(0.2)
                    if before:
                        raise AssertionError("Disabled Maven startup imported fixture resources")
                    invoke("groovy.maven.importSelected", {"rootUri": unrelated.as_uri(), "pomUris": [(unrelated / "pom.xml").as_uri()], "offline": True},
                           {unrelated / "pom.xml"}, "exact unrelated POM import")
                    if membership() != {"unrelated"}:
                        raise AssertionError("Exact selected-POM import expanded unrelated roots")
                    receipt["checks"].append("disabled startup and old exact-POM import do not expand the reactor")
                invoke("groovy.maven.importReactor", root_request, expected_poms, "cold restart reactor hydration" if restart else "fresh effective reactor root import")
                if membership() != expected_all:
                    raise AssertionError("Reactor expansion imported inactive/ignored/foreign or omitted effective nested modules")
                receipt["checks"].append("host-active profile, nested and parentless modules imported; inactive and ignored subtrees excluded")
                invoke("groovy.maven.importReactor", root_request, expected_poms, "repeated current reactor import")
                if not restart:
                    unlink = client.request("workspace/executeCommand", {"command": "groovy.maven.unlinkSelected", "arguments": [
                        {"rootUri": reactor.as_uri(), "projectUris": [(reactor / "library").as_uri()]}]}, timeout=30)
                    journey["responses"].append({"label": "selected library unlink", "result": unlink})
                    if membership() != expected_all - {"reactor/library"}:
                        raise AssertionError("Selected library unlink changed unrelated modules")
                    excluded_library = {**root_request, "excludedProjectUris": [ignored.as_uri(), (reactor / "library").as_uri()]}
                    excluded_result = client.request("workspace/executeCommand", {"command": "groovy.maven.importReactor", "arguments": [excluded_library]}, timeout=120)
                    journey["responses"].append({"label": "excluded required library", "request": excluded_library, "result": excluded_result})
                    if (excluded_result.get("selectedImportPhaseComplete") is not True or excluded_result.get("fullSynchronizationKnown") is not False
                            or {Path(unquote(urlparse(uri).path)).resolve() for uri in excluded_result.get("importedPomUris", [])}
                                != expected_poms - {reactor / "library/pom.xml"}
                            or membership() != expected_all - {"reactor/library"}):
                        raise AssertionError("Excluded required library was reimported or reactor receipt overclaims")
                    started = time.monotonic()
                    deadline = started + 20
                    while True:
                        classpath = client.request("workspace/executeCommand", {"command": "java.project.getSettings", "arguments": [
                            (reactor / "application").as_uri(), ["org.eclipse.jdt.ls.core.classpathEntries"]]}, timeout=30).get("org.eclipse.jdt.ls.core.classpathEntries", [])
                        replacements = [entry for entry in classpath if entry.get("kind") == 1 and str(entry.get("path", "")).endswith("library-7.44.jar")]
                        if replacements and all(Path(entry["path"]).is_file() for entry in replacements) and not any(
                                entry.get("kind") == 2 and str(entry.get("path", "")).endswith("library") for entry in classpath):
                            break
                        if time.monotonic() >= deadline:
                            raise AssertionError("Excluded required library did not use its cached artifact")
                        time.sleep(0.2)
                    journey["excludedLibraryClasspath"] = classpath
                    journey["excludedLibraryObservationSeconds"] = time.monotonic() - started
                    receipt["checks"].append("exact library exclusion preserves unrelated modules and resolves the prepared cached dependency without downloading")
                    invoke("groovy.maven.importReactor", root_request, expected_poms, "mixed existing and missing reactor repair")
                    if membership() != expected_all:
                        raise AssertionError("Mixed reactor repair did not restore exact ownership")
                    for label, request in [
                        ("non-file root POM", {**root_request, "pomUri": "https://invalid.example/pom.xml"}),
                        ("foreign root POM", {**root_request, "pomUri": (foreign / "pom.xml").as_uri()}),
                        ("online request", {**root_request, "offline": False}),
                        ("unknown request key", {**root_request, "unknown": True}),
                        ("duplicate exclusions", {**root_request, "excludedProjectUris": [ignored.as_uri()] * 2}),
                        ("foreign exclusion", {**root_request, "excludedProjectUris": [foreign.as_uri()]}),
                        ("excluded root", {**root_request, "excludedProjectUris": [reactor.as_uri()]}),
                        ("oversized exclusions", {**root_request, "excludedProjectUris": [ignored.as_uri()] * 129}),
                        ("missing effective model", {**root_request, "excludedProjectUris": []}),
                    ]:
                        if label != "missing effective model" or not arguments.pom_masks_only:
                            reject(request, label)
                    root_pom = reactor / "pom.xml"
                    nested_pom = nested / "pom.xml"
                    for label, text in [
                        ("duplicate declared module", root_contents.replace('<module>application</module>', '<module>application</module><module>application</module>')),
                        ("escaping declared module", root_contents.replace('<module>application</module>', '<module>../../foreign</module>')),
                        ("oversized effective modules", root_contents.replace('<module>application</module>', ''.join(f'<module>module-{index}</module>' for index in range(129)))),
                        ("missing module", root_contents.replace('<module>application</module>', '<module>not-present</module>')),
                        ("missing cached parent", root_contents.replace(coordinates, '<parent><groupId>fixture.missing</groupId><artifactId>missing</artifactId><version>7.44</version><relativePath/></parent>')),
                    ]:
                        try:
                            root_pom.write_text(text)
                            reject(root_request, label)
                        finally:
                            root_pom.write_text(root_contents)
                    for label, text in [
                        ("oversized root input", root_contents + "<!--" + "x" * (1024 * 1024) + "-->"),
                        ("oversized module text", root_contents.replace('<module>application</module>', '<module>' + 'x' * 4097 + '</module>')),
                    ]:
                        try:
                            root_pom.write_text(text)
                            reject(root_request, label)
                        finally:
                            root_pom.write_text(root_contents)
                    alias = reactor / "alias"
                    try:
                        alias.symlink_to(foreign, target_is_directory=True)
                        root_pom.write_text(root_contents.replace('<module>application</module>', '<module>alias</module>'))
                        reject(root_request, "symlink module escapes initialized root")
                    finally:
                        root_pom.write_text(root_contents)
                        alias.unlink()
                    budget_directories = []
                    try:
                        for index in range(9):
                            module = reactor / f"budget-{index}"
                            module.mkdir()
                            budget_directories.append(module)
                            (module / "pom.xml").write_text(f'<project><modelVersion>4.0.0</modelVersion>{coordinates}<artifactId>budget-{index}</artifactId></project><!--' + 'x' * 950000 + '-->')
                        root_pom.write_text(root_contents.replace('<module>application</module>', ''.join(f'<module>budget-{index}</module>' for index in range(9))))
                        reject(root_request, "aggregate reactor input byte budget")
                    finally:
                        root_pom.write_text(root_contents)
                        for module in budget_directories:
                            shutil.rmtree(module)
                    nested_original = nested_pom.read_bytes()
                    try:
                        nested_pom.write_text(nested_original.decode().replace('<module>leaf</module>', '<module>..</module>'))
                        reject(root_request, "cyclic nested module")
                    finally:
                        nested_pom.write_bytes(nested_original)
                    application = reactor / "application"
                    application_pom = application / "pom.xml"
                    application_original = application_pom.read_bytes()
                    for options in [{}, {"replace_root": True}, {"ownership_change": application}, {"input_change": application}]:
                        before = membership()
                        try:
                            qualify_unlink_wait(client, reactor, root_request, "groovy.maven.importReactor", receipt, journey, **options)
                        finally:
                            if "ownership_change" in options:
                                client.request("workspace/executeCommand", {"command": "groovy.fixture.unlink.natures", "arguments": [
                                    {"rootUri": reactor.as_uri(), "projectUri": application.as_uri(), "dual": False}]}, timeout=30)
                            application_pom.write_bytes(application_original)
                        if membership() != before:
                            raise AssertionError("Held reactor guard changed resource membership")
                    invoke("groovy.maven.importReactor", root_request, expected_poms, "repair after bounded and held guard failures")
                if arguments.pom_masks_only:
                    command = "groovy.maven.importReactorWithMasks"
                    if command not in commands:
                        raise AssertionError("Versioned mask capability is unavailable")
                    contract = {"version": 1, "syntax": "idea-mask-v1", "pathFormat": "portable-absolute-v1", "patterns": []}
                    def masked(patterns, exclusions=()):
                        return {**root_request, "excludedProjectUris": [path.as_uri() for path in exclusions],
                                "pomMasks": {**contract, "patterns": patterns}}
                    def apply_masks(request, imported, ignored_poms, label):
                        result = client.request("workspace/executeCommand", {"command": command, "arguments": [request]}, timeout=120)
                        journey["responses"].append({"label": label, "command": command, "request": request, "result": result})
                        actual_imported = {Path(unquote(urlparse(uri).path)).resolve() for uri in result.get("importedPomUris", [])}
                        actual_ignored = {Path(unquote(urlparse(uri).path)).resolve() for uri in result.get("ignoredPomUris", [])}
                        expected_members = {"unrelated"} | {"reactor" if pom.parent == reactor else "reactor/" + pom.parent.relative_to(reactor).as_posix() for pom in imported}
                        actual_members = membership()
                        journey["responses"][-1]["observedMembership"] = sorted(actual_members)
                        if (result.get("version") != 2 or result.get("selectedImportPhaseComplete") is not True
                                or result.get("fullSynchronizationKnown") is not False or actual_imported != imported
                                or actual_ignored != ignored_poms or actual_members != expected_members
                                or result.get("reactorPomUri") != root_request["pomUri"]
                                or {Path(unquote(urlparse(uri).path)).resolve() for uri in result.get("excludedProjectUris", [])}
                                    != {Path(unquote(urlparse(uri).path)).resolve() for uri in request["excludedProjectUris"]}):
                            raise AssertionError("Masked phase receipt or resource membership differs")
                        receipt["checks"].append(label + ": exact v2 imported/ignored facts and membership; full sync unknown")
                    def cached_descendants(case, imported_poms, missing_ancestor):
                        if arguments.masked_models_only != case:
                            return
                        before = membership()
                        deadline = time.monotonic() + 20
                        while True:
                            try:
                                model = client.request("workspace/executeCommand", {"command": "groovy.maven.projects", "arguments": [
                                    {"rootUri": reactor.as_uri(), "cachedOnly": True}]}, timeout=60)
                                break
                            except RuntimeError as failure:
                                journey.setdefault("cachedModelAvailabilityRetries", []).append(str(failure))
                                if time.monotonic() >= deadline:
                                    raise
                                time.sleep(0.2)
                        actual = {Path(unquote(urlparse(project["pomUri"]).path)).resolve() for project in model["projects"]}
                        unavailable = {Path(unquote(urlparse(uri).path)).resolve() for uri in model.get("unavailablePomUris", [])}
                        journey["responses"].append({"label": "cached descendants beneath unavailable " + case,
                            "model": model, "expectedPoms": sorted(str(pom) for pom in imported_poms),
                            "actualPoms": sorted(str(pom) for pom in actual), "missingAncestor": str(missing_ancestor)})
                        if membership() != before:
                            raise AssertionError("Cached descendant read changed workspace membership")
                        if missing_ancestor not in unavailable:
                            raise AssertionError("Unavailable ancestor was hidden or invented as a cached model")
                        if actual != imported_poms:
                            raise AssertionError("Cached Maven read hides still-imported descendants beneath an unavailable " + case)
                        receipt["checks"].append("cached " + case + " omission retains exact active models, unavailable identity and workspace membership")
                    invalid_contracts = [
                        None, {**contract, "version": 2}, {**contract, "version": "1"},
                        {**contract, "syntax": "glob"}, {**contract, "pathFormat": "native"}, {**contract, "extra": True},
                        {**contract, "patterns": "*"}, {**contract, "patterns": [None]},
                        {**contract, "patterns": [""]}, {**contract, "patterns": ["*", "*"]},
                        {**contract, "patterns": ["private-canary\n"]}, {**contract, "patterns": ["private-canary\u2028"]},
                        {**contract, "patterns": ["x" * 4097]}, {**contract, "patterns": ["λ" * 2049]},
                        {**contract, "patterns": [str(index) for index in range(129)]},
                        {**contract, "patterns": [str(index) + "x" * 4092 for index in range(17)]},
                    ]
                    for index, invalid in enumerate(invalid_contracts):
                        reject({**root_request, "pomMasks": invalid}, f"invalid mask contract {index}", command)
                    reject(masked(["*/library/pom.xml"]), "unknown masked request key", "groovy.maven.importReactor")
                    library_pom = reactor / "library/pom.xml"
                    ignored_poms = {ignored / "pom.xml", ignored / "descendant/pom.xml"}
                    mask_ignored = ["*/ignored/pom.xml", "*/ignored/descendant/pom.xml"]
                    apply_masks(masked(["*/library/pom.xml"], [ignored]), expected_poms - {library_pom}, {library_pom}, "combined exact exclusion and required-library mask")
                    deadline = time.monotonic() + 20
                    while True:
                        classpath = client.request("workspace/executeCommand", {"command": "java.project.getSettings", "arguments": [
                            (reactor / "application").as_uri(), ["org.eclipse.jdt.ls.core.classpathEntries"]]}, timeout=30).get("org.eclipse.jdt.ls.core.classpathEntries", [])
                        replacements = [entry for entry in classpath if entry.get("kind") == 1 and str(entry.get("path", "")).endswith("library-7.44.jar")]
                        if replacements and all(Path(entry["path"]).is_file() for entry in replacements) and not any(
                                entry.get("kind") == 2 and str(entry.get("path", "")).endswith("library") for entry in classpath):
                            break
                        if time.monotonic() >= deadline:
                            raise AssertionError("Masked required module did not become a cached library")
                        time.sleep(0.2)
                    journey["maskedLibraryClasspath"] = classpath
                    receipt["checks"].append("masked required library resolves from prepared cache without external network")
                    apply_masks(masked(mask_ignored), expected_poms, ignored_poms, "pattern-only selection and required-module recovery")
                    apply_masks(masked(mask_ignored + ["*/nested/pom.xml"]), expected_poms - {nested / "pom.xml"}, ignored_poms | {nested / "pom.xml"}, "masked aggregator retains nonmatching child")
                    cached_descendants("aggregate", expected_poms - {nested / "pom.xml"}, nested / "pom.xml")
                    apply_masks(masked(mask_ignored + [str(reactor / "pom.xml")]), expected_poms - {reactor / "pom.xml"}, ignored_poms | {reactor / "pom.xml"}, "masked root retains nonmatching modules")
                    cached_descendants("root", expected_poms - {reactor / "pom.xml"}, reactor / "pom.xml")
                    all_poms = expected_poms | ignored_poms
                    apply_masks(masked(["*"]), set(), all_poms, "all matched reactor is known empty and retains unrelated root")
                    apply_masks(masked([]), all_poms, set(), "empty masks repair all effective modules")
                    if not restart:
                        application = reactor / "application"
                        application_pom = application / "pom.xml"
                        original = application_pom.read_bytes()
                        for options in [{}, {"replace_root": True}, {"ownership_change": application}, {"input_change": application}]:
                            before = membership()
                            try:
                                qualify_unlink_wait(client, reactor, masked(["*/library/pom.xml"]), command, receipt, journey, **options)
                            finally:
                                if "ownership_change" in options:
                                    client.request("workspace/executeCommand", {"command": "groovy.fixture.unlink.natures", "arguments": [
                                        {"rootUri": reactor.as_uri(), "projectUri": application.as_uri(), "dual": False}]}, timeout=30)
                                application_pom.write_bytes(original)
                            if membership() != before:
                                raise AssertionError("Held masked guard mutated imported resources")
                        apply_masks(masked([]), all_poms, set(), "masked failure/cancellation repair")
                    apply_masks(masked(mask_ignored), expected_poms, ignored_poms, "restore baseline membership before restart")
                    current_poms = {path.relative_to(projects).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in projects.rglob("pom.xml")}
                    if current_poms != receipt["fixturePomSha256"]:
                        raise AssertionError("Masked import changed fixture POM inputs")
                    if settings_path.read_text() != settings_contents:
                        raise AssertionError("Masked import changed host Maven settings")
                    receipt["checks"].append("masked phase preserves all fixture POMs and host settings")
                current_sources = {path.relative_to(projects).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in projects.rglob("*.java")}
                if current_sources != source_hashes:
                    raise AssertionError("Reactor import changed original or generated source bytes")
                receipt["checks"].append("original and generated source hashes preserved")
            except Exception:
                evidence = arguments.receipt.parent / f"failed-reactor-{time.time_ns()}"
                evidence.mkdir()
                log = data / ".metadata/.log"
                if log.exists():
                    shutil.copy2(log, evidence / "jdtls.log")
                client.stderr.seek(0)
                (evidence / "stderr.log").write_bytes(client.stderr.read())
                raise
            finally:
                client.close()
                arguments.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))



def probe_default_roots(arguments):
    provider = arguments.provider.resolve()
    if hashlib.sha256(provider.read_bytes()).hexdigest() != arguments.sha256:
        raise AssertionError("Provider binary differs from the source build")
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"provider_sha256": arguments.sha256, "checks": [], "journeys": [], "limits": [
        "Pinned headless JDTLS scanner comparison; no installed extension, GUI, remote or release qualification",
        "Production receives bounded visible candidates; the independent scanner walks only disposable test fixtures",
        "Existing linked projects are retained separately; discovery never claims full synchronization",
    ]}
    with tempfile.TemporaryDirectory(prefix="groovy-maven-default-roots-") as directory:
        temporary = Path(directory).resolve()
        root = temporary / "projects"
        root.mkdir()
        repository = temporary / "repository"
        shutil.copytree(arguments.maven_repository.resolve(), repository, copy_function=shutil.copy2,
                        ignore=shutil.ignore_patterns("*.lock"))
        names = ["aggregate", "aggregate/child", "aggregate/unlinked", "standalone", "opaque",
                 "blocked", "blocked/included", "target/generated", "node_modules/package", ".metadata/hidden",
                 "archetype-resources/template", "META-INF/maven/sample", "deep/a/b/c/d", "deep/a/b/c/e/f"]
        poms = []
        for index, name in enumerate(names):
            pom = root / name / "pom.xml"
            pom.parent.mkdir(parents=True)
            modules = "<modules><module>child</module></modules>" if name == "aggregate" else ""
            pom.write_text("invalid XML intentionally unread by discovery" if name == "opaque" else
                           '<project><modelVersion>4.0.0</modelVersion><groupId>dev.made.groovy.fixture.defaultroots</groupId>'
                           f'<artifactId>module{index}</artifactId><version>1</version><packaging>pom</packaging>{modules}</project>')
            poms.append(pom)
        source_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in poms}
        settings_file = temporary / "settings.xml"
        settings_file.write_text('<settings><offline>true</offline><localRepository>' + escape(str(repository)) + '</localRepository></settings>')
        settings_hash = hashlib.sha256(settings_file.read_bytes()).hexdigest()
        bundle, source = unlink_fixture_bundle(arguments, temporary, default_root_modes=True)
        (arguments.receipt.parent / "DefaultRootsFixtureHandler.java").write_bytes(source.read_bytes())
        receipt["fixtureBundleSha256"] = hashlib.sha256(bundle.read_bytes()).hexdigest()
        command = "groovy.maven.defaultImportRoots"
        for configuration, exclusions in [("baseline", None), ("negation", ["**/blocked", "!**/blocked/included", "**/standalone"])]:
            settings = {"java": {"import": {"maven": {"enabled": False, "offline": {"enabled": True}},
                                                "gradle": {"enabled": False}},
                                  "configuration": {"maven": {"userSettings": str(settings_file)}}}}
            if exclusions is not None:
                settings["java"]["import"]["exclusions"] = exclusions
            data = temporary / ("data-" + configuration)
            for restart in [False, True]:
                journey = {"configuration": configuration, "restart": restart, "responses": []}
                receipt["journeys"].append(journey)
                client = client_class()(arguments.launcher.resolve(), arguments.java_home.resolve(), root, data,
                                        deny_external_network=True)
                def execute(name, request):
                    return client.request("workspace/executeCommand", {"command": name, "arguments": [request]}, timeout=90)
                def membership():
                    return imported_resource_membership(client, root)
                def discover_roots(label, candidates=poms, retained=()):
                    before = membership()
                    oracle = execute("groovy.fixture.unlink.defaultRootsOracle", {"rootUri": root.as_uri()})
                    if oracle["automaticMavenImportEnabled"] is not False:
                        raise AssertionError("Discovery silently enabled automatic Maven import")
                    if exclusions is not None and oracle["exclusions"] != exclusions:
                        raise AssertionError("Oracle used different provider exclusions")
                    request = {"rootUri": root.as_uri(), "candidatePomUris": [path.as_uri() for path in candidates], "cachedOnly": True}
                    response = execute(command, request)
                    candidate_set = {path.as_uri() for path in candidates}
                    expected = {(Path(unquote(urlparse(uri).path)) / "pom.xml").as_uri() for uri in oracle["directories"]} & candidate_set
                    retained_set = {path.as_uri() for path in retained} & candidate_set
                    if (response.get("version") != 1 or response.get("rootUri").rstrip("/") != root.as_uri()
                            or response.get("cachedOnly") is not True or response.get("readOnly") is not True
                            or response.get("partial") is not False or response.get("fullSynchronizationKnown") is not False
                            or set(response.get("rootPomUris", [])) != expected | retained_set
                            or set(response.get("retainedPomUris", [])) != retained_set
                            or len(response["rootPomUris"]) != len(set(response["rootPomUris"]))):
                        raise AssertionError("Bounded roots differ from actual SDK traversal/current linked roots")
                    if membership() != before:
                        raise AssertionError("Read-only root query changed imported projects")
                    journey["responses"].append({"label": label, "request": request, "oracle": oracle, "result": response})
                    receipt["checks"].append(configuration + (" restart: " if restart else " fresh: ") + label)
                    return response
                def reject(request, label):
                    before = membership()
                    try:
                        execute(command, request)
                    except RuntimeError as failure:
                        message = str(failure)
                        if "Maven default-root discovery failed" not in message or str(temporary) in message or "private-canary" in message:
                            raise AssertionError("Discovery failure is not sanitized") from failure
                        journey["responses"].append({"label": label, "error": message})
                    else:
                        raise AssertionError("Invalid default-root request was accepted")
                    if membership() != before:
                        raise AssertionError("Rejected discovery changed imported projects")
                    receipt["checks"].append(label + ": sanitized refusal without import")
                try:
                    initialized = client.request("initialize", {"processId": os.getpid(), "rootUri": root.as_uri(),
                        "workspaceFolders": [{"uri": root.as_uri(), "name": root.name}],
                        "capabilities": {"workspace": {"workspaceFolders": True, "configuration": True}},
                        "initializationOptions": {"bundles": [str(provider), str(bundle)], "settings": settings,
                                                  "workspaceFolders": [root.as_uri()]}}, timeout=90)
                    if command not in initialized["capabilities"]["executeCommandProvider"]["commands"]:
                        raise AssertionError("Provider did not advertise default-root discovery")
                    client.notify("initialized", {})
                    retained = [root / "aggregate/pom.xml", root / "aggregate/child/pom.xml", root / "aggregate/unlinked/pom.xml"] if restart else []
                    if not restart and membership():
                        raise AssertionError("Disabled automatic import linked a fixture before discovery")
                    response = discover_roots("default exclusions, nested roots, depth boundary and unread invalid XML", retained=retained)
                    if not restart and configuration == "baseline" and set(response["rootPomUris"]) != {
                            (root / name / "pom.xml").as_uri() for name in ["aggregate", "standalone", "opaque", "blocked", "deep/a/b/c/d"]}:
                        raise AssertionError("Baseline oracle did not exercise exclusions and the exact depth boundary")
                    if not restart and (root / "aggregate/unlinked/pom.xml").as_uri() in response["rootPomUris"]:
                        raise AssertionError("Unlinked nested POM became a default root")
                    discover_roots("empty candidate set remains empty", [], [])
                    if not restart:
                        for name, import_command in [("aggregate", "groovy.maven.importReactor"), ("aggregate/unlinked", "groovy.maven.importSelected")]:
                            request = {"rootUri": root.as_uri(), "offline": True}
                            request.update({"pomUri": (root / name / "pom.xml").as_uri(), "excludedProjectUris": []}
                                           if import_command.endswith("importReactor") else {"pomUris": [(root / name / "pom.xml").as_uri()]})
                            execute(import_command, request)
                        if set(membership()) != {"aggregate", "aggregate/child", "aggregate/unlinked"}:
                            raise AssertionError("Explicit fixture imports do not have exact owned membership")
                        retained = [root / name / "pom.xml" for name in ["aggregate", "aggregate/child", "aggregate/unlinked"]]
                    discover_roots("existing linked descendants retained without promoting unrelated POMs", retained=retained)
                    nature_request = {"rootUri": root.as_uri(), "projectUri": (root / "aggregate").as_uri()}
                    execute("groovy.fixture.unlink.natures", {**nature_request, "plain": True})
                    try:
                        discover_roots("non-Maven imported root excludes new detection; owned children remain", retained=retained[1:])
                    finally:
                        execute("groovy.fixture.unlink.natures", nature_request)
                    discover_roots("restored Maven ownership recovers discovery", retained=retained)
                    request = {"rootUri": root.as_uri(), "candidatePomUris": [path.as_uri() for path in poms], "cachedOnly": True}
                    foreign = temporary / "private-canary/pom.xml"
                    foreign.parent.mkdir(exist_ok=True); foreign.write_text("private-canary")
                    alias = root / "alias"; alias.symlink_to(root / "standalone", target_is_directory=True)
                    oversized = root / "oversized/pom.xml"
                    oversized.parent.mkdir(); oversized.write_bytes(b"x" * (1024 * 1024 + 1))
                    linked_pom = root / "linked-pom/pom.xml"
                    linked_pom.parent.mkdir(); linked_pom.symlink_to(root / "standalone/pom.xml")
                    try:
                        invalid = [({**request, "cachedOnly": False}, "noncached query"),
                                   ({**request, "extra": True}, "unknown key"),
                                   ({key: value for key, value in request.items() if key != "cachedOnly"}, "missing key"),
                                   ({**request, "rootUri": foreign.parent.as_uri()}, "uninitialized scope"),
                                   ({**request, "candidatePomUris": [poms[0].as_uri()] * 2}, "duplicate candidate"),
                                   ({**request, "candidatePomUris": [foreign.as_uri()]}, "foreign candidate"),
                                   ({**request, "candidatePomUris": [(alias / "pom.xml").as_uri()]}, "aliased candidate"),
                                   ({**request, "candidatePomUris": [(root / "missing/pom.xml").as_uri()]}, "missing candidate"),
                                   ({**request, "candidatePomUris": [(root / "standalone").as_uri()]}, "non-POM candidate"),
                                   ({**request, "candidatePomUris": [poms[0].as_uri()] * 33}, "candidate count budget"),
                                   ({**request, "candidatePomUris": [oversized.as_uri()]}, "POM metadata size budget"),
                                   ({**request, "candidatePomUris": [linked_pom.as_uri()]}, "symlink POM"),
                                   ({**request, "candidatePomUris": [(root / "private-canary\n/pom.xml").as_uri()]}, "control path"),
                                   ({**request, "candidatePomUris": "not-a-list"}, "candidate type"),
                                   ({**request, "candidatePomUris": [(root / "aggregate/child/pom.xml").as_uri()]}, "incomplete ancestor coverage")]
                        for value, label in invalid:
                            if label == "incomplete ancestor coverage":
                                # An already linked child is intentionally retained, so use an unlinked opaque ancestor instead.
                                value = {**request, "candidatePomUris": [(root / "blocked/included/pom.xml").as_uri()]}
                                if exclusions is not None:
                                    continue
                            reject(value, label)
                    finally:
                        alias.unlink(); linked_pom.unlink(); oversized.unlink()
                        linked_pom.parent.rmdir(); oversized.parent.rmdir()
                    if source_hashes != {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in poms} or settings_hash != hashlib.sha256(settings_file.read_bytes()).hexdigest():
                        raise AssertionError("Discovery changed POM or host settings bytes")
                    receipt["checks"].append("POM bytes and offline/automatic-import settings preserved")
                finally:
                    evidence = arguments.receipt.parent / (configuration + ("-restart" if restart else "-fresh"))
                    evidence.mkdir(exist_ok=True)
                    log = data / ".metadata/.log"
                    if log.exists():
                        shutil.copy2(log, evidence / "jdtls.log")
                    client.stderr.seek(0)
                    (evidence / "stderr.log").write_bytes(client.stderr.read())
                    client.close()
                    arguments.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        source = temporary / "DefaultRootsCancellation.java"
        source.write_text('''import java.util.List;
import org.eclipse.core.runtime.NullProgressMonitor;
public final class DefaultRootsCancellation {
 public static void main(String[] arguments) throws Exception {
  var monitor = new NullProgressMonitor(); monitor.setCanceled(true);
  try { new dev.made.groovy.MavenImportHandler().executeCommand("groovy.maven.defaultImportRoots", List.of(), monitor);
   throw new AssertionError("Cancelled request succeeded");
  } catch (IllegalStateException failure) {
   if (!failure.getMessage().equals("Maven default-root discovery cancelled; no projects were imported. Refresh before retrying.") || failure.getCause() != null) throw new AssertionError("Cancellation not sanitized", failure);
  }
  System.out.println("Pre-cancellation rejected before SDK service access without cause chain");
 }
}
''')
        direct_receipt = {"checks": []}
        run_direct_maven_check(argparse.Namespace(**{**vars(arguments), "receipt": arguments.receipt.parent / "direct-cancellation.json"}),
                               temporary, source, direct_receipt, "DefaultRootsCancellation")
        receipt["checks"].extend(direct_receipt["checks"])
        receipt["directCancellation"] = direct_receipt
        arguments.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))

def probe_dependency_graph(arguments):
    provider = arguments.provider.resolve()
    if hashlib.sha256(provider.read_bytes()).hexdigest() != arguments.sha256:
        raise AssertionError("Provider binary differs from the source build")
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"provider_sha256": arguments.sha256, "checks": [], "journeys": [], "limits": [
        "Headless pinned Maven collection only; product admission, shared tree, native, remote and release remain open",
        "Coordinates and mediation are metadata; source origins and artifact readiness remain unknown",
        "Same-profile restart explicitly hydrates selected cached POMs through existing import; it does not prove automatic startup hydration",
    ]}
    with tempfile.TemporaryDirectory(prefix="groovy-maven-graph-") as directory:
        temporary = Path(directory)
        root = (temporary / "project").resolve()
        root.mkdir()
        repository = temporary / "repository"
        shutil.copytree(arguments.maven_repository.resolve(), repository, copy_function=cache_copy,
                        ignore=shutil.ignore_patterns("*.lock"))
        group = "dev.made.groovy.fixture.graph"
        group_directory = repository / group.replace(".", "/")
        if group_directory.exists():
            raise AssertionError("Graph fixture coordinates collide with the cache seed")

        def dependency(name, version="1", extra=""):
            return f"<dependency><groupId>{group}</groupId><artifactId>{name}</artifactId><version>{version}</version>{extra}</dependency>"

        def pom(name, dependencies="", version="1", extra="", packaging="jar"):
            return (f"<project><modelVersion>4.0.0</modelVersion><groupId>{group}</groupId><artifactId>{name}</artifactId>"
                    f"<version>{version}</version><packaging>{packaging}</packaging>{extra}"
                    f"<dependencies>{dependencies}</dependencies></project>\n")

        cache_poms = {
            ("left", "1"): pom("left", dependency("shared") + dependency("managed") + dependency("excluded")
                                + dependency("branch")
                                + dependency("transitive-optional", extra="<optional>true</optional>")),
            ("right", "1"): pom("right", dependency("shared", "2") + dependency("branch", extra=f"<exclusions><exclusion><groupId>{group}</groupId><artifactId>leaf</artifactId></exclusion></exclusions>")),
            ("branch", "1"): pom("branch", dependency("leaf")), ("leaf", "1"): pom("leaf"),
            ("shared", "1"): pom("shared"), ("shared", "2"): pom("shared", version="2"),
            ("managed", "1"): pom("managed"), ("managed", "2"): pom("managed", version="2"),
            ("cycle-a", "1"): pom("cycle-a", dependency("cycle-b")),
            ("cycle-b", "1"): pom("cycle-b", dependency("cycle-a")),
            ("optional", "1"): pom("optional"), ("provided", "1"): pom("provided"),
            ("test", "1"): pom("test"), ("excluded", "1"): pom("excluded"),
            ("transitive-optional", "1"): pom("transitive-optional"),
        }
        for (name, version), contents in cache_poms.items():
            destination = group_directory / name / version
            destination.mkdir(parents=True)
            (destination / f"{name}-{version}.pom").write_text(contents)
        management = '<dependencyManagement><dependencies>' + dependency("managed", "2") + '</dependencies></dependencyManagement>'
        exclusion = f'<exclusions><exclusion><groupId>{group}</groupId><artifactId>excluded</artifactId></exclusion></exclusions>'
        dependencies = (dependency("left", extra=exclusion) + dependency("right") + dependency("cycle-a")
                        + dependency("optional", extra="<optional>true</optional>")
                        + dependency("provided", extra="<scope>provided</scope>")
                        + dependency("test", extra="<scope>test</scope>")
                        + dependency("library", extra="<type>pom</type>"))
        (root / "pom.xml").write_text(pom("root", extra='<modules><module>application</module><module>library</module></modules>', packaging="pom"))
        for name in ["application", "library"]:
            module = root / name
            module.mkdir()
            (module / "pom.xml").write_text(pom(name, dependencies if name == "application" else "",
                                                extra=management if name == "application" else "", packaging="pom"))
        source = root / "application/src/main/java/evidence/Preserved.java"
        source.parent.mkdir(parents=True)
        source.write_text("package evidence; public class Preserved {}\n")
        generated = root / "application/target/generated-sources/fixture/evidence/Generated.java"
        generated.parent.mkdir(parents=True)
        generated.write_text("package evidence; public class Generated {}\n")
        original_sources = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in root.rglob("*.java")}
        original_poms = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in root.rglob("pom.xml")}
        receipt["sourceSha256"] = original_sources
        receipt["pomSha256"] = original_poms
        receipt["cachedPomSha256"] = {path.relative_to(repository).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in group_directory.rglob("*.pom")}
        settings_path = temporary / "settings.xml"
        settings_path.write_text('<settings><offline>true</offline><localRepository>' + escape(str(repository))
                                 + '</localRepository></settings>\n')
        settings = {"java": {"import": {"maven": {"enabled": True, "offline": {"enabled": True}}, "gradle": {"enabled": False}},
                             "configuration": {"maven": {"userSettings": str(settings_path)}}}}
        data = temporary / "data"
        application_pom = root / "application/pom.xml"
        request = {"rootUri": root.as_uri(), "pomUri": application_pom.as_uri(), "cachedOnly": True, "offline": True}
        bundle = graph_fixture_bundle(arguments, temporary)
        for restart in [False, True]:
            client = client_class()(arguments.launcher.resolve(), arguments.java_home.resolve(), root, data,
                                    deny_external_network=True)
            journey = {"restart": restart}
            receipt["journeys"].append(journey)
            try:
                initialized = client.request("initialize", {"processId": os.getpid(), "rootUri": root.as_uri(),
                    "workspaceFolders": [{"uri": root.as_uri(), "name": root.name}],
                    "capabilities": {"workspace": {"workspaceFolders": True, "configuration": True}},
                    "initializationOptions": {"bundles": [str(provider), str(bundle)], "settings": settings}}, timeout=120)
                if "groovy.maven.dependencyGraph" not in initialized["capabilities"]["executeCommandProvider"]["commands"]:
                    raise AssertionError("Maven graph command is unavailable")
                client.notify("initialized", {})
                if restart:
                    journey["coldMetadata"] = discover(client, root)
                    journey["hydrationImport"] = client.request("workspace/executeCommand", {"command": "groovy.maven.importSelected", "arguments": [{
                        "rootUri": root.as_uri(), "pomUris": [(root / path / "pom.xml").as_uri() for path in [".", "application", "library"]],
                        "offline": True}]}, timeout=70)
                    receipt["checks"].append("explicit selected M2E hydration before same-profile cached graph restart")
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    metadata = discover(client, root, details=True)
                    if len(metadata["projects"]) == 3:
                        break
                    time.sleep(0.2)
                else:
                    raise AssertionError("Maven cached fixture did not import")

                def query(value=request):
                    return client.request("workspace/executeCommand", {"command": "groovy.maven.dependencyGraph", "arguments": [value]}, timeout=70)

                response = query()
                journey["response"] = response
                if response["version"] != 2 or not response["offline"] or not response["cachedOnly"] or response["pomUri"] != application_pom.as_uri():
                    raise AssertionError("Maven graph contract identity differs")
                if not response["collectionComplete"] or not response["structuralCoverageKnown"] or response["originsKnown"] or response["artifactsKnown"]:
                    raise AssertionError("Maven graph coverage differs")
                components = {component["id"]: component for component in response["components"]}
                if len(components) != len(response["components"]):
                    raise AssertionError("Maven component identities collide")
                edges = response["edges"]
                if any(edge["from"] not in components or edge["component"] not in components or edge["selected"] not in components for edge in edges):
                    raise AssertionError("Maven graph references an absent component")
                shared = [edge for edge in edges if components[edge["component"]]["artifactId"] == "shared"]
                if len(shared) < 2 or not any(edge["omitted"] for edge in shared) or {components[edge["selected"]]["version"] for edge in shared} != {"1"}:
                    raise AssertionError("Maven nearest/first conflict mediation differs")
                managed = [edge for edge in edges if components[edge["component"]]["artifactId"] == "managed"]
                if len(managed) != 1 or components[managed[0]["component"]]["version"] != "2" or managed[0]["premanagedVersion"] != "1" or not managed[0]["managedBits"]:
                    raise AssertionError("Maven premanaged dependency facts differ")
                direct = {components[edge["component"]]["artifactId"]: edge for edge in edges if edge["direct"]}
                if not direct["optional"]["optional"] or direct["provided"]["scope"] != "provided" or direct["test"]["scope"] != "test":
                    raise AssertionError("Maven direct scope/optionality differs")
                names = {component["artifactId"] for component in components.values()}
                if "excluded" in names or "transitive-optional" in names or not direct["left"]["exclusions"]:
                    raise AssertionError("Maven exclusions or transitive optional semantics differ")
                if "cycle-a" not in names or "cycle-b" not in names or response["cycleCount"] < 1:
                    raise AssertionError("Maven cycle facts are absent")
                if "library" not in names:
                    raise AssertionError("Maven workspace module dependency is absent")
                branch_edges = [edge for edge in edges if components[edge["component"]]["artifactId"] == "branch"]
                leaf_edges = [edge for edge in edges if components[edge["component"]]["artifactId"] == "leaf"]
                if len(branch_edges) < 2 or not leaf_edges or not any(edge["exclusions"] for edge in branch_edges):
                    raise AssertionError("Fixture did not retain both exclusion contexts")
                if not response["occurrencesKnown"]:
                    raise AssertionError("Maven occurrence coverage is unknown")
                nodes = {node["id"]: node["component"] for node in response["nodes"]}
                if len(nodes) != len(response["nodes"]) or nodes[response["rootNode"]] != response["rootComponent"]:
                    raise AssertionError("Occurrence identity collision or root mismatch")
                for edge in edges:
                    if nodes.get(edge["fromNode"]) != edge["from"] or nodes.get(edge["targetNode"]) != edge["component"] or nodes.get(edge["selectedNode"]) != edge["selected"]:
                        raise AssertionError("Maven occurrence/component references differ")
                    if edge["direct"] != (edge["fromNode"] == response["rootNode"]):
                        raise AssertionError("Maven occurrence directness differs")
                outgoing = {}
                for edge in edges:
                    outgoing.setdefault(edge["fromNode"], []).append(edge)
                for children in outgoing.values():
                    if [edge["position"] for edge in children] != list(range(len(children))):
                        raise AssertionError("Actual ordered children are not retained")
                left_branch = next(edge for edge in branch_edges if components[edge["from"]]["artifactId"] == "left")
                right_branch = next(edge for edge in branch_edges if components[edge["from"]]["artifactId"] == "right")
                if left_branch["targetNode"] == right_branch["targetNode"] or not right_branch["omitted"] or right_branch["selectedNode"] != left_branch["targetNode"]:
                    raise AssertionError("Same-coordinate occurrence/winner identities were conflated")
                if not any(components[edge["component"]]["artifactId"] == "leaf" for edge in outgoing.get(left_branch["targetNode"], [])) or outgoing.get(right_branch["targetNode"], []):
                    raise AssertionError("Excluded omitted path inherited another occurrence's children")
                if response["cycleOccurrencesKnown"] or len(response["cycles"]) != response["cycleCount"]:
                    raise AssertionError("Cycle occurrence coverage was invented or cycle count differs")
                cyclic_names = [{components[step["component"]]["artifactId"] for step in cycle["cyclic"]} for cycle in response["cycles"]]
                if not any({"cycle-a", "cycle-b"}.issubset(names) for names in cyclic_names):
                    raise AssertionError("Actual Resolver cycle descriptors are absent")
                receipt["checks"].extend(["bounded occurrence identities and component/winner references",
                    "actual ordered child positions and directness", "same-coordinate exclusion contexts stay separate",
                    "omitted path jumps to actual winner without inheriting its children", "actual cut-cycle descriptors with occurrence mapping explicitly unknown"])
                if query() != response or discover(client, root, details=True) != metadata:
                    raise AssertionError("Graph repeat or cached model compatibility differs")
                receipt["checks"].extend(["current cached POM identity and unknown artifact/origin coverage",
                    "stable coordinate identities and valid path references", "diamond nearest-first conflict winner",
                    "managed version with actual premanagement data", "direct scopes and optionality", "exclusions and transitive optional semantics",
                    "cycle metadata and workspace module dependency", "repeat and cached details compatibility"])
                if restart and response != receipt["journeys"][0]["response"]:
                    raise AssertionError("Same-profile graph restart differs")
                receipt["checks"].append("same-profile graph journey preserved")
                if not restart:
                    failures = [dict(request, offline=False), dict(request, cachedOnly=False), dict(request, extra="secret"),
                                dict(request, pomUri=(temporary / "foreign.xml").as_uri()), dict(request, pomUri=application_pom.as_uri() + "#private-canary"),
                                dict(request, pomUri=(root / "unimported/pom.xml").as_uri())]
                    (temporary / "foreign.xml").write_text(pom("foreign", packaging="pom"))
                    (root / "unimported").mkdir()
                    (root / "unimported/pom.xml").write_text(pom("unimported", packaging="pom"))
                    (root / "escape.xml").symlink_to(temporary / "foreign.xml")
                    failures.append(dict(request, pomUri=(root / "escape.xml").as_uri()))
                    for value in failures:
                        try:
                            query(value)
                        except RuntimeError as failure:
                            if any(private in str(failure) for private in [str(temporary), "private-canary", "secret"]):
                                raise AssertionError("Maven invalid-request error exposes private input") from failure
                        else:
                            raise AssertionError("Invalid Maven graph request succeeded")
                    shutil.rmtree(root / "unimported")
                    (root / "escape.xml").unlink()
                    receipt["checks"].append("seven strict/offline/foreign/unimported/symlink request guards")
                    checks = client.request("workspace/executeCommand", {"command": "groovy.fixture.maven.graphChecks", "arguments": [request]}, timeout=70)
                    journey["fixtureChecks"] = checks
                    receipt["checks"].extend(checks["checks"])
                    repaired_import = client.request("workspace/executeCommand", {"command": "groovy.maven.importSelected",
                        "arguments": [{"rootUri": root.as_uri(), "pomUris": [application_pom.as_uri()], "offline": True}]}, timeout=70)
                    journey["staleRepairImport"] = repaired_import
                    recovery_deadline = time.monotonic() + 60
                    recovery_failures = []
                    while time.monotonic() < recovery_deadline:
                        try:
                            repaired_graph = query()
                        except RuntimeError as failure:
                            if str(temporary) in str(failure) or "private-canary" in str(failure):
                                raise AssertionError("Stale recovery exposes private input") from failure
                            recovery_failures.append(str(failure))
                            time.sleep(0.2)
                            continue
                        if repaired_graph != response:
                            raise AssertionError("Explicit import recovered a different graph")
                        break
                    else:
                        raise AssertionError("Explicit import did not recover restored stale POM graph")
                    journey["recoveryFailures"] = recovery_failures
                    receipt["checks"].append("explicit M2E import repairs stale POM graph")
                    missing_response = client.request("workspace/executeCommand", {"command": "groovy.fixture.maven.graphMissing", "arguments": [request]}, timeout=70)
                    journey["missingResponse"] = missing_response
                    if missing_response["collectionComplete"] or missing_response["structuralCoverageKnown"] or missing_response["problem"] != "offline-metadata-unavailable-check-cache-and-refresh":
                        raise AssertionError("Missing Maven metadata was claimed complete")
                    sanitized = dict(missing_response)
                    sanitized.pop("rootUri"); sanitized.pop("pomUri")
                    if str(temporary) in json.dumps(sanitized) or "private-canary" in json.dumps(sanitized):
                        raise AssertionError("Missing metadata exposes private resolver data")
                    receipt["checks"].append("missing offline metadata has fixed partial category without resolver errors")
                    if query() != response:
                        raise AssertionError("Maven graph did not recover after isolated missing-metadata query")
                    receipt["checks"].append("graph recovery after bounded/stale/cancel/missing cases")
                current_sources = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in root.rglob("*.java")}
                current_poms = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in root.rglob("pom.xml")}
                if current_sources != original_sources or current_poms != original_poms or list(group_directory.rglob("*.jar")) or list(root.rglob("TEST-*.xml")):
                    raise AssertionError("Metadata collection changed source/POMs or acquired fixture Jars/test reports")
                receipt["checks"].append("source/generated/POM preservation and no fixture artifact/test report acquisition")
                journey["passed"] = True
            finally:
                evidence = arguments.receipt.parent / ("restart" if restart else "initial")
                evidence.mkdir(exist_ok=True)
                client.stderr.seek(0)
                (evidence / "stderr.log").write_bytes(client.stderr.read())
                log = data / ".metadata/.log"
                if log.exists():
                    shutil.copy2(log, evidence / "jdtls.log")
                client.close()
                arguments.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def graph_fixture_bundle(arguments, directory):
    source = Path(directory) / "GraphFixtureHandler.java"
    source.write_text('''package dev.made.groovy.fixture;
import java.util.*;
import java.nio.file.*;
import java.net.URI;
import java.lang.reflect.*;
import org.eclipse.core.runtime.IProgressMonitor;
import org.eclipse.core.runtime.NullProgressMonitor;
import org.eclipse.jdt.ls.core.internal.IDelegateCommandHandler;
import org.eclipse.m2e.core.MavenPlugin;
import org.eclipse.aether.artifact.DefaultArtifact;
import org.eclipse.aether.graph.*;
import org.eclipse.aether.collection.*;
import org.osgi.framework.FrameworkUtil;
public class GraphFixtureHandler implements IDelegateCommandHandler {
 private Class<?> providerClass(String name) throws Exception {
  return Arrays.stream(FrameworkUtil.getBundle(MavenPlugin.class).getBundleContext().getBundles())
   .filter(bundle->bundle.getSymbolicName().equals("dev.made.groovy.gradle.model")).findFirst().orElseThrow().loadClass(name);
 }
 private Object graph(List<Object> arguments,IProgressMonitor monitor) throws Exception {
  Class<?> handler=providerClass("dev.made.groovy.MavenModelHandler");
  try { return handler.getMethod("executeCommand",String.class,List.class,IProgressMonitor.class)
   .invoke(handler.getConstructor().newInstance(),"groovy.maven.dependencyGraph",arguments,monitor); }
  catch(InvocationTargetException failure) { if(failure.getCause() instanceof Exception cause)throw cause;throw failure; }
 }
 public Object executeCommand(String command,List<Object> arguments,IProgressMonitor monitor) throws Exception {
  Map<?,?> request=(Map<?,?>)arguments.get(0);
  Path pom=Path.of(new URI((String)request.get("pomUri"))).toRealPath();
  var facade=MavenPlugin.getMavenProjectRegistry().getProjects().stream()
   .filter(value->value.getPom().getLocation().toFile().toPath().equals(pom)).findFirst().orElseThrow();
  var model=facade.getMavenProject();
  if(command.equals("groovy.fixture.maven.graphMissing")) {
   var copy=new org.apache.maven.model.Dependency(); copy.setGroupId("dev.made.groovy.fixture.graph");
   copy.setArtifactId("never-cached");copy.setVersion("1");
   var existing=model.getDependencies();
   try { model.setDependencies(List.of(copy));return graph(arguments,monitor); }
   finally { model.setDependencies(existing); }
  }
  if(!command.equals("groovy.fixture.maven.graphChecks")) throw new IllegalArgumentException();
  List<String> checks=new ArrayList<>();
  Class<?> reader=providerClass("dev.made.groovy.MavenModelHandler$Reader");
  Class<?> budgetClass=providerClass("dev.made.groovy.MavenModelHandler$Reader$CollectionBudget");
  var readerConstructor=reader.getDeclaredConstructor();readerConstructor.setAccessible(true);
  var budgetConstructor=budgetClass.getDeclaredConstructor(IProgressMonitor.class);budgetConstructor.setAccessible(true);
  var projection=reader.getDeclaredMethod("graphProjection",Path.class,Path.class,CollectResult.class,budgetClass);projection.setAccessible(true);
  var root=new DefaultDependencyNode(new DefaultArtifact("fixture:root:pom:1"));
  var child=new DefaultDependencyNode(new Dependency(new DefaultArtifact("fixture:child:jar:1"),"compile"));
  var collected=new CollectResult(new CollectRequest()).setRoot(root);
  for(int mode=0;mode<3;mode++) {
   if(mode==0) root.setChildren(Collections.nCopies(16385,child));
   if(mode==1) { List<DependencyNode> nodes=new ArrayList<>();for(int i=0;i<4097;i++)nodes.add(new DefaultDependencyNode(new Dependency(new DefaultArtifact("fixture:component"+i+":jar:1"),"compile")));root.setChildren(nodes); }
   if(mode==2) root.setChildren(List.of(new DefaultDependencyNode(new Dependency(new DefaultArtifact("fixture:"+"x".repeat(4097)+":jar:1"),"compile"))));
   try { projection.invoke(readerConstructor.newInstance(),pom.getParent().getParent(),pom,collected,budgetConstructor.newInstance(monitor));throw new AssertionError("Oversized graph accepted"); }
   catch(InvocationTargetException rejected) { if(!(rejected.getCause() instanceof IllegalStateException))throw rejected; }
  }
  checks.add("edge/component/text graph bounds reject before admission");
  root.setChildren(List.of(child));
  child.setData(org.eclipse.aether.util.graph.transformer.ConflictResolver.NODE_DATA_WINNER,new DefaultDependencyNode(new Dependency(new DefaultArtifact("fixture:foreign:jar:1"),"compile")));
  try { projection.invoke(readerConstructor.newInstance(),pom.getParent().getParent(),pom,collected,budgetConstructor.newInstance(monitor));throw new AssertionError("Foreign winner occurrence accepted"); }
  catch(InvocationTargetException rejected){if(!(rejected.getCause() instanceof IllegalStateException))throw rejected;}
  child.setData(org.eclipse.aether.util.graph.transformer.ConflictResolver.NODE_DATA_WINNER,null);
  checks.add("unknown winner occurrence rejected instead of fabricated");

  root.setChildren(Collections.nCopies(10,child));
  NullProgressMonitor cancelled=new NullProgressMonitor(){int calls;public boolean isCanceled(){return ++calls>4;}};
  try { projection.invoke(readerConstructor.newInstance(),pom.getParent().getParent(),pom,collected,budgetConstructor.newInstance(cancelled));throw new AssertionError("In-flight projection cancellation ignored"); }
  catch(InvocationTargetException rejected) { if(!(rejected.getCause() instanceof IllegalStateException))throw rejected; }
  checks.add("in-flight graph projection monitor cancellation");
  NullProgressMonitor preCancelled=new NullProgressMonitor();preCancelled.setCanceled(true);
  try { graph(arguments,preCancelled);throw new AssertionError("Pre-cancelled graph accepted"); }
  catch(IllegalStateException rejected){if(!rejected.getMessage().contains("cancelled"))throw rejected;}
  checks.add("pre-cancelled graph command rejects before host access");
  Object budget=budgetConstructor.newInstance(monitor);
  var traverse=budgetClass.getDeclaredMethod("traverse");traverse.setAccessible(true);
  for(int i=0;i<16384;i++)traverse.invoke(budget);
  try { traverse.invoke(budget);throw new AssertionError("Collection traversal limit ignored"); }
  catch(InvocationTargetException rejected){if(!(rejected.getCause() instanceof IllegalStateException))throw rejected;}
  var check=budgetClass.getDeclaredMethod("check");check.setAccessible(true);
  try { check.invoke(budget);throw new AssertionError("Collector could hide traversal limit"); }
  catch(InvocationTargetException rejected){if(!(rejected.getCause() instanceof IllegalStateException))throw rejected;}
  checks.add("collection limit remains latched if resolver catches failure");
  Class<?> traverserClass=providerClass("dev.made.groovy.MavenModelHandler$Reader$BoundedTraverser");
  var traverserConstructor=traverserClass.getDeclaredConstructor(DependencyTraverser.class,budgetClass,int.class);traverserConstructor.setAccessible(true);
  var derive=traverserClass.getDeclaredMethod("deriveChildTraverser",DependencyCollectionContext.class);derive.setAccessible(true);
  Object depthBudget=budgetConstructor.newInstance(monitor);
  Object traverser=traverserConstructor.newInstance(null,depthBudget,0);
  for(int i=0;i<256;i++)traverser=derive.invoke(traverser,new Object[]{null});
  try { derive.invoke(traverser,new Object[]{null});throw new AssertionError("Depth limit ignored"); }
  catch(InvocationTargetException rejected){if(!(rejected.getCause() instanceof IllegalStateException))throw rejected;}
  try { check.invoke(depthBudget);throw new AssertionError("Collector could hide depth limit"); }
  catch(InvocationTargetException rejected){if(!(rejected.getCause() instanceof IllegalStateException))throw rejected;}
  checks.add("collection depth limit remains latched");
  NullProgressMonitor traversalCancelled=new NullProgressMonitor(){int calls;public boolean isCanceled(){return ++calls>1;}};
  Object cancelledBudget=budgetConstructor.newInstance(traversalCancelled);
  Object cancelTraverser=traverserConstructor.newInstance(null,cancelledBudget,0);
  var traverseDependency=traverserClass.getDeclaredMethod("traverseDependency",Dependency.class);traverseDependency.setAccessible(true);
  traverseDependency.invoke(cancelTraverser,child.getDependency());
  try { traverseDependency.invoke(cancelTraverser,child.getDependency());throw new AssertionError("In-flight traversal cancellation ignored"); }
  catch(InvocationTargetException rejected){if(!(rejected.getCause() instanceof IllegalStateException))throw rejected;}
  checks.add("in-flight collection traversal monitor cancellation");
  var deadline=budgetClass.getDeclaredField("deadline");deadline.setAccessible(true);
  Object expired=budgetConstructor.newInstance(monitor);deadline.setLong(expired,System.nanoTime()-1);
  try { check.invoke(expired);throw new AssertionError("Collection deadline ignored"); }
  catch(InvocationTargetException rejected){if(!(rejected.getCause() instanceof IllegalStateException))throw rejected;}
  checks.add("collection deadline rejected");
  var dependencies=model.getDependencies();
  var config=facade.createExecutionContext();
  config.execute((context,progress)->{
   Map<?,?> originalConfig=new HashMap<>(context.getRepositorySession().getConfigProperties());
   try { graph(arguments,progress); } catch(Exception failure) { throw new IllegalStateException("Fixture nested graph failed",failure); }
   if(model.getDependencies()!=dependencies||!context.getRepositorySession().getConfigProperties().equals(originalConfig))throw new AssertionError("Shared Maven model/session mutated");
   return null;
  },monitor);
  checks.add("cached model and host repository configuration preserved");
  var inputFile=facade.getPom();long originalStamp=inputFile.getModificationStamp();long originalTime=inputFile.getLocalTimeStamp();
  byte[] original=Files.readAllBytes(pom);
  try {
   inputFile.setContents(new java.io.ByteArrayInputStream((new String(original,java.nio.charset.StandardCharsets.UTF_8)+"<!-- private-canary -->").getBytes(java.nio.charset.StandardCharsets.UTF_8)),true,false,monitor);
   if(!facade.isStale())throw new AssertionError("Fixture did not create stale facade");
   try { graph(arguments,monitor);throw new AssertionError("Stale model accepted"); }
   catch(IllegalStateException rejected){if(rejected.getMessage().contains("private-canary")||rejected.getMessage().contains(pom.toString()))throw rejected;}
  } finally {
   inputFile.setContents(new java.io.ByteArrayInputStream(original),true,false,monitor);inputFile.setLocalTimeStamp(originalTime);inputFile.revertModificationStamp(originalStamp);
  }
  checks.add("stale cached facade rejection and POM restoration");
  return Map.of("checks",checks);
 }
}
''')
    plugins = arguments.launcher.resolve().parent.parent / "plugins"
    runtime = Path(directory) / "runtime"
    runtime.mkdir()
    with zipfile.ZipFile(plugins / "org.eclipse.m2e.maven.runtime_3.9.1600.20260522-0510.jar") as archive:
        for name in archive.namelist():
            if name.startswith("jars/") and name.endswith(".jar"):
                (runtime / Path(name).name).write_bytes(archive.read(name))
    classes = Path(directory) / "fixture-classes"
    classes.mkdir()
    classpath = os.pathsep.join(str(path) for path in [arguments.provider.resolve(), *plugins.glob("*.jar"), *runtime.glob("*.jar")])
    (arguments.receipt.parent / source.name).write_bytes(source.read_bytes())
    compilation = subprocess.run([str(arguments.java_home / "bin/javac"), "--release", "17", "-Xlint:all", "-Werror", "-classpath", classpath,
                    "-d", str(classes), str(source)], capture_output=True)
    (arguments.receipt.parent / "fixture-compilation.log").write_bytes(compilation.stdout + compilation.stderr)
    if compilation.returncode != 0:
        raise RuntimeError("Graph fixture compilation failed; inspect retained fixture-compilation.log")
    bundle = Path(directory) / "graph-fixture.jar"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("META-INF/MANIFEST.MF", "Manifest-Version: 1.0\nBundle-ManifestVersion: 2\nBundle-SymbolicName: dev.made.groovy.fixture.graph;singleton:=true\nBundle-Version: 1.0.0\nRequire-Bundle: org.eclipse.jdt.ls.core,org.eclipse.core.runtime,org.eclipse.core.resources,org.eclipse.m2e.core,org.eclipse.m2e.maven.runtime,org.eclipse.osgi\n\n")
        archive.writestr("plugin.xml", '<plugin><extension point="org.eclipse.jdt.ls.core.delegateCommandHandler"><delegateCommandHandler class="dev.made.groovy.fixture.GraphFixtureHandler"><command id="groovy.fixture.maven.graphChecks"/><command id="groovy.fixture.maven.graphMissing"/></delegateCommandHandler></extension></plugin>')
        for path in classes.rglob("*.class"):
            archive.writestr(path.relative_to(classes).as_posix(), path.read_bytes())
    (arguments.receipt.parent / source.name).write_bytes(source.read_bytes())
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Opt-in cached Maven JDTLS metadata probe; no external network")
    for name in ["launcher", "java-home", "provider", "maven-repository", "receipt"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--cached-ancestry-only", action="store_true", help="Direct bounded cached-descendant queue checks")
    parser.add_argument("--pom-mask-contract-only", action="store_true", help="Direct bounded production POM-mask contract checks")
    parser.add_argument("--profile-projection-only", action="store_true",
                        help="Direct bounded cached-profile method checks; no JDTLS or build worker")
    parser.add_argument("--import-selection-only", action="store_true",
                        help="Opt-in upstream selection command and same-profile restart qualification")
    parser.add_argument("--parentless-reactor", action="store_true",
                        help="Selection-only qualification of aggregated modules without POM inheritance")
    parser.add_argument("--use-groovy-import", action="store_true",
                        help="Qualify the development M2E selected-import delegate (and pre-cancellation in metadata mode)")
    parser.add_argument("--exclusion-only", action="store_true",
                        help="Opt-in root/module exclusions, required dependency, selected precedence and orphan observations")
    parser.add_argument("--guarded-unlink-only", action="store_true",
                        help="Qualify guarded source-preserving unlink and repaired selected import")
    parser.add_argument("--resource-membership-only", action="store_true",
                        help="Qualify existing non-Java imported resource enumeration, unlink and repair")
    parser.add_argument("--required-library-only", action="store_true",
                        help="Observe cached/missing required-library classpaths after guarded unlink and restart")
    parser.add_argument("--container-models-only", action="store_true",
                        help="Qualify cached container/rooted Maven models with required-library and restart journeys")
    parser.add_argument("--hydrate-models-only", action="store_true",
                        help="Qualify exact existing/mixed selected Maven hydration and cold-model restart")
    parser.add_argument("--masked-models-only", choices=["aggregate", "root"], help="Qualify cached active Maven models beneath the selected unavailable ancestor")
    parser.add_argument("--pom-masks-only", action="store_true", help="Qualify versioned POM masks through the existing reactor fixture")
    parser.add_argument("--reactor-root-only", action="store_true",
                        help="Qualify explicit effective Maven reactor expansion through the guarded M2E phase")
    parser.add_argument("--default-roots-only", action="store_true", help="Read-only default Maven root discovery against the pinned SDK scanner")
    parser.add_argument("--dependency-graph-only", action="store_true", help="Qualify explicit offline cached Maven dependency collection")
    arguments = parser.parse_args()
    if arguments.default_roots_only and any(value for name, value in vars(arguments).items()
            if name not in ["default_roots_only", "launcher", "java_home", "provider", "maven_repository", "receipt", "sha256"]):
        parser.error("--default-roots-only is an independent provider qualification mode")
    if arguments.cached_ancestry_only and any(value for name, value in vars(arguments).items()
            if name not in ["cached_ancestry_only", "launcher", "java_home", "provider", "maven_repository", "receipt", "sha256"]):
        parser.error("--cached-ancestry-only is an independent direct-method qualification mode")
    if arguments.pom_mask_contract_only and any(value for name, value in vars(arguments).items()
            if name not in ["pom_mask_contract_only", "launcher", "java_home", "provider", "maven_repository", "receipt", "sha256"]):
        parser.error("--pom-mask-contract-only is an independent direct-method qualification mode")
    if arguments.masked_models_only and not (arguments.reactor_root_only and arguments.pom_masks_only):
        parser.error("--masked-models-only requires --reactor-root-only --pom-masks-only")
    if arguments.pom_masks_only and not arguments.reactor_root_only:
        parser.error("--pom-masks-only requires --reactor-root-only")
    if arguments.profile_projection_only and any(value for name, value in vars(arguments).items()
                                                if name not in ["profile_projection_only", "launcher", "java_home", "provider",
                                                                "maven_repository", "receipt", "sha256"]):
        parser.error("--profile-projection-only is an independent direct-method qualification mode")
    if arguments.dependency_graph_only and any([arguments.reactor_root_only, arguments.hydrate_models_only, arguments.container_models_only, arguments.required_library_only, arguments.import_selection_only, arguments.parentless_reactor, arguments.use_groovy_import, arguments.exclusion_only, arguments.guarded_unlink_only, arguments.resource_membership_only]):
        parser.error("--dependency-graph-only is an independent provider qualification mode")
    if arguments.reactor_root_only and any([arguments.hydrate_models_only, arguments.container_models_only, arguments.required_library_only, arguments.import_selection_only, arguments.parentless_reactor, arguments.use_groovy_import, arguments.exclusion_only, arguments.guarded_unlink_only, arguments.resource_membership_only]):
        parser.error("--reactor-root-only is an independent provider qualification mode")
    if arguments.hydrate_models_only and any([arguments.container_models_only, arguments.required_library_only, arguments.import_selection_only, arguments.parentless_reactor, arguments.use_groovy_import, arguments.exclusion_only, arguments.guarded_unlink_only, arguments.resource_membership_only]):
        parser.error("--hydrate-models-only is an independent provider qualification mode")
    if arguments.container_models_only and any([arguments.required_library_only, arguments.import_selection_only, arguments.parentless_reactor, arguments.use_groovy_import, arguments.exclusion_only, arguments.guarded_unlink_only, arguments.resource_membership_only]):
        parser.error("--container-models-only is an independent provider qualification mode")
    if arguments.required_library_only and any([arguments.import_selection_only, arguments.parentless_reactor, arguments.use_groovy_import, arguments.exclusion_only, arguments.guarded_unlink_only, arguments.resource_membership_only]):
        parser.error("--required-library-only is an independent provider qualification mode")
    if arguments.resource_membership_only and (not arguments.import_selection_only or not arguments.use_groovy_import
            or arguments.exclusion_only or arguments.guarded_unlink_only):
        parser.error("--resource-membership-only requires --import-selection-only --use-groovy-import without other specialized modes")
    if arguments.guarded_unlink_only and (not arguments.import_selection_only or not arguments.use_groovy_import or arguments.exclusion_only):
        parser.error("--guarded-unlink-only requires --import-selection-only --use-groovy-import without --exclusion-only")
    if arguments.exclusion_only and not arguments.import_selection_only:
        parser.error("--exclusion-only requires --import-selection-only")
    if arguments.parentless_reactor and not arguments.import_selection_only:
        parser.error("--parentless-reactor requires --import-selection-only")
    if arguments.default_roots_only:
        probe_default_roots(arguments)
    elif arguments.cached_ancestry_only:
        probe_cached_ancestry(arguments)
    elif arguments.pom_mask_contract_only:
        probe_pom_masks(arguments)
    elif arguments.profile_projection_only:
        probe_profile_projection(arguments)
    elif arguments.dependency_graph_only:
        probe_dependency_graph(arguments)
    elif arguments.reactor_root_only:
        probe_reactor_root(arguments)
    elif arguments.hydrate_models_only:
        probe_required_library(arguments, container_models=True, hydrate_models=True)
    elif arguments.container_models_only:
        probe_required_library(arguments, container_models=True)
    elif arguments.required_library_only:
        probe_required_library(arguments)
    elif arguments.import_selection_only:
        probe_import_selection(arguments)
    else:
        probe(arguments)
