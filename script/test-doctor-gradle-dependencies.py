#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

REPOSITORY = Path(__file__).resolve().parent.parent


def run(arguments):
    executable = arguments.gradle.resolve(strict=True)
    java_home = arguments.java_home.resolve(strict=True)
    script = REPOSITORY / "tooling/jvm/gradle-model/doctor-dependencies.gradle"
    receipt = {"checks": [], "adapter_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
               "limitations": ["Offline disposable Gradle 8.13 fixture, not native UI/remote evidence",
                               "Selected component graph; repository origins and artifact classifiers remain unknown",
                               "Gradle owns resolution/processes; no added execution manager or installed audit tool"]}
    with tempfile.TemporaryDirectory(prefix="made-doctor-gradle-report-") as directory:
        root = Path(directory).resolve()
        project = root / "project"
        project.mkdir()
        (project / "settings.gradle").write_text("rootProject.name = 'doctor-fixture'\ninclude 'library'\n")
        (project / "library").mkdir()
        (project / "library/build.gradle").write_text("plugins { id 'java-library' }\n")
        (project / "build.gradle").write_text("""plugins { id 'java' }
allprojects { group = 'evidence'; version = '1.0'; repositories { maven { url = uri(rootProject.file('repository')) } } }
dependencies {
    implementation 'evidence:shared:1.0'
    implementation 'evidence:shared:2.0'
    implementation project(':library')
    if (providers.gradleProperty('doctorFailure').orNull == 'true') { implementation 'evidence:missing:9.0' }
    if (providers.gradleProperty('doctorFiles').orNull == 'true') { implementation files('private-name.jar') }
    if (providers.gradleProperty('doctorBound').orNull == 'true') { implementation('evidence:' + 'x' * 600 + ':1.0') }
}
""")
        for artifact, version, dependency in [("shared", "1.0", None), ("shared", "2.0", "leaf"), ("leaf", "1.0", None)]:
            location = project / f"repository/evidence/{artifact}/{version}"
            location.mkdir(parents=True)
            dependencies = "" if dependency is None else f"<dependencies><dependency><groupId>evidence</groupId><artifactId>{dependency}</artifactId><version>1.0</version></dependency></dependencies>"
            (location / f"{artifact}-{version}.pom").write_text(f"<project><modelVersion>4.0.0</modelVersion><groupId>evidence</groupId><artifactId>{artifact}</artifactId><version>{version}</version>{dependencies}</project>")
        sentinel = project / "sentinel.txt"
        sentinel.write_text("preserve unrelated input\n")
        environment = dict(os.environ, JAVA_HOME=str(java_home))
        command = [str(executable), "--offline", "--no-daemon", "--console=plain",
                   "--gradle-user-home", str(root / "gradle-home"), "--init-script", str(script),
                   ":_madeDoctorDependencies", "-PmadeDoctorConfiguration=runtimeClasspath"]

        def execute(name, properties=(), expected=0):
            output = root / f"{name}.json"
            result = subprocess.run([*command, f"-PmadeDoctorReport={output}", *properties],
                                    cwd=project, env=environment, capture_output=True, text=True, timeout=90)
            (arguments.receipt.parent / f"{name}.log").write_text(result.stdout + result.stderr)
            if (result.returncode == 0) != (expected == 0):
                raise AssertionError(f"Unexpected Gradle Task outcome for {name}; inspect its bounded fixture log")
            if expected:
                if output.exists():
                    raise AssertionError("Rejected Task created a successful report")
                return None
            raw = output.read_bytes()
            if len(raw) > 8 * 1024 * 1024:
                raise AssertionError("Adapter exceeded its report limit")
            shutil.copyfile(output, arguments.receipt.parent / f"{name}.json")
            return json.loads(raw)

        success = execute("selected")
        if success["version"] != 1 or success["gradleVersion"] != "8.13" or success["partial"] or success["unresolvedDependencies"]:
            raise AssertionError("Successful selected component report differs")
        selected = next(index for index, node in enumerate(success["nodes"]) if node["kind"] == "module" and node["module"] == "shared")
        if success["nodes"][selected]["version"] != "2.0":
            raise AssertionError("Conflict winner was not the actual selected version")
        requests = {edge["requested"] for edge in success["edges"] if edge["child"] == selected}
        if not {"evidence:shared:1.0", "evidence:shared:2.0"}.issubset(requests):
            raise AssertionError("Requested and selected versions were conflated")
        leaf = next(index for index, node in enumerate(success["nodes"]) if node["module"] == "leaf")
        if not any(edge["parent"] == selected and edge["child"] == leaf for edge in success["edges"]):
            raise AssertionError("Transitive introducing path missing")
        if not any(node["kind"] == "project" and node["projectPath"] == ":library" for node in success["nodes"]):
            raise AssertionError("Project dependency was conflated with a registry module")
        receipt["checks"].extend(["actual conflict selection", "requested edges", "transitive path", "project dependency"])
        failed = execute("unresolved", ["-PdoctorFailure=true"])
        if not failed["partial"] or failed["unresolvedDependencies"] != 1 or not any(edge["requested"] == "evidence:missing:9.0" for edge in failed["edges"]):
            raise AssertionError("Failed dependency was silently skipped")
        files = execute("files", ["-PdoctorFiles=true"])
        if not files["partial"] or not any(node["kind"] == "unknown" for node in files["nodes"]):
            raise AssertionError("File dependency was falsely reported complete")
        if "private-name.jar" in json.dumps(files) or str(root) in json.dumps(success):
            raise AssertionError("Adapter retained an absolute path or unselected file name")
        execute("bounds", ["-PdoctorBound=true"], expected=1)
        existing = root / "existing.json"
        existing.write_text("preserve existing artifact")
        result = subprocess.run([*command, f"-PmadeDoctorReport={existing}"], cwd=project, env=environment,
                                capture_output=True, text=True, timeout=90)
        if result.returncode == 0 or existing.read_text() != "preserve existing artifact":
            raise AssertionError("Existing report was overwritten")
        linked = root / "linked-output"
        linked.symlink_to(root, target_is_directory=True)
        linked_report = linked / "forbidden.json"
        result = subprocess.run([*command, f"-PmadeDoctorReport={linked_report}"], cwd=project, env=environment,
                                capture_output=True, text=True, timeout=90)
        if result.returncode == 0 or (root / "forbidden.json").exists():
            raise AssertionError("Symlinked report parent was accepted")
        if sentinel.read_text() != "preserve unrelated input\n":
            raise AssertionError("Unrelated project input changed")
        receipt["checks"].extend(["unresolved dependency preserved", "file dependency incomplete", "bounded fields", "no existing artifact overwrite", "symlinked output rejected", "unrelated input preserved"])
        receipt["disposable_root"] = str(root)
    receipt["cleanup"] = "disposable project/home/results removed after fixture; Gradle single-use daemons own normal shutdown"
    arguments.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the Made Doctor Gradle adapter offline with an existing Gradle 8.13/JDK")
    parser.add_argument("--gradle", type=Path, required=True)
    parser.add_argument("--java-home", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    options = parser.parse_args()
    options.receipt.parent.mkdir(parents=True, exist_ok=True)
    run(options)
