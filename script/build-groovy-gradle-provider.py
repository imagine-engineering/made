#!/usr/bin/env python3

import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile
import zipfile


REPOSITORY = Path(__file__).resolve().parent.parent
SOURCE = REPOSITORY / "tooling/jvm/gradle-model"
PINNED_ARTIFACTS = {
    "org.eclipse.buildship.core_3.1.12.v20260901-1217-s.jar":
        "b2894c6b3e7d9778d8bf08c27391cc0448da62123de4769b0787072d1d1ab1aa",
    "org.eclipse.jdt.ls.core_1.61.0.202609031315.jar":
        "b382afc38a89e41959a56aee532ae567c7f804c3e4d4f13444114468789ec032",
    "org.eclipse.lsp4j_1.0.0.v20260209-1721.jar":
        "ccd78893facc6bfcc359d56cba05d3d5b85eb41e4c40d4b4215ca45db5f416d9",
    "org.eclipse.m2e.core_2.7.800.20260522-0510.jar":
        "966345e9c87fd50adacb2785ff9486e851bdcaea02a089215a18bca956e33382",
    "org.eclipse.m2e.maven.runtime_3.9.1600.20260522-0510.jar":
        "2aaf8786ecf333a5f8f910bac6e4ae04b243d5dcb811eb4aa79585b9fd4263c7",
    "org.gradle.toolingapi_8.9.0.v20260901-1217-s.jar":
        "124d1147714d9b70bab136479b4501b6906ca43321b39d4bbf626d6c73d99fc1",
}
PINNED_GRADLE_ARTIFACTS = {
    "lib/plugins/gradle-language-java-8.13.jar": "9371bc64403916fa200f015912d0ecfe83d2a2b4bd0975c3cc44581d10f9a28f",
    "lib/plugins/gradle-language-jvm-8.13.jar": "54083c9261459ef2aa0f0adea3bf3a312768a0827525ec247cf964a87917ae2e",
    "lib/gradle-core-8.13.jar": "0a69dc9e20abce45bbe11135351dca180dd6b98b77274fd2b347655d0cd6fa41",
    "lib/gradle-core-api-8.13.jar": "bfc8223d06d15d5d85096894bd8714f4a5ad3a4cf3d819b6378ddeed9607a6d4",
    "lib/plugins/gradle-plugins-java-base-8.13.jar": "f9f5e455cccb5b90d607d39b3296c80551953bc2da318c2dc7b6ea5e51eb3793",
    "lib/plugins/gradle-platform-jvm-8.13.jar": "ec66fe7844956f42e7c78781663c76a388aa4fb32b9e2ed86544a3f5e2457aaf",
}


def build(jdtls, java_home, gradle_home, component_annotations, output):
    plugins = jdtls / "plugins"
    for name, expected in PINNED_ARTIFACTS.items():
        artifact = plugins / name
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pinned JDTLS development artifact differs: {name}")
    for name, expected in PINNED_GRADLE_ARTIFACTS.items():
        if hashlib.sha256((gradle_home / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pinned Gradle development artifact differs: {name}")
    if hashlib.sha256(component_annotations.read_bytes()).hexdigest() != "42c080de090896f45d2a4e980a261d5b6addeee07ec4b201f01de95ab562b448":
        raise ValueError("Pinned compile-only org.osgi.service.component.annotations:1.5.1 differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="groovy-gradle-compile-") as directory:
        workspace = Path(directory)
        classes = workspace / "classes"
        classes.mkdir()
        maven = workspace / "maven"
        maven.mkdir()
        with zipfile.ZipFile(plugins / "org.eclipse.m2e.maven.runtime_3.9.1600.20260522-0510.jar") as archive:
            for name in archive.namelist():
                if name.startswith("jars/") and name.endswith(".jar"):
                    (maven / Path(name).name).write_bytes(archive.read(name))
        classpath = ":".join(str(path) for path in sorted([
            component_annotations, *plugins.glob("*.jar"), *gradle_home.glob("lib/*.jar"),
            *gradle_home.glob("lib/plugins/*.jar"), *maven.glob("*.jar"),
        ]))
        subprocess.run([
            str(java_home / "bin/javac"), "--release", "17", "-Xlint:all", "-Werror",
            "-classpath", classpath, "-d", str(classes), *(str(path) for path in sorted(SOURCE.glob("*.java"))),
        ], check=True)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            entries = {"META-INF/MANIFEST.MF": (SOURCE / "MANIFEST.MF").read_bytes().rstrip() + b"\n\n",
                       "plugin.xml": (SOURCE / "plugin.xml").read_bytes()}
            entries.update({path.relative_to(classes).as_posix(): path.read_bytes()
                            for path in sorted(classes.rglob("*.class"))})
            for name, content in entries.items():
                entry = zipfile.ZipInfo(name, date_time=(2026, 10, 3, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(entry, content)
    print(f"Development provider: {output}")
    print(f"SHA-256: {hashlib.sha256(output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the local Gradle JDTLS delegate; does not download or publish artifacts")
    parser.add_argument("--jdtls", type=Path, required=True)
    parser.add_argument("--java-home", type=Path, required=True)
    parser.add_argument("--gradle-home", type=Path, required=True)
    parser.add_argument("--component-annotations", type=Path, required=True,
                        help="Pinned compile-only org.osgi.service.component.annotations:1.5.1; never embedded or downloaded")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    build(arguments.jdtls.resolve(), arguments.java_home.resolve(), arguments.gradle_home.resolve(), arguments.component_annotations.resolve(), arguments.output.resolve())
