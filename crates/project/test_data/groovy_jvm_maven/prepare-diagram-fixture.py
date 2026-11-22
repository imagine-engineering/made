#!/usr/bin/env python3
import argparse
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZipInfo


def main():
    parser = argparse.ArgumentParser(description="Prepare a disposable offline Maven diagram fixture")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve(strict=True)
    root = directory / "graph-project"
    repository = (directory / "maven-home/repository").resolve(strict=True)
    group = "dev.made.groovy.fixture.diagram"
    cached_group = repository / group.replace(".", "/")
    cached_group.mkdir(parents=True)

    def dependency(name, version="1", extra=""):
        return (
            f"<dependency><groupId>{group}</groupId><artifactId>{name}</artifactId>"
            f"<version>{version}</version>{extra}</dependency>"
        )

    def pom(name, dependencies="", version="1", extra="", packaging="jar"):
        return (
            f"<project><modelVersion>4.0.0</modelVersion><groupId>{group}</groupId>"
            f"<artifactId>{name}</artifactId><version>{version}</version>"
            f"<packaging>{packaging}</packaging>{extra}"
            f"<dependencies>{dependencies}</dependencies></project>\n"
        )

    def write_artifact(path):
        with ZipFile(path, "w") as artifact:
            artifact.writestr(ZipInfo("META-INF/MANIFEST.MF"), "Manifest-Version: 1.0\n\n")

    for name, version, dependencies in [
        ("left", "1", dependency("shared")),
        ("right", "1", dependency("shared", "2")),
        ("shared", "1", dependency("uncached")),
        ("shared", "2", ""),
        ("test-only", "1", ""),
    ]:
        descriptor = cached_group / name / version / f"{name}-{version}.pom"
        descriptor.parent.mkdir(parents=True)
        descriptor.write_text(pom(name, dependencies, version))
        write_artifact(descriptor.with_suffix(".jar"))

    uncached = cached_group / "uncached/1/uncached-1.jar"
    uncached.parent.mkdir(parents=True)
    write_artifact(uncached)

    java_build = (
        "<properties><maven.compiler.release>17</maven.compiler.release></properties>"
        "<build><plugins><plugin><groupId>org.apache.maven.plugins</groupId>"
        "<artifactId>maven-compiler-plugin</artifactId><version>3.14.1</version>"
        "</plugin></plugins></build>"
    )

    source = root / "application/src/main/java/evidence/GraphProbe.java"
    source.parent.mkdir(parents=True)
    source.write_text("package evidence; public class GraphProbe {}\n")
    (root / "pom.xml").write_text(
        pom("root", extra="<modules><module>application</module></modules>", packaging="pom")
    )
    (root / "application/pom.xml").write_text(
        pom(
            "application",
            dependency("left")
            + dependency("right")
            + dependency("test-only", extra="<scope>test</scope>"),
            extra=java_build,
        )
    )
    (directory / "settings.xml").write_text(
        f"<settings><offline>true</offline><localRepository>{escape(str(repository))}"
        "</localRepository></settings>\n"
    )


if __name__ == "__main__":
    main()
