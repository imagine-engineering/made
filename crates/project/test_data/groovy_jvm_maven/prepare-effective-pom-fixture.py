#!/usr/bin/env python3
import argparse
from pathlib import Path
from shutil import copy2
from xml.sax.saxutils import escape


def main():
    parser = argparse.ArgumentParser(description="Prepare an offline Maven Effective POM fixture")
    parser.add_argument("directory", type=Path)
    directory = parser.parse_args().directory.resolve(strict=True)
    repository = (directory / "maven-home/repository").resolve(strict=True)
    project = directory / "effective-pom-project"
    project.mkdir()

    fixture = Path(__file__).resolve().parent
    copy2(fixture / "mvnw", project / "mvnw")
    wrapper = Path(".mvn/wrapper/maven-wrapper.properties")
    (project / wrapper.parent).mkdir(parents=True)
    copy2(fixture / wrapper, project / wrapper)
    source = project / "src/main/java/evidence/EffectivePomProbe.java"
    source.parent.mkdir(parents=True)
    source.write_text("package evidence; public class EffectivePomProbe {}\n")
    (project / "pom.xml").write_text(
        """<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>evidence</groupId>
  <artifactId>effective-pom-fixture</artifactId>
  <version>1</version>
  <properties>
    <maven.compiler.release>17</maven.compiler.release>
    <effective.fixture>base</effective.fixture>
  </properties>
  <build><plugins>
    <plugin>
      <groupId>org.apache.maven.plugins</groupId>
      <artifactId>maven-compiler-plugin</artifactId>
      <version>3.14.1</version>
    </plugin>
    <plugin>
      <groupId>org.apache.maven.plugins</groupId>
      <artifactId>maven-help-plugin</artifactId>
      <version>3.5.2</version>
      <configuration><output>target/groovy-effective.xml</output></configuration>
    </plugin>
  </plugins></build>
  <profiles><profile>
    <id>evidence-selected</id>
    <properties><effective.fixture>selected</effective.fixture></properties>
  </profile></profiles>
</project>
"""
    )
    (directory / "settings with spaces.xml").write_text(
        f"<settings><offline>true</offline><localRepository>{escape(str(repository))}"
        "</localRepository><activeProfiles><activeProfile>evidence-selected</activeProfile>"
        "</activeProfiles></settings>\n"
    )


if __name__ == "__main__":
    main()
