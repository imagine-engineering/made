#!/usr/bin/env python3
import argparse
from pathlib import Path
from shutil import copy2
from xml.sax.saxutils import escape


def main():
    parser = argparse.ArgumentParser(description="Prepare an offline Maven Effective POM fixture")
    parser.add_argument("directory", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--source-archives", action="store_true", help="prepare the source archive fixture instead"
    )
    mode.add_argument(
        "--ordered-cancellation", action="store_true", help="include a cancellable Surefire test"
    )
    args = parser.parse_args()
    directory = args.directory.resolve(strict=True)
    repository = (directory / "maven-home/repository").resolve(strict=True)
    project = directory / ("source-archive-project" if args.source_archives else "effective-pom-project")
    project.mkdir()

    fixture = Path(__file__).resolve().parent
    copy2(fixture / "mvnw", project / "mvnw")
    wrapper = Path(".mvn/wrapper/maven-wrapper.properties")
    (project / wrapper.parent).mkdir(parents=True)
    copy2(fixture / wrapper, project / wrapper)
    probe = "ArtifactProbe" if args.source_archives else "EffectivePomProbe"
    source = project / "src/main/java/evidence" / f"{probe}.java"
    source.parent.mkdir(parents=True)
    source.write_text(f"package evidence; public class {probe} {{}}\n")
    if args.source_archives:
        pom = """<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>evidence</groupId>
  <artifactId>artifact-goals-fixture</artifactId>
  <version>1</version>
  <properties><maven.compiler.release>17</maven.compiler.release></properties>
  <dependencies><dependency>
    <groupId>org.apache.commons</groupId>
    <artifactId>commons-lang3</artifactId>
    <version>3.17.0</version>
  </dependency></dependencies>
  <build><plugins><plugin>
    <groupId>org.apache.maven.plugins</groupId>
    <artifactId>maven-dependency-plugin</artifactId>
    <version>3.10.0</version>
  </plugin></plugins></build>
  <profiles><profile>
    <id>artifact-javadocs</id>
    <build><plugins><plugin>
      <groupId>org.apache.maven.plugins</groupId>
      <artifactId>maven-dependency-plugin</artifactId>
      <configuration><classifier>javadoc</classifier><type>jar</type></configuration>
    </plugin></plugins></build>
  </profile></profiles>
</project>
"""
        active_profiles = ""
    else:
        pom = """<project xmlns="http://maven.apache.org/POM/4.0.0">
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
  </profile><profile>
    <id>evidence-alternative</id>
    <properties><effective.fixture>alternative</effective.fixture></properties>
  </profile></profiles>
</project>
"""
        active_profiles = (
            "<activeProfiles><activeProfile>evidence-selected</activeProfile></activeProfiles>"
        )
    if args.ordered_cancellation:
        pom = pom.replace(
            "  <build>",
            """  <dependencies><dependency>
    <groupId>org.junit.jupiter</groupId>
    <artifactId>junit-jupiter</artifactId>
    <version>5.11.4</version>
    <scope>test</scope>
  </dependency></dependencies>
  <build>""",
            1,
        ).replace(
            "  </plugins></build>",
            """    <plugin>
      <groupId>org.apache.maven.plugins</groupId>
      <artifactId>maven-surefire-plugin</artifactId>
      <version>3.5.2</version>
    </plugin>
  </plugins></build>""",
            1,
        )
        test_source = project / "src/test/java/evidence/OrderedTaskTest.java"
        test_source.parent.mkdir(parents=True)
        test_source.write_text(r"""package evidence;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import static org.junit.jupiter.api.Assertions.assertFalse;

class OrderedTaskTest {
    @Test
    @Timeout(60)
    void runsOrderedTest() throws Exception {
        Path target = Path.of("target");
        Path wait = target.resolve("await-cancellation");
        Files.writeString(target.resolve("ordered-test-pid"), Long.toString(ProcessHandle.current().pid()));
        System.out.println("Groovy ordered Maven test started");
        while (Files.exists(wait)) {
            Thread.sleep(100);
        }
        assertFalse(Files.exists(target.resolve("force-test-failure")), "Intentional Groovy ordered Maven test failure");
        Files.writeString(target.resolve("ordered-test-completed"), "completed\n");
    }
}
""")
    (project / "pom.xml").write_text(pom)
    (directory / "settings with spaces.xml").write_text(
        f"<settings><offline>true</offline><localRepository>{escape(str(repository))}"
        f"</localRepository>{active_profiles}</settings>\n"
    )


if __name__ == "__main__":
    main()
