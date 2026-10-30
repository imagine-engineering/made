# Groovy JVM Gradle fixture

This two-module project has a Kotlin library, a Java application that depends
on it, a generated Java source root, and one plain JUnit Jupiter method. It is
an integration fixture for Groovy development, not a product package.

The wrapper pins Gradle 8.13. Its first invocation needs the distribution in
the host Gradle cache or network access. The Kotlin plugin is pinned to 2.0.21;
JUnit Jupiter is pinned to 5.11.4.

The checked-in wrapper JAR's SHA-256 is
`81a82aaea5abcc8ff68b3dfcb58b3c3c429378efd98e7433460610fecd7ae45f`,
matching [Gradle's published 8.13 wrapper checksum](https://gradle.org/release-checksums/).
Gradle's [Wrapper guide](https://docs.gradle.org/current/userguide/gradle_wrapper.html)
expects the generated JAR and scripts to be committed with a project.

A populated cache permits this selected test to run with network access
disabled:

```sh
./gradlew --offline --no-daemon :application:clean :application:test --tests evidence.GreetingTest.messageIsAvailable
```

The test report is written to
`application/build/test-results/test/TEST-evidence.GreetingTest.xml` and the
generated source to
`application/build/generated/sources/groovyFixture/java/evidence/BuildMessage.java`.
The copy under `reports/` removes only Gradle's timestamp, hostname and elapsed
time so its parsed identity is deterministic in a source test.
The test's class and method are deliberately unambiguous. No provider may infer
that a display name or parameterized case is an invocable method from this
fixture alone.

Opening this project in an LSP can execute Gradle build scripts; the host
worktree must be trusted first. A standalone Kotlin LSP may use its own Gradle
cache even when the shell's cache is populated. This fixture's presence does
not establish Maven, Tests UI, Debug, remote, or package support.

## Tool-window parity extensions

`-PgroovyParity=true` links the independent `included` build. Task paths, groups,
and options must come from the Gradle provider, not from parsing this README.

```sh
./gradlew --offline -PgroovyParity=true :application:parityArguments --message 'two words' :groovy-parity-included:parityIncluded
./gradlew --offline :application:parityFailure
./gradlew --offline :application:parityWait
```

The first command must emit the exact supplied message and included-build marker.
The second must fail with the deliberate fixture error; cancel the third after
its started marker and verify the owned Task terminates without cancelling other
builds. Existing generated roots and selected JUnit identity remain required.
This fixture extension is not evidence that tool-window discovery, cancellation,
remote execution or Debug already works.

The opt-in `parityGraph` configuration resolves two versions from the fixture's
local metadata repository, without external artifacts. Run
`-PgroovyParity=true :application:dependencies --configuration parityGraph` and
require Gradle's selected `2.0` plus the requested `1.0` edge. The included build
contains a second `evidence/Main.java` to disambiguate source navigation.

## Captured configuration graph

`dependency-graph-v1.json` copies the first baseline response in the source-bound
10.7 Gradle provider receipt (`gradle-resolution-graph-20261005/attempt-6/receipt.json`,
SHA-256 `c6d8eb21bf08b8ee67990f7454b5ed2d90846fdb1a6c85aa521b749ba240b28a`). Only outer rootUri,
buildRootUri and moduleUri bindings are normalized to synthetic test paths.
The graphJson bytes, composite/project/module identities, selectors, variants,
selection causes, rich constraint, cycle and unresolved metadata remain unchanged.
This is a Rust admission fixture; it does not qualify product/native/platform parity.
