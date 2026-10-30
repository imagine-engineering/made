# Groovy JVM Maven fixture

This two-module project has a Kotlin library, a Java application that depends
on it, one Java Debug main and one plain JUnit Jupiter method. It is an integration fixture for
Groovy development, not a product package or a generated Maven project.

The wrapper uses Apache Maven 3.9.16 and Maven Wrapper 3.3.4 `only-script`.
The scripts were extracted from the wrapper distribution published on Maven
Central. Its SHA-1 sidecar matched the downloaded archive
(`640f7c39ee902b630ada7e77086a6ddb9fceae56`); the script's local SHA-256
was `6cb584c2bc907b849a0b931d8266d3ff3214cdd3127115ed4f49fb7176413d36`.
The wrapper properties bind Maven 3.9.16 to its exact SHA-256 (cross-checked
against Apache's [published SHA-512](https://maven.apache.org/download.cgi))
so a downloaded distribution is checked before extraction. The Kotlin plugin is
pinned to 2.0.21, Maven Surefire to 3.5.2, and JUnit Jupiter to 5.11.4.

After a successful online dependency fetch, the selected method runs from the
populated cache with Maven offline:

```sh
./mvnw -B -ntp -o -pl application -am -Dsurefire.failIfNoSpecifiedTests=false -Dtest=evidence.GreetingTest#messageIsAvailable test
```

Surefire writes `application/target/surefire-reports/TEST-evidence.GreetingTest.xml`.
The copy under `reports/` removes machine properties and elapsed times while
preserving the actual class, method, and result identity for a source test.
The class and method are deliberately unambiguous. The fixture does not justify
inferring an invocable method from arbitrary display names or parameterized
cases.

With `-Dmaven.repo.local` pointing at an empty directory, the same `-o`
command fails before compilation and names the missing
`maven-resources-plugin:3.4.0`. This verifies an incomplete-cache failure of
Maven itself; it does not establish Groovy's user-facing error flow.

Opening this project in an LSP can execute Maven build logic; the host
worktree must be trusted first. A standalone LSP may use its own Maven cache.
This fixture does not establish product Tasks, Tests UI, product Debug, remote,
denied-network import, or package support.

The `DebugMain.java` main is a stable breakpoint target for
`script/test-groovy-java-debug-provider.py`. The probe copies this fixture,
compiles that main with debug symbols, imports it through JDTLS with the
specified Java Debug bundle, and requires a verified breakpoint and stopped
frame on line 7. It needs an external JDTLS archive, a compatible JDK and the
adapter jar; the jar's expected SHA-256 must be supplied. This standalone
provider check does not exercise Copper's Debug panel or a packaged product.

From the repository root, after obtaining a compatible JDTLS archive and
the [pinned Java Debug jar](https://github.com/zed-industries/java-debug/releases/tag/0.53.2):

```sh
python3 script/test-groovy-java-debug-provider.py \
  --jdtls-launcher /path/to/jdtls/bin/jdtls \
  --java-home /path/to/jdk-21-or-newer \
  --debug-jar /path/to/com.microsoft.java.debug.plugin-0.53.2.jar \
  --expected-sha256 5275195905015ce786fc6318c8d039fef43a1fada1d03acdec24c69a3b9ba83c
```

## Tool-window parity extensions

The inactive `parity` profile adds `parity-module`; `parity-cycle` deliberately
adds a cyclic reactor. Profile discovery must distinguish available from active
profiles and update current module identities after switching and reimport.

```sh
./mvnw -B -ntp -o -Pparity validate
./mvnw -B -ntp -o -Pparity-cycle validate
```

The first validates a nine-project reactor including the parent. The second must
fail before compilation with a cyclic-reference error; it must not publish a
successful Task/test result. Default selected-test fixtures remain unchanged.
The profile adds a local diamond with two requested versions of `evidence:parity-shared`
and a reactor annotation processor. `-Pparity package` generates
`application/target/generated-sources/annotations/evidence/GeneratedGreeting.java`.
After installing only into an isolated repository, pinned dependency-plugin 3.7.0
can export the mediated graph for `parity-module`. Full provider metadata, graph
projection and native/profile/restart qualification still need evidence.

## Captured Maven graph v2 decoder fixture

`dependency-graph-v2.json` is the real pinned 0.18.0 Resolver response from
`maven-occurrence-context-20261005/attempt-2/receipt.json`, first journey.
The source receipt SHA-256 is `b03821249e392c8a8f9d78e7d3a70affb027af415e5bc94b5a6f094a90e30af0`.
Only rootUri and pomUri are replaced with stable synthetic test URIs; component,
occurrence, edge and cut-cycle facts are unchanged. Decoder tests qualify the
Rust boundary, not a new provider/native/remote/release journey. The original
source-bound evidence remains in the immutable occurrence archive.
