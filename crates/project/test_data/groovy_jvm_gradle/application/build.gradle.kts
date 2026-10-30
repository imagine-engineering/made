plugins {
    java
}

dependencies {
    implementation(project(":library"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.11.4")
}

val generateBuildMessage by tasks.registering {
    val outputDirectory = layout.buildDirectory.dir("generated/sources/groovyFixture/java")
    outputs.dir(outputDirectory)
    doLast {
        val source = outputDirectory.get().file("evidence/BuildMessage.java").asFile
        source.parentFile.mkdirs()
        source.writeText("package evidence; public class BuildMessage { public static String value() { return \"generated\"; } }\n")
    }
}

sourceSets.main {
    java.srcDir(generateBuildMessage)
}

tasks.test {
    useJUnitPlatform()
}

abstract class ParityArguments : DefaultTask() {
    @get:Input
    @get:org.gradle.api.tasks.options.Option(option = "message", description = "Fixture message")
    abstract val message: Property<String>

    init {
        group = "groovy parity"
        message.convention("default")
    }

    @TaskAction
    fun report() {
        println("GROOVY_PARITY_MESSAGE=" + message.get())
    }
}

tasks.register<ParityArguments>("parityArguments")

tasks.register("parityFailure") {
    group = "groovy parity"
    doLast { throw GradleException("Intentional Groovy parity fixture failure") }
}

tasks.register("parityWait") {
    group = "groovy parity"
    doLast {
        println("GROOVY_PARITY_WAIT_STARTED")
        Thread.sleep(60000)
    }
}

if (providers.gradleProperty("groovyParity").orNull == "true") {
    repositories { maven { url = uri(rootProject.file("dependency-repository")) } }
    val parityGraph by configurations.creating
    dependencies {
        add(parityGraph.name, "evidence:parity-shared:1.0@pom")
        add(parityGraph.name, "evidence:parity-shared:2.0@pom")
    }
}
