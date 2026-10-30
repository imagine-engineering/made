pluginManagement {
    repositories {
        gradlePluginPortal()
    }
}

rootProject.name = "groovy-kotlin-evidence"
include("library", "application")

if (providers.gradleProperty("groovyParity").orNull == "true") {
    includeBuild("included") { name = "groovy-parity-included" }
}
