plugins { java }

group = "evidence"
version = "1.0"

tasks.register("parityIncluded") {
    group = "groovy parity"
    doLast { println("GROOVY_PARITY_INCLUDED") }
}
