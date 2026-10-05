// Desktop command-line front end: `scan --out <dir> <image>...` (SPEC section 5).
plugins {
    alias(libs.plugins.kotlin.jvm)
    application
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

kotlin {
    compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) }
}

dependencies {
    implementation(project(":core"))
    // Desktop runtimes (include native libraries for Windows / Linux / macOS).
    implementation(libs.opencv.desktop)
    implementation(libs.onnxruntime.desktop)
}

application {
    applicationName = "scan"
    mainClass.set("io.github.pedrogush.cardfscan.cli.MainKt")
    applicationDefaultJvmArgs = listOf("-Xmx2g")
}

// Make `./gradlew :cli:run --args="..."` resolve relative paths from the folder you ran it in
// (and find the bundled model + the shared name index by default).
tasks.named<JavaExec>("run") {
    workingDir = rootDir.parentFile
    systemProperty("cardfscan.modelsDir", rootDir.resolve("models").absolutePath)
    systemProperty("cardfscan.repoRoot", rootDir.parentFile.absolutePath)
}

// The installed script (build/install/scan/bin/scan) gets the model folder copied next to lib/.
distributions {
    main {
        contents {
            from(rootDir.resolve("models")) { into("models") }
        }
    }
}
