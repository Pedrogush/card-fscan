import org.jetbrains.kotlin.gradle.dsl.JvmTarget

// The scanning pipeline as a plain JVM library, shared by the CLI and the Android app.
plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.serialization)
}

java {
    // Java 17 bytecode runs on desktop JVMs and is accepted by Android's D8 compiler.
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

kotlin {
    compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
}

dependencies {
    // `api` = also visible to modules that depend on :core (they use the result types).
    api(libs.kotlinx.serialization.json)

    // OpenCV and ONNX Runtime have the same Java API on desktop and Android, so :core
    // only compiles against them ("compileOnly"); each app ships the runtime it needs.
    compileOnly(libs.opencv.desktop)
    compileOnly(libs.onnxruntime.desktop)

    testImplementation(libs.opencv.desktop)
    testImplementation(libs.onnxruntime.desktop)
    testImplementation(kotlin("test"))
    testImplementation(platform(libs.junit.bom))
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.launcher)
}

tasks.test {
    useJUnitPlatform()
    // Tests read shared fixtures (testdata/, models/) relative to the repo.
    systemProperty("cardfscan.repoRoot", rootDir.parentFile.absolutePath)
    systemProperty("cardfscan.modelsDir", rootDir.resolve("models").absolutePath)
    maxHeapSize = "1g"
}
