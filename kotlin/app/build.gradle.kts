import com.android.build.api.variant.ApplicationAndroidComponentsExtension

// Android app: Jetpack Compose UI around the :core pipeline.
plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "io.github.pedrogush.cardfscan"
    compileSdk = 36
    // Used only to strip debug symbols from the native libraries (OpenCV, ONNX Runtime).
    ndkVersion = "29.0.14206865"

    defaultConfig {
        applicationId = "io.github.pedrogush.cardfscan"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"
        // Real phones are arm64. Build with -Pabis=arm64-v8a,x86_64 to also run on the emulator.
        val abis = (project.findProperty("abis") as String?)?.split(",") ?: listOf("arm64-v8a")
        ndk { abiFilters += abis }
    }

    buildFeatures { compose = true }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    packaging {
        // Compress native libraries inside the APK: a much smaller download, at the cost of
        // extracting them once at install time.
        jniLibs { useLegacyPackaging = true }
    }
}

dependencies {
    implementation(project(":core"))
    implementation(libs.opencv.android)
    implementation(libs.onnxruntime.android)

    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.material3)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.activity.compose)
    implementation(libs.lifecycle.viewmodel.compose)
    implementation(libs.lifecycle.runtime.compose)
}

/**
 * Copies the OCR model and the shared name index into the APK's assets at build time, so
 * neither is duplicated in git. Declared as a typed task so Gradle can cache it.
 */
abstract class CopyScannerAssets : DefaultTask() {
    @get:InputFiles
    abstract val sources: ConfigurableFileCollection

    @get:OutputDirectory
    abstract val outputDir: DirectoryProperty

    @TaskAction
    fun copy() {
        val out = outputDir.get().asFile
        out.deleteRecursively()
        out.mkdirs()
        sources.files.forEach { it.copyTo(out.resolve(it.name), overwrite = true) }
    }
}

val copyScannerAssets = tasks.register<CopyScannerAssets>("copyScannerAssets") {
    val repo = rootDir.parentFile
    sources.from(
        rootDir.resolve("models/latin_PP-OCRv5_rec_mobile.onnx"),
        repo.resolve("testdata/names/names_v1.json"),
    )
    outputDir.set(layout.buildDirectory.dir("generated/scannerAssets"))
}

extensions.getByType(ApplicationAndroidComponentsExtension::class.java).onVariants { variant ->
    variant.sources.assets?.addGeneratedSourceDirectory(copyScannerAssets, CopyScannerAssets::outputDir)
}
