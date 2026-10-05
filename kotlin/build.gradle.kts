// Root build file: only declares plugin versions (from gradle/libs.versions.toml)
// so the sub-projects can apply them without repeating the version.
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.jvm) apply false
    alias(libs.plugins.kotlin.serialization) apply false
    alias(libs.plugins.kotlin.compose) apply false
}
