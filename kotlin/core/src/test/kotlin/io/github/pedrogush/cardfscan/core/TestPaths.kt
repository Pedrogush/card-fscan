package io.github.pedrogush.cardfscan.core

import io.github.pedrogush.cardfscan.core.match.NameIndex
import java.io.File

/** Locations of shared fixtures, passed in by Gradle (see core/build.gradle.kts). */
object TestPaths {
    val repoRoot = File(System.getProperty("cardfscan.repoRoot") ?: "..")
    val modelsDir = File(System.getProperty("cardfscan.modelsDir") ?: "models")
    val namesDir = File(repoRoot, "testdata/names")

    /** names_v1.json or names_v1.json.gz, whichever exists. */
    val nameIndexFile: File? =
        listOf("names_v1.json", "names_v1.json.gz").map { File(namesDir, it) }.firstOrNull { it.exists() }

    // `by lazy` computes the value on first use and caches it: the 8 MB index loads once per test run.
    val nameIndex: NameIndex by lazy {
        NameIndex.load(nameIndexFile ?: error("name index not found in $namesDir"))
    }
}
