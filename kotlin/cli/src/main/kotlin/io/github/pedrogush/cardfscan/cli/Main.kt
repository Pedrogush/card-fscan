package io.github.pedrogush.cardfscan.cli

import io.github.pedrogush.cardfscan.core.CardScanner
import io.github.pedrogush.cardfscan.core.ScanOptions
import io.github.pedrogush.cardfscan.core.match.NameIndex
import io.github.pedrogush.cardfscan.core.ocr.PaddleTextRecognizer
import java.io.File
import kotlin.system.exitProcess

const val DEFAULT_MODEL = "latin_PP-OCRv5_rec_mobile.onnx"

private const val USAGE = """usage: scan --out <dir> [options] <image>...

Writes <dir>/<image basename>.json for every image (SPEC section 4).

options:
  --index <file>     name index (default: testdata/names/names_v1.json[.gz] found from the repo)
  --model <file>     PP-OCR recognition model (default: models/$DEFAULT_MODEL)
  --threads <n>      ONNX Runtime threads (default 2)
  --debug <dir>      dump warped columns and slot crops here
  --preset <name>    crop/enhance stages to use (see ScanOptions.PRESETS)
"""

/** Parsed command line. A `data class` gets equals/hashCode/toString/copy for free. */
data class CliArgs(
    val out: File,
    val images: List<File>,
    val index: File?,
    val model: File?,
    val threads: Int,
    val debug: File?,
    val preset: String?,
)

fun parseArgs(args: List<String>): CliArgs {
    val rest = if (args.firstOrNull() == "scan") args.drop(1) else args
    var out: File? = null
    var index: File? = null
    var model: File? = null
    var threads = 2
    var debug: File? = null
    var preset: String? = null
    val images = mutableListOf<File>()
    val it = rest.iterator()
    while (it.hasNext()) {
        when (val a = it.next()) {
            "--out" -> out = File(it.next())
            "--index" -> index = File(it.next())
            "--model" -> model = File(it.next())
            "--threads" -> threads = it.next().toInt()
            "--debug" -> debug = File(it.next())
            "--preset" -> preset = it.next()
            "-h", "--help" -> usage(0)
            else -> if (a.startsWith("--")) usage(2, "unknown option $a") else images += File(a)
        }
    }
    if (out == null || images.isEmpty()) usage(2, "need --out and at least one image")
    return CliArgs(out, images, index, model, threads, debug, preset)
}

// `Nothing` is the return type of a function that never returns (it always exits or throws).
private fun usage(code: Int, message: String? = null): Nothing {
    message?.let { System.err.println("error: $it") }
    System.err.print(USAGE)
    exitProcess(code)
}

fun main(args: Array<String>) {
    val cli = parseArgs(args.toList())
    nu.pattern.OpenCV.loadLocally() // unpacks and loads OpenCV's native library (desktop only)

    val t0 = System.nanoTime()
    val indexFile = cli.index ?: Locations.defaultIndex() ?: usage(2, "name index not found; pass --index")
    val modelFile = cli.model ?: Locations.defaultModel() ?: usage(2, "OCR model not found; pass --model")
    val index = NameIndex.load(indexFile)
    val recognizer = PaddleTextRecognizer(modelFile.readBytes(), threads = cli.threads)
    System.err.println("loaded ${index.size} names and ${modelFile.name} in ${(System.nanoTime() - t0) / 1_000_000} ms")

    cli.out.mkdirs()
    val stages = cli.preset?.let { ScanOptions.PRESETS[it] ?: usage(2, "unknown preset $it") } ?: ScanOptions.DEFAULT_STAGES
    val scanner = CardScanner(index, recognizer, ScanOptions(stages = stages, debugDir = cli.debug))
    val times = mutableListOf<Long>()
    var failures = 0
    // `use` closes the recognizer (and its native ONNX session) when the block ends.
    recognizer.use {
        for (image in cli.images) {
            try {
                val report = scanner.scanFile(image)
                File(cli.out, "${image.name}.json").writeText(report.toJson() + "\n")
                val total = report.timingMs.getValue("total")
                times += total
                val slots = report.columns.flatMap { it.slots }
                val auto = slots.count { it.status == "auto" }
                println("${image.name}: ${report.config} ${report.status} auto $auto/${slots.size} in $total ms ${report.timingMs}")
            } catch (e: Exception) {
                failures++
                System.err.println("${image.name}: FAILED: $e")
            }
        }
    }
    if (times.isNotEmpty()) {
        println("photos ${times.size}, mean ${times.average().toLong()} ms, max ${times.max()} ms")
    }
    if (failures > 0) exitProcess(1)
}

/** Default file locations, so the common case needs no options. */
object Locations {
    private fun repoRoot(): File? =
        System.getProperty("cardfscan.repoRoot")?.let(::File)
            ?: generateSequence(File("").absoluteFile) { it.parentFile }.firstOrNull { File(it, "spec/SPEC.md").exists() }

    fun defaultIndex(): File? = repoRoot()?.let { root ->
        listOf("names_v1.json", "names_v1.json.gz").map { File(root, "testdata/names/$it") }.firstOrNull { it.exists() }
    }

    fun defaultModel(): File? {
        val candidates = buildList {
            System.getProperty("cardfscan.modelsDir")?.let { add(File(it, DEFAULT_MODEL)) }
            // Installed layout: <app>/lib/cli.jar next to <app>/models/.
            val jar = File(CliArgs::class.java.protectionDomain.codeSource.location.toURI())
            add(File(jar.parentFile.parentFile, "models/$DEFAULT_MODEL"))
            repoRoot()?.let { add(File(it, "kotlin/models/$DEFAULT_MODEL")) }
        }
        return candidates.firstOrNull { it.exists() }
    }
}
