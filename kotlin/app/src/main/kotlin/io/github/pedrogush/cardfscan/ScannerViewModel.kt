package io.github.pedrogush.cardfscan

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import io.github.pedrogush.cardfscan.core.CardScanner
import io.github.pedrogush.cardfscan.core.ScanReport
import io.github.pedrogush.cardfscan.core.match.NameIndex
import io.github.pedrogush.cardfscan.core.ocr.PaddleTextRecognizer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import org.opencv.core.MatOfByte
import org.opencv.imgcodecs.Imgcodecs

/** Everything the screen shows. Compose redraws whenever a new copy is emitted. */
data class UiState(
    val loading: Boolean = true,
    val busy: Boolean = false,
    val message: String = "Loading OCR model and card names…",
    val report: ScanReport? = null,
)

/**
 * Holds the scanner and the current result. A ViewModel survives screen rotation, so the
 * 16 MB of model + names is loaded once, not every time the activity is recreated.
 */
class ScannerViewModel(app: Application) : AndroidViewModel(app) {
    private val _state = MutableStateFlow(UiState())
    val state: StateFlow<UiState> = _state.asStateFlow()

    private var scanner: CardScanner? = null

    init {
        // viewModelScope cancels this coroutine if the ViewModel is destroyed;
        // Dispatchers.Default runs the heavy loading off the UI thread.
        viewModelScope.launch(Dispatchers.Default) {
            try {
                check(OpenCVLoader.initLocal()) { "OpenCV native library failed to load" }
                val assets = app.assets
                val t0 = System.currentTimeMillis()
                val index = assets.open("names_v1.json").use { NameIndex.load(it) }
                val model = assets.open("latin_PP-OCRv5_rec_mobile.onnx").use { it.readBytes() }
                scanner = CardScanner(index, PaddleTextRecognizer(model, threads = 4))
                val ms = System.currentTimeMillis() - t0
                _state.update { it.copy(loading = false, message = "Ready (${index.size} names loaded in $ms ms). Pick or take a photo.") }
            } catch (e: Throwable) {
                _state.update { it.copy(loading = false, message = "Startup failed: $e") }
            }
        }
    }

    fun scan(uri: Uri) {
        val s = scanner ?: return
        _state.update { it.copy(busy = true, message = "Scanning…", report = null) }
        viewModelScope.launch {
            val result = withContext(Dispatchers.Default) {
                runCatching {
                    val bytes = getApplication<Application>().contentResolver.openInputStream(uri)!!.use { it.readBytes() }
                    val image = Imgcodecs.imdecode(MatOfByte(*bytes), Imgcodecs.IMREAD_COLOR)
                    require(!image.empty()) { "could not decode the photo" }
                    try {
                        s.scan(image, uri.lastPathSegment ?: "photo.jpg")
                    } finally {
                        image.release()
                    }
                }
            }
            // `fold` handles both outcomes of runCatching: success or the caught exception.
            result.fold(
                onSuccess = { report ->
                    val slots = report.columns.flatMap { it.slots }
                    val auto = slots.count { it.status == "auto" }
                    val msg = "${report.config} ${report.status}: $auto/${slots.size} auto in ${report.timingMs["total"]} ms"
                    _state.update { it.copy(busy = false, message = msg, report = report) }
                },
                onFailure = { e -> _state.update { it.copy(busy = false, message = "Scan failed: ${e.message}") } },
            )
        }
    }
}
