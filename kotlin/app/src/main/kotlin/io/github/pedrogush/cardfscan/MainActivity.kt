package io.github.pedrogush.cardfscan

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import io.github.pedrogush.cardfscan.core.ColumnReport
import io.github.pedrogush.cardfscan.core.ScanReport
import java.io.File

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(Modifier.fillMaxSize()) { ScannerScreen() }
            }
        }
    }
}

/**
 * Phase-1 screen: pick a photo (Photo Picker) or take one with the system camera app at
 * full resolution, scan it, and list what was read.
 */
@Composable
fun ScannerScreen(vm: ScannerViewModel = viewModel()) {
    // `by` delegates to the State object: reading `state` subscribes this composable to changes.
    val state by vm.state.collectAsStateWithLifecycle()
    val context = androidx.compose.ui.platform.LocalContext.current
    var pendingPhoto by remember { mutableStateOf<Uri?>(null) }

    val pickPhoto = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) vm.scan(uri)
    }
    val takePhoto = rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) { saved ->
        val uri = pendingPhoto
        if (saved && uri != null) vm.scan(uri)
    }

    Column(Modifier.fillMaxSize().safeDrawingPadding().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("card-fscan", style = MaterialTheme.typography.headlineSmall)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            val enabled = !state.loading && !state.busy
            Button(enabled = enabled, onClick = {
                pickPhoto.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
            }) { Text("Pick photo") }
            Button(enabled = enabled, onClick = {
                val file = File(context.cacheDir, "photos/shot_${System.currentTimeMillis()}.jpg")
                file.parentFile?.mkdirs()
                val uri = FileProvider.getUriForFile(context, "${context.packageName}.files", file)
                pendingPhoto = uri
                takePhoto.launch(uri)
            }) { Text("Take photo") }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (state.loading || state.busy) CircularProgressIndicator()
            Text(state.message)
        }
        state.report?.let { ReportList(it) }
    }
}

@Composable
private fun ReportList(report: ScanReport) {
    LazyColumn(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(4.dp)) {
        items(report.columns) { column -> ColumnCard(column) }
    }
}

@Composable
private fun ColumnCard(column: ColumnReport) {
    val ok = column.status == "ok"
    Column(Modifier.fillMaxWidth().padding(vertical = 6.dp)) {
        val stop = if (column.stopCard) ", stop card" else ""
        Text(
            "Column ${column.column}: ${column.status} (${column.nCards} cards$stop)" + (column.reason?.let { " - $it" } ?: ""),
            style = MaterialTheme.typography.titleMedium,
            color = if (ok) Color(0xFF2E7D32) else Color(0xFFC62828),
        )
        for (slot in column.slots) {
            val label = when (slot.status) {
                "auto" -> slot.name
                "review" -> "? ${slot.rawText}  →  ${slot.candidates.firstOrNull()?.name ?: ""}"
                else -> "(empty)"
            }
            Text(
                "%2d  %s".format(slot.slot, label),
                style = MaterialTheme.typography.bodySmall,
                color = if (slot.status == "auto") Color.Unspecified else Color(0xFFEF6C00),
            )
        }
    }
}
