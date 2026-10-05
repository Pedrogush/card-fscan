package io.github.pedrogush.cardfscan.core.ocr

/**
 * Reads the `metadata_props` (key/value strings) of an ONNX model straight from its
 * protobuf bytes.
 *
 * Why not `OrtSession.metadata`? ONNX Runtime's Java binding passes strings through JNI's
 * "modified UTF-8", which corrupts characters outside the Basic Multilingual Plane. The
 * PP-OCR latin dictionary ends with two such characters (mathematical italic letters), so the
 * dictionary came back two entries short and the space class was silently lost.
 *
 * Protobuf wire format in brief: a message is a sequence of (tag, value) pairs; the tag is a
 * varint `fieldNumber << 3 | wireType`. ModelProto field 14 is `metadata_props`, a repeated
 * StringStringEntryProto whose field 1 is the key and field 2 the value.
 */
object OnnxMetadata {
    private const val MODEL_METADATA_PROPS = 14

    fun read(model: ByteArray): Map<String, String> {
        val out = LinkedHashMap<String, String>()
        val top = Reader(model, 0, model.size)
        while (top.hasMore()) {
            val tag = top.varint()
            val field = (tag ushr 3).toInt()
            val wire = (tag and 7).toInt()
            if (field == MODEL_METADATA_PROPS && wire == WIRE_LEN) {
                val len = top.varint().toInt()
                val entry = Reader(model, top.pos, top.pos + len)
                var key = ""
                var value = ""
                while (entry.hasMore()) {
                    val t = entry.varint()
                    if ((t and 7).toInt() != WIRE_LEN) {
                        entry.skip(t)
                        continue
                    }
                    val n = entry.varint().toInt()
                    val s = String(model, entry.pos, n, Charsets.UTF_8)
                    entry.pos += n
                    when ((t ushr 3).toInt()) {
                        1 -> key = s
                        2 -> value = s
                    }
                }
                out[key] = value
                top.pos += len
            } else {
                top.skip(tag)
            }
        }
        return out
    }

    private const val WIRE_VARINT = 0
    private const val WIRE_I64 = 1
    private const val WIRE_LEN = 2
    private const val WIRE_I32 = 5

    /** A cursor over bytes[pos until end]. */
    private class Reader(val bytes: ByteArray, var pos: Int, val end: Int) {
        fun hasMore() = pos < end

        fun varint(): Long {
            var result = 0L
            var shift = 0
            while (true) {
                val b = bytes[pos++].toInt() and 0xFF
                result = result or ((b and 0x7F).toLong() shl shift)
                if (b and 0x80 == 0) return result
                shift += 7
            }
        }

        /** Skips the value of a field whose tag was just read. */
        fun skip(tag: Long) {
            when ((tag and 7).toInt()) {
                WIRE_VARINT -> varint()
                WIRE_I64 -> pos += 8
                WIRE_LEN -> {
                    // Not `pos += varint()`: that would read `pos` before varint() advances it.
                    val len = varint().toInt()
                    pos += len
                }
                WIRE_I32 -> pos += 4
                else -> error("unsupported protobuf wire type in tag $tag")
            }
        }
    }
}
