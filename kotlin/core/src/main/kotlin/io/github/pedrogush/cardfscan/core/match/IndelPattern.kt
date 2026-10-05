package io.github.pedrogush.cardfscan.core.match

/**
 * `ratio(a, b) = 100 * (1 - indel(a, b) / (len(a) + len(b)))`, the score of SPEC section 3
 * (equal to `rapidfuzz.fuzz.ratio`).
 *
 * The insert/delete-only edit distance satisfies `indel = len(a) + len(b) - 2 * LCS(a, b)`,
 * so we only need the length of the longest common subsequence. [IndelPattern] computes it with
 * the bit-parallel algorithm of Hyyrö (2004): one machine word holds the DP column for 64
 * characters of the pattern, so comparing a ~20-char query against a ~20-char key costs
 * about 20 word operations. That makes brute force over the whole ~60k-key index cheap.
 *
 * Normalised keys only contain `a-z`, `0-9` and space (37 symbols), which keeps the
 * per-symbol bit masks in a tiny array.
 */
class IndelPattern(val text: String) {
    private val blocks = (text.length + 63) / 64
    /** masks[symbol * blocks + block]: bit i set when text[64*block + i] == symbol. */
    private val masks = LongArray(ALPHABET * blocks)

    init {
        text.forEachIndexed { i, c ->
            val sym = symbolOf(c)
            if (sym >= 0) masks[sym * blocks + i / 64] = masks[sym * blocks + i / 64] or (1L shl (i % 64))
        }
    }

    /** Length of the longest common subsequence of this pattern and [other]. */
    fun lcs(other: ByteArray): Int {
        if (text.isEmpty() || other.isEmpty()) return 0
        return if (blocks == 1) lcsSingle(other) else lcsMulti(other)
    }

    /** ratio(pattern, other) as rapidfuzz computes it, rounded through float32 like the reference. */
    fun ratio(other: ByteArray): Float {
        val lenSum = text.length + other.size
        if (lenSum == 0) return 100f
        val indel = lenSum - 2 * lcs(other)
        return (100.0 * (1.0 - indel.toDouble() / lenSum)).toFloat()
    }

    private fun lcsSingle(other: ByteArray): Int {
        var v = -1L // all ones
        for (b in other) {
            val sym = symbolOfByte(b)
            val m = if (sym >= 0) masks[sym] else 0L
            val u = v and m
            v = (v + u) or (v and m.inv())
        }
        val used = if (text.length == 64) -1L else (1L shl text.length) - 1
        return java.lang.Long.bitCount(v.inv() and used)
    }

    private fun lcsMulti(other: ByteArray): Int {
        val v = LongArray(blocks) { -1L }
        for (b in other) {
            val sym = symbolOfByte(b)
            var carry = 0L
            for (k in 0 until blocks) {
                val m = if (sym >= 0) masks[sym * blocks + k] else 0L
                val vk = v[k]
                val u = vk and m
                // 64-bit add with carry in and out (unsigned comparison detects overflow).
                val sum1 = vk + u
                val c1 = java.lang.Long.compareUnsigned(sum1, vk) < 0
                val sum = sum1 + carry
                val c2 = carry != 0L && sum == 0L
                carry = if (c1 || c2) 1L else 0L
                v[k] = sum or (vk and m.inv())
            }
        }
        var count = 0
        for (k in 0 until blocks) {
            val bitsHere = minOf(64, text.length - 64 * k)
            val used = if (bitsHere == 64) -1L else (1L shl bitsHere) - 1
            count += java.lang.Long.bitCount(v[k].inv() and used)
        }
        return count
    }

    companion object {
        private const val ALPHABET = 37

        fun symbolOf(c: Char): Int = when (c) {
            in 'a'..'z' -> c - 'a'
            in '0'..'9' -> 26 + (c - '0')
            ' ' -> 36
            else -> -1
        }

        private fun symbolOfByte(b: Byte): Int = symbolOf(b.toInt().toChar())

        /** Convenience for tests: ratio of two already-normalised strings. */
        fun ratio(a: String, b: String): Float = IndelPattern(a).ratio(b.toByteArray(Charsets.US_ASCII))
    }
}
