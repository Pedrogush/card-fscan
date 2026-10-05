package io.github.pedrogush.cardfscan.core.match

import java.text.Normalizer

/**
 * Cleaning and normalisation of OCR text, exactly as SPEC section 3 and
 * tools/reference_match.py define them.
 */
object TextNormalizer {
    private const val TRAILING_JUNK = "{}()[]0123456789@#*%&+=|\\/"

    /** True when [s] contains at least one Unicode letter (Python's `str.isalpha`). */
    fun hasLetter(s: String): Boolean = s.codePoints().anyMatch(Character::isLetter)

    /**
     * Strips mana-cost residue from the end of an OCR line:
     * drop trailing whitespace-separated tokens without letters, then strip a
     * trailing run of [TRAILING_JUNK] characters and whitespace.
     */
    fun clean(raw: String): String {
        // ArrayDeque is Kotlin's double-ended queue; removeLast() pops from the end.
        val tokens = ArrayDeque(raw.split(WHITESPACE).filter { it.isNotEmpty() })
        while (tokens.isNotEmpty() && !hasLetter(tokens.last())) tokens.removeLast()
        val joined = tokens.joinToString(" ")
        return joined.trimEnd { it in TRAILING_JUNK || it.isWhitespace() }.trim()
    }

    /** True for text in capitals (4+ letters, at least 90% upper case), like a headline banner. */
    fun looksLikeHeadline(raw: String): Boolean {
        val letters = raw.filter { it.isLetter() }
        return letters.length >= 4 && letters.count { it.isUpperCase() } >= 0.9 * letters.length
    }

    /** NFKD, drop combining marks, lowercase, map anything outside `[a-z0-9 ]` to a space, collapse spaces. */
    fun normalize(s: String): String {
        val decomposed = Normalizer.normalize(s, Normalizer.Form.NFKD)
        val sb = StringBuilder(decomposed.length)
        decomposed.codePoints().forEach { cp ->
            if (!isCombining(cp)) sb.appendCodePoint(cp)
        }
        // `buildString` creates a StringBuilder, runs the block on it, and returns the String.
        val mapped = buildString {
            for (c in sb.toString().lowercase()) {
                append(if (c in 'a'..'z' || c in '0'..'9' || c == ' ') c else ' ')
            }
        }
        return mapped.split(' ').filter { it.isNotEmpty() }.joinToString(" ")
    }

    /** Python's `unicodedata.combining(c) != 0` is, for all practical input, a non-spacing / enclosing / spacing mark. */
    private fun isCombining(cp: Int): Boolean = when (Character.getType(cp).toByte()) {
        Character.NON_SPACING_MARK, Character.ENCLOSING_MARK, Character.COMBINING_SPACING_MARK -> true
        else -> false
    }

    /** Python's `str.split()` with no argument splits on any run of Unicode whitespace. */
    private val WHITESPACE = Regex("[\\s\\p{Z}\\u001c-\\u001f\\u0085]+")
}
