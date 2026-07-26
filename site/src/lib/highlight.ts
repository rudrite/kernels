// Build-time syntax highlighting for CodeWalk. Shiki renders the whole file
// as one HTML string with one <span class="line"> per source line; we split
// that string back into per-line fragments so CodeWalk can position, number,
// and dim/highlight lines independently. The highlighting itself comes
// straight from shiki's grammar for `lang`: nothing here invents color or
// token data.
import { codeToHtml } from 'shiki'

/**
 * Highlights `code` as `lang` with the vesper theme and returns one HTML
 * string per source line (each a shiki `<span class="line">…</span>`
 * fragment, stripped of the surrounding `<pre><code>`). Array order matches
 * source line numbers 1:1, so `lines[i]` is line `i + 1`.
 */
export async function highlightLines(code: string, lang: string): Promise<string[]> {
  const html = await codeToHtml(code, { lang, theme: 'vesper' })
  return splitLineSpans(html)
}

// Shiki emits each line as a top-level `<span class="line">…</span>`, with
// token spans nested inside it. A non-greedy regex would stop at the first
// inner `</span>` instead of the line's own closing tag, so we scan and
// track open/close depth to find where each line span actually ends.
function splitLineSpans(html: string): string[] {
  const openTag = '<span class="line">'
  const lines: string[] = []
  let searchFrom = 0

  for (;;) {
    const start = html.indexOf(openTag, searchFrom)
    if (start === -1) break

    let depth = 1
    let cursor = start + openTag.length
    while (depth > 0) {
      const nextOpen = html.indexOf('<span', cursor)
      const nextClose = html.indexOf('</span>', cursor)
      if (nextClose === -1) {
        throw new Error('highlightLines: unbalanced <span> tags in shiki output')
      }
      if (nextOpen !== -1 && nextOpen < nextClose) {
        depth += 1
        cursor = nextOpen + '<span'.length
      } else {
        depth -= 1
        cursor = nextClose + '</span>'.length
      }
    }

    lines.push(html.slice(start, cursor))
    searchFrom = cursor
  }

  return lines
}
