#!/usr/bin/env node
// Voice gate: scans authored prose sources for the mechanical voice
// violations (the bans a regex can catch; the judgment layer lives in
// docs/VOICE.md). Exits non-zero on any hit. Generated corpora are machine
// dumps, not prose, and are excluded by name.
import { execSync } from 'node:child_process'
import { readFileSync } from 'node:fs'

const PROSE_ROOTS = [
  'README.md',
  'CURRICULUM.md',
  'site/src/data/',
  'site/src/pages/',
  'site/src/components/',
  'site/src/layouts/',
  'site/src/exhibits/',
]

// Machine dumps and generated indexes: verbatim compiler output, extracted lab
// metadata. Their text is evidence, not authored prose.
const EXCLUDE = [
  /-corpus\.json$/,
  /^site\/src\/data\/(ir-maps|ir-attention|hlo-pairs|transform-gallery|labs\.generated)\.json$/,
  /\.test\.ts$/, // the voice tests name the patterns they ban
]

const PATTERNS = [
  { name: 'em-dash', re: /—/ },
  { name: 'double-hyphen as dash', re: /\s--\s/ },
  { name: 'en-dash as dash', re: /\s–\s/ },
  { name: 'punchline hook', re: /here'?s the (catch|kicker|magic|trick|secret)|the whole trick|that'?s the (secret|magic|trick)\b|and that'?s it\./i },
  { name: 'report scaffold', re: /In this (section|chapter|page),? (we|you)\b/ },
  { name: 'report scaffold', re: /it is worth noting|it should be noted|needless to say/i },
  { name: 'reassurance / banned idiom', re: /don'?t worry|defaults are sensible|fat pipe|rides along|bumping into/i },
  { name: 'bold-label stamp', re: /\*\*[A-Z][^*\n]{0,40}\.\*\*\s/ },
]

const root = execSync('git rev-parse --show-toplevel', { encoding: 'utf8' }).trim()
process.chdir(root)

const tracked = execSync('git ls-files --cached --others --exclude-standard', { encoding: 'utf8' })
  .trim()
  .split('\n')
  .filter(Boolean)
const files = tracked.filter(
  (f) => PROSE_ROOTS.some((r) => (r.endsWith('/') ? f.startsWith(r) : f === r)) && !EXCLUDE.some((re) => re.test(f)),
)

const failures = []
for (const path of files) {
  const lines = readFileSync(path, 'utf8').split('\n')
  lines.forEach((line, i) => {
    for (const p of PATTERNS) {
      if (p.re.test(line)) failures.push(`${path}:${i + 1}: ${p.name}`)
    }
  })
}

if (failures.length) {
  console.error('voice: BLOCKED\n' + failures.map((f) => `  ${f}`).join('\n'))
  process.exit(1)
}
console.log(`voice: clean (${files.length} prose files)`)
