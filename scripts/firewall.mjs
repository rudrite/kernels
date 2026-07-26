#!/usr/bin/env node
// Firewall: scans tracked files (and site/dist when present) for strings that
// must never appear in this public repo. Exits non-zero on any hit.
// The scrub rails themselves are excluded: they must name the patterns to block them.
import { execSync } from 'node:child_process'
import { readFileSync, existsSync, readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'

const SELF_EXCLUDE = new Set(['scripts/firewall.mjs', '.githooks/commit-msg'])
const BINARY_EXT = /\.(png|jpg|jpeg|gif|webp|ico|woff2?|ttf|otf|pdf|mp4)$/i

// Assembled from fragments so this file never contains a bare blocked string.
const frag = (a, b) => a + b
const PATTERNS = [
  { name: frag('cla', 'ude'), re: new RegExp(frag('cla', 'ude'), 'i') },
  { name: frag('anthro', 'pic'), re: new RegExp(frag('anthro', 'pic'), 'i') },
  { name: frag('carto', 'graph'), re: new RegExp(frag('carto', 'graph'), 'i') },
  { name: frag('co-authored', '-by'), re: new RegExp(frag('co-authored', '-by'), 'i') },
  { name: 'local home path', re: /\/Users\/[a-z]+/i },
  { name: 'personal email', re: new RegExp(frag('mridulsahu', '01')) },
  { name: 'api key shape', re: /sk-[a-zA-Z0-9-]{12,}/ },
]

function scanText(path, text, failures) {
  for (const p of PATTERNS) {
    if (p.re.test(text)) failures.push(`${path}: matches "${p.name}"`)
  }
}

const failures = []

// Always operate from the repo root, wherever the script was invoked from.
const root = execSync('git rev-parse --show-toplevel', { encoding: 'utf8' }).trim()
process.chdir(root)

const tracked = execSync('git ls-files', { encoding: 'utf8' }).trim().split('\n').filter(Boolean)
for (const path of tracked) {
  if (SELF_EXCLUDE.has(path) || BINARY_EXT.test(path)) continue
  scanText(path, readFileSync(path, 'utf8'), failures)
}

// History scan: commit messages and author identities.
const log = execSync('git log --format="%H %an %ae%n%B"', { encoding: 'utf8' })
scanText('git history', log, failures)

// Built output, when present.
const dist = 'site/dist'
if (existsSync(dist)) {
  const walk = (dir) => {
    for (const entry of readdirSync(dir)) {
      const full = join(dir, entry)
      if (statSync(full).isDirectory()) walk(full)
      else if (!BINARY_EXT.test(full)) scanText(full, readFileSync(full, 'utf8'), failures)
    }
  }
  walk(dist)
}

if (failures.length) {
  console.error('firewall: BLOCKED\n' + failures.map((f) => `  ${f}`).join('\n'))
  process.exit(1)
}
console.log(`firewall: clean (${tracked.length} tracked files${existsSync(dist) ? ' + dist' : ''} + history)`)
