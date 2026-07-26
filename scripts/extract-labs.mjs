#!/usr/bin/env node
// Notebooks are the single source of truth for lab code. This script extracts
// cells tagged `site:show` from every notebook under labs/ into a generated
// JSON the site renders. Runs before every site build; output is gitignored.
import { execSync } from 'node:child_process'
import { readFileSync, writeFileSync, mkdirSync, readdirSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'

const root = execSync('git rev-parse --show-toplevel', { encoding: 'utf8' }).trim()
const labsDir = join(root, 'labs')
const outFile = join(root, 'site/src/data/labs.generated.json')

const notebooks = []
const walk = (dir) => {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry)
    if (statSync(full).isDirectory()) walk(full)
    else if (entry.endsWith('.ipynb')) notebooks.push(full)
  }
}
walk(labsDir)

const out = {}
for (const path of notebooks) {
  const nb = JSON.parse(readFileSync(path, 'utf8'))
  const meta = nb.metadata?.lab
  if (!meta?.designator) {
    console.error(`extract-labs: ${path} has no metadata.lab.designator`)
    process.exit(1)
  }
  const cells = nb.cells
    .filter((c) => c.cell_type === 'code' && (c.metadata?.tags ?? []).includes('site:show'))
    .map((c) => ({
      code: c.source.join(''),
      lane: (c.metadata?.tags ?? []).includes('algorithm') ? 'algorithm' : (c.metadata?.tags ?? []).includes('schedule') ? 'schedule' : null,
    }))
  out[meta.designator] = {
    file: relative(root, path),
    title: meta.title,
    time: meta.time,
    hardware: meta.hardware,
    cells,
  }
}

mkdirSync(join(root, 'site/src/data'), { recursive: true })
writeFileSync(outFile, JSON.stringify(out, null, 1))
console.log(`extract-labs: ${notebooks.length} notebook(s) → ${Object.keys(out).length} lab(s)`)
