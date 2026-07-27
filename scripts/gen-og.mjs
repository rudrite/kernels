#!/usr/bin/env node
// Open-graph cards for the pages that share one. The kernel path's stage
// cards were made before the site grew siblings; this renders the same
// design for every path and workshop surface so a shared link says what it
// is. Fonts come from the site's own woff files, so the cards match the
// pages they represent.
//
// Rasterizing uses qlmanage, which ships with macOS. It always renders an
// SVG into a square canvas, so the card is drawn into the middle band of a
// 1200x1200 image and cropped back out; scaling it instead distorts the
// type. Cards are checked in, so this only runs when the set changes.
import { execSync } from 'node:child_process'
import { mkdtempSync, readFileSync, writeFileSync, copyFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const root = execSync('git rev-parse --show-toplevel', { encoding: 'utf8' }).trim()
const fonts = join(root, 'site/node_modules/@fontsource')
const outDir = join(root, 'site/public/og')

const b64 = (p) => readFileSync(p).toString('base64')
const FACE = (family, file, weight) =>
  `@font-face{font-family:'${family}';font-weight:${weight};src:url(data:font/woff;base64,${b64(file)}) format('woff')}`

const FACES = [
  FACE('Plex Mono', join(fonts, 'ibm-plex-mono/files/ibm-plex-mono-latin-400-normal.woff'), 400),
  FACE('Plex Serif', join(fonts, 'ibm-plex-serif/files/ibm-plex-serif-latin-600-normal.woff'), 600),
].join('')

// One card per surface: the status line, the copper eyebrow, the headline.
const CARDS = {
  jax: { status: 'path 02 · open', eyebrow: 'twelve chapters · across the language', head: 'Learn to think in JAX.' },
  xla: { status: 'path 03 · open', eyebrow: 'fourteen chapters · through the compiler', head: 'Learn how XLA works.' },
  pytorch: { status: 'path 04 · open', eyebrow: 'twelve chapters · landing on tpu', head: 'Learn to think in PyTorch.' },
  gym: { status: 'the workshop', eyebrow: 'one floor per path · every answer measured', head: 'Fluency is drills, not prose.' },
  mistakes: { status: 'the workshop', eyebrow: 'one wing per path · errors kept verbatim', head: 'Meet every failure in a toy first.' },
}

const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

const BAND = 285   // (1200 - 630) / 2: where the card sits in the square canvas

const svg = ({ status, eyebrow, head }) => `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1200" viewBox="0 0 1200 1200">
<style>${FACES}
.mono{font-family:'Plex Mono',monospace}
.serif{font-family:'Plex Serif',serif;font-weight:600}
.label{font-size:15px;letter-spacing:.18em;fill:#6f767d}
.mark{font-size:31px;fill:#d6d9dc}
.status{font-size:28px;fill:#d6d9dc}
.eyebrow{font-size:21px;letter-spacing:.16em;fill:#c88a70}
.head{font-size:74px;fill:#d6d9dc}
.foot{font-size:20px;fill:#6f767d}
.copper{fill:#c88a70}
</style>
<rect width="1200" height="1200" fill="#16181c"/>
<g transform="translate(0 ${BAND})">
<line x1="0" y1="124" x2="1200" y2="124" stroke="#2c3138" stroke-width="1"/>
<line x1="347" y1="0" x2="347" y2="124" stroke="#2c3138" stroke-width="1"/>
<line x1="0" y1="552" x2="1200" y2="552" stroke="#2c3138" stroke-width="1"/>
<text class="mono label" x="34" y="42">SITE</text>
<text class="mono mark" x="34" y="86">rudr<tspan class="copper">i</tspan>te kernels</text>
<text class="mono label" x="383" y="42">STATUS</text>
<text class="mono status" x="383" y="86">${esc(status)}</text>
<text class="mono eyebrow" x="64" y="294">${esc(eyebrow.toUpperCase())}</text>
<text class="serif head" x="62" y="392">${esc(head)}</text>
<text class="mono foot" x="64" y="601">kernels.rudrite.com</text>
<text class="mono foot copper" x="1136" y="601" text-anchor="end">// kernels, derived</text>
</g>
</svg>`

const tmp = mkdtempSync(join(tmpdir(), 'og-'))
for (const [slug, card] of Object.entries(CARDS)) {
  const svgPath = join(tmp, `${slug}.svg`)
  writeFileSync(svgPath, svg(card))
  execSync(`qlmanage -t -s 1200 -o ${tmp} ${svgPath}`, { stdio: 'ignore' })
  const rendered = join(tmp, `${slug}.svg.png`)
  execSync(`sips -c 630 1200 ${rendered} --out ${rendered}`, { stdio: 'ignore' })   // crop the band, never scale
  copyFileSync(rendered, join(outDir, `${slug}.png`))
  console.log(`gen-og: ${slug}.png`)
}
rmSync(tmp, { recursive: true, force: true })
