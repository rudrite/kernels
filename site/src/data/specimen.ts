// The specimen, as the site reads it: one manifest, one text per captured
// level, resolved at build time from the files bench/specimen/capture.py
// committed. Nothing here compiles anything or reaches the network, and no
// level is invented; a level the capture could not reach arrives as pending
// and stays that way until a real dump lands.
import manifest from '../../../bench/specimen/manifest.json'

export interface SpecimenEntry {
  id: string
  /** Floor on the hourglass, 0 at the source. */
  level: number
  /** Which column of the map the artifact belongs to. */
  lane: string
  title: string
  status: 'captured' | 'pending'
  /** Captured only: repo-relative path, artifact language, and its measurements. */
  file?: string
  lang?: string
  lines?: number
  bytes?: number
  sha256?: string
  produced_by?: string
  /** Set only where the capture run's versions do not apply, as on the
   * paste-back level, which came off another machine entirely. */
  versions?: Record<string, string>
  /** Pending only: the capture that will fill this level. */
  pending?: string
}

export interface SpecimenManifest {
  specimen: {
    id: string
    title: string
    source: string
    shapes: string
    seed: number
    checksum: string
  }
  capture: {
    command: string
    cwd: string
    backend: string
    devices: string[]
    platform: string
    versions: Record<string, string>
    scrub: string
  }
  levels: SpecimenEntry[]
}

export const SPECIMEN = manifest as SpecimenManifest

const RAW = import.meta.glob('../../../bench/specimen/artifacts/*', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>

/** Shiki grammar per artifact language; mlir has none and gets MlirCode. */
export const GRAMMAR: Record<string, string> = {
  python: 'python',
  jaxpr: 'haskell',
  hlo: 'txt',
  llvm: 'llvm',
  ptx: 'txt',
  sass: 'txt',
}

export interface SpecimenLevel extends SpecimenEntry {
  /** The artifact itself, empty on a pending level. */
  text: string
}

/** Every level in descent order, each captured one carrying its artifact. */
export const SPECIMEN_LEVELS: SpecimenLevel[] = SPECIMEN.levels.map((entry) => {
  if (entry.status === 'pending') return { ...entry, text: '' }
  const key = Object.keys(RAW).find((p) => p.endsWith(`/${entry.file!.split('/').pop()}`))
  if (!key) throw new Error(`specimen: the manifest claims ${entry.file}, which is not on disk`)
  return { ...entry, text: RAW[key]! }
})
