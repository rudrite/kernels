// The exhibit engine's contract: exhibits play traces. A trace is a list of
// discrete frames computed by a pure generator from real block/schedule
// parameters. The renderer never invents anything; kernel state appears as
// formulas, never as fabricated numbers.

export interface HbmTile {
  id: string
  /** Which logical array this tile belongs to (drawn as one labeled group). */
  group: string
  row: number
  col: number
}

export interface DmaEdge {
  tile: string
  toSlot: string
  /** hbm = local memory traffic (steel); ici = arrives from a neighbor chip. */
  kind: 'hbm' | 'ici'
}

export interface TraceFrame {
  /** One mono line: what the machine does in this step. */
  caption: string
  /** Tile ids being read this frame (lit in the HBM strata). */
  reads: string[]
  /** Slot occupancy after this frame: slot id to tile id (or null = empty). */
  slots: Record<string, string | null>
  /** Active compute this frame, or null while the unit waits. */
  compute: { unit: 'MXU' | 'VPU'; expr: string } | null
  /** Carried state after this frame, as formulas (never invented values). */
  state?: Record<string, string>
  /** Transfers in flight this frame. */
  dma: DmaEdge[]
}

export interface TraceDoc {
  id: string
  title: string
  chip: string
  /** VMEM slot ids in display order. */
  slotIds: string[]
  /** Carried-state ids in display order (flash/ring kernels). */
  stateIds: string[]
  /** The HBM tile grid, grouped by logical array. */
  hbmTiles: HbmTile[]
  /** When true, some tiles arrive over ICI from a neighbor chip. */
  remoteGroups: string[]
  /** Tiles the schedule never visits (masked out); drawn struck-through. */
  deadTiles?: string[]
  frames: TraceFrame[]
  /** Where these frames come from: the honesty line rendered on the panel. */
  provenance: string
}
