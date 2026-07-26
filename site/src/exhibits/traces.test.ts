import { describe, expect, it } from 'vitest'
import { matmulTrace } from './matmul'
import { flashTrace, ringTrace } from './flash'

describe('matmulTrace', () => {
  it('emits prologue + one frame per K block + epilogue', () => {
    expect(matmulTrace(4).frames).toHaveLength(6)
    expect(matmulTrace(2).frames).toHaveLength(4)
  })

  it('alternates slot pairs (double buffering)', () => {
    const frames = matmulTrace(4).frames
    expect(frames[1]!.slots['a0']).toBe('A0')
    expect(frames[2]!.slots['a1']).toBe('A1')
    expect(frames[3]!.slots['a0']).toBe('A2')
  })

  it('computes on every steady-state frame and flushes in the epilogue', () => {
    const frames = matmulTrace(3).frames
    expect(frames[0]!.compute).toBeNull()
    for (let k = 1; k <= 3; k++) expect(frames[k]!.compute?.unit).toBe('MXU')
    expect(frames.at(-1)!.slots['acc']).toContain('HBM')
  })
})

describe('flashTrace / ringTrace', () => {
  it('carries (m, l, acc) formulas on every compute frame', () => {
    for (const f of flashTrace(4).frames) {
      if (f.compute?.unit === 'MXU') {
        expect(f.state?.['m']).toContain('max')
        expect(f.state?.['l']).toContain('rowsum')
        expect(f.state?.['acc']).toContain('e^')
      }
    }
  })

  it('ring is the identical schedule with transport swapped to ICI', () => {
    const flash = flashTrace(4)
    const ring = ringTrace(4)
    expect(ring.frames).toHaveLength(flash.frames.length)
    expect(ring.remoteGroups).toEqual(['K', 'V'])
    expect(flash.remoteGroups).toEqual([])
    // Every KV transfer differs only in transport; Q always comes from HBM.
    for (let i = 0; i < flash.frames.length; i++) {
      const fd = flash.frames[i]!.dma
      const rd = ring.frames[i]!.dma
      expect(rd).toHaveLength(fd.length)
      for (let j = 0; j < fd.length; j++) {
        expect(rd[j]!.tile).toBe(fd[j]!.tile)
        expect(rd[j]!.toSlot).toBe(fd[j]!.toSlot)
        expect(rd[j]!.kind).toBe(fd[j]!.tile === 'Q' ? 'hbm' : 'ici')
      }
    }
    // State formulas are transport-independent.
    for (let i = 0; i < flash.frames.length; i++) {
      expect(ring.frames[i]!.state).toEqual(flash.frames[i]!.state)
    }
  })

  it('never materializes scores: no S tile exists in HBM', () => {
    for (const t of flashTrace(4).hbmTiles) expect(t.id).not.toMatch(/^S/)
  })
})
