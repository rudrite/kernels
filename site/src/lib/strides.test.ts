// The view explorer computes strides in the browser, which is only
// defensible if the arithmetic matches the library it claims to explain.
// The corpus was produced by running these chains on torch 2.2.2 and
// recording what came back, so this suite is the instrument checked
// against real torch rather than against my expectations of it.
import { describe, expect, it } from 'vitest'
import corpus from '../data/pytorch/stride-corpus.json'
import { applyAll, isContiguous, rowMajorStrides, type Op } from './strides'

interface Item {
  code: string
  shape: number[]
  stride: number[]
  contiguous: boolean
}

const ITEMS = (corpus as { torch: string; items: Item[] }).items

/** Parse the corpus's generated source back into a base shape and ops. */
function parse(code: string): { shape: number[]; ops: Op[] } | null {
  const base = code.match(/reshape\(([\d, ]+)\)/)
  if (!base) return null
  const shape = base[1]!.split(',').map((n) => Number(n.trim())).filter((n) => !Number.isNaN(n))
  const chain = code.slice(code.indexOf('y = x') + 5)
  const ops: Op[] = []
  const tokens = chain.match(/\.t\(\)|\.transpose\(\d+, \d+\)|\.permute\([\d, ]+\)|\.unsqueeze\(\d+\)|\[1:\]|\[:, ::2\]|\.flatten\(\)|\.contiguous\(\)/g)
  if (!tokens) return null
  for (const t of tokens) {
    if (t === '.t()') ops.push({ kind: 't' })
    else if (t.startsWith('.transpose')) {
      const [a, b] = t.match(/\d+/g)!.map(Number)
      ops.push({ kind: 'transpose', a: a!, b: b! })
    } else if (t.startsWith('.permute')) {
      ops.push({ kind: 'permute', order: t.match(/\d+/g)!.map(Number) })
    } else if (t.startsWith('.unsqueeze')) {
      ops.push({ kind: 'unsqueeze', dim: Number(t.match(/\d+/)![0]) })
    } else if (t === '[1:]') ops.push({ kind: 'sliceRows', start: 1 })
    else if (t === '[:, ::2]') ops.push({ kind: 'strideCols', step: 2 })
    else if (t === '.flatten()') ops.push({ kind: 'flatten' })
    else if (t === '.contiguous()') ops.push({ kind: 'contiguous' })
  }
  return { shape, ops }
}

describe('stride arithmetic against real torch', () => {
  const cases = ITEMS.map((item) => ({ item, parsed: parse(item.code) })).filter((c) => c.parsed)

  it('covers the whole corpus', () => {
    expect(cases.length).toBe(ITEMS.length)
    expect(cases.length).toBeGreaterThan(40)
  })

  it('reproduces every shape torch reported', () => {
    for (const { item, parsed } of cases) {
      const v = applyAll(parsed!.shape, parsed!.ops)
      expect(v.shape, item.code).toEqual(item.shape)
    }
  })

  it('reproduces every stride torch reported', () => {
    for (const { item, parsed } of cases) {
      const v = applyAll(parsed!.shape, parsed!.ops)
      expect(v.stride, item.code).toEqual(item.stride)
    }
  })

  it('agrees with torch about contiguity', () => {
    for (const { item, parsed } of cases) {
      const v = applyAll(parsed!.shape, parsed!.ops)
      expect(isContiguous(v), item.code).toBe(item.contiguous)
    }
  })
})

describe('row-major strides', () => {
  it('steps the last axis by one and multiplies outward', () => {
    expect(rowMajorStrides([3, 4])).toEqual([4, 1])
    expect(rowMajorStrides([2, 3, 4])).toEqual([12, 4, 1])
  })
})
