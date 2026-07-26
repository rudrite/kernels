// Frame-to-line map for the tiled matmul walk. Ties pipelinedMatmulTrace's
// six frames (site/src/exhibits/matmul.ts, kBlocks = 4, called as
// matmulTrace(4) in gym.astro) to line ranges in the walk's own code string
// (site/src/data/walks/matmul.ts). Ranges are 1-indexed and inclusive, the
// convention SyncedWalk uses (CodeWalk walks use 0-indexed ranges; these are separate consumers), so `lines[i]` (from
// lib/highlight.ts, where lines[i] is source line i + 1) lines up directly.
//
// Verified against both files: pipelinedMatmulTrace(4) pushes 1 prologue
// frame, kBlocks (4) compute frames, and 1 epilogue frame, for 6 total; the
// walk's code string was split on '\n' and its lines printed with 1-indexed
// numbers to check each range names the statement it claims to. This map is
// specific to kBlocks = 4: a different kBlocks changes the frame count
// (kBlocks + 2) but not the code, so a kBlocks = 6 walk would need six
// compute-frame entries repeating the same [2, 6] range, not a new range.
export const MATMUL_FRAME_LINES: [number, number][] = [
  // frame 0 - prologue: "grid step 0 · DMA A0, B0 into the first slot pair;
  // the MXU has no data yet and waits". The lines that decide which A and B
  // tiles land in VMEM for a grid step are the two BlockSpecs in in_specs,
  // lines 14-17 of the walk's code ("in_specs=[" through the closing "],").
  // The prologue frame's own DMA is exactly the first application of that
  // mapping (i, kk) -> A0 and (kk, j) -> B0, so this is the block that
  // explains it, not the grid=... line above it.
  [14, 17],

  // frame 1 - compute step for k = 0: "MXU consumes A0·B0 while DMA
  // prefetches A1, B1". k = 0 is the one reduction step where the kernel's
  // pl.when(k == 0) branch actually fires, so this frame's line range spans
  // the full per-step kernel body: reading which K-block this is (line 2),
  // the init check and zeroing (lines 3-5), and the accumulate (line 6).
  [2, 6],

  // frame 2 - compute step for k = 1: same kernel body executes each grid
  // step (pallas_call re-invokes matmul_kernel once per step); k = 1 skips
  // the pl.when branch at runtime, but the source lines driving this frame
  // are the same lines 2-6, since that is the whole reduction-step body.
  [2, 6],

  // frame 3 - compute step for k = 2. Same reasoning as frame 2.
  [2, 6],

  // frame 4 - compute step for k = 3, the last reduction step for this
  // output tile (kBlocks = 4, so k runs 0..3). Same reasoning as frame 2.
  [2, 6],

  // frame 5 - epilogue: "the accumulator flushes to HBM as the C output
  // block; slots retire". The line that says where a finished output tile
  // goes is out_specs, line 18 ("out_specs=pl.BlockSpec((bm, bn), lambda
  // i, j, kk: (i, j)),"). Unlike in_specs this is a single statement, so
  // the range is one line, not a block.
  [18, 18],
]
