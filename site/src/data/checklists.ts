// Self-test checklists, one per track stage. Each item is a concrete thing
// the reader should be able to do, not a topic to recognize. Grounded in
// track.ts theory/goal and CURRICULUM.md's checkpoints for the same stage id.

export const CHECKLISTS: Record<string, string[]> = {
  machine: [
    'Compute the ridge point (FLOPs per byte) for a TPU generation from its published FLOP/s and HBM bandwidth, without looking either number up twice.',
    'Given an op and its shapes, classify it compute-bound or memory-bound in under a minute, and say which chip generation you assumed.',
    'Name what lives in VMEM versus HBM versus SMEM, and say why VMEM residency is the entire point of a Pallas kernel.',
    'State the (8, 128) tiling lattice rule from memory, and say what changes for bf16 packing.',
    'Estimate a step-time floor for a forward pass from chip constants alone, given batch size, sequence length, and parameter count, no profiler needed.',
    'Given a roofline prediction and a measured XProf number that disagree by more than 2x, name a concrete reason for the miss instead of shrugging at "overhead."',
    'Tell the difference between ICI and DCN, and say which one a collective inside one pod uses.',
  ],
  pallas: [
    "Write a BlockSpec's index map for a tiled matmul from scratch, and say which grid axis it responds to.",
    'Explain why the TPU grid is a software pipeline and not a batch of parallel threads, and what breaks if a block is sized too large for double buffering to hide the transfer.',
    'Given a fused kernel (softmax, LayerNorm), say why it needs only one HBM round trip while the naive XLA chain needs several.',
    'Point to the carried accumulator in a tiled matmul kernel and explain why K must be the innermost grid dimension, not the outermost.',
    'Benchmark a kernel correctly (warmup, block_until_ready, median-of-N, chip stated) without being handed the checklist first.',
    'Given a production Pallas kernel you have never seen, mark every line algorithm or schedule on a first read.',
    'Read a profile of your own kernel, find a pipeline bubble, and say what block-size change would close it.',
    'Predict where a kernel will fail to compile from a VMEM budget estimate, before running it.',
  ],
  ir: [
    'Decode the dimension_numbers of any dot_general on sight: which axes contract, which batch.',
    'Given a JAX function, produce its jaxpr, StableHLO, and optimized HLO yourself, and map one equation across all three.',
    "Find the HBM spill in naive attention's HLO dump and estimate its size in bytes from the sequence length and dtype alone.",
    'Explain in one sentence why fusion cannot fix the spill you just found, and what would.',
    'Given a StableHLO op you have never seen, tell whether it is a real computation or compiler bookkeeping (a broadcast, a convert, a reshape).',
    "Pull one real operator from a production model library and name what its jaxpr carries that your toy version doesn't (remat markers, sharding constraints, dtype casts).",
    'Tell a lax.scan from an unrolled lax.associative_scan just by reading the jaxpr, with no scan primitive present in the unrolled case.',
  ],
  kernels: [
    'Derive online softmax from the two-pass definition on a blank page, ending at the (m, l, acc) triple, without checking a reference.',
    'Prove the online-softmax combine is associative and commutative, and say in one sentence why that proof is what makes ring attention correct later.',
    'Write flash attention forward from your own derivation, then diff it against Splash or Tokamax and classify every difference as algorithm, schedule, or feature.',
    'Derive dQ, dK, dV for flash attention on paper, including the D = rowsum(dO ⊙ O) identity, before wiring custom_vjp.',
    'Explain why a causal mask becomes a loop bound instead of an added tensor, and what determines the resulting speedup.',
    'Given a differential test failing at 1e-3 forward but passing at 1e-2 grads in bf16, say which failure mode that pattern suggests.',
    "Read Splash attention's mask machinery and state what scalar prefetch is doing, in your own words.",
    'Point to the loop bound in a ragged or paged kernel and say which runtime data (a page table, a lengths array) shapes it.',
  ],
  distributed: [
    'Name the three parts of a remote-DMA collective (the async push, the destination VMEM write, the semaphore signal) and say which one keeps the MXU free.',
    'Build a ring all-gather from ppermute hops and a carried accumulator, and validate it bitwise against lax.all_gather.',
    'Explain exposed communication in one sentence, and name the pattern from stage 1 that cures it.',
    'Construct a deadlock on purpose by reordering a semaphore wait in interpret mode, and explain exactly which send is waiting on which recv.',
    "Read a profile and point to where a hop overlaps compute, and where it doesn't.",
    'Explain why ring attention needs no new derivation, only the stage 3 monoid folded over a ring, and name the property that makes that composition sound.',
    'Describe what a collective actually is, in one sentence, without using the word "primitive."',
  ],
  capstone: [
    'State the gap your kernel closes, and name the two baselines (the XLA floor and the hand-tuned ceiling) you benchmark against.',
    'Produce a differential test suite covering values and grads, corner cases enumerated, passing at 1e-3 forward and 1e-2 grads in bf16.',
    'Show a benchmark record where every number states chip, dtype, shapes, and method, for both the floor and the ceiling comparison.',
    "Point to the review comments that changed your kernel, and explain what each one caught that your own testing didn't.",
    'Defend every schedule decision (block sizes, grid order, what got remat-ed) with a measured consequence, not a guess.',
    'Write the derivation section so a reader with no prior context could rebuild your kernel from it alone.',
    'Confirm the upstream PR is public with the review discussion linked, not merely "ready to send."',
  ],
}
