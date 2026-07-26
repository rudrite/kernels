// The mastery ledger: every piece of content on the site belongs to exactly
// one chapter as an ordered work item. The path builds mastery by pacing a
// reader through these; nothing on the site is off-course.
//
// kinds: read (the chapter itself), instrument (interact with an exhibit),
// drill (a gym streak target, auto-checked), walk (a guided code walk),
// museum (mistake exhibits), lab (auto-checked from mark-as-run),
// checklist (auto-checked when every can-you item is ticked).

export type AutoRule =
  | { type: 'streak'; key: string; goal: number }
  | { type: 'labs'; ids: string[] }
  | { type: 'checklist'; stage: string; total: number }

export interface WorkItem {
  id: string
  label: string
  href?: string
  auto?: AutoRule
}

export const MASTERY: Record<string, WorkItem[]> = {
  'l:source': [
    { id: 'read', label: 'read the chapter: what tracing takes away' },
    { id: 'trace', label: 'trace one of your own functions with jax.make_jaxpr and read what survived' },
  ],
  'l:jaxpr': [
    { id: 'read', label: 'read the chapter: equations, shapes, the f32 upcast' },
    { id: 'oracle', label: 'shape oracle: streak of 5', href: '/gym#drills', auto: { type: 'streak', key: 'gym.shape.streak', goal: 5 } },
  ],
  'l:stablehlo': [
    { id: 'read', label: 'read the chapter: MLIR, dialects, dimension_numbers' },
    { id: 'decoder', label: 'dot_general decoder: streak of 5', href: '/gym#drills', auto: { type: 'streak', key: 'gym.dot.streak', goal: 5 } },
    { id: 'xray', label: 'corpus x-ray: narrate two programs you have not seen', href: '/gym#xray' },
  ],
  'l:xla': [
    { id: 'read', label: 'read the chapter: fusion and its exact limit' },
    { id: 'spot', label: 'spot the decision: streak of 3', href: '/gym#drills', auto: { type: 'streak', key: 'gym.spot.streak', goal: 3 } },
  ],
  'l:gap': [
    { id: 'read', label: 'read the chapter: the two modes, and why the crossing is on foot' },
  ],
  'l:pallas': [
    { id: 'read', label: 'read the chapter: refs, BlockSpecs, the grid' },
    { id: 'sandbox', label: 'BlockSpec sandbox: produce one legal and one illegal schedule, explain both', href: '/s/pallas#instruments' },
  ],
  'l:mosaic': [
    { id: 'read', label: 'read the chapter: what lowers below pallas_call' },
    { id: 'mxray', label: 'mosaic x-ray: find your index maps compiled into functions', href: '/gym#mosaic' },
  ],
  'l:tpu': [
    { id: 'read', label: 'read the chapter: the four numbers' },
    { id: 'roofline', label: 'roofline playground: find the shape where matmul goes memory-bound', href: '/s/machine#instruments' },
  ],
  'l:ici': [
    { id: 'read', label: 'read the chapter: rings, remote DMA, semaphores' },
    { id: 'ring', label: 'EX·04: watch the same schedule grow longer arrows', href: '/s/distributed#instruments' },
  ],
  's:machine': [
    { id: 'read', label: 'read the chapter and the two scaling-book chapters' },
    { id: 'lab', label: 'run LAB·0.1 on a Colab TPU', href: 'https://colab.research.google.com/github/rudrite/kernels/blob/main/labs/stage-0/lab-0.1-rooflines.ipynb', auto: { type: 'labs', ids: ['LAB·0.1'] } },
    { id: 'canyou', label: 'tick every can-you item honestly', auto: { type: 'checklist', stage: 'machine', total: 7 } },
  ],
  's:pallas': [
    { id: 'read', label: 'read the chapter; step EX·02 against EX·07 until the difference is boring' },
    { id: 'walks', label: 'step both guided walks, then the synced walk (EX·12)', href: '/s/pallas#walks' },
    { id: 'museum', label: 'read all six mistake exhibits; predict each error before revealing it', href: '/mistakes' },
    { id: 'labs', label: 'run LAB·1.1 through LAB·1.4', href: '#labs', auto: { type: 'labs', ids: ['LAB·1.1', 'LAB·1.2', 'LAB·1.3', 'LAB·1.4'] } },
    { id: 'canyou', label: 'tick every can-you item honestly', auto: { type: 'checklist', stage: 'pallas', total: 8 } },
  ],
  's:ir': [
    { id: 'read', label: 'read the chapter; find the spill in EX·05 yourself' },
    { id: 'decoder10', label: 'dot_general decoder: streak of 10 (the fluency bar)', href: '/gym#drills', auto: { type: 'streak', key: 'gym.dot.streak', goal: 10 } },
    { id: 'oracle10', label: 'shape oracle: streak of 10', href: '/gym#drills', auto: { type: 'streak', key: 'gym.shape.streak', goal: 10 } },
    { id: 'spot5', label: 'spot the decision: streak of 5', href: '/gym#drills', auto: { type: 'streak', key: 'gym.spot.streak', goal: 5 } },
    { id: 'labs', label: 'run LAB·2.1 and LAB·2.2', href: '#labs', auto: { type: 'labs', ids: ['LAB·2.1', 'LAB·2.2'] } },
    { id: 'canyou', label: 'tick every can-you item honestly', auto: { type: 'checklist', stage: 'ir', total: 7 } },
  ],
  's:kernels': [
    { id: 'read', label: 'read the chapter; derive the monoid on paper before anything else' },
    { id: 'walks', label: 'step the flash and causal walks, then the production Splash walk', href: '/s/kernels#walks' },
    { id: 'labs', label: 'run LAB·3.1 through LAB·3.4, blind-build rules observed', href: '#labs', auto: { type: 'labs', ids: ['LAB·3.1', 'LAB·3.2', 'LAB·3.3', 'LAB·3.4'] } },
    { id: 'canyou', label: 'tick every can-you item honestly', auto: { type: 'checklist', stage: 'kernels', total: 8 } },
  ],
  's:distributed': [
    { id: 'read', label: 'read the chapter; step into the deadlock in EX·10 on purpose' },
    { id: 'mesh', label: 'EX·11: compute one hop time by hand, confirm against the caption', href: '/s/distributed#instruments' },
    { id: 'labs', label: 'run LAB·4.1 and LAB·4.2', href: '#labs', auto: { type: 'labs', ids: ['LAB·4.1', 'LAB·4.2'] } },
    { id: 'canyou', label: 'tick every can-you item honestly', auto: { type: 'checklist', stage: 'distributed', total: 7 } },
  ],
  's:capstone': [
    { id: 'read', label: 'read the chapter; pick the gap and write the derivation first' },
    { id: 'build', label: 'forward and backward, differential-tested, benchmarked against floor and ceiling' },
    { id: 'upstream', label: 'send it upstream; link the review from the site' },
    { id: 'canyou', label: 'tick every can-you item honestly', auto: { type: 'checklist', stage: 'capstone', total: 7 } },
  ],
}
