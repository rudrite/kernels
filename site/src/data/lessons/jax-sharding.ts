// New file: site/src/data/lessons/jax-sharding.ts
// Sharding at the language level, under the survey chapter 10 teaches: the
// mesh and the spec read fluently, device_put and commitment, jit as the
// entry to the partitioner, and with_sharding_constraint as an instruction
// that survives into the compiled module. Every printed value ran on this
// machine (jax 0.4.38, jaxlib 0.4.38, CPU, eight host-platform devices via
// XLA_FLAGS=--xla_force_host_platform_device_count=8, 2026-08-15). The
// collectives a given pair of specs produces belong to LAB·J4; the pass that
// inserts them belongs to /xla/spmd. Neither is retold here.
import type { UnitLessons } from './index'

export const JAX_SHARDING_LESSONS: UnitLessons[] = [
  {
    unit: 'jax:sharding',
    lessons: [
      {
        id: 'mesh-and-spec',
        num: 1,
        title: 'The mesh and the spec',
        lede: 'Two objects decide where every byte of a sharded array sits, and neither of them is the array. Read them fluently and nothing downstream has to be guessed at.',
        goal: 'Given a mesh, a PartitionSpec and a global shape, state the shard shape and the index range one device holds, and name which of four errors a malformed spec raises.',
        sections: [
          {
            h: 'the mesh is the devices you already have, named',
            ps: [
              'Three different lines below build the same mesh, and JAX agrees they are the same. `jax.make_mesh((4, 2), ("data", "model"))` is the short form; a `Mesh` wrapped around `mesh_utils.create_device_mesh` is the older one; a `Mesh` over a plain reshaped array of `jax.devices()` is what both come down to. A mesh has no identity past the devices it holds and the names it gives their axes.',
              'That equality is worth more than it first looks. A `NamedSharding` built against a mesh inside a helper compares equal to one built at the call site, so passing meshes around does not quietly leave you with two shardings that disagree about the same layout.',
              'Everything about a mesh reads back. `mesh.shape` is an ordered mapping from axis name to length, `mesh.axis_names` is the tuple you passed, and `mesh.size` is the device count. No array has appeared yet, which is the point: a mesh is a grid of devices, and you can build one before deciding what to put on it.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU, eight host-platform devices): three constructions, one mesh',
              lang: 'python',
              text: `import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import numpy as np
import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh

a = jax.make_mesh((4, 2), ("data", "model"))
b = Mesh(mesh_utils.create_device_mesh((4, 2)), ("data", "model"))
c = Mesh(np.array(jax.devices()).reshape(4, 2), ("data", "model"))

print(a)
print(a.shape, a.axis_names, a.size)
print(a == b, a == c)

# Mesh('data': 4, 'model': 2)
# OrderedDict({'data': 4, 'model': 2}) ('data', 'model') 8
# True True`,
            },
          },
          {
            h: 'a spec is positional, and it may be short',
            ps: [
              "The entries of a `PartitionSpec` line up with the axes of the array, in order. Entry i names the mesh axis that splits array axis i, `None` leaves that axis whole on every device, and a tuple of names splits one array axis across several mesh axes at once.",
              '`shard_shape` answers the question without allocating anything. Hand it a global shape and it returns the shape one device ends up holding, so `(8, 16)` under `P("data", "model")` gives every device a `(2, 8)` block, and `P(("data", "model"), None)` splits the first axis eight ways for a `(1, 16)` block instead.',
              'A spec shorter than the array\'s rank is legal, and the entries you left off are read as `None`. `P("data")` on a two-axis array means `P("data", None)`, which is why the last two rows of the run below print the same shape. A spec longer than the rank is not legal, and the fourth section has the sentence it raises.',
              '>> The spec is indexed by array axis. The names inside it are mesh axes.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU, eight host-platform devices): seven specs against one shape',
              lang: 'python',
              text: `import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
from jax.sharding import NamedSharding, PartitionSpec as P

mesh = jax.make_mesh((4, 2), ("data", "model"))

for spec in [P("data", "model"), P("data", None), P(None, "model"), P(),
             P(("data", "model"), None), P("model", "data"), P("data")]:
    print(spec, NamedSharding(mesh, spec).shard_shape((8, 16)))

# PartitionSpec('data', 'model') (2, 8)
# PartitionSpec('data', None) (2, 16)
# PartitionSpec(None, 'model') (8, 8)
# PartitionSpec() (8, 16)
# PartitionSpec(('data', 'model'), None) (1, 16)
# PartitionSpec('model', 'data') (4, 4)
# PartitionSpec('data',) (2, 16)`,
            },
            table: {
              caption: 'the same seven, read in words; shard shapes measured on jax 0.4.38 CPU for a global (8, 16) on a (4, 2) mesh named ("data", "model")',
              cols: ['spec', 'shard shape', 'what one device holds'],
              rows: [
                ['P("data", "model")', '(2, 8)', 'a tile: two rows, eight columns'],
                ['P("data", None)', '(2, 16)', 'two whole rows'],
                ['P(None, "model")', '(8, 8)', 'eight whole columns, all rows'],
                ['P()', '(8, 16)', 'the entire array, on all eight devices'],
                ['P(("data", "model"), None)', '(1, 16)', 'one whole row: both mesh axes split axis 0'],
                ['P("model", "data")', '(4, 4)', 'the axes crossed the other way'],
                ['P("data")', '(2, 16)', 'the same as P("data", None): a short spec pads'],
              ],
            },
          },
          {
            h: 'every shard is a slice you can print',
            ps: [
              '`.addressable_shards` turns the arrangement into eight objects you can look at. Each one carries the device it sits on, the index tuple naming which slice of the global array it is, and the local array itself.',
              'The index is an ordinary tuple of Python slices, so a shard\'s place in the global array is something you read rather than something you remember. Device 0 holds rows 0 to 2 and columns 0 to 8, device 1 holds the same rows and the next eight columns, and device 2 has moved down a row block.',
              'Printing the first element of each shard against a flat `arange` confirms the tiling: 0, then 8, then 32. Device order runs along the last mesh axis first, so `model` advances every device and `data` advances every two.',
              'Chapter 10 draws this same layout as a picture with `jax.debug.visualize_array_sharding`. The shard list is the version you can assert on in a test, which is the reason to know both.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU, eight host-platform devices): the first three of eight shards',
              lang: 'python',
              text: `import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

mesh = jax.make_mesh((4, 2), ("data", "model"))
x = jax.device_put(jnp.arange(128.).reshape(8, 16), NamedSharding(mesh, P("data", "model")))

print(x.shape, len(x.addressable_shards))
for sh in x.addressable_shards[:3]:
    print(sh.device, sh.index, sh.data.shape, sh.data[0, 0].item())

# (8, 16) 8
# TFRT_CPU_0 (slice(0, 2, None), slice(0, 8, None)) (2, 8) 0.0
# TFRT_CPU_1 (slice(0, 2, None), slice(8, 16, None)) (2, 8) 8.0
# TFRT_CPU_2 (slice(2, 4, None), slice(0, 8, None)) (2, 8) 32.0`,
            },
          },
          {
            h: 'four ways a spec is wrong, in the words it uses',
            ps: [
              'Each of these messages names the quantity that failed, which makes them worth reading instead of pattern-matching. An axis that does not divide gives you the divisor and the size in one clause: dimension 0 should be divisible by 4, but it is equal to 6.',
              'A name that is not in the mesh gets caught before any data moves, and the message prints the mesh\'s axis names so the typo is visible next to what you meant. A repeated name is rejected as well, since one mesh axis cannot split two array axes at once; the devices it names would have to be in two places.',
              'The fourth is the rank rule from two sections up, seen from the failing side. A three-entry spec applied to a two-axis array reports that the sharding is only valid for values of rank at least 3. Short specs pad on the right; long ones raise.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU, eight host-platform devices): four malformed placements, messages verbatim and wrapped to fit',
              lang: 'python',
              text: `import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

mesh = jax.make_mesh((4, 2), ("data", "model"))
cube = jax.make_mesh((2, 2, 2), ("a", "b", "c"))

cases = [(mesh, P("data", "model"), (6, 16)), (mesh, P("batch", None), (8, 16)),
         (mesh, P("data", "data"), (8, 16)), (cube, P("a", "b", "c"), (8, 16))]
for m, spec, shape in cases:
    try:
        jax.device_put(jnp.ones(shape), NamedSharding(m, spec))
    except ValueError as err:
        print(err)
        print()

# One of device_put args was given the sharding of NamedSharding(mesh=Mesh('data': 4,
# 'model': 2), spec=PartitionSpec('data', 'model'), memory_kind=unpinned_host), which
# implies that the global size of its dimension 0 should be divisible by 4, but it is
# equal to 6 (full shape: (6, 16))
#
# Resource axis: batch of PartitionSpec('batch', None) is not found in mesh:
# ('data', 'model').
#
# A single NamedSharding spec specification can map every mesh axis to at most one
# positional dimension, but PartitionSpec('data', 'data') has duplicate entries for
# \`data\`
#
# One of device_put args is incompatible with its sharding annotation
# NamedSharding(mesh=Mesh('a': 2, 'b': 2, 'c': 2), spec=PartitionSpec('a', 'b', 'c'),
# memory_kind=unpinned_host): Sharding NamedSharding(mesh=Mesh('a': 2, 'b': 2,
# 'c': 2), spec=PartitionSpec('a', 'b', 'c'), memory_kind=unpinned_host) is only
# valid for values of rank at least 3, but was applied to a value of rank 2.`,
            },
            table: {
              caption: 'the four failures and the clause that identifies each; messages quoted from the run above (jax 0.4.38 CPU)',
              cols: ['what went wrong', 'the clause to look for', 'what to change'],
              rows: [
                ['the axis does not divide', 'should be divisible by 4, but it is equal to 6', 'the shape, the mesh axis length, or which axis carries it'],
                ['the name is not on the mesh', 'is not found in mesh: (\'data\', \'model\')', 'the spelling, or the axis names you built the mesh with'],
                ['one mesh axis used twice', 'has duplicate entries for `data`', 'split one array axis by two mesh axes instead: P(("data", "model"), None)'],
                ['more entries than axes', 'only valid for values of rank at least 3', 'drop entries, or reshape before placing'],
              ],
            },
          },
          {
            h: 'a spec is still a tuple here, and later it is not',
            ps: [
              'On this version a `PartitionSpec` inherits from tuple, so `len`, indexing and equality against a plain tuple all work, and a lot of code leans on that without meaning to.',
              'The changelog records two steps away from it. Release 0.6.1 stopped `PartitionSpec` inheriting from a tuple, and 0.10.0 stopped it reporting itself equal to one, with the instruction to convert tuples to specs before comparing. Anything that stores a spec as a tuple, or asserts against a tuple in a test, is the code that will break on the upgrade rather than the code that uses specs normally.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): tuple behavior that later releases remove, per the JAX changelog for 0.6.1 and 0.10.0',
              lang: 'python',
              text: `from jax.sharding import PartitionSpec as P

spec = P("data", None)
print(isinstance(spec, tuple), len(spec), spec[0], spec[1])
print(spec == ("data", None))

# True 2 data None
# True`,
            },
          },
        ],
        readings: [
          { label: 'jax.sharding', url: 'https://docs.jax.dev/en/latest/jax.sharding.html', note: 'Mesh, NamedSharding and PartitionSpec with their signatures, in one page' },
          { label: 'jax.make_mesh', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.make_mesh.html', note: 'the short constructor, and what it does about device order' },
          { label: 'JAX changelog', url: 'https://docs.jax.dev/en/latest/changelog.html', note: 'search PartitionSpec: 0.6.1 drops the tuple base, 0.10.0 drops tuple equality' },
        ],
        check: [
          {
            q: 'An (8, 16) array on a (4, 2) mesh is placed with P(("data", "model"), None). What shape does one device hold?',
            a: '(1, 16). Both mesh axes ride on array axis 0, so it splits eight ways, and axis 1 carries None, which leaves all sixteen columns on every device.',
          },
          {
            q: 'You pass P("data") for a two-axis array. Is that an error, and what does it mean?',
            a: 'Not an error. A spec shorter than the rank pads with None on the right, so it means P("data", None) and each device keeps the whole second axis. A spec longer than the rank does raise, reporting that the sharding is only valid for values of rank at least 3.',
          },
          {
            q: 'Why does P("data", "data") raise when P(("data", "model"), None) does not?',
            a: 'Because one mesh axis cannot split two array axes at once, which is what the duplicate-entries message says. Splitting one array axis by two mesh axes is the legal direction, and it is written as a tuple inside a single entry.',
          },
        ],
        work: [
          { id: 'shard-shapes-by-hand', label: 'write the shard shape for four specs of your own before you run shard_shape on any of them', href: '#a-spec-is-positional-and-it-may-be-short' },
          { id: 'read-the-index', label: 'print addressable_shards for one array of your own and say which slice each device holds before you look at the output', href: '#every-shard-is-a-slice-you-can-print' },
        ],
      },
      {
        id: 'placing-an-array',
        num: 2,
        title: 'Placing an array on purpose',
        lede: 'Two operands can disagree about their layout and still add. They cannot disagree about which devices they are on. Knowing which of those two you have is the difference between a reshard and an exception.',
        goal: 'Predict what comes back when two differently sharded arrays meet, place a whole pytree in one call, and state what a reshard costs in copies.',
        sections: [
          {
            h: 'two layouts meet, and one of them gives way',
            ps: [
              'Place the same values twice on the same mesh under opposite specs, add them, and nothing complains. The result takes the first operand\'s spec, and swapping the operands swaps the answer, so `a + b` comes back `P("data", "model")` while `b + a` comes back `P("model", "data")`.',
              'Order is not the rule, though, which is why guessing here is a bad habit. Add a replicated array to a sharded one and the sharded spec wins in both orders. Read the result\'s spec rather than working out who should have won.',
              'What the operands may not disagree about is which devices they sit on. An array spread over all eight plus an array placed on device 3 raises, and the message prints both device id lists, `[0, 1, 2, 3, 4, 5, 6, 7]` against `[3]`. A layout difference gets resolved. A device-set difference does not.',
              'The flag underneath that refusal, `committed`, and the placement rules it carries belong to the arrays chapter, which reads its docstring directly. What a mesh adds is the second thing two operands can now differ by.',
              '>> Layouts negotiate. Device sets do not.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU, eight host-platform devices): four sums that work and one that raises; the message is wrapped to fit',
              lang: 'python',
              text: `import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

mesh = jax.make_mesh((4, 2), ("data", "model"))
u = jnp.ones((8, 16))
a = jax.device_put(u, NamedSharding(mesh, P("data", "model")))
b = jax.device_put(u, NamedSharding(mesh, P("model", "data")))
r = jax.device_put(u, NamedSharding(mesh, P()))
one = jax.device_put(u, jax.devices()[3])

print((a + b).sharding.spec, (b + a).sharding.spec)
print((a + r).sharding.spec, (r + a).sharding.spec)
try:
    a + one
except ValueError as err:
    print(err)

# PartitionSpec('data', 'model') PartitionSpec('model', 'data')
# PartitionSpec('data', 'model') PartitionSpec('data', 'model')
# Received incompatible devices for jitted computation. Got argument x of add with
# shape float32[8,16] and device ids [0, 1, 2, 3, 4, 5, 6, 7] on platform CPU and
# argument y of add with shape float32[8,16] and device ids [3] on platform CPU`,
            },
          },
          {
            h: 'one call places a whole tree',
            ps: [
              '`device_put` takes pytrees, and it takes shardings in two shapes. Pass a single sharding and every leaf gets it. Pass a tree of shardings with the same structure and each leaf gets its own, which is how a parameter tree ends up with weights and biases split differently.',
              'The single-sharding form goes further than it looks because specs are positional and short specs pad. `P("data")` sends an `(8, 16)` weight and a `(16,)` bias to the same mesh axis without either leaf having to know its own rank.',
              'The shard shapes in the second call are where the two forms separate. The weight splits both ways down to `(2, 8)`, while the bias splits only across `model`, down to `(8,)`. Same call, same mesh, two different answers per leaf.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU, eight host-platform devices): one sharding for a tree, then one per leaf',
              lang: 'python',
              text: `import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

mesh = jax.make_mesh((4, 2), ("data", "model"))
tree = {"w": jnp.ones((8, 16)), "b": jnp.zeros(16)}

one = jax.device_put(tree, NamedSharding(mesh, P("data")))
print({k: (v.sharding.spec, v.addressable_shards[0].data.shape) for k, v in one.items()})

each = jax.device_put(tree, {"w": NamedSharding(mesh, P("data", "model")),
                            "b": NamedSharding(mesh, P("model"))})
print({k: (v.sharding.spec, v.addressable_shards[0].data.shape) for k, v in each.items()})

# {'b': (PartitionSpec('data',), (4,)), 'w': (PartitionSpec('data',), (2, 16))}
# {'b': (PartitionSpec('model',), (8,)), 'w': (PartitionSpec('data', 'model'), (2, 8))}`,
            },
          },
          {
            h: 'resharding is a copy, and the old array survives',
            ps: [
              'Handing `device_put` an array that is already sharded is legal, and outside `jit` it is the way data moves between layouts. Shard shapes go from `(2, 8)` to `(4, 4)`, every value stays where it belongs in the global array, and an elementwise comparison of the two arrays is True everywhere.',
              'Afterwards the original is untouched, still `(8, 16)` under its old spec, because chapter 1\'s rule about arrays being values does not stop applying when there are eight devices. Nothing moved in place. A second array exists, and the first one lives until nothing refers to it.',
              'That is the cost worth holding on to. A reshard is bytes across links plus two copies resident at once, so a reshard inside a training step is paying both, every step, for a layout decision that could have been made before the loop.',
              '>> There is no move. There is a new array, and the old one until you drop it.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU, eight host-platform devices): one array, two layouts, same values',
              lang: 'python',
              text: `import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

mesh = jax.make_mesh((4, 2), ("data", "model"))
x = jax.device_put(jnp.arange(128.).reshape(8, 16), NamedSharding(mesh, P("data", "model")))
y = jax.device_put(x, NamedSharding(mesh, P("model", "data")))

print(x.addressable_shards[0].data.shape, y.addressable_shards[0].data.shape)
print(bool(jnp.all(x == y)))
print(x.sharding.spec, y.sharding.spec)

# (2, 8) (4, 4)
# True
# PartitionSpec('data', 'model') PartitionSpec('model', 'data')`,
            },
          },
          {
            h: 'replicated is a sharding, not the absence of one',
            ps: [
              '`P()` splits no array axis, so every device holds the whole thing. `shard_shape` returns the global shape, `is_fully_replicated` is True, and there are still eight shards; they are eight identical copies rather than eight pieces.',
              'Counting those copies is the memory argument. A replicated array costs eight times what the same array costs under `P("data", "model")` on this mesh, and the next lesson measures exactly that on a pair of matrices big enough for the number to matter.',
              'On this machine `global_shards` and `addressable_shards` both return eight, because one process owns every device, and `is_fully_addressable` says so directly. They separate on multi-host runs, which chapter 12 owns; here the two are interchangeable.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU, eight host-platform devices): eight shards, each the whole array',
              lang: 'python',
              text: `import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

mesh = jax.make_mesh((4, 2), ("data", "model"))
r = jax.device_put(jnp.arange(128.).reshape(8, 16), NamedSharding(mesh, P()))

print(r.is_fully_replicated, r.addressable_shards[0].data.shape)
print(len(r.addressable_shards), len(r.global_shards), r.is_fully_addressable)

# True (8, 16)
# 8 8 True`,
            },
          },
        ],
        readings: [
          { label: 'jax.device_put', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html', note: 'the signature, including the pytree-of-shardings form' },
          { label: 'JAX FAQ', url: 'https://docs.jax.dev/en/latest/faq.html', note: 'the placement section: committed and uncommitted arrays, in the docs\' own words' },
          { label: 'Distributed arrays and automatic parallelization', url: 'https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html', note: 'the same placements as a notebook, with the pictures' },
        ],
        check: [
          {
            q: 'Two arrays on the same mesh carry opposite specs. What happens when you add them?',
            a: 'The add succeeds and the result carries one of the two specs, the first operand\'s when both are sharded. A layout difference is resolved for you; a difference in which devices the operands sit on raises instead, naming both device id lists.',
          },
          {
            q: 'device_put reshards an array to a different spec. What happens to the array you passed in?',
            a: 'Nothing. It stays valid under its old spec, because device_put returns a new array instead of moving one, and both copies are resident until the old one is dropped.',
          },
          {
            q: 'What does P() cost compared with P("data", "model") on an eight-device mesh?',
            a: 'Eight times the bytes. P() replicates, so each device holds the whole array, while P("data", "model") gives each device one eighth of it as a tile. Both are shardings; only one of them splits anything.',
          },
        ],
        work: [
          { id: 'layout-collision', label: 'print the spec of every array entering the first jitted call in a script of your own, and name the pairs that would have to be reconciled', href: '#two-layouts-meet-and-one-of-them-gives-way' },
          { id: 'reshard-hunt', label: 'find one reshard that happens inside a loop and decide whether the layout or the loop is the thing to change', href: '#resharding-is-a-copy-and-the-old-array-survives' },
        ],
      },
      {
        id: 'jit-partitions',
        num: 3,
        title: 'jit is where the partitioner runs',
        lede: 'Nothing in a jitted function body mentions a mesh, and every intermediate still gets a layout. Three lines of Python read back what was decided, and one of them prices it.',
        goal: 'Predict the output sharding of a jitted call from its input shardings, say what in_shardings and out_shardings each do to a committed argument, and get a per-device byte count for two specs without running either.',
        sections: [
          {
            h: 'the output sharding is a fact you can read',
            ps: [
              'Give a jitted function a sharded input and the output comes back sharded, with nothing in the body naming a mesh. Six calls on one `(8, 16)` input show the pattern: an elementwise op keeps both names, summing over an axis drops that axis\'s name, transposing swaps the two, and a matmul against a `P("model", None)` weight leaves the result split on `data` alone.',
              'Reading the spec back is cheap enough to do as a habit. One call plus `out.sharding.spec` answers what the whole propagation settled on, which beats reasoning it out, because the reasoning has a hole in it: `(8, 16)` under `P("data", "model")` reshaped to `(16, 8)` comes back `P("data")`, not what the two-name input suggests.',
              'How the compiler works any of this out belongs to the XLA path\'s SPMD chapter, which reads the pass that does the propagation and the rewrite. What belongs here is the observation itself. Input shardings are the only thing you stated, and every intermediate got one anyway.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU, eight host-platform devices): six ops, one sharded input, the spec that came back',
              lang: 'python',
              text: `import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

mesh = jax.make_mesh((4, 2), ("data", "model"))
x = jax.device_put(jnp.ones((8, 16)), NamedSharding(mesh, P("data", "model")))
w = jax.device_put(jnp.ones((16, 32)), NamedSharding(mesh, P("model", None)))

for label, f, args in [("a * 2", lambda a: a * 2, (x,)),
                       ("a.sum(0)", lambda a: a.sum(0), (x,)),
                       ("a.sum(1)", lambda a: a.sum(1), (x,)),
                       ("a.T", lambda a: a.T, (x,)),
                       ("a.reshape(16, 8)", lambda a: a.reshape(16, 8), (x,)),
                       ("a @ b", lambda a, b: a @ b, (x, w))]:
    print(label.ljust(17), jax.jit(f)(*args).sharding.spec)

# a * 2             PartitionSpec('data', 'model')
# a.sum(0)          PartitionSpec('model',)
# a.sum(1)          PartitionSpec('data',)
# a.T               PartitionSpec('model', 'data')
# a.reshape(16, 8)  PartitionSpec('data',)
# a @ b             PartitionSpec('data',)`,
            },
            table: {
              caption: 'the same six, with the reason; every spec measured on jax 0.4.38 CPU, input (8, 16) under P("data", "model") and w (16, 32) under P("model", None)',
              cols: ['call', 'output spec', 'what happened to the names'],
              rows: [
                ['a * 2', 'P("data", "model")', 'both kept: an elementwise op moves nothing'],
                ['a.sum(0)', 'P("model")', 'the summed axis went, and its name with it'],
                ['a.sum(1)', 'P("data")', 'the same rule on the other axis'],
                ['a.T', 'P("model", "data")', 'the names followed their axes'],
                ['a.reshape(16, 8)', 'P("data")', 'one name survived the new shape, the other did not'],
                ['a @ b', 'P("data")', 'the contracted axis carried "model" and was contracted away'],
              ],
            },
          },
          {
            h: 'in_shardings will not move a committed argument',
            ps: [
              '`out_shardings` does what its name says. Ask for `P("model", "data")` on the way out and that is what comes back, whatever propagation would have chosen for you.',
              '`in_shardings` behaves differently, and the difference is worth learning once rather than discovering under a deadline. On an uncommitted argument it places the array, so a plain `jnp.ones` gets laid across the mesh on the way in. On a committed argument whose sharding disagrees, it raises, saying the sharding passed to pjit does not match the sharding on the respective arg.',
              'The refusal is the useful half. An input reshard is real data movement, and JAX makes you write the `device_put` that performs it instead of folding it invisibly into the call that was supposed to be the compiled step.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU, eight host-platform devices): forced out, placed in, and the mismatch that raises; the message is wrapped to fit',
              lang: 'python',
              text: `import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

mesh = jax.make_mesh((4, 2), ("data", "model"))
here = NamedSharding(mesh, P("data", "model"))
there = NamedSharding(mesh, P("model", "data"))

x = jax.device_put(jnp.ones((8, 16)), here)
print(jax.jit(lambda a: a * 2, out_shardings=there)(x).sharding.spec)
print(jax.jit(lambda a: a * 2, in_shardings=here)(jnp.ones((8, 16))).sharding.spec)
try:
    jax.jit(lambda a: a * 2, in_shardings=there)(x)
except ValueError as err:
    print(err)

# PartitionSpec('model', 'data')
# PartitionSpec('data', 'model')
# Sharding passed to pjit does not match the sharding on the respective arg. Got pjit
# sharding: NamedSharding(mesh=Mesh('data': 4, 'model': 2), spec=PartitionSpec('model',
# 'data'), memory_kind=unpinned_host),
# arg sharding: NamedSharding(mesh=Mesh('data': 4, 'model': 2),
# spec=PartitionSpec('data', 'model'), memory_kind=unpinned_host) for arg shape:
# float32[8,16]`,
            },
          },
          {
            h: 'the compiled object tells you what it decided',
            ps: [
              'Lowering and compiling ahead of the run gives you the same answers without executing anything. `jax.jit(f).lower(x).compile()` hands back an object whose `input_shardings` and `output_shardings` are the shardings the executable was actually built for.',
              'This is the form of the question to put in a test. It costs a compile and no execution, it accepts `ShapeDtypeStruct` inputs with shardings attached so no array has to exist, and a wrong shape fails there rather than three steps into a training loop.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU, eight host-platform devices): the shardings the executable was built for',
              lang: 'python',
              text: `import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

mesh = jax.make_mesh((4, 2), ("data", "model"))
x = jax.device_put(jnp.ones((8, 16)), NamedSharding(mesh, P("data", "model")))

compiled = jax.jit(lambda a: a.sum(1)).lower(x).compile()
print(compiled.input_shardings[0][0].spec)
print(compiled.output_shardings.spec)

# PartitionSpec('data', 'model')
# PartitionSpec('data',)`,
            },
          },
          {
            h: 'what a spec is worth, in bytes per device',
            ps: [
              'The compiled object also carries a memory analysis, and that is where a spec stops being a preference and becomes a number. Two 512 by 512 float32 matrices multiplied on the eight-device mesh, fully replicated, hand each device 2097152 bytes of arguments and produce 1048576 bytes of output. Sharded `P("data", None)` against `P(None, "model")`, the same call hands each device 786432 bytes and produces 131072.',
              'Both numbers check out by hand. Replicated, every device holds both whole matrices, which is 2 x 512 x 512 x 4 bytes. Sharded, a device holds a (128, 512) slice of the first and a (512, 256) slice of the second, so 262144 plus 524288 is 786432, and its output tile is (128, 256) at four bytes each, which is 131072.',
              'What the analysis does not price is the traffic. Both plans compute the same product, and the sharded one buys its smaller footprint with communication. The chapter\'s reading list points at the cost model for that trade, and LAB·J4 is where you read which collective a given pair of specs produced.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU, eight host-platform devices): per-device bytes for two specs, from a compile and no execution',
              lang: 'python',
              text: `import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

mesh = jax.make_mesh((4, 2), ("data", "model"))
matmul = jax.jit(lambda a, b: a @ b)

for spec_a, spec_b in [(P(), P()), (P("data", None), P(None, "model"))]:
    A = jax.ShapeDtypeStruct((512, 512), jnp.float32, sharding=NamedSharding(mesh, spec_a))
    B = jax.ShapeDtypeStruct((512, 512), jnp.float32, sharding=NamedSharding(mesh, spec_b))
    m = matmul.lower(A, B).compile().memory_analysis()
    print(spec_a, spec_b, m.argument_size_in_bytes, m.output_size_in_bytes)

# PartitionSpec() PartitionSpec() 2097152 1048576
# PartitionSpec('data', None) PartitionSpec(None, 'model') 786432 131072`,
            },
          },
        ],
        readings: [
          { label: 'jax.jit', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.jit.html', note: 'in_shardings and out_shardings, with the rules for what may be omitted' },
          { label: 'Ahead-of-time lowering and compilation', url: 'https://docs.jax.dev/en/latest/aot.html', note: 'lower, compile, and the analyses hanging off the compiled object' },
          { label: 'jax.ShapeDtypeStruct', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.ShapeDtypeStruct.html', note: 'the input stand-in that carries a sharding, so nothing has to be allocated' },
        ],
        check: [
          {
            q: 'An (8, 16) input sharded P("data", "model") goes into jit(lambda a: a.sum(0)). What comes out?',
            a: 'A result sharded P("model"). Axis 0 was summed away, so the name that split it goes too, and the surviving axis keeps the mesh axis it already had.',
          },
          {
            q: 'Why does in_shardings raise on a committed argument instead of resharding it?',
            a: 'Because an input reshard is real data movement, and JAX will not do it silently inside the compiled call. Either device_put the array first, or hand the function an uncommitted array, which in_shardings will place.',
          },
          {
            q: 'Where does a per-device byte count for a given pair of specs come from?',
            a: 'From memory_analysis() on the object jit(f).lower(...).compile() returns. It reports argument, output and temp sizes per device, so two candidate specs can be compared before either one executes.',
          },
        ],
        work: [
          { id: 'propagation-table', label: 'predict the output spec for five ops of your own on one sharded input, then check each with out.sharding.spec', href: '#the-output-sharding-is-a-fact-you-can-read' },
          { id: 'bytes-before-you-run', label: 'compare two input specs for one step function by memory_analysis alone, without executing either', href: '#what-a-spec-is-worth-in-bytes-per-device' },
        ],
      },
      {
        id: 'pinning-a-layout',
        num: 4,
        title: 'Pinning a layout, and where this course stops',
        lede: 'with_sharding_constraint is not a check and not a hint. It is an instruction that survives into the module XLA compiles, and you can read it there in six words of annotation.',
        goal: 'Place a constraint and predict its effect on the intermediate and on the result, read an mhlo.sharding string back out of a lowered module, and name which sharding questions this course answers and which belong elsewhere.',
        sections: [
          {
            h: 'the constraint is an instruction in the program',
            ps: [
              'Put `with_sharding_constraint` in the middle of a function and the jaxpr grows an equation for it. `sharding_constraint` is a primitive the same way `mul` and `reduce_sum` are, and it carries the whole `NamedSharding` in its parameters, plus a set of unconstrained dimensions that section three fills in.',
              'What it constrains is a value, not a variable and not a function. It says the value bound at this point is laid out this way, and everything downstream propagates from there.',
              'Which is why the second run below matters more than it looks. Pinning `a * 2` to `P("data", None)` changes how that intermediate is stored, and the sum after it still comes back `P("data")`, exactly as it does without the constraint. A constraint controls the value you named. It does not dictate the result.',
            ],
            code: {
              caption: 'the constrained function traced, verbatim (verified, jax 0.4.38 CPU, eight host-platform devices); the fold holds the program that printed it and the two output specs',
              lang: 'haskell',
              text: `{ lambda ; a:f32[8,16]. let
    b:f32[8,16] = mul a 2.0
    c:f32[8,16] = sharding_constraint[
      layout=None
      resource_env=ResourceEnv(mesh=Mesh())
      sharding=NamedSharding(mesh=Mesh('data': 4, 'model': 2), spec=PartitionSpec('data', None), memory_kind=unpinned_host)
      unconstrained_dims=set()
    ] b
    d:f32[8] = reduce_sum[axes=(1,)] c
  in (d,) }`,
              full: {
                label: 'the two programs that produced it, and the specs they return',
                text: `import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
from jax.lax import with_sharding_constraint
from jax.sharding import NamedSharding, PartitionSpec as P

mesh = jax.make_mesh((4, 2), ("data", "model"))
x = jax.device_put(jnp.ones((8, 16)), NamedSharding(mesh, P("data", "model")))
rows = NamedSharding(mesh, P("data", None))

def f(a):
    b = a * 2
    b = with_sharding_constraint(b, rows)
    return b.sum(1)

print(jax.make_jaxpr(f)(x))

plain = jax.jit(lambda a: (a * 2).sum(1))
pinned = jax.jit(f)
print(plain(x).sharding.spec, pinned(x).sharding.spec)

# { lambda ; a:f32[8,16]. let
#     b:f32[8,16] = mul a 2.0
#     c:f32[8,16] = sharding_constraint[
#       layout=None
#       resource_env=ResourceEnv(mesh=Mesh())
#       sharding=NamedSharding(mesh=Mesh('data': 4, 'model': 2), spec=PartitionSpec('data', None), memory_kind=unpinned_host)
#       unconstrained_dims=set()
#     ] b
#     d:f32[8] = reduce_sum[axes=(1,)] c
#   in (d,) }
# PartitionSpec('data',) PartitionSpec('data',)`,
              },
            },
          },
          {
            h: 'how the annotation reaches the compiler',
            ps: [
              'Lower the same function and the constraint is still there, as `stablehlo.custom_call @Sharding` sitting between the multiply and the reduce, with an `mhlo.sharding` string attached. The function\'s argument carries a string of its own in the signature, in the same notation.',
              'Read that notation once and it stops being noise. `{devices=[4,2]<=[8]}` says the array is tiled four by two over the eight devices taken in order. `<=[4,2]T(1,0)` says take that same four by two grid transposed, which is what naming the axes the other way round produces. And `last_tile_dim_replicate` marks a trailing tile dimension that is not an array axis at all, so `{devices=[4,1,2]<=[8] last_tile_dim_replicate}` reads as four tiles down axis 0, one across axis 1, each tile held by two devices.',
              'The table below is the whole vocabulary for a two-axis array on this mesh, measured one spec at a time. The printer that produces these strings lives in XLA\'s `hlo_sharding.cc`, which settles any case a table this size does not cover.',
            ],
            code: {
              caption: 'the constrained function lowered, verbatim (verified, jax 0.4.38 CPU, eight host-platform devices)',
              lang: 'mlir',
              text: `module @jit_f attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8x16xf32> {mhlo.sharding = "{devices=[4,2]<=[8]}"}) -> (tensor<8xf32> {jax.result_info = ""}) {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x16xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<8x16xf32>
    %2 = stablehlo.custom_call @Sharding(%1) {backend_config = "", mhlo.sharding = "{devices=[4,1,2]<=[8] last_tile_dim_replicate}"} : (tensor<8x16xf32>) -> tensor<8x16xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%2 init: %cst_0) applies stablehlo.add across dimensions = [1] : (tensor<8x16xf32>, tensor<f32>) -> tensor<8xf32>
    return %3 : tensor<8xf32>
  }
}`,
            },
            table: {
              cols: ['spec', 'mhlo.sharding on the argument', 'in words'],
              caption: 'six specs placed on an (8, 16) argument, annotation strings read out of jax.jit(...).lower(x).as_text() (verified, jax 0.4.38 CPU, eight host-platform devices)',
              rows: [
                ['P("data", "model")', '{devices=[4,2]<=[8]}', 'four by two tiles, devices in order'],
                ['P("data", None)', '{devices=[4,1,2]<=[8] last_tile_dim_replicate}', 'four row blocks, each on two devices'],
                ['P(None, "model")', '{devices=[1,2,4]<=[4,2]T(1,0) last_tile_dim_replicate}', 'two column blocks, each on four devices, grid transposed'],
                ['P()', '{replicated}', 'no tiling at all'],
                ['P(("data", "model"), None)', '{devices=[8,1]<=[8]}', 'eight row blocks, one column block'],
                ['P("model", "data")', '{devices=[2,4]<=[4,2]T(1,0)}', 'two by four tiles, grid transposed'],
              ],
            },
          },
          {
            h: 'leave one axis unconstrained on purpose',
            ps: [
              '`P.UNCONSTRAINED` in an entry says you have an opinion about the other axes and none about this one. The annotation records it as `backend_config = "unspecified_dims=[1]"`, and propagation fills the gap, so constraining axis 0 to `data` and leaving axis 1 open returns a result still split `P("data", "model")`.',
              'Set that against pinning both axes to nothing. `P()` in the middle of a function forces the value to be replicated at that point, the annotation becomes `{replicated}`, and the output comes back `P()`. One of the two leaves a decision to the compiler; the other takes it away.',
              'The distinction is easy to lose because `None` and `UNCONSTRAINED` sit in the same slot. `None` is a request: keep this axis whole on every device. `UNCONSTRAINED` is the absence of a request, and an axis you left alone can come back split.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU, eight host-platform devices): one axis left open, then both axes pinned to replicated; the two custom_call lines are wrapped to fit',
              lang: 'python',
              text: `import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
from jax.lax import with_sharding_constraint
from jax.sharding import NamedSharding, PartitionSpec as P

mesh = jax.make_mesh((4, 2), ("data", "model"))
x = jax.device_put(jnp.ones((8, 16)), NamedSharding(mesh, P("data", "model")))

def pinned(a):
    return with_sharding_constraint(a * 2, NamedSharding(mesh, P("data", P.UNCONSTRAINED)))

def gathered(a):
    return with_sharding_constraint(a * 2, NamedSharding(mesh, P()))

for f in (pinned, gathered):
    line = [l for l in jax.jit(f).lower(x).as_text().splitlines() if "@Sharding" in l][0]
    print(line.strip())
    print("   out:", jax.jit(f)(x).sharding.spec)

# %2 = stablehlo.custom_call @Sharding(%1) {backend_config = "unspecified_dims=[1]",
#   mhlo.sharding = "{devices=[4,1,2]<=[8] last_tile_dim_replicate}"} :
#   (tensor<8x16xf32>) -> tensor<8x16xf32>
#    out: PartitionSpec('data', 'model')
# %2 = stablehlo.custom_call @Sharding(%1) {backend_config = "",
#   mhlo.sharding = "{replicated}"} : (tensor<8x16xf32>) -> tensor<8x16xf32>
#    out: PartitionSpec()`,
            },
          },
          {
            h: 'a request the compiler is allowed to decline',
            ps: [
              'Ask for something impossible and nothing raises. Slicing six rows out of the array and constraining that to `P("data", None)` asks for six rows split four ways, which has no answer, and the call still returns.',
              'The annotation goes into the module exactly as written, `{devices=[4,1,2]<=[8] last_tile_dim_replicate}` on a `tensor<6x16xf32>`. The executable that comes back holds the result replicated, `P()`, with a `(6, 16)` shard on every device. The constraint was carried to the compiler, and the compiler declined it.',
              'So a constraint is not an assertion, and treating it as one is how a program ends up quietly replicating a value it meant to split. The check that catches this is the one from lesson three: read the output sharding back, or read the compiled executable\'s, and compare it with what you asked for.',
              '>> A constraint states an intent. Only the compiled program states an outcome.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU, eight host-platform devices): an unsatisfiable constraint, annotated and then ignored; the custom_call line is wrapped to fit',
              lang: 'python',
              text: `import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
from jax.lax import with_sharding_constraint
from jax.sharding import NamedSharding, PartitionSpec as P

mesh = jax.make_mesh((4, 2), ("data", "model"))
x = jax.device_put(jnp.ones((8, 16)), NamedSharding(mesh, P("data", "model")))

# six rows cannot be split four ways
f = jax.jit(lambda a: with_sharding_constraint(a[:6], NamedSharding(mesh, P("data", None))))
line = [l for l in f.lower(x).as_text().splitlines() if "@Sharding" in l][0]
print(line.strip())
print(f(x).sharding.spec, f(x).addressable_shards[0].data.shape)

# %1 = stablehlo.custom_call @Sharding(%0) {backend_config = "",
#   mhlo.sharding = "{devices=[4,1,2]<=[8] last_tile_dim_replicate}"} :
#   (tensor<6x16xf32>) -> tensor<6x16xf32>
# PartitionSpec() (6, 16)`,
            },
          },
          {
            h: 'outside jit it moves bytes instead',
            ps: [
              'Calling `with_sharding_constraint` on a real array, outside any trace, does not raise on this version. The array comes back with the spec you asked for, committed, with the shard shape that spec implies, which is `device_put` in different clothing.',
              'That convenience hides a real difference. Inside a trace there is a value the compiler has not laid out yet, and the call annotates it. Outside, there is nothing left to annotate, so the call moves data, with everything lesson two said about what a reshard costs.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU, eight host-platform devices): the same call, no trace in sight',
              lang: 'python',
              text: `import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
from jax.lax import with_sharding_constraint
from jax.sharding import NamedSharding, PartitionSpec as P

mesh = jax.make_mesh((4, 2), ("data", "model"))
x = jax.device_put(jnp.ones((8, 16)), NamedSharding(mesh, P("data", "model")))

y = with_sharding_constraint(x, NamedSharding(mesh, P(None, "model")))
print(y.sharding.spec, y.committed, y.addressable_shards[0].data.shape)

# PartitionSpec(None, 'model') True (8, 8)`,
            },
          },
          {
            h: 'where this course stops',
            ps: [
              'Four questions about sharding have separate homes, and two of them are here. What a mesh and a spec mean, and what `device_put` does with them, is lesson one and lesson two. What `jit` propagates and what a spec costs per device is lesson three. Chapter 10 adds the level below both, `shard_map`, where the function runs once per device and you name the collectives yourself.',
              'How the propagation and the rewrite actually happen is the XLA path\'s SPMD chapter, which reads the pass that does it. What a collective is as an agreement between devices, and how that agreement deadlocks, is its collectives chapter. The kernel path\'s distributed stage builds one out of raw remote DMAs, which is the same operation three layers down.',
              'Which spec to choose is a cost question, and the scaling book on the chapter\'s reading list carries that arithmetic. LAB·J4 is the runnable boundary between the two: eight devices on one laptop, and the question of which collective a given pair of specs produced, which is the first question this vocabulary makes it possible to ask.',
            ],
          },
        ],
        readings: [
          { label: 'jax.lax.with_sharding_constraint', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.lax.with_sharding_constraint.html', note: 'the signature, and the note that it is meaningful inside jit' },
          { label: 'hlo_sharding.cc in openxla/xla', url: 'https://github.com/openxla/xla/blob/main/xla/hlo/ir/hlo_sharding.cc', note: 'the printer for every annotation string in this lesson; read Print for the grammar' },
          { label: 'HLO operation semantics', url: 'https://openxla.org/xla/operation_semantics', note: 'where the collectives named by the other two courses are specified exactly' },
        ],
        check: [
          {
            q: 'What does with_sharding_constraint become in the lowered module?',
            a: 'A stablehlo.custom_call @Sharding wrapping the value, carrying the requested layout in an mhlo.sharding string. It is an instruction inside the program, and the function arguments carry the same notation in the signature.',
          },
          {
            q: 'How do you read {devices=[4,1,2]<=[8] last_tile_dim_replicate}?',
            a: 'Four tiles down array axis 0, one across axis 1, over the eight devices in order, and the trailing 2 is not an array axis: it says each tile is held by two devices. That is what P("data", None) compiles to on a (4, 2) mesh.',
          },
          {
            q: 'Your constraint asked for a split the shapes cannot support. What happens?',
            a: 'Nothing raises. The annotation is emitted into the module as written, and the partitioner declines it, so the value comes back replicated instead. Reading the output sharding back is the only way to notice.',
          },
        ],
        work: [
          { id: 'read-the-annotation', label: 'lower one function of your own and translate every mhlo.sharding string in it back into the PartitionSpec that produced it', href: '#how-the-annotation-reaches-the-compiler' },
          { id: 'constraint-audit', label: 'take one constraint you have written and check the compiled output sharding against what the constraint asked for', href: '#a-request-the-compiler-is-allowed-to-decline' },
        ],
      },
    ],
  },
]
