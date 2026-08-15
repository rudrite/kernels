// New file: site/src/data/lessons/jax-random.ts
// What a key holds and what the three key operations hash, below the survey
// chapter 8 teaches: the two words, the counter each of bits, split and
// fold_in feeds to threefry, the symptoms of reuse, and the two disciplines
// for threading keys through vmap and scan. Every printed value ran on this
// machine (jax 0.4.38 CPU, jaxlib 0.4.38, Python 3.12, 2026-08-15). Every
// source excerpt is verbatim from jax/_src/prng.py in the installed wheel,
// which is byte-identical to the jax-v0.4.38 tag, so the quoted line numbers
// hold against the public tree. jax 0.5.0 turned jax_threefry_partitionable
// on by default; every caption whose numbers move under that flag says so.
import type { UnitLessons } from './index'

export const JAX_RANDOM_LESSONS: UnitLessons[] = [
  {
    unit: 'jax:random',
    lessons: [
      {
        id: 'two-words-and-a-hash',
        num: 1,
        title: 'Two words and a hash',
        lede: 'A key prints its own contents: two unsigned 32-bit words and a type tag. Every sampler, every split and every fold_in is one hash function applied to those two words and a counter you can work out in advance.',
        goal: 'Say what a typed key holds and what its dtype forbids, name the count array that bits, split and fold_in each feed to the hash, and predict which of two draws from one key share values.',
        sections: [
          {
            h: 'what jax.random.key(0) actually holds',
            ps: [
              "Ask a key to print itself and it shows you both halves of what it is. The repr names an array of shape () with dtype key<fry>, and underneath it prints two unsigned 32-bit words: 0 and 0. A seed of zero becomes the pair (0, 0) literally, and `jax.random.key_data` hands those words back as a plain uint32 array any time you want to look.",
              "The older constructor returns the same two words with the type peeled off. `jax.random.PRNGKey(0)` is uint32 of shape (2,), and it equals key_data of the typed key entry for entry. Nothing about the randomness differs between the two calls. What differs is whether the array carries its own type.",
              "That type is a field on the value, not a convention you hold in your head. `jax.random.key_impl` reports threefry2x32 for this key, so the key carries its own generator. A uint32 key carries no such tag, and the module docs are direct about the consequence: legacy keys do not carry information about the RNG implementation, so a global configuration setting decides which algorithm runs.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): one key, printed four ways',
              lang: 'python',
              text: 'import jax\n\nk = jax.random.key(0)\nprint(repr(k))\nprint(k.dtype, k.shape, jax.random.key_impl(k))\nprint(jax.random.key_data(k))\nprint(jax.random.PRNGKey(0), jax.random.PRNGKey(0).dtype, jax.random.PRNGKey(0).shape)\n\n# Array((), dtype=key<fry>) overlaying:\n# [0 0]\n# key<fry> () threefry2x32\n# [0 0]\n# [0 0] uint32 (2,)',
            },
          },
          {
            h: 'the type is a guardrail',
            ps: [
              "Add one to a typed key and JAX refuses, with `add does not accept dtypes key<fry>, int32.` There is no arithmetic on a key, because arithmetic on a key is never what anyone meant. The legal moves are all key moves: split it, fold an integer into it, hand it to a sampler, or read its raw words out.",
              "Do the same thing to a legacy key and it goes through. `jax.random.PRNGKey(0) + 1` returns [1 1], an ordinary uint32 array that every sampling function will accept as a key. The result is a seed nobody chose, produced by a line that looks like a typo and runs like an instruction.",
              "Getting back to plain integers is still one call, for the times a serializer needs them. key_data pulls the two words out, wrap_key_data puts them back, and the round trip returns the pair unchanged.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): what the two key types allow',
              lang: 'python',
              text: 'import jax\n\nk = jax.random.key(0)\ntry:\n    k + 1\nexcept TypeError as err:\n    print(err)\nprint(jax.random.PRNGKey(0) + 1)\nprint(jax.random.key_data(jax.random.wrap_key_data(jax.random.key_data(k))))\n\n# add does not accept dtypes key<fry>, int32.\n# [1 1]\n# [0 0]',
            },
            table: {
              caption: 'the two constructors, every row measured on this machine (verified, jax 0.4.38 CPU)',
              cols: ['question', 'jax.random.key(0)', 'jax.random.PRNGKey(0)'],
              rows: [
                ['dtype', 'key<fry>', 'uint32'],
                ['shape', '()', '(2,)'],
                ['the raw words', '[0 0], through key_data', '[0 0], the array itself'],
                ['which generator', 'threefry2x32, recorded on the value', 'not recorded; read from jax_default_prng_impl'],
                ['key + 1', 'TypeError: add does not accept dtypes key<fry>, int32.', '[1 1], accepted by every sampler'],
                ['visible to the reuse checker', 'yes', 'no, which lesson two measures'],
              ],
            },
          },
          {
            h: 'a draw is a hash of the key and a counter',
            ps: [
              "Trace a call to `jax.random.bits` and the entire program is one equation. The jaxpr holds a single primitive, random_bits, and the requested shape sits inside the brackets as an attribute rather than flowing in as a value. The shape is not packaging around the draw. It is an input to it.",
              "Under that primitive, threefry2x32 hashes the key against a count array, and for a 32-bit request that count array is an iota: 0, 1, 2, and so on, one entry per word. Word i of the output depends on the key and on i, and on nothing else in the array. That independence is why a draw parallelizes and why no element waits on its neighbour.",
              "Two requests with the same element count therefore line up exactly. A (2, 2) normal draw reshaped to (4,) is elementwise equal to the (4,) draw, because the reshape happens after the same iota was hashed. Requests with different element counts are a different story, and the next section is the mechanism behind it.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the whole jaxpr of a draw, then two shapes that agree. jax 0.5.0 defaults jax_threefry_partitionable on, which changes these numbers and not the jaxpr',
              lang: 'python',
              text: 'import jax\n\nk = jax.random.key(0)\nprint(jax.make_jaxpr(lambda key: jax.random.bits(key, (4,)))(k))\nprint(jax.random.bits(k, (4,)))\nprint(jax.random.normal(k, (2, 2)).reshape(4))\nprint(jax.random.normal(k, (4,)))\n\n# { lambda ; a:key<fry>[]. let\n#     b:u32[4] = random_bits[bit_width=32 shape=(4,)] a\n#   in (b,) }\n# [4146024105  967050713 2718843009 1272950319]\n# [ 1.8160863  -0.75488514  0.33988908 -0.53483534]\n# [ 1.8160863  -0.75488514  0.33988908 -0.53483534]',
            },
          },
          {
            h: 'the count array gets halved before it is hashed',
            ps: [
              "Threefry hashes two words at a time, so the count array is split down the middle and the two halves go in as a pair. An odd-sized request gets a zero appended first and the extra output dropped at the end. Four lines of `threefry_2x32` do all of it.",
              "Run the halving by hand for a four-word request and a three-word one. The four-word counts split into [0, 1] and [2, 3], so position 0 is hashed against 2 and position 1 against 3. The three-word counts pad to [0, 1] and [2, 0], so position 0 is still hashed against 2 while position 1 now meets a zero. That is exactly the pattern the run shows: entries 0 and 2 match across the two requests, entry 1 does not.",
              "Which means the overlap you can see is an artifact of one implementation choice, not a rule. Under the partitionable flag that jax 0.5.0 turns on by default, the counter is built per element instead of per half, and a three-word request becomes a strict prefix of a four-word one. Do not build anything on either behaviour; build on the shape being part of the draw.",
              '>> The same key with a different shape is a different draw.',
            ],
            code: {
              caption: 'verbatim, jax/_src/prng.py:1082-1090 at the jax-v0.4.38 tag, the tail of threefry_2x32, then a session on this machine (jax 0.4.38 CPU). Under jax 0.5.0 defaults the shorter request is a prefix of the longer one instead',
              lang: 'python',
              text: '  if odd_size:\n    x = list(jnp.split(jnp.concatenate([count.ravel(), np.uint32([0])]), 2))\n  else:\n    x = list(jnp.split(count.ravel(), 2))\n\n  x = threefry2x32_p.bind(key1, key2, x[0], x[1])\n  out = jnp.concatenate(x)\n  assert out.dtype == np.uint32\n  return lax.reshape(out[:-1] if odd_size else out, count.shape)\n\n# >>> jax.random.bits(k, (4,))\n# Array([4146024105,  967050713, 2718843009, 1272950319], dtype=uint32)\n# >>> jax.random.bits(k, (3,))\n# Array([4146024105, 1351547692, 2718843009], dtype=uint32)\n# >>> jax.random.normal(k, (4,))\n# Array([ 1.8160863 , -0.75488514,  0.33988908, -0.53483534], dtype=float32)\n# >>> jax.random.normal(k, (3,))\n# Array([ 1.8160863 , -0.48262316,  0.33988908], dtype=float32)',
            },
          },
          {
            h: 'split is those same bits, retyped',
            ps: [
              "Three lines implement splitting, and none of them is new machinery. `_threefry_split_original` takes an iota twice as long as the number of children you asked for, hashes it with the same threefry call every sampler uses, and reshapes the result into pairs. A child key is two random words wearing the key type.",
              "The session proves it end to end. The eight words behind `split(k, 4)` are elementwise equal to `bits(k, (8,))`, taken from the same key. Splitting is not a separate source of randomness sitting next to sampling. It is sampling, with the output labelled as keys.",
              "The count you pass therefore changes every child, not just the extra ones. `split(k, 2)` gives [4146024105, 967050713] and [2718843009, 1272950319], which are the first four words of the eight above only because 2 happens to divide 4 evenly here; ask for a different width and the halving from the previous section re-pairs everything. Treat the number of children as part of the draw, the same way you treat a sampler's shape.",
            ],
            code: {
              caption: 'verbatim, jax/_src/prng.py:1104-1108 at the jax-v0.4.38 tag, then a session on this machine (jax 0.4.38 CPU). jax 0.5.0 defaults to the partitionable split below it in the file, whose words differ',
              lang: 'python',
              text: '@partial(jit, static_argnums=(1,), inline=True)\ndef _threefry_split_original(key, shape) -> typing.Array:\n  num = math.prod(shape)\n  counts = lax.iota(np.uint32, num * 2)\n  return lax.reshape(threefry_2x32(key, counts), (*shape, 2))\n\n# >>> jax.random.key_data(jax.random.split(k, 4)).reshape(-1)\n# Array([2285895361, 1501764800, 1518642379, 4090693311,  433833334,\n#        4221794875,  839183663, 3740430601], dtype=uint32)\n# >>> jax.random.bits(k, (8,))\n# Array([2285895361, 1501764800, 1518642379, 4090693311,  433833334,\n#        4221794875,  839183663, 3740430601], dtype=uint32)\n# >>> jax.random.key_data(jax.random.split(k, 2))\n# Array([[4146024105,  967050713],\n#        [2718843009, 1272950319]], dtype=uint32)',
            },
          },
          {
            h: 'fold_in hashes the index instead',
            ps: [
              "Folding an integer in is the same hash against a different counter. `_threefry_fold_in` calls threefry_2x32 with `threefry_seed(data)`, and threefry_seed of a small integer is the pair (0, i), which is exactly what `PRNGKey(i)` prints. So `fold_in(k, 3)` hashes the key against the counts (0, 3), while `split(k, n)` hashes it against an iota. Different counters, one hash, no shared state between them.",
              "The counter also explains a property worth banking. fold_in's counts depend on i alone, never on how many siblings exist or on which of them ran first. Ask for the key of item 3 and you get the same two words whether you asked for items 0 through 7 or for item 3 by itself.",
              "One measurement makes the difference concrete. Turning jax_threefry_partitionable on, the flag jax 0.5.0 ships enabled, changes the words that split returns; the words that fold_in returns are identical to the run below, for every index 0 through 7 on this machine. Draws taken from those keys still move, because random_bits changed too. The key derivation is what stayed put.",
            ],
            code: {
              caption: 'verbatim, jax/_src/prng.py:1118-1124 at the jax-v0.4.38 tag, then a session on this machine (jax 0.4.38 CPU); these two words are unchanged under jax_threefry_partitionable',
              lang: 'python',
              text: 'def threefry_fold_in(key: typing.Array, data: typing.Array) -> typing.Array:\n  assert not data.shape\n  return _threefry_fold_in(key, jnp.uint32(data))\n\n@jit\ndef _threefry_fold_in(key, data):\n  return threefry_2x32(key, threefry_seed(data))\n\n# >>> jax.random.PRNGKey(3)\n# Array([0, 3], dtype=uint32)\n# >>> jax.random.key_data(jax.random.fold_in(k, 3))\n# Array([2467461003, 3840466878], dtype=uint32)',
            },
          },
        ],
        readings: [
          { label: 'the jax.random module reference', url: 'https://docs.jax.dev/en/latest/jax.random.html', note: 'the key-type note and the implementation table, both quoted in this arc' },
          { label: 'JEP 9263: typed key arrays', url: 'https://docs.jax.dev/en/latest/jep/9263-typed-keys.html', note: 'why the dtype exists and what it was allowed to forbid' },
          { label: 'prng.py at the jax-v0.4.38 tag', url: 'https://github.com/jax-ml/jax/blob/jax-v0.4.38/jax/_src/prng.py', note: 'threefry_2x32 at 1060, the two split paths at 1104, fold_in at 1118' },
        ],
        check: [
          {
            q: 'A key prints as dtype key<fry> over the words [0 0]. What does that dtype buy over the uint32 pair PRNGKey returns?',
            a: 'It records the generator on the value, so the sampler does not consult jax_default_prng_impl, and it refuses arithmetic. PRNGKey(0) + 1 returns [1 1] and every sampler accepts it as a key; k + 1 raises add does not accept dtypes key<fry>, int32.',
          },
          {
            q: 'Why do jax.random.normal(k, (3,)) and jax.random.normal(k, (4,)) disagree at the second entry and agree at the first and third?',
            a: 'Because the count array is halved before hashing. Four counts pair as (0,2) and (1,3); three counts pad and pair as (0,2) and (1,0), so only the position facing a changed partner moves. Under jax 0.5.0 defaults the counter is per element and the shorter draw is a prefix instead.',
          },
          {
            q: 'split(k, 2) and the first two children of split(k, 4) hold different words. Where does that come from?',
            a: 'split hashes an iota of length 2n, so the count array itself depends on n. The eight words behind split(k, 4) are elementwise equal to bits(k, (8,)), and asking for a different number of children hashes a different set of counts.',
          },
        ],
        work: [
          { id: 'predict-the-counter', label: 'take three sampling calls from your own code and write down the count array each one will hash before you run anything', href: '#a-draw-is-a-hash-of-the-key-and-a-counter' },
          { id: 'words-by-hand', label: 'print key_data for one key, its four children, and fold_in of 0 through 3, then say which of those words you could have predicted from the source', href: '#split-is-those-same-bits-retyped' },
        ],
      },
      {
        id: 'reuse-is-silent',
        num: 2,
        title: 'Reuse is silent',
        lede: 'The mistakes museum collects programs that stop, each exhibit pairing a snippet with the error text it printed. Reusing a key stops nothing. The run finishes, the loss goes down, and two things that were supposed to be independent are the same array.',
        goal: 'Recognize key reuse from its symptoms rather than from an error, measure what it costs an estimator, and turn the silent failure into a raised exception before it reaches anyone else.',
        sections: [
          {
            h: 'the failure that prints nothing',
            ps: [
              "Initialize two weight matrices from one key and they come back identical. Not similar, not correlated: the same nine numbers, twice, because a key plus a shape determines the bits and nothing else about the two calls differed.",
              "No check fires. The shapes are right, the dtypes are right, the values are plausible draws from a normal, and a model built on them will train. It started from a smaller family of initializations than the code claims, and no output says so.",
              "Which is why this one has no exhibit next door. The museum's cases, `inplace-assign` and `tracer-bool` among them, all end in a message you can paste into a search box. There is no message to quote here, so the rest of this lesson is about the symptoms you can measure instead.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): two layers, one key. jax 0.5.0 defaults change the numbers and not the equality',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\nkey = jax.random.key(0)\nw1 = jax.random.normal(key, (3, 3))\nw2 = jax.random.normal(key, (3, 3))\nprint(bool(jnp.all(w1 == w2)))\nprint(w1)\n\n# True\n# [[-0.3721109   0.26423115 -0.18252768]\n#  [-0.7368197   0.44973662 -0.1521442 ]\n#  [-0.67135346 -0.5908641   0.73168886]]',
            },
          },
          {
            h: 'not correlated, identical',
            ps: [
              "Correlated is the usual word for what reuse does, and it undersells the damage. Draw ten thousand normals from a key, draw ten thousand more from the same key, and the correlation comes back 1.0 to fourteen decimal places while the largest elementwise difference is exactly 0.0. The second draw is not a correlated sample of the first. It is the first.",
              "Split the key and the same measurement lands where independence lands. The two children give -0.0055 over ten thousand samples, and a parent's draw measured against one of its children gives -0.0033. Both are the size of sampling noise at n = 10000, and neither is evidence of anything except that the keys were different.",
              "So the test for reuse is not statistical. It is equality, and it is cheap: assert that no two draws you meant to be independent are elementwise equal.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU, numpy 2.2.6): correlation on 10000 samples, reused against split',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\nimport numpy as np\n\nkey = jax.random.key(0)\nk1, k2 = jax.random.split(key)\nx = jax.random.normal(key, (10000,))\ny = jax.random.normal(key, (10000,))\nu = jax.random.normal(k1, (10000,))\nv = jax.random.normal(k2, (10000,))\ncorr = lambda a, b: float(np.corrcoef(np.asarray(a), np.asarray(b))[0, 1])\nprint(corr(x, y), float(jnp.max(jnp.abs(x - y))))\nprint(corr(u, v))\nprint(corr(x, u))\n\n# 0.9999999999999998 0.0\n# -0.0054627058855592735\n# -0.0033143140158316963',
            },
          },
          {
            h: 'the estimator stops converging',
            ps: [
              "Averaging is where reuse turns into a number somebody eventually notices. Eight independent draws of 2048 normals, averaged elementwise, give a mean array with standard deviation 0.352, which is 1 over the square root of 8, or 0.3536, as it should be. Eight draws from one key, averaged the same way, give 0.978.",
              "Nothing got slower and nothing warned. The estimator simply stopped converging at the rate the code was written to assume, which is the shape this bug takes in a Monte Carlo estimate, a dropout ensemble, an augmentation pipeline, or a bootstrap.",
              "If you only remember one diagnostic from this lesson, remember this one. Averaging N samples should shrink the spread by a factor near the square root of N. When it does not, look at the keys before you look at the model.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the standard deviation of a mean over eight draws, split against reused',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\nkey = jax.random.key(0)\nkeys = jax.random.split(key, 8)\nindependent = jax.vmap(lambda k: jax.random.normal(k, (2048,)))(keys).mean(0)\nreused = jnp.stack([jax.random.normal(key, (2048,)) for _ in range(8)]).mean(0)\nprint(float(independent.std()), float(reused.std()), float(1 / jnp.sqrt(8.0)))\n\n# 0.3524690568447113 0.9781718850135803 0.3535533845424652',
            },
          },
          {
            h: 'the loop that forgets to advance',
            ps: [
              "The idiom in the jax.random docstring is one line long and every character of it carries weight: `key, subkey = jax.random.split(key)`. The left-hand `key` is a rebinding, and dropping it turns the loop into three copies of one step.",
              "The stale loop below draws the integers 8, 45, 78, 7 three times over. It calls split on every iteration, so the code looks like it is doing the right thing; what it never does is move the name forward, so the same parent produces the same child each time round.",
              "A key is a value like any other, which means Python will not stop you from reading a stale one. The discipline is a naming discipline: once a key has been split, the parent name should point at the new key, and a key you have already sampled from should never appear again.",
            ],
            code: {
              caption: 'verbatim, the basic-usage block of the jax.random module docstring as printed by the local jax 0.4.38 install (its trailing doctest directive trimmed), then a run on this machine (jax 0.4.38 CPU)',
              lang: 'python',
              text: '>>> seed = 1701\n>>> num_steps = 100\n>>> key = jax.random.key(seed)\n>>> for i in range(num_steps):\n...   key, subkey = jax.random.split(key)\n...   params = compiled_update(subkey, params, next(batches))\n\nimport jax\nimport jax.numpy as jnp\n\nkey = jax.random.key(0)\n\nk = key\nstale = []\nfor _ in range(3):\n    stale.append(jax.random.randint(jax.random.split(k)[1], (4,), 0, 100))\n\nk = key\nfresh = []\nfor _ in range(3):\n    k, sub = jax.random.split(k)\n    fresh.append(jax.random.randint(sub, (4,), 0, 100))\n\nprint(jnp.stack(stale))\nprint(jnp.stack(fresh))\n\n# [[ 8 45 78  7]\n#  [ 8 45 78  7]\n#  [ 8 45 78  7]]\n# [[ 8 45 78  7]\n#  [81 73 74 37]\n#  [43 35 56 43]]',
            },
          },
          {
            h: 'make it raise',
            ps: [
              "JAX ships a checker for exactly this failure, switched off by default. Wrap the code in `jax.debug_key_reuse(True)` and a consumed key becomes a KeyReuseError, with two messages depending on where the reuse happened: `In pjit, argument 0 is already consumed.` when both draws are inside one jitted function, and `Previously-consumed key passed to jit-compiled function at index 0` when a key crosses the boundary twice.",
              "Repeating a draw on purpose is still allowed, as long as you say so. `jax.random.clone` returns a key the checker treats as fresh, and the values that come out match the original exactly, which is usually the reason you wanted it: replaying the same noise for a comparison.",
              "The checker has one blind spot, and it is the sharpest argument in the arc for typed keys. It tracks key<fry> values, so a program built on `jax.random.PRNGKey` sails through: two draws from the same uint32 key under the checker return identical arrays and raise nothing. The module's own documentation calls the checker experimental and says that in the future we will likely enable it by default, which is a good reason to run it now and find out what it says about your code.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): both KeyReuseError messages, the deliberate replay, and the legacy-key blind spot',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\ndef init(key):\n    return jax.random.normal(key, (3, 3)), jax.random.normal(key, (3,))\n\nwith jax.debug_key_reuse(True):\n    try:\n        jax.jit(init)(jax.random.key(0))\n    except jax.errors.KeyReuseError as err:\n        print(err)\n    key = jax.random.key(0)\n    a = jax.random.normal(key, (2,))\n    try:\n        jax.random.normal(key, (2,))\n    except jax.errors.KeyReuseError as err:\n        print(str(err).splitlines()[0])\n    print(bool(jnp.all(a == jax.random.normal(jax.random.clone(key), (2,)))))\n\n    raw = jax.random.PRNGKey(0)\n    print(bool(jnp.all(jax.random.normal(raw, (2,)) == jax.random.normal(raw, (2,)))))\n\n# In pjit, argument 0 is already consumed.\n# See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.KeyReuseError\n# Previously-consumed key passed to jit-compiled function at index 0\n# True\n# True',
            },
          },
        ],
        readings: [
          { label: 'jax.experimental.key_reuse', url: 'https://docs.jax.dev/en/latest/jax.experimental.key_reuse.html', note: 'the checker, the context manager, and what the team plans to do with the default' },
          { label: 'jax.errors, the KeyReuseError entry', url: 'https://docs.jax.dev/en/latest/errors.html', note: 'both messages, next to every other error this course quotes' },
          { label: 'jax.random.clone', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.random.clone.html', note: 'the one-line escape for a repeat you actually meant' },
        ],
        check: [
          {
            q: 'Two calls to jax.random.normal with one key produced arrays whose correlation is 1.0. What is the real relationship between them?',
            a: 'They are the same array. The largest elementwise difference is exactly 0.0, because a key plus a shape determines the bits, so the second call recomputed the first. Independent draws from split keys measure around -0.005 at ten thousand samples.',
          },
          {
            q: 'You average eight sampled estimates and the standard deviation of the mean barely falls. What do you check first?',
            a: 'The keys. Averaging eight independent draws of 2048 normals gives 0.352, close to 1 over the square root of 8; averaging eight draws from one key gives 0.978, because the average of one sample with itself is that sample.',
          },
          {
            q: 'Why does the reuse checker stay quiet about a program built on jax.random.PRNGKey?',
            a: 'It tracks typed key<fry> values, and a legacy key is a plain uint32 array with no consumption to track. Under jax.debug_key_reuse(True) two draws from one PRNGKey return identical arrays and raise nothing at all.',
          },
        ],
        work: [
          { id: 'run-under-the-checker', label: 'run one of your own sampling scripts under jax.debug_key_reuse(True) and fix everything it raises before you change anything else', href: '#make-it-raise' },
          { id: 'assert-inequality', label: 'add an equality assertion to one sampling test of your own: no two draws that are meant to be independent may be elementwise equal', href: '#not-correlated-identical' },
        ],
      },
      {
        id: 'keys-through-vmap-and-scan',
        num: 3,
        title: 'Keys through vmap and scan',
        lede: 'A map wants one key per row and a loop wants one key per step. Get either wrong and the program still runs at full speed, sampling the same noise everywhere it goes.',
        goal: 'Thread keys through vmap and scan both ways, name the condition under which a vmapped sampler is exactly the loop it replaces, and say which of the two scan disciplines survives a checkpoint restore.',
        sections: [
          {
            h: 'one key per row, decided outside the map',
            ps: [
              "Split first, map second. Split a key into eight, vmap a sampler over the resulting key array, and the result is elementwise equal to the Python loop that draws from each key in turn. Not statistically indistinguishable. Equal.",
              "That exactness belongs to the default generator, not to vmap. The jax.random reference states it as a property row in its implementation table, `exact jax.vmap over keys`, ticked for threefry with and without the partitionable flag and blank for both rbg implementations.",
              "The counterexample runs here. Build eight rbg keys, vmap a scalar normal over them, and all eight values come from the first key alone: the vmapped result is elementwise equal to `jax.random.normal(keys[0], (8,))`, exactly as the docstring warns. Nothing raises, the shape is right, and the seven other keys were never consulted. If you ever switch implementation for TPU speed, this is the line whose meaning changes.",
            ],
            code: {
              caption: 'verbatim, the rbg paragraph of the jax.random module docstring as printed by the local jax 0.4.38 install (comment markers added, closing sentence trimmed), then a run on this machine (jax 0.4.38 CPU)',
              lang: 'python',
              text: '# Additionally, both ``"rbg"`` and ``"unsafe_rbg"`` behave unusually\n# under ``jax.vmap``. When vmapping a random function over a batch\n# of keys, its output values can differ from its true map over the\n# same keys. Instead, under ``vmap``, the entire batch of output\n# random numbers is generated from only the first key in the input\n# key batch. For example, if ``keys`` is a vector of 8 keys, then\n# ``jax.vmap(jax.random.normal)(keys)`` equals\n# ``jax.random.normal(keys[0], shape=(8,))``.\n\nimport jax\nimport jax.numpy as jnp\n\nkeys = jax.random.split(jax.random.key(0), 8)\nbatched = jax.vmap(lambda k: jax.random.normal(k, (3,)))(keys)\nlooped = jnp.stack([jax.random.normal(k, (3,)) for k in keys])\nprint(batched.shape, bool(jnp.all(batched == looped)))\n\nrbg = jax.random.split(jax.random.key(0, impl=\'rbg\'), 8)\nprint(jax.vmap(jax.random.normal)(rbg))\nprint(jax.random.normal(rbg[0], (8,)))\n\n# (8, 3) True\n# [-1.7768954  -1.3365983   0.21346289 -0.16086003 -0.16176912 -0.2579535\n#   0.8256621   0.03783947]\n# [-1.7768954  -1.3365983   0.21346289 -0.16086003 -0.16176912 -0.2579535\n#   0.8256621   0.03783947]',
            },
          },
          {
            h: 'a key with in_axes=None is one draw copied',
            ps: [
              "Hand vmap a single key with in_axes=None and it does precisely what you asked: broadcasts one key to every row. All eight rows then draw the same three numbers, the output shape is the shape you expected, and no rule was broken anywhere.",
              "The museum's `vmap-axes` exhibit is the loud version of an in_axes mistake, where two arguments disagree about the batch size and vmap says so in the error. This one has no disagreement to report. A key is a shape-() value, so None is a legal thing to say about it.",
              "Passing the split key array instead, with the default in_axes of 0, is what you meant, and the equality assertion from the previous lesson is how you keep it that way: no two rows elementwise equal.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the broadcast key, then the split key array through the same function',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\nkey = jax.random.key(0)\nrows = jax.vmap(lambda k, x: jax.random.normal(k, (3,)) + x, in_axes=(None, 0))(key, jnp.zeros((8, 3)))\nprint(rows[:2])\nprint(bool(jnp.all(rows == rows[0])))\n\nkeys = jax.random.split(key, 8)\nok = jax.vmap(lambda k, x: jax.random.normal(k, (3,)) + x)(keys, jnp.zeros((8, 3)))\nprint(bool(jnp.all(ok == ok[0])))\n\n# [[ 1.8160863  -0.48262316  0.33988908]\n#  [ 1.8160863  -0.48262316  0.33988908]]\n# True\n# False',
            },
          },
          {
            h: 'the two scan disciplines hash different counters',
            ps: [
              "Chapter 8 names both scan disciplines and calls them equally correct: carry the key and split it every step, or fold the step index into a base key that never moves. What the chapter does not say is that the two feed different counters to the same hash, so a loop that switches from one to the other changes every number it draws.",
              "The first lesson printed both counters, and the loop below runs them six times each. The carried form draws from the second child of `split`, whose two words that same lesson printed, and every later step splits whatever the previous step handed on. The folded form hashes the pair (0, i) against a key that never changes, so step i is fixed by i alone. On this version the two streams share nothing: twelve distinct values on each side and a shared count of zero.",
              "The carried form works only because a key is a legal scan carry. `scan` requires the carry to come back with the shape and dtype it went in with, and a key satisfies that: after six steps the carry still prints as dtype key<fry>, shape (), while the two words inside it have moved from [0 0] to [3110407274 4280739360]. Which discipline you want is decided by what happens when a run resumes partway through, and the next section measures that.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): six steps of noise, carried against folded, then the count of values the two streams share and the key the carry ended on. Under jax 0.5.0 defaults every printed number moves, the boolean still holds, and that count is 2, because partitionable split derives child i as fold_in(key, i), which lands the carried step zero on the same key the folded step one uses',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\nbase = jax.random.key(0)\n\ndef carried(key, steps):\n    def step(k, _):\n        k, sub = jax.random.split(k)\n        return k, jax.random.normal(sub, (2,))\n    return jax.lax.scan(step, key, None, length=steps)\n\ndef folded(key, start, steps):\n    def step(_, i):\n        return None, jax.random.normal(jax.random.fold_in(key, i), (2,))\n    return jax.lax.scan(step, None, jnp.arange(start, start + steps))\n\nend_key, noise_a = carried(base, 6)\n_, noise_b = folded(base, 0, 6)\nprint(noise_a)\nprint(noise_b)\nprint(bool(jnp.all(noise_a[0] == jax.random.normal(jax.random.split(base)[1], (2,)))))\nprint(len(set(map(float, noise_a.reshape(-1))) & set(map(float, noise_b.reshape(-1)))))\nprint(end_key.dtype, end_key.shape, jax.random.key_data(end_key))\n\n# [[ 0.19307722 -0.52678293]\n#  [ 0.00870701 -0.04888523]\n#  [-0.89105326 -0.66184473]\n#  [ 1.2091267   2.3117282 ]\n#  [ 0.91798526  0.48332107]\n#  [-0.21526915 -0.41612625]]\n# [[-0.48121497 -0.01837499]\n#  [ 1.4321449   1.3629805 ]\n#  [ 1.1727159  -1.8279221 ]\n#  [-0.05104028  0.68906456]\n#  [ 0.05196318 -0.67536026]\n#  [ 0.8082805  -0.21244425]]\n# True\n# 0\n# key<fry> () [3110407274 4280739360]',
            },
          },
          {
            h: 'the resume test',
            ps: [
              "Run six steps, then try to restart at step three, and the two disciplines come apart. The folded form needs two things: the base key and the integer 3. The three noise vectors that come back are elementwise equal to steps three through five of the uninterrupted run.",
              "The carried form needs the key as it stood after step two, which means the key is part of what a checkpoint has to contain. Hand the function the original base key instead, and step three of the resumed run is elementwise equal to step zero of the first run. The loop restarted its stream, and the numbers look exactly as random as they did before.",
              "Chapter twelve makes the rule out of this: params, optimizer state, step and key travel together. The run below is the test that fails when the key is left behind, and it is worth writing once for your own loop, because it is three assertions and it catches a class of bug that no loss curve will show you.",
              '>> A resumed run that samples fresh noise from step zero is not a resumed run.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the same two loops as above, resumed at step three three ways. All three booleans hold under jax 0.5.0 defaults too, on numbers that differ',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\nbase = jax.random.key(0)\n\ndef carried(key, steps):\n    def step(k, _):\n        k, sub = jax.random.split(k)\n        return k, jax.random.normal(sub, (2,))\n    return jax.lax.scan(step, key, None, length=steps)\n\ndef folded(key, start, steps):\n    def step(_, i):\n        return None, jax.random.normal(jax.random.fold_in(key, i), (2,))\n    return jax.lax.scan(step, None, jnp.arange(start, start + steps))\n\n_, whole_a = carried(base, 6)\n_, whole_b = folded(base, 0, 6)\n\nkey_at_3, _ = carried(base, 3)\n_, from_base = carried(base, 3)\n_, from_saved = carried(key_at_3, 3)\n_, resumed_b = folded(base, 3, 3)\n\nprint(bool(jnp.all(from_base[0] == whole_a[0])))\nprint(bool(jnp.all(from_saved == whole_a[3:])))\nprint(bool(jnp.all(resumed_b == whole_b[3:])))\n\n# True\n# True\n# True',
            },
            table: {
              caption: 'the two scan disciplines, every row measured by the runs in this lesson (verified, jax 0.4.38 CPU)',
              cols: ['question', 'carry the key and split', 'fold the step index in'],
              rows: [
                ['what the loop carries', 'the key, as a scan carry of dtype key<fry>', 'nothing; the base key is a closure constant'],
                ['what step k needs', 'the key as it stood after step k-1', 'the base key and the integer k'],
                ['resume from a checkpoint', 'correct only if the key was saved beside the params', 'correct from the step number alone'],
                ['index may be a traced value', 'no: the count in split becomes a shape', 'yes: fold_in only hashes the integer'],
                ['under jax_threefry_partitionable', 'the child keys change', 'the derived keys are unchanged; the draws still move'],
              ],
            },
          },
          {
            h: 'one stream per example and per step',
            ps: [
              "Two folds compose. Fold the example id into the base key, fold the step into the result, and every pair of coordinates gets its own key with no carry to thread and no split width to decide in advance. Twelve draws across a three by four grid come back as twelve distinct values.",
              "The reason this composes is a staticness rule you can see in the error. `split` takes a count that turns into a shape, so under jit it demands a concrete Python integer and says `Shapes must be 1D sequences of concrete values of integer type` when it does not get one. `fold_in` only hashes its argument, so a traced integer is fine. Anything indexed by a runtime quantity, a step, an example id, a layer number, is fold_in work.",
              "Coordinates you can name are coordinates you can resume from, which is the property the previous section measured. Deriving a key from where a piece of work sits, rather than from how many draws happened before it, is what makes a sampler restartable at any point.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): a key per (example, step) pair, then what split refuses under jit',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\nkey = jax.random.key(0)\n\ndef stream(example, step):\n    return jax.random.fold_in(jax.random.fold_in(key, example), step)\n\ngrid = jax.vmap(lambda e: jax.vmap(lambda s: jax.random.normal(stream(e, s), ()))(jnp.arange(4)))(jnp.arange(3))\nprint(grid)\nprint(len(set(map(float, grid.reshape(-1)))))\n\ntry:\n    jax.jit(lambda n: jax.random.split(key, n))(3)\nexcept TypeError as err:\n    print(str(err).splitlines()[0])\n\n# [[-0.9984514   0.4432096  -0.2424067  -1.1119174 ]\n#  [-0.04697571  1.2683467   1.816136   -0.6499298 ]\n#  [ 1.3010085   0.18747099  1.159638    0.10436483]]\n# 12\n# Shapes must be 1D sequences of concrete values of integer type, got (Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace>,).',
            },
          },
        ],
        readings: [
          { label: 'Distributed arrays and automatic parallelization', url: 'https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html', note: 'the RNG section the module docs point at for jax_threefry_partitionable' },
          { label: 'jax.lax.scan', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html', note: 'the carry contract a key has to satisfy, and it does' },
          { label: 'the jax changelog, 0.5.0', url: 'https://docs.jax.dev/en/latest/changelog.html', note: 'the line that flipped jax_threefry_partitionable on by default, with its upgrade note' },
        ],
        check: [
          {
            q: 'You vmap a sampler over eight split keys. What makes the result exactly equal to eight separate calls?',
            a: 'The threefry implementation. The jax.random reference lists exact vmap over keys as a property of threefry only; under rbg the whole batch comes from the first key, so a vmapped scalar normal over eight rbg keys equals normal(keys[0], (8,)).',
          },
          {
            q: 'Every row of a vmapped dropout mask is identical and nothing raised. What is the most likely cause?',
            a: 'One key was passed with in_axes=None, so vmap broadcast it and every row drew the same numbers. A key is a shape-() value, so None is legal and no size disagreement exists for vmap to report.',
          },
          {
            q: 'A run resumes at step three and its noise repeats step zero. Which key discipline was in use, and what was missing from the checkpoint?',
            a: 'The carried form, splitting a key held in the scan carry. That key is state, and it was not saved with the params, so the loop restarted its stream. Folding the step index into a fixed base key resumes from the step number alone.',
          },
        ],
        work: [
          { id: 'break-the-map', label: 'prove the rows of one vmapped sampler of your own differ, then break it on purpose with in_axes=None and watch nothing complain', href: '#a-key-with-in-axes-none-is-one-draw-copied' },
          { id: 'resume-test', label: 'convert one carried-key scan of your own to fold_in over the step index, then run the resume test against the uninterrupted output', href: '#the-resume-test' },
        ],
      },
    ],
  },
]
