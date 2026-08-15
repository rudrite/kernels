// New file: site/src/data/lessons/jax-performance.ts
// Honest measurement below the survey chapter 11 teaches: what a timing loop
// actually caught, what a running process is still holding in device bytes,
// what jax.checkpoint changes and where that change stops being visible, and
// which instrument answers which performance question. Every printed value ran
// on this machine (jax 0.4.38, CPU, one CpuDevice, 2026-08-15) and is quoted as
// printed. Timings are wall clock on a loaded laptop: each caption says which
// statistic it reports, and the run-to-run spread is stated where it matters.
import type { UnitLessons } from './index'

export const JAX_PERFORMANCE_LESSONS: UnitLessons[] = [
  {
    unit: 'jax:performance',
    lessons: [
      {
        id: 'what-the-clock-caught',
        num: 1,
        title: 'What the clock caught',
        lede: 'The chapter names three lies and hands you a harness that kills all three. Run it and one question is still open: the number it returns is a rate, not a latency, and on the same fifty calls those two differ by a factor of two.',
        goal: 'Say which part of a JAX call a given timing loop measured, choose deliberately between reporting a best case and a sustained rate, and state what one compile costs in units of steady steps.',
        sections: [
          {
            h: 'dispatch costs the same whatever you ask for',
            ps: [
              "Chapter 1 established that JAX dispatches asynchronously, so a line returns before the device is done. The consequence is easier to feel as a measurement than as a sentence. Time the same jitted matmul at three sizes, once without blocking and once with, and the unblocked column barely moves while the blocked column grows with the work.",
              "The middle column below spans 156.4 to 5929.7 microseconds, a factor of 38 for a factor of 512 more arithmetic. The left column sits between 14.9 and 18.4 microseconds the whole way. That left number is the cost of a Python call plus an enqueue, and it is nearly independent of the program you enqueued.",
              "So a benchmark that forgets to block is not merely off by a constant. It reports a number that tracks your dispatch path and not your computation, which is why an unblocked benchmark can show a rewrite getting faster when the rewrite did nothing at all.",
              '>> An unblocked timing loop measures your Python, not your program.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the same jitted matmul at three sizes, timed twice; the two timing columns are the minimum of 25 single calls in microseconds, the last is their ratio',
              lang: 'python',
              text: 'import time\n\nimport jax\nimport jax.numpy as jnp\n\nf = jax.jit(lambda x: x @ x)\n\ndef best(g, x, r=25):\n    return min(g(x) for _ in range(r))\n\ndef dispatch(x):\n    t0 = time.perf_counter()\n    f(x)\n    return time.perf_counter() - t0\n\ndef compute(x):\n    t0 = time.perf_counter()\n    f(x).block_until_ready()\n    return time.perf_counter() - t0\n\nfor n in (128, 512, 1024):\n    x = jnp.ones((n, n))\n    f(x).block_until_ready()\n    d, c = best(dispatch, x), best(compute, x)\n    print(f"{n:5d} {d * 1e6:7.1f} {c * 1e6:9.1f} {c / d:7.0f}")\n\n#   128    14.9     156.4      11\n#   512    18.4     736.4      40\n#  1024    16.5    5929.7     359',
            },
          },
          {
            h: 'one compile is worth a hundred steps',
            ps: [
              "The first call to a jitted function traces it, lowers it, compiles it, and then runs it. Timing that call gives you an upper bound on the compile rather than the compile itself, and for a six-layer MLP on this machine the bound is 651.6 milliseconds against a steady step of 5.9 milliseconds.",
              "Divide and the number stops being an abstract complaint about warmup. One compile of this function costs 111 steady steps. Across four runs of the same script the ratio landed between 105 and 190, because the steady step varies more than the compile does when the machine is busy.",
              "That ratio is the exchange rate for a mid-run recompile. LAB·J2 is where you find out which part of the cache key moved and drive the log to silence; this is what one entry in that log costs you before you go looking.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the first call against the mean of twenty steady calls, one run of four',
              lang: 'python',
              text: 'import time\n\nimport jax\nimport jax.numpy as jnp\n\ndef mlp(p, x):\n    for w, b in p:\n        x = jnp.tanh(x @ w + b)\n    return x.sum()\n\nkey = jax.random.key(0)\np = [(jax.random.normal(k, (512, 512)) * 0.05, jnp.zeros(512))\n     for k in jax.random.split(key, 6)]\nx = jnp.ones((256, 512))\nf = jax.jit(mlp)\n\nt0 = time.perf_counter()\nf(p, x).block_until_ready()\nfirst = time.perf_counter() - t0\n\nt1 = time.perf_counter()\nfor _ in range(20):\n    out = f(p, x)\nout.block_until_ready()\nsteady = (time.perf_counter() - t1) / 20\n\nprint(f"first call {first * 1e3:7.1f} ms")\nprint(f"steady     {steady * 1e3:7.1f} ms")\nprint(f"ratio      {first / steady:7.0f}")\n\n# first call   651.6 ms\n# steady         5.9 ms\n# ratio          111',
            },
          },
          {
            h: 'where the block sits decides what the number means',
            ps: [
              "Take fifty calls of one function and move a single line. Block once after the loop and the mean is 727 microseconds per call. Block inside the loop, on every call, and the mean is 1524. Same function, same fifty calls, same array.",
              "The gap is async dispatch doing its job. When nothing blocks mid-loop, the host runs ahead and enqueues the next call while the device is still on the current one. When every call blocks, the host has nothing queued and each step pays the full round trip before the next one is submitted.",
              "Neither number is a lie, and picking between them is a question about your program rather than about JAX. A training loop that fires steps back to back is described by the first. A service that waits on each result before deciding what to do next is described by the second. Say which one you measured, because a reader cannot tell from the number.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): fifty calls timed twice, the block moved one line; each figure is the minimum of seven such loops',
              lang: 'python',
              text: 'import time\n\nimport jax\nimport jax.numpy as jnp\n\nf = jax.jit(lambda x: x @ x)\nx = jnp.ones((512, 512))\nf(x).block_until_ready()\n\ndef block_at_the_end(k=50):\n    t0 = time.perf_counter()\n    for _ in range(k):\n        out = f(x)\n    out.block_until_ready()\n    return (time.perf_counter() - t0) / k\n\ndef block_every_call(k=50):\n    t0 = time.perf_counter()\n    for _ in range(k):\n        f(x).block_until_ready()\n    return (time.perf_counter() - t0) / k\n\na = min(block_at_the_end() for _ in range(7)) * 1e6\nb = min(block_every_call() for _ in range(7)) * 1e6\nprint(f"block at the end {a:6.0f} us/call")\nprint(f"block every call {b:6.0f} us/call")\n\n# block at the end    727 us/call\n# block every call   1524 us/call',
            },
          },
          {
            h: 'min or mean, and say which',
            ps: [
              "Two sections back, a 512 matmul measured 736.4 microseconds. One section back, the same matmul on the same machine measured 1524. Both are honest, and they disagree because one is the minimum of 25 isolated calls and the other is the mean of 50 back-to-back ones.",
              "A minimum reports the best the machine managed with the scheduler out of the way, which is the right statistic when you are comparing two implementations and want the noise gone. A mean over a loop reports what a run of that length actually delivered, which is the right statistic when you are quoting throughput to someone who will plan around it. The method is not a footnote to the number; it is half of the number.",
              "The spread is worth knowing before you celebrate a change. Five consecutive runs of the identical script, on this machine, put the 1024 blocked figure between 5929.7 and 7167.1 microseconds, a 21 percent range with nothing changed. A difference smaller than that is not a result yet.",
            ],
            table: {
              caption: 'the n = 1024 row of the first script, five consecutive runs of the same file (verified, jax 0.4.38 CPU, 15 August 2026); microseconds',
              cols: ['run', 'dispatch', 'compute', 'ratio'],
              rows: [
                ['1', '16.6', '6171.4', '373'],
                ['2', '18.5', '7167.1', '386'],
                ['3', '21.0', '6245.6', '298'],
                ['4', '16.5', '5929.7', '359'],
                ['5', '26.7', '6705.9', '251'],
              ],
            },
          },
        ],
        readings: [
          { label: 'Asynchronous dispatch', url: 'https://docs.jax.dev/en/latest/async_dispatch.html', note: 'the mechanism the first and third sections measure, stated by the source' },
          { label: 'JAX FAQ · benchmarking JAX code', url: 'https://docs.jax.dev/en/latest/faq.html#benchmarking-jax-code', note: 'the official notes, including the transfer costs a CPU run cannot show you' },
          { label: 'jax.block_until_ready', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.block_until_ready.html', note: 'the pytree form; the method on an array blocks one leaf, this blocks them all' },
        ],
        check: [
          {
            q: 'A timing loop reports nearly the same microseconds for a 128 and a 1024 matmul. What did it measure?',
            a: 'Dispatch. Nothing blocked on the result, so the clock caught the Python call and the enqueue, which cost 14.9 to 18.4 microseconds at every size on this machine while the blocked figure grew from 156.4 to 5929.7.',
          },
          {
            q: 'Why do the same fifty calls report 727 microseconds one way and 1524 the other?',
            a: 'Because of where the block sits. Blocking once at the end lets the host run ahead and keep the device fed, which reports a sustained rate; blocking every call pays the round trip per step, which reports a latency. Both are true of different programs.',
          },
          {
            q: 'Your function compiles in 651.6 ms and steps in 5.9 ms. What does one mid-run recompile cost?',
            a: 'About 111 steps, and the first-call figure is an upper bound because it also includes tracing, lowering and one execution. LAB·J2 is where you find which part of the cache key moved.',
          },
        ],
        work: [
          { id: 'three-numbers', label: 'time one op of your own three ways (never blocked, blocked once at the end, blocked every call) and write one sentence per number saying what it describes', href: '#where-the-block-sits-decides-what-the-number-means' },
          { id: 'compile-in-steps', label: 'measure one of your own compiles in units of steady steps, then state how many steps a single mid-run recompile costs you', href: '#one-compile-is-worth-a-hundred-steps' },
        ],
      },
      {
        id: 'what-the-process-holds',
        num: 2,
        title: 'What the process is still holding',
        lede: 'Ask how much device memory a JAX program is using and the usual answer is a guess. There is a counting answer instead, one line of arithmetic over every array still alive, and you can run it from inside your own training loop.',
        goal: 'Count what a running JAX program holds in device bytes, say what keeps an array alive after its Python name is gone, and report what a donated buffer saved in bytes rather than in seconds.',
        sections: [
          {
            h: 'count what is alive, in bytes',
            ps: [
              "Two arrays, one of them derived from the other, and a question about what that costs. `jax.live_arrays()` answers it directly: it returns every array currently alive on the devices, so the whole ledger is a sum of `nbytes` over that list. One 2048 by 2048 float32 array is 16 MiB, and after `y = x * 2` the ledger reads two arrays and 32 MiB.",
              "`del y` takes it straight back to one array and 16 MiB. The device buffer is released when the last Python reference to the array goes, which is what makes a ledger printed at the top of a training loop a fair account of what that loop is holding at that moment.",
              "One thing the list does not include is the memory a call needs while it is running. The temporary arena the compiled program reserves is not a `jax.Array` and never appears here, and lesson three reads that number off the compiled program instead. Two instruments, two questions, and a figure from one is not an answer to the other.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the ledger either side of one derived array; each pair is (live arrays, total MiB)',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\ndef live():\n    arrays = jax.live_arrays()\n    return len(arrays), sum(a.nbytes for a in arrays) // 1024 ** 2\n\nx = jnp.ones((2048, 2048)).block_until_ready()\nprint("x alive          ", live())\ny = (x * 2.0).block_until_ready()\nprint("y = x * 2 alive  ", live())\ndel y\nprint("after del y      ", live())\n\n# x alive           (1, 16)\n# y = x * 2 alive   (2, 32)\n# after del y       (1, 16)',
            },
          },
          {
            h: 'a name you dropped is not memory you freed',
            ps: [
              "Close a jitted function over an array and that array gets a second holder your code never names. Add a 16 MiB constant to a 16 MiB input, call the function once, and the ledger reads three arrays and 48 MiB while your program has two names.",
              "Now `del c`, and nothing moves. The count stays at three and the total stays at 48 MiB, because the jit cache is holding both the original array and a copy of it, and neither is reachable from a name in your program. `jax.clear_caches()` is what returns the ledger to a single array and 16 MiB.",
              "Pass the same array as an argument instead of capturing it and the ledger matches what the code says: two arrays after the call, one once the name is gone. So when a long-running process holds more than the arrays you can point at, cached programs and the constants they captured are the first place to look.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): a captured constant against the same array passed in, with gc.collect() before every reading so the count does not depend on when collection runs',
              lang: 'python',
              text: 'import gc\n\nimport jax\nimport jax.numpy as jnp\n\ndef live():\n    arrays = jax.live_arrays()\n    return len(arrays), sum(a.nbytes for a in arrays) // 1024 ** 2\n\nx = jnp.ones((2048, 2048)).block_until_ready()\n\nc = jnp.full((2048, 2048), 7.0).block_until_ready()\ncaptured = jax.jit(lambda p: p + c)\ncaptured(x).block_until_ready()\ngc.collect()\nprint("captured, after one call", live())\ndel c\ngc.collect()\nprint("captured, after del c   ", live())\njax.clear_caches()\ngc.collect()\nprint("captured, after clear   ", live())\n\nk = jnp.full((2048, 2048), 7.0).block_until_ready()\npassed = jax.jit(lambda p, q: p + q)\npassed(x, k).block_until_ready()\ngc.collect()\nprint("argument, after one call", live())\ndel k\ngc.collect()\nprint("argument, after del k   ", live())\n\n# captured, after one call (3, 48)\n# captured, after del c    (3, 48)\n# captured, after clear    (1, 16)\n# argument, after one call (2, 32)\n# argument, after del k    (1, 16)',
            },
          },
          {
            h: 'a donation, read off the ledger',
            ps: [
              "One keyword changes that arithmetic, and its own lesson is elsewhere. `donate_argnums` is told at /jax/jit/giving-an-argument-away: what the promise says, how to prove before anything runs that the compiler accepted it, what happens to the array you gave away, and the two mismatches that get a donation refused. What belongs in a performance chapter is the measurement.",
              "Run the same update twice, plain and donating, counting either side of the call. Without the keyword one 16 MiB array goes in and two come out, because your name still holds the old parameters and the new ones are a fresh allocation. With it, one goes in and one comes out, so the peak stays at 16 MiB rather than doubling.",
              "Then time the two loops and nothing separates them. Fifty rotations at this size measured 3508 microseconds per call plain against 3447 donating in one run, and across five runs the plain figure landed between 3184 and 4021 while the donated one landed between 3185 and 4561. The ranges overlap, so the honest report is that this machine cannot tell them apart. What donation saves is bytes; turning bytes into time takes a device where the second allocation and the larger footprint cost something.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): live arrays and their total either side of one update, with the keyword and without',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\ndef live():\n    arrays = jax.live_arrays()\n    return len(arrays), sum(a.nbytes for a in arrays) // 1024 ** 2\n\nfor donate in (False, True):\n    kwargs = {"donate_argnums": 0} if donate else {}\n    update = jax.jit(lambda p: p * 0.99, **kwargs)\n    p = jnp.ones((2048, 2048))\n    p.block_until_ready()\n    before = live()\n    new = update(p).block_until_ready()\n    print(f"donate={donate!s:5s} before {before}  after {live()}")\n    del p, new\n\n# donate=False before (1, 16)  after (2, 32)\n# donate=True  before (1, 16)  after (1, 16)',
            },
            table: {
              caption: 'the same rotation loop timed both ways (verified, jax 0.4.38 CPU, 15 August 2026): fifty calls at 2048 by 2048, each figure the minimum of seven such loops, five consecutive runs of one script',
              cols: ['run', 'plain, us/call', 'donated, us/call'],
              rows: [
                ['1', '3243', '4561'],
                ['2', '3613', '4343'],
                ['3', '3508', '3447'],
                ['4', '3184', '3399'],
                ['5', '4021', '3185'],
              ],
            },
          },
          {
            h: 'a donating function will not survive a benchmark loop',
            ps: [
              "Put a donating function inside the harness from lesson one and the second iteration fails. The first call consumes the input, and every call after it hands the runtime a buffer that no longer exists, so the rejection arrives at the argument, before any of your program runs.",
              "The message names the buffer and not the line, and it comes back as a `ValueError` from the call itself: `INVALID_ARGUMENT: Invalid buffer passed: buffer has been deleted or donated.` A harness that catches broadly will log that as a bad measurement when it is really a statement about the loop's shape.",
              "The fix is not a flag. Feed the loop the value the previous call produced, `p = update(p)`, and the promise is true every time, because the new parameters really did replace the old ones. A training loop has that shape already, which is why donation fits there and not in a microbenchmark that calls `f(x)` twenty times with the same `x`.",
              '>> Donation and a repeat-the-same-argument benchmark are incompatible by construction, not by accident.',
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): three calls with the same argument, then the rotation that makes the promise true',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\n\nupdate = jax.jit(lambda p: p * 0.99, donate_argnums=0)\n\np = jnp.ones((4,))\nfor i in range(3):\n    try:\n        update(p).block_until_ready()\n        print("call", i, "ok")\n    except ValueError as err:\n        print("call", i, err)\n\np = jnp.ones((4,))\nfor _ in range(3):\n    p = update(p)\nprint(p.block_until_ready())\n\n# call 0 ok\n# call 1 INVALID_ARGUMENT: Invalid buffer passed: buffer has been deleted or donated.\n# call 2 INVALID_ARGUMENT: Invalid buffer passed: buffer has been deleted or donated.\n# [0.97029907 0.97029907 0.97029907 0.97029907]',
            },
          },
        ],
        readings: [
          { label: 'jax.live_arrays', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.live_arrays.html', note: 'the one call this lesson counts with, and what the source says it returns' },
          { label: 'jax.clear_caches', url: 'https://docs.jax.dev/en/latest/_autosummary/jax.clear_caches.html', note: 'the call that empties the internal caches holding a captured constant' },
          { label: 'Buffer donation', url: 'https://docs.jax.dev/en/latest/buffer_donation.html', note: 'which arguments qualify; the JAX-side surface is the jit chapter lesson, this one only measures the result' },
        ],
        check: [
          {
            q: 'Your process holds 48 MiB of device arrays and only two of them have names. Where are the others?',
            a: 'In the jit cache. One call over a closed-over 16 MiB constant left three arrays and 48 MiB on the ledger, deleting the name changed neither figure, and jax.clear_caches() brought it back to one array and 16 MiB. Passing the array as an argument never creates the second owner.',
          },
          {
            q: 'How do you show that a donation saved bytes in a process that is already running?',
            a: 'Count live arrays and their total either side of the call. The plain update went from one array and 16 MiB to two arrays and 32; the donating one stayed at one array and 16 MiB. The compile-time proof, the aliased byte count and the lowered signature, is the jit chapter lesson on giving an argument away.',
          },
          {
            q: 'Why does the same donating function work in a training loop and fail on the second call of a benchmark harness?',
            a: 'Because a training loop rotates the value, p = update(p), so the donated buffer really is dead. A harness calls f(x) repeatedly with the same x, and the second call hands the runtime a buffer that was consumed, which it rejects with Invalid buffer passed.',
          },
        ],
        work: [
          { id: 'ledger-your-loop', label: 'print the live-array ledger at the top of your own training loop and name every entry no variable of yours refers to', href: '#a-name-you-dropped-is-not-memory-you-freed' },
          { id: 'donation-in-bytes', label: 'measure one donation of your own either side of the call, report the saving in bytes, and say whether your hardware turns it into time', href: '#a-donation-read-off-the-ledger' },
        ],
      },
      {
        id: 'residuals-you-refuse-to-keep',
        num: 3,
        title: 'The residuals you refuse to keep',
        lede: 'Chapter 4 stated the trade rematerialization makes. Measuring it is a separate skill, and the first thing measuring teaches is that the instrument which shows the change is not the instrument that shows the saving.',
        goal: 'List the residuals a backward pass will hold before running anything, say what jax.checkpoint and a saveable policy each changed in that list, and pick the instrument that answers a given performance question.',
        sections: [
          {
            h: 'list the residuals before you run anything',
            ps: [
              "Reverse mode holds values from the forward pass until the backward pass consumes them, and `print_saved_residuals` prints that list without executing the program. Four tanh layers over a 256 by 512 activation save nine arrays: two per layer, plus one from the final sum.",
              "Each line names the shape, the primitive that produced the value, and the source position it came from, so the list is a map back into your own code rather than a total. Nine arrays of `f32[256,512]` is 4.5 MiB at this size, and the same function at sixteen layers saves thirty-three of them.",
              "Wrap the same function in `jax.checkpoint` and the list collapses to the two arguments. No activation survives the forward pass; the backward pass will recompute what it needs. That is the change remat makes, printed rather than argued.",
              "One import detail on this version: `jax.ad_checkpoint` is not reachable as an attribute of `jax`, and reaching for it raises `AttributeError: module 'jax' has no attribute 'ad_checkpoint'. Did you mean: 'checkpoint'?`. Import the name from the submodule, as below.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the residual list with and without remat, run through python3 -c, which is why the source column reads <string>',
              lang: 'python',
              text: 'import jax\nimport jax.numpy as jnp\nfrom jax.ad_checkpoint import print_saved_residuals\n\nW = jax.random.normal(jax.random.key(0), (512, 512)) * 0.05\nx = jnp.ones((256, 512))\n\ndef stack(W, x):\n    h = x\n    for _ in range(4):\n        h = jnp.tanh(h @ W)\n    return jnp.sum(h ** 2)\n\nprint_saved_residuals(stack, W, x)\nprint("--- under jax.checkpoint")\nprint_saved_residuals(jax.checkpoint(stack), W, x)\n\n# f32[512,512] from the argument W\n# f32[256,512] from the argument x\n# f32[256,512] output of tanh from <string>:11:12 (stack)\n# f32[256,512] output of sub from <string>:11:12 (stack)\n# f32[256,512] output of tanh from <string>:11:12 (stack)\n# f32[256,512] output of sub from <string>:11:12 (stack)\n# f32[256,512] output of tanh from <string>:11:12 (stack)\n# f32[256,512] output of sub from <string>:11:12 (stack)\n# f32[256,512] output of tanh from <string>:11:12 (stack)\n# f32[256,512] output of sub from <string>:11:12 (stack)\n# f32[256,512] output of mul from <string>:12:19 (stack)\n# --- under jax.checkpoint\n# f32[512,512] from the argument W\n# f32[256,512] from the argument x',
            },
          },
          {
            h: 'the arena the compiler reserved did not move',
            ps: [
              "The obvious next step is to ask the compiled program whether it needs less memory now, and on this backend the answer is no. `memory_analysis().temp_size_in_bytes` reports the same temporary arena with remat and without, at every depth tried: 2.0 MiB at two layers, 4.5 at four, 8.5 at eight, 18.0 at sixteen, identical in both columns.",
              "Read what that number is before deciding it is wrong. The arena is what buffer assignment reserved for temporaries in the CPU executable. Dropping the residuals at the JAX level means the backward pass recomputes those values, and recomputed values need somewhere to live too, so the high-water mark the compiler reserved did not fall.",
              "The lesson is about instruments, not about remat. The jaxpr-level list responded to the change and the compiled arena did not, and if you had only run the second one you would have concluded that `jax.checkpoint` does nothing. On an accelerator the number that decides whether a model fits is the allocator's peak, which comes from a device memory profile, and this CPU run cannot produce that figure. LAB·J3 covers the correctness half: both gradients agree.",
              '>> Report the number the instrument actually produces, and name the instrument next to it.',
            ],
            table: {
              caption: 'the same tanh stack at four depths (verified, jax 0.4.38 CPU): saved activations counted from print_saved_residuals, arena from memory_analysis().temp_size_in_bytes',
              cols: ['layers', 'activations saved, plain', 'activations saved, checkpoint', 'arena, plain', 'arena, checkpoint'],
              rows: [
                ['2', '5', '0', '2.0 MiB', '2.0 MiB'],
                ['4', '9', '0', '4.5 MiB', '4.5 MiB'],
                ['8', '17', '0', '8.5 MiB', '8.5 MiB'],
                ['16', '33', '0', '18.0 MiB', '18.0 MiB'],
              ],
            },
          },
          {
            h: 'a policy is the setting between all and nothing',
            ps: [
              "`jax.checkpoint` with no policy saves nothing and recomputes everything. A policy names which values may be kept, and `dots_saveable` is the common one: keep the matmul outputs, recompute the cheap elementwise work around them.",
              "On an eight-layer gelu stack that policy is the only variant that moved the arena, from 7.50 MiB down to 2.00, and it bought that with 0.35 percent more flops by the compiler's own count, 3.112G against 3.123G. On the four-layer tanh stack the identical policy moved neither number.",
              "Two programs, one policy, opposite outcomes. Which values are worth keeping depends on what the backward pass actually needs and on what the compiler was going to do anyway, so a policy is a thing to measure per program rather than a setting to carry between them.",
            ],
            table: {
              caption: 'flops from cost_analysis and arena from memory_analysis, both read off the compiled program with nothing executed (verified, jax 0.4.38 CPU); the flops call itself is chapter 3 territory',
              cols: ['program', 'variant', 'flops', 'arena'],
              rows: [
                ['4-layer tanh', 'plain', '1.479G', '4.50 MiB'],
                ['4-layer tanh', 'jax.checkpoint', '1.479G', '4.50 MiB'],
                ['4-layer tanh', 'dots_saveable', '1.480G', '4.50 MiB'],
                ['8-layer gelu', 'plain', '3.112G', '7.50 MiB'],
                ['8-layer gelu', 'jax.checkpoint', '3.112G', '7.50 MiB'],
                ['8-layer gelu', 'dots_saveable', '3.123G', '2.00 MiB'],
              ],
            },
          },
          {
            h: 'the dump answers what got built',
            ps: [
              "Set `XLA_FLAGS=--xla_dump_to=DIR` and compile, and the compiler writes the module out. Nothing has to execute: the script below calls `.lower(...).compile()` and prints a line saying so, and ten files land for that one module, including the HLO before and after optimization, the buffer assignment, and a memory usage report.",
              "The report opens with the total the compiler reserved and then breaks it down by allocation. For a 256 by 256 tanh-matmul-sum it reserves 768.0 KiB: a 512.0 KiB temporary, the 256.0 KiB parameter, and a handful of four-byte allocations under that.",
              "That is a memory answer available before you own the hardware, which is what makes the dump the first instrument to reach for when the question is structural. The XLA path reads these same files pass by pass; here the point is only that they exist without a run.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): a compile with no execution, the files it wrote, and the head of the memory report',
              lang: 'text',
              text: '$ cat s9.py\nimport jax\nimport jax.numpy as jnp\n\ndef step(p):\n    return jnp.tanh(p @ p).sum()\n\njax.jit(step).lower(jnp.ones((256, 256))).compile()\nprint("compiled; nothing ran")\n\n$ XLA_FLAGS=--xla_dump_to=dump2 python3 s9.py && ls dump2 | grep jit_step\ncompiled; nothing ran\nmodule_0005.jit_step.before_optimizations.txt\nmodule_0005.jit_step.cpu_after_optimizations-buffer-assignment.txt\nmodule_0005.jit_step.cpu_after_optimizations-memory-usage-report.txt\nmodule_0005.jit_step.cpu_after_optimizations.txt\nmodule_0005.jit_step.ir-no-opt.part_00.ll\nmodule_0005.jit_step.ir-no-opt.part_01.ll\nmodule_0005.jit_step.ir-with-opt.part_00.ll\nmodule_0005.jit_step.ir-with-opt.part_01.ll\nmodule_0005.jit_step.obj-file.part_00.o\nmodule_0005.jit_step.obj-file.part_01.o\n\n$ head -9 dump2/module_0005.jit_step.cpu_after_optimizations-memory-usage-report.txt\nTotal bytes used: 786452 (768.0KiB)\n\nAllocations sorted by size:\n\ncumulative_size; total_size - cumulative_size; allocation\n------------------------------------------------------------------------------\n  512.0KiB( 67%);   256.0KiB; allocation 6: size 512.0KiB, preallocated-temp:\n  768.0KiB(100%);        20B; allocation 0: size 256.0KiB, parameter 0, shape |f32[256,256]| at ShapeIndex {}:\n  768.0KiB(100%);        16B; allocation 1: size 4B, output shape is |f32[]|, maybe-live-out:',
            },
          },
          {
            h: 'the trace answers where the time went',
            ps: [
              "A profiler trace is the other kind of answer, and it needs the program to actually run. `jax.profiler.trace(dir)` writes two files per capture, a gzipped Chrome-format JSON for Perfetto and an xplane protobuf for XProf, and `TraceAnnotation` puts your own label on a region so you can find it in either.",
              "The capture is plain JSON underneath, so you can check what landed without opening a viewer. The events carry HLO op names with the module they came from, and the annotation is there as its own event. Two things in that output are worth reading carefully rather than quoting onward.",
              "First, the annotated region measures 5.97 milliseconds while the ten steps it wraps take far longer, because the annotation covers the dispatch loop and the block sits outside it. Lesson one predicted exactly that shape. Second, the `dot.3` events across all ten calls total 2.67 milliseconds, while the same ten steps take between 91.9 and 366.9 milliseconds of wall clock on this machine. Summing raw event durations from a CPU capture is not an accounting of the time; the work is spread across threadpool events, and aggregating them correctly is what XProf is for.",
              "One more line to expect: the capture logs `Can't import tensorflow.python.profiler.trace` at error level when TensorFlow is absent, and writes both files regardless.",
            ],
            code: {
              caption: 'run it (verified, jax 0.4.38 CPU): the capture, the files it wrote with the host name elided, and the totals read straight out of the JSON',
              lang: 'python',
              text: 'import collections\nimport glob\nimport gzip\nimport json\n\nimport jax\nimport jax.numpy as jnp\n\nx = jnp.ones((1024, 1024))\nstep = jax.jit(lambda a: jnp.tanh(a @ a).sum())\nstep(x).block_until_ready()\n\nwith jax.profiler.trace("tr2"):\n    with jax.profiler.TraceAnnotation("ten steps"):\n        for _ in range(10):\n            out = step(x)\n    out.block_until_ready()\n\npath = sorted(glob.glob("tr2/plugins/profile/*/*.trace.json.gz"))[-1]\nevents = json.load(gzip.open(path))["traceEvents"]\ntracks = {e["pid"]: e["args"]["name"] for e in events if e.get("name") == "process_name"}\nprint(len(events), "events on tracks", sorted(tracks.values()))\n\ntotal = collections.defaultdict(float)\ncount = collections.Counter()\nfor e in events:\n    if e.get("ph") == "X" and not e["name"].startswith("$"):\n        total[e["name"]] += e.get("dur", 0)\n        count[e["name"]] += 1\nfor name, dur in sorted(total.items(), key=lambda kv: -kv[1])[:6]:\n    print(f"{dur / 1000:8.2f} ms  x{count[name]:<5d} {name}")\n\n# tr2/plugins/profile/2026_08_15_05_20_33/<host>.trace.json.gz\n# tr2/plugins/profile/2026_08_15_05_20_33/<host>.xplane.pb\n#\n# 29497 events on tracks [/host:CPU]\n#    77.25 ms  x10    ThunkExecutor::Execute (wait for completion)\n#    11.70 ms  x20    PjitFunction(<lambda>)\n#    10.99 ms  x10    call\n#    10.94 ms  x10    tanh.4.clone\n#     5.97 ms  x1     ten steps\n#     3.52 ms  x10    call.1',
            },
          },
          {
            h: 'which instrument, which question',
            ps: [
              "Every instrument this arc reached for is in the table below, ordered by what it costs rather than by what it is called. The first four need no execution at all: a trace or a compile is enough, which is what lets you ask them about hardware you do not have in front of you.",
              "The split worth carrying is the dump against the trace. A dump tells you what the compiler built, and a compile is the whole price. A trace tells you what the machine then did with that program, and it costs a real run on the real device. Ask the dump why a program has the shape it has; ask the trace where a run's time or memory actually went.",
            ],
            table: {
              caption: 'the instruments this arc used, and what each one needs before it can answer; the flops call belongs to chapter 3, the compile log to LAB·J2, and the aliased-byte proof of a donation to the jit chapter lesson',
              cols: ['the question', 'the instrument', 'what has to happen first'],
              rows: [
                ['how many flops is this program', 'compiled.cost_analysis(), chapter 3', 'a lowering and a compile'],
                ['how much does the executable reserve for temporaries', 'compiled.memory_analysis().temp_size_in_bytes', 'a lowering and a compile'],
                ['what will the backward pass hold', 'print_saved_residuals', 'tracing only'],
                ['what did the compiler actually build', 'XLA_FLAGS=--xla_dump_to=DIR', 'a compile, no execution'],
                ['what is this process holding right now', 'jax.live_arrays(), summed over nbytes', 'the arrays exist, in this process'],
                ['why is it still recompiling', 'jax_log_compiles, driven to silence in LAB·J2', 'the real loop, running'],
                ['where did the wall clock go', 'jax.profiler.trace, read in XProf or Perfetto', 'the real steps, running'],
                ['what is the peak device memory', 'jax.profiler.save_device_memory_profile', 'the real steps, on the real device'],
              ],
            },
          },
        ],
        readings: [
          { label: 'Gradient checkpointing', url: 'https://docs.jax.dev/en/latest/gradient-checkpointing.html', note: 'policies in full, and print_saved_residuals used the way this lesson uses it' },
          { label: 'Profiling JAX programs', url: 'https://docs.jax.dev/en/latest/profiling.html', note: 'the capture, the viewers, and the annotations that make a trace navigable' },
          { label: 'Device memory profiling', url: 'https://docs.jax.dev/en/latest/device_memory_profiling.html', note: 'the peak this CPU run cannot show you, and how to read one where it matters' },
        ],
        check: [
          {
            q: 'How do you find out what a backward pass will hold, without running the program?',
            a: 'print_saved_residuals from jax.ad_checkpoint prints the list from tracing alone, naming each value by shape, primitive and source position. A four-layer tanh stack saves nine activations; under jax.checkpoint the list is just the arguments.',
          },
          {
            q: 'jax.checkpoint emptied the residual list and temp_size_in_bytes did not change. What does that tell you?',
            a: 'That the two instruments answer different questions. The residual list is what remat controls at the JAX level; the arena is what buffer assignment reserved in the CPU executable, and recomputed values still need room. The peak that decides whether a model fits comes from a device memory profile instead.',
          },
          {
            q: 'You want to know why a program has the shape it has, and you have no accelerator. Which instrument?',
            a: 'The dump: XLA_FLAGS=--xla_dump_to writes the HLO, the buffer assignment and a memory usage report from a compile alone, with nothing executed. A profiler trace answers the other question, where a real run spent its time, and needs the run.',
          },
        ],
        work: [
          { id: 'residual-ledger', label: 'print the saved residuals for a function you actually train, add up the bytes, and name which of them a saveable policy could drop', href: '#list-the-residuals-before-you-run-anything' },
          { id: 'instrument-first', label: 'take one performance question you have right now, name the instrument that answers it, and state what has to run before it can', href: '#which-instrument-which-question' },
        ],
      },
    ],
  },
]
