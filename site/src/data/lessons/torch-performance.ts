// New file: site/src/data/lessons/torch-performance.ts
// Honest measurement below the survey chapter 9 teaches: what four timing
// methods report for one call and which of them argues back, what one real
// profiler trace from this machine says about a training step, and what the
// CUDA-only instruments do on a machine that has no CUDA. Every printed value
// ran on the course machine (torch 2.2.2, CPU, CPython 3.12.0, 15 August 2026)
// and is quoted as printed; wall-clock figures are medians with their spread,
// because the machine is a loaded laptop and the spread is large. Every source
// excerpt is verbatim from that install, whose benchmark, profiler and cuda
// files are byte-identical to the pytorch v2.2.2 tag. No accelerator was
// involved, and the third lesson says what that costs the claim.
import type { UnitLessons } from './index'

export const TORCH_PERFORMANCE_LESSONS: UnitLessons[] = [
  {
    unit: 'pt:performance',
    lessons: [
      {
        id: 'four-numbers-one-call',
        num: 1,
        title: 'Four numbers for one call',
        lede: 'Time one Linear four ways and the answers run from 0.20 to 1.45 milliseconds. Two of those four methods agree with each other, and only one of them tells you that this machine cannot support the digits it just printed.',
        goal: 'Measure one torch call so the number survives a second run: say what a cold call and a warm loop each caught, read the median and interquartile range torch.utils.benchmark computes for you, and state what the thread count did to the figure before you quote it.',
        sections: [
          {
            h: 'the same call, four numbers',
            ps: [
              "The chapter builds a harness by hand, and it is the right thing to build once, because writing it is how you learn what it defends against. Run four variants of it against one `nn.Linear(512, 512)` on a batch of 64 and the first column disagrees with the other three by a factor of seven.",
              "A single cold call measured 1.451 milliseconds on the first of five runs, and stayed between 1.015 and 1.740 across all five. The warm loop in the same script never rose above 0.234. So a first call costs four to eight times the steady figure here, and none of that difference is compute: it is one-time allocation, the first pass through a fresh dispatch path, and whatever the operating system does the first time a page gets touched.",
              "The other three columns are the interesting part, because they agree. The cold loop, the warm loop and `torch.utils.benchmark.Timer` landed inside 0.194 to 0.250 milliseconds on every one of the five runs, and the run-to-run spread of each one is about 20 percent. So the choice between a hand loop and the shipped instrument is not a question of accuracy on this program. It is a question of which one tells you the spread instead of hiding it, which is the next section.",
              ">> A cold call and a warm loop are two different measurements of one function.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU, CPython 3.12.0): four methods against one Linear; the stdout shown is the first of five runs of the identical file',
              lang: 'python',
              text: 'import time\n\nimport torch\nimport torch.nn as nn\nimport torch.utils.benchmark as benchmark\n\ntorch.manual_seed(0)\nlin = nn.Linear(512, 512)\nx = torch.randn(64, 512)\n\ndef one_call():\n    t0 = time.perf_counter()\n    lin(x)\n    return (time.perf_counter() - t0) * 1e3\n\ndef loop(n=50):\n    t0 = time.perf_counter()\n    for _ in range(n):\n        lin(x)\n    return (time.perf_counter() - t0) / n * 1e3\n\nfirst = one_call()\ncold_loop = loop()\nfor _ in range(50):\n    lin(x)\nwarm_loop = loop()\nm = benchmark.Timer(stmt="lin(x)", globals={"lin": lin, "x": x},\n                    num_threads=torch.get_num_threads()).blocked_autorange(min_run_time=2.0)\n\nprint(f"one cold call         {first:7.3f} ms")\nprint(f"50-call mean, cold    {cold_loop:7.3f} ms")\nprint(f"50-call mean, warm    {warm_loop:7.3f} ms")\nprint(f"Timer median          {m.median * 1e3:7.3f} ms   iqr {m.iqr * 1e3:6.3f}")\n\n# ---- stdout ----\n# one cold call           1.451 ms\n# 50-call mean, cold      0.207 ms\n# 50-call mean, warm      0.196 ms\n# Timer median            0.197 ms   iqr  0.049',
            },
            table: {
              caption: 'five consecutive runs of that same file (verified, torch 2.2.2 CPU, 15 August 2026); milliseconds per call',
              cols: ['run', 'one cold call', '50-call mean, cold', '50-call mean, warm', 'Timer median', 'Timer iqr'],
              rows: [
                ['1', '1.451', '0.207', '0.196', '0.197', '0.049'],
                ['2', '1.740', '0.205', '0.194', '0.211', '0.051'],
                ['3', '1.051', '0.230', '0.234', '0.241', '0.069'],
                ['4', '1.064', '0.250', '0.223', '0.231', '0.013'],
                ['5', '1.015', '0.250', '0.227', '0.234', '0.022'],
              ],
            },
          },
          {
            h: 'the instrument prints its own confidence',
            ps: [
              "`torch.utils.benchmark.Timer` is not a nicer wrapper around `perf_counter`. It picks a block size by warming up until the timer call itself accounts for under 0.1 percent of the measurement, runs blocks until it has spent `min_run_time`, and returns a `Measurement` that carries every block time rather than one average. That is what a hand loop cannot give you back, because a hand loop has thrown the distribution away by the time it divides.",
              "Point it at one training step of a small MLP here and it answers 1.39 milliseconds, then says on the next line that the interquartile range is 41.6 percent of that. The warning is generated, not written by a human: `common.py` sets two thresholds, a tenth and a quarter of the median, and picks between two sentences depending on which one the measurement crossed. Four runs at eight threads crossed the higher one every time.",
              "The same object reports `significant_figures` as 2, estimated from the interquartile region alone so the tails cannot flatter it. Two digits is what this measurement is entitled to. Quoting 1.394 milliseconds off this run means publishing noise in the third digit.",
              "Provenance ages, which is the other reason to report the spread. The chapter's harness measured 0.142 milliseconds for a Linear of this shape on the day it was written; the same shape measures 0.194 to 0.250 today on the same machine. Both are honest. A wall-clock figure is a fact about a machine on a day, so a comparison belongs inside one run and never across two.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): one training step through Timer, printed exactly as the Measurement prints itself',
              lang: 'text',
              text: '$ cat step_timer.py\nimport torch\nimport torch.nn as nn\nimport torch.utils.benchmark as benchmark\n\ntorch.manual_seed(0)\nmodel = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10))\nopt = torch.optim.SGD(model.parameters(), lr=0.1)\nx, y = torch.randn(64, 512), torch.randint(0, 10, (64,))\nloss_fn = nn.CrossEntropyLoss()\n\ndef step():\n    opt.zero_grad(set_to_none=True)\n    loss_fn(model(x), y).backward()\n    opt.step()\n\nm = benchmark.Timer(stmt="step()", globals={"step": step},\n                    label="one training step", sub_label="64 x 512 MLP",\n                    num_threads=torch.get_num_threads()).blocked_autorange(min_run_time=3.0)\nprint(m)\nprint("has_warnings:", m.has_warnings, "| significant_figures:", m.significant_figures)\n\n$ python3 step_timer.py\none training step: 64 x 512 MLP\n  Median: 1.39 ms\n  IQR:    0.58 ms (1.18 to 1.76)\n  1898 measurements, 1 runs per measurement, 8 threads\n  WARNING: Interquartile range is 41.6% of the median measurement.\n           This suggests significant environmental influence.\nhas_warnings: True | significant_figures: 2',
            },
            table: {
              caption: 'the same file four times, then twice more with num_threads=1 (verified, torch 2.2.2 CPU, 15 August 2026); a warning fired on all six, and the last two crossed only the lower threshold, which is a different sentence',
              cols: ['run', 'threads', 'median', 'iqr as % of median', 'which warning', 'significant figures'],
              rows: [
                ['1', '8', '1.39 ms', '41.6', 'significant environmental influence', '2'],
                ['2', '8', '1.56 ms', '42.3', 'significant environmental influence', '2'],
                ['3', '8', '1.50 ms', '34.6', 'significant environmental influence', '2'],
                ['4', '8', '1.69 ms', '35.8', 'significant environmental influence', '2'],
                ['5', '1', '2.42 ms', '22.7', 'could indicate system fluctuation', '2'],
                ['6', '1', '2.26 ms', '24.3', 'could indicate system fluctuation', '2'],
              ],
            },
          },
          {
            h: 'the default is one thread and your program is not',
            ps: [
              "`Timer.__init__` takes `num_threads: int = 1`, and that default is the most common way a torch benchmark ends up describing a program nobody runs. This process has `torch.get_num_threads()` at 8. Construct a Timer without saying otherwise and it measures your matmul with seven of those threads idle.",
              "The gap is not small. A 512 by 512 matmul measured 4072 microseconds at one thread and 1122 at eight, and the default-constructed Timer in the same script reported 4089, sitting with the one-thread column exactly as the signature says it should. Read that as the cost inside an eight-thread training loop and you are off by a factor of 3.6.",
              "Threading also moves the spread, in the direction you would not guess. The single-threaded runs of the training step above had the tighter interquartile range, 22.7 and 24.3 percent against 34.6 to 42.3 at eight threads, and that is the difference between the two warning sentences the last section showed. Eight threads on a laptop contend with whatever else the laptop is doing. Fewer threads measured slower and measured steadier.",
              "So the thread count is part of the measurement, in the same way the shapes and the dtype are. Say it next to the number. `torch.set_num_threads` sets the intra-op pool for the process; `num_threads` on a Timer sets it for that measurement only and restores it after.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the default Timer next to explicit thread counts, one 512 by 512 matmul',
              lang: 'python',
              text: 'import torch\nimport torch.utils.benchmark as benchmark\n\nprint("torch.get_num_threads()", torch.get_num_threads())\nx = torch.randn(512, 512)\ndefault = benchmark.Timer(stmt="x @ x", globals={"x": x})\nprint("Timer default num_threads:", default._task_spec.num_threads)\nr = default.blocked_autorange(min_run_time=1.0)\nprint(f"default timer  median {r.median * 1e6:8.1f} us")\nfor nt in (1, 2, 4, 8):\n    m = benchmark.Timer(stmt="x @ x", globals={"x": x}, num_threads=nt).blocked_autorange(min_run_time=1.0)\n    print(f"threads={nt}      median {m.median * 1e6:8.1f} us  iqr {m.iqr * 1e6:7.1f}")\n\n# ---- stdout ----\n# torch.get_num_threads() 8\n# Timer default num_threads: 1\n# default timer  median   4088.8 us\n# threads=1      median   4071.7 us  iqr   459.5\n# threads=2      median   2297.9 us  iqr   430.9\n# threads=4      median   1331.9 us  iqr   123.7\n# threads=8      median   1122.0 us  iqr    97.7',
              full: {
                text: 'import torch\nimport torch.nn as nn\nimport torch.utils.benchmark as benchmark\n\ntorch.manual_seed(0)\nlin = nn.Linear(512, 512)\nresults = []\nfor batch in (1, 16, 64, 256):\n    x = torch.randn(batch, 512)\n    for nt in (1, 8):\n        results.append(\n            benchmark.Timer(\n                stmt="lin(x)",\n                globals={"lin": lin, "x": x},\n                label="nn.Linear(512, 512)",\n                sub_label=f"batch {batch}",\n                description=f"{nt} thread" + ("s" if nt > 1 else ""),\n                num_threads=nt,\n            ).blocked_autorange(min_run_time=1.0)\n        )\nbenchmark.Compare(results).print()\n\n# ---- stdout ----\n# [--------- nn.Linear(512, 512) ----------]\n#                  |  1 thread  |  8 threads\n# 1 threads: -------------------------------\n#       batch 1    |     52.6   |\n#       batch 16   |    198.5   |\n#       batch 64   |    594.8   |\n#       batch 256  |   2252.0   |\n# 8 threads: -------------------------------\n#       batch 1    |            |     42.1\n#       batch 16   |            |     88.5\n#       batch 64   |            |    198.8\n#       batch 256  |            |    687.3\n#\n# Times are in microseconds (us).',
                label: 'the sweep as a Compare table, four batch sizes by two thread counts',
              },
            },
          },
          {
            h: 'the floor under every op',
            ps: [
              "Underneath all of this there is a cost that does not shrink when the work does. One elementwise add on a one-element tensor took 8.70 microseconds here. On 64 elements, 8.94. On 4096 elements, 9.84. The tensor grew by four thousand times and the call got 13 percent more expensive.",
              "The Python around it is not the cost. An empty function call in the same loop measured 0.068 microseconds and `x.dim()` measured 0.112, so what the 9 microseconds buys is the dispatch itself: the operator lookup, the shape and dtype checks, the output allocation, the kernel entry. In-place `add_` came in at 6.96, and the missing two microseconds are roughly the allocation the out-of-place version pays.",
              "Past 4096 elements the arithmetic finally takes over, 93.72 microseconds at 262144 and 729.96 at a million. The figure to carry out of this is the crossover, not the constant. Below a few thousand elements per op you are timing PyTorch's dispatcher, and no faster kernel can help you there.",
              "That is the same wall the compile chapters attack from the other side. A model that spends its time in thousands of small ops is what `torch.compile` fuses, and the lesson arc at /pytorch/dynamo counts how many graphs you actually got. A benchmark that cannot see the floor cannot tell you whether fusion is worth attempting.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU, one thread): the per-op floor, and where the work finally outruns it',
              lang: 'python',
              text: 'import time\n\nimport torch\n\ntorch.set_num_threads(1)\nx1 = torch.ones(1)\n\ndef loop(fn, n=20000):\n    fn()\n    t0 = time.perf_counter()\n    for _ in range(n):\n        fn()\n    return (time.perf_counter() - t0) / n * 1e6\n\ndef nothing():\n    pass\n\nprint(f"python call only  {min(loop(nothing) for _ in range(5)):7.3f} us")\nprint(f"x.dim()           {min(loop(x1.dim) for _ in range(5)):7.3f} us")\nprint(f"x + 1.0           {min(loop(lambda: x1 + 1.0) for _ in range(5)):7.3f} us")\nprint(f"x.add_(0.0)       {min(loop(lambda: x1.add_(0.0)) for _ in range(5)):7.3f} us")\n\ndef sized(numel, n):\n    x = torch.ones(numel)\n    y = x + 1.0\n    t0 = time.perf_counter()\n    for _ in range(n):\n        y = x + 1.0\n    return (time.perf_counter() - t0) / n * 1e6\n\nfor numel in (1, 64, 4096, 262144, 1048576):\n    n = 5000 if numel < 262144 else 300\n    print(f"add on {numel:9d} elements {min(sized(numel, n) for _ in range(5)):9.2f} us")\n\n# ---- stdout ----\n# python call only    0.068 us\n# x.dim()             0.112 us\n# x + 1.0             9.120 us\n# x.add_(0.0)         6.960 us\n# add on         1 elements      8.70 us\n# add on        64 elements      8.94 us\n# add on      4096 elements      9.84 us\n# add on    262144 elements     93.72 us\n# add on   1048576 elements    729.96 us',
            },
          },
          {
            h: 'the traps that live on other devices',
            ps: [
              "Nothing measured above needed a synchronize, because CPU dispatch runs the op before the line returns. The chapter names the two backends where that stops being true, and the jax path's lesson at /jax/performance/what-the-clock-caught measures the async version of the same mistake in detail: what an unblocked loop actually catches, why blocking every call and blocking once report different true numbers, and when to quote a minimum against a mean. Those traps are told once, there, and they transfer to torch without modification.",
              "What torch adds is that the instrument already handles the first of them. When the module is imported into a process where CUDA is built and available, `timer` is redefined to synchronize before reading the clock, and that function is the default `timer` argument of every Timer you construct. The harness the chapter writes by hand is the harness `Timer` already is, on whichever device you run it.",
              "The lazy backend is the one case where no instrument saves you. A torch_xla tensor records instead of computing, so a timer around it measures graph construction until something forces a sync, and LAB·P5 is where that boundary gets counted call by call on real hardware.",
            ],
            code: {
              caption: 'verbatim, torch/utils/benchmark/utils/timer.py:16-19 and :129-133 at torch 2.2.2, byte-identical to the v2.2.2 tag',
              lang: 'python',
              text: 'if torch.backends.cuda.is_built() and torch.cuda.is_available():\n    def timer() -> float:\n        torch.cuda.synchronize()\n        return timeit.default_timer()\n\n# and from the Timer docstring, on the timer argument:\n#\n#     Callable which returns the current time. If PyTorch was built\n#     without CUDA or there is no GPU present, this defaults to\n#     `timeit.default_timer`; otherwise it will synchronize CUDA before\n#     measuring the time.',
            },
          },
        ],
        readings: [
          { label: 'timer.py at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/utils/benchmark/utils/timer.py', note: 'the synchronizing timer, the num_threads default, and what blocked_autorange does to pick a block size' },
          { label: 'common.py at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/utils/benchmark/utils/common.py', note: 'Measurement statistics: the two interquartile thresholds at lines 26 and 27, and the significant-figure estimate' },
          { label: 'Benchmarking recipe', url: 'https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html', note: 'the official walkthrough, including the timeit comparison this lesson skips' },
          { label: 'CPU threading', url: 'https://docs.pytorch.org/docs/2.2/notes/cpu_threading_torchscript_inference.html', note: 'intra-op against inter-op pools, and which one set_num_threads moves' },
        ],
        check: [
          {
            q: 'Your benchmark warms up, loops fifty times and divides. What is it still not telling you?',
            a: 'The distribution. A mean has thrown the spread away by the time it divides. Timer keeps every block time, reports a median with the interquartile range, and prints one of two warnings when that range passes a tenth or a quarter of the median. On this training step it reported 41.6 percent and cut the answer to two significant figures.',
          },
          {
            q: 'A colleague reports 4089 microseconds for a matmul that your training loop runs in 1122. Where is the disagreement?',
            a: 'Almost certainly the thread count. Timer defaults to num_threads=1 while the process default here is 8, and that one matmul measured 4072 microseconds at one thread against 1122 at eight. Pass num_threads explicitly and state it beside the number.',
          },
          {
            q: 'Why can a faster kernel fail to make a small model faster on CPU?',
            a: 'Because below a few thousand elements per op the dispatch is the cost, not the arithmetic. An add measured 8.70 microseconds on one element and 9.84 on 4096, while an empty Python call measured 0.068. Fusing the ops away is what moves that, which is the compile chapter, not the kernel.',
          },
        ],
        work: [
          { id: 'four-methods', label: 'measure one call of your own all four ways in a single script, then run that script five times and write down the range of each column', href: '#the-same-call-four-numbers' },
          { id: 'own-floor', label: 'find the crossover on your own machine: the element count at which one op stops costing what an empty op costs', href: '#the-floor-under-every-op' },
        ],
      },
      {
        id: 'reading-one-real-trace',
        num: 2,
        title: 'Reading one real trace',
        lede: 'Capture five steps of a small training loop and the biggest row in the table is not an operator. It is the annotation you wrapped the step in, holding six hundred microseconds that belong to no kernel at all.',
        goal: 'Read a torch.profiler table of your own training step: separate self time from total time, name the host-side gap and say what is in it, find the allocation that dominates the step from the shape-grouped view, and pair a backward node with the forward op that produced it.',
        sections: [
          {
            h: 'the schedule decides what the table contains',
            ps: [
              "Profiling every step of a run would record the noisy first steps along with the steady ones, so `torch.profiler.schedule` divides the steps into phases and the profiler only keeps one of them. Ask for `wait=2, warmup=2, active=5, repeat=1` and the schedule is a plain function of the step index that you can call yourself before you profile anything.",
              "Printing it for ten steps shows the shape directly: two NONE, two WARMUP, four RECORD, one RECORD_AND_SAVE, then NONE for anything past the single cycle. The last active step carries the save because that is where `on_trace_ready` fires.",
              "The counterpart in your loop is `prof.step()`, one call per iteration, which is what advances that index. Forget it and the schedule never moves off its first phase. The proof that the schedule worked is in the table's call counts: ten steps ran, the `train_step` row says 5.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the phase for each of ten steps, then what survived into the table',
              lang: 'python',
              text: 'import torch\nimport torch.nn as nn\nfrom torch.profiler import ProfilerActivity, profile, record_function, schedule\n\ntorch.manual_seed(0)\nmodel = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10))\nopt = torch.optim.SGD(model.parameters(), lr=0.1)\nx, y = torch.randn(64, 512), torch.randint(0, 10, (64,))\nloss_fn = nn.CrossEntropyLoss()\n\ndef step():\n    opt.zero_grad(set_to_none=True)\n    loss_fn(model(x), y).backward()\n    opt.step()\n\nsched = schedule(wait=2, warmup=2, active=5, repeat=1)\nprint("phase per step:", [sched(i).name for i in range(10)])\n\nwith profile(activities=[ProfilerActivity.CPU], schedule=sched,\n             record_shapes=True, profile_memory=True) as prof:\n    for _ in range(10):\n        with record_function("train_step"):\n            step()\n        prof.step()\n\nrows = {e.key: e.count for e in prof.key_averages()}\nprint("train_step rows kept:", rows["train_step"], "of 10 steps run")\n\n# ---- stdout ----\n# phase per step: [\'NONE\', \'NONE\', \'WARMUP\', \'WARMUP\', \'RECORD\', \'RECORD\', \'RECORD\', \'RECORD\', \'RECORD_AND_SAVE\', \'NONE\']\n# train_step rows kept: 5 of 10 steps run',
            },
          },
          {
            h: 'the largest row in the table is not an op',
            ps: [
              "Sort that capture by self CPU time and `train_step` is at the top with 31.20 percent, 3.094 milliseconds across five steps. Nothing in that row is an operator. Self time is what a region spent outside all of its children, so the annotation's self time is exactly the part of the step that ran between ATen calls: Python, module `__call__`, the autograd engine deciding what to run next, the optimizer's bookkeeping before it dispatches anything.",
              "That is the host-side gap the chapter asks you to name, and here it is 0.619 milliseconds of a step whose whole recorded cost is 1.942. Across five runs of the same file the share stayed between 28.1 and 31.6 percent while the milliseconds moved by half. On an accelerator that same third of the step is time the device spends idle unless the host runs far enough ahead, which is why it is worth extracting rather than eyeballing.",
              "The total column reads the opposite way and the two must not be confused. The row for `autograd::engine::evaluate_function: AddmmBackward0` shows 1.56 percent self against 21.00 percent total, because the total includes the backward node and the `aten::mm` calls under it. Self answers where the time went; total answers what a subtree cost. Sorting by the wrong one is how a wrapper ends up looking like the bottleneck.",
              "Read the counts as a check on your own model of the step. `aten::addmm` fires 10 times over 5 steps, which is the two forward linears. `aten::mm` fires 15, three per step, and the shape-grouped view names them: the second linear needs a gradient for its input and one for its weight, the first linear needs only the weight one, because the batch it was handed is a leaf that requires no gradient.",
            ],
            code: {
              caption: 'verbatim stdout from the capture above (torch 2.2.2 CPU, 15 August 2026), printed with key_averages().table(sort_by="self_cpu_time_total", row_limit=10); one run of five, and the call counts and byte figures were identical in all five',
              lang: 'text',
              text: '-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  \n                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls  \n-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  \n                                             train_step        31.20%       3.094ms        97.92%       9.710ms       1.942ms       1.02 Mb    -652.54 Kb             5  \n                                            aten::addmm        16.13%       1.599ms        17.44%       1.729ms     172.900us     652.50 Kb     652.50 Kb            10  \n                                               aten::mm        14.70%       1.458ms        14.70%       1.458ms      97.200us       5.72 Mb       5.72 Mb            15  \n                                Optimizer.step#SGD.step         7.95%     788.000us        11.88%       1.178ms     235.600us           0 b           0 b             5  \n                                             aten::add_         3.93%     390.000us         3.93%     390.000us      19.500us           0 b           0 b            20  \n                      Optimizer.zero_grad#SGD.zero_grad         2.63%     261.000us         2.63%     261.000us      52.200us      -4.09 Mb      -4.09 Mb             5  \n                                          ProfilerStep*         2.08%     206.000us       100.00%       9.916ms       1.983ms       1.02 Mb           0 b             5  \n                                                aten::t         1.94%     192.000us         3.52%     349.000us       7.756us           0 b           0 b            45  \n                                              aten::sum         1.63%     162.000us         1.78%     177.000us      17.700us      10.20 Kb      10.20 Kb            10  \n    autograd::engine::evaluate_function: AddmmBackward0         1.56%     155.000us        21.00%       2.082ms     208.200us       5.10 Mb    -652.50 Kb            10  \n-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  \nSelf CPU time total: 9.916ms',
            },
          },
          {
            h: 'what each flag costs, and one that will not run here',
            ps: [
              "Recording is not free, and the honest way to size the cost is with the instrument from lesson one rather than with the profiler's own totals. Timing the same step through Timer, inside and outside a `profile()` block, put the ratio at 1.29, 1.18 and 1.21 across three rounds of one script. Call it a fifth to a quarter, on this step, on this machine.",
              "That is small enough to trust the proportions in the table and too large to quote the absolute milliseconds anywhere. The five profiled captures reported 1.638 to 2.377 milliseconds per step, while Timer put the same unwatched function at 1.420 to 1.619. Take shares of time from the profiler and take the time itself from a benchmark.",
              "`record_shapes` and `profile_memory` earned their place in this capture, and neither one moved the ratio measurably. `with_stack` is the flag that would attribute every op to the Python line that made it, and on this pair of versions it refuses. A single profiled optimizer step raises an internal assert from the Python stack replay, reproducibly, at n equal to 1 as well as 50, while the same flag over a plain forward and backward with no optimizer succeeds. Treat it as a version fact about torch 2.2.2 with CPython 3.12, not as a statement about the flag.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU, CPython 3.12.0): the observer cost through Timer, and the flag that refuses; the CI build path inside the assert message is elided, nothing else is',
              lang: 'text',
              text: '$ python3 observer.py       # Timer around the same step, three rounds, one script\nrun 1  plain  1.420 ms   under profile()  1.838 ms   ratio  1.29x\nrun 2  plain  1.427 ms   under profile()  1.682 ms   ratio  1.18x\nrun 3  plain  1.619 ms   under profile()  1.959 ms   ratio  1.21x\n\n$ python3 with_stack.py     # one optimizer step under with_stack=True\npython 3.12.0 | torch 2.2.2\nn=1: RuntimeError: !stack.empty() INTERNAL ASSERT FAILED at ".../torch/csrc/autograd/profiler_python.cpp":969, please report a bug to PyTorch. Python replay stack is empty.\nn=5: RuntimeError: !stack.empty() INTERNAL ASSERT FAILED at ".../torch/csrc/autograd/profiler_python.cpp":969, please report a bug to PyTorch. Python replay stack is empty.\nn=50: RuntimeError: !stack.empty() INTERNAL ASSERT FAILED at ".../torch/csrc/autograd/profiler_python.cpp":969, please report a bug to PyTorch. Python replay stack is empty.',
            },
          },
          {
            h: 'the biggest allocation in the step has a shape',
            ps: [
              "With `record_shapes=True` the same capture can be grouped by input shape, and with `profile_memory=True` it carries bytes. Together they answer a question the flat table cannot: which single call allocated the most, and for which operand.",
              "The winner in this step is an `aten::mm` on `[[512, 64], [64, 512]]`, 5.00 Mb over five steps, one megabyte per step. Read the shapes and it names itself. That is the transposed activation times the incoming gradient, which is the weight gradient of the first linear, and it is larger than the activations because the weight is 512 by 512 while the batch is only 64 rows.",
              "Two rows carry negative memory and they are not errors. `Optimizer.zero_grad#SGD.zero_grad` shows -4.09 Mb self, because `set_to_none=True` drops the gradient tensors and the profiler accounts the release to whoever released it. The `train_step` annotation nets out at -652.54 Kb for the same reason: over a whole step, this loop frees slightly more than it keeps.",
              "None of these bytes are device bytes. The columns say CPU Mem, they come from the CPU allocator, and the third lesson is about why the CUDA equivalents on this machine answer zero rather than refusing.",
            ],
            code: {
              caption: 'the same capture again, printed with key_averages(group_by_input_shape=True).table(sort_by="self_cpu_memory_usage", row_limit=6); verbatim',
              lang: 'text',
              text: '-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------------------------------------  \n                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls                                Input Shapes  \n-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------------------------------------  \n                                               aten::mm        12.08%       1.198ms        12.08%       1.198ms     239.600us       5.00 Mb       5.00 Mb             5                      [[512, 64], [64, 512]]  \n                                            aten::addmm        13.89%       1.377ms        14.83%       1.471ms     294.200us     640.00 Kb     640.00 Kb             5      [[512], [64, 512], [512, 512], [], []]  \n                                        aten::clamp_min         0.91%      90.000us         0.91%      90.000us      18.000us     640.00 Kb     640.00 Kb             5                             [[64, 512], []]  \n                                               aten::mm         1.04%     103.000us         1.04%     103.000us      20.600us     640.00 Kb     640.00 Kb             5                       [[64, 10], [10, 512]]  \n                               aten::threshold_backward         1.11%     110.000us         1.11%     110.000us      22.000us     640.00 Kb     640.00 Kb             5                  [[64, 512], [64, 512], []]  \n                                               aten::mm         1.58%     157.000us         1.58%     157.000us      31.400us     100.00 Kb     100.00 Kb             5                       [[10, 64], [64, 512]]  \n-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------------------------------------  \nSelf CPU time total: 9.916ms',
            },
          },
          {
            h: 'every backward event carries a forward sequence number',
            ps: [
              "Export the capture with `export_chrome_trace` and it is plain JSON, so you can check what landed without opening a viewer. This one holds 784 events over five steps: 540 operator events, 20 annotations, 146 memory events from `profile_memory`, and 70 events in a category called `fwdbwd`.",
              "Those 70 are flow events, and they exist to answer the question a flat table cannot. Every operator event carries a `Sequence number` in its args, and the backward node that consumes it carries the same one. Sequence 29 holds `aten::addmm` with `Fwd thread id` 0 and `AddmmBackward0` with `Fwd thread id` 1, which is the pairing drawn as an arrow when the trace is opened in Perfetto.",
              "That integer is not new here. It is the same `_sequence_nr()` the autograd lesson at /pytorch/autograd/the-node-behind-grad-fn reads off a graph node, seen from the other end: the tape assigns it when the forward op runs, and the profiler stamps it on both events so a trace can be walked back into your own code.",
              ">> A backward row you cannot attribute is a forward op you have not looked up yet.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the tail of the same script, reading the trace it just exported',
              lang: 'python',
              text: '# the tail of the capture script from the first section, same process\nimport collections\nimport json\n\nprof.export_chrome_trace("trace.json")\nev = json.load(open("trace.json"))["traceEvents"]\nprint("trace events:", len(ev), "| by category:", dict(collections.Counter(e.get("cat") for e in ev)))\n\npairs = collections.defaultdict(list)\nfor e in ev:\n    if e.get("cat") == "cpu_op" and e.get("args", {}).get("Sequence number", -1) >= 0:\n        pairs[e["args"]["Sequence number"]].append((e["name"], e["args"]["Fwd thread id"]))\nfor seq in sorted(pairs)[:3]:\n    print("seq", seq, pairs[seq])\n\n# ---- stdout ----\n# trace events: 784 | by category: {\'user_annotation\': 20, \'cpu_op\': 540, \'fwdbwd\': 70, \'cpu_instant_event\': 146, None: 7, \'Trace\': 1}\n# seq 28 [(\'aten::linear\', 0), (\'aten::t\', 0), (\'autograd::engine::evaluate_function: TBackward0\', 1), (\'TBackward0\', 1)]\n# seq 29 [(\'aten::addmm\', 0), (\'autograd::engine::evaluate_function: AddmmBackward0\', 1), (\'AddmmBackward0\', 1)]\n# seq 30 [(\'aten::relu\', 0), (\'autograd::engine::evaluate_function: ReluBackward0\', 1), (\'ReluBackward0\', 1)]',
            },
          },
        ],
        readings: [
          { label: 'torch.profiler reference', url: 'https://docs.pytorch.org/docs/2.2/profiler.html', note: 'every argument this lesson set, and the ones it did not' },
          { label: 'profiler.py at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/profiler/profiler.py', note: 'schedule as a plain function of the step index, at lines 318 to 350, and the four actions above it' },
          { label: 'Perfetto UI', url: 'https://ui.perfetto.dev/', note: 'where the exported json opens; the fwdbwd events are the arrows between forward and backward' },
        ],
        check: [
          {
            q: 'The top row of your profile is a record_function annotation, not an operator. What is in it?',
            a: 'Host work that ran between ATen calls: Python, module __call__, the autograd engine, optimizer bookkeeping. Self time excludes children, so that row is the host-side gap. In this capture it was 3.094 ms over five steps, 31 percent of the recorded step, and it stayed between 28 and 32 percent across five runs.',
          },
          {
            q: 'Why should you take shares of time from the profiler but not milliseconds?',
            a: 'Because recording costs a fifth to a quarter of this step: Timer put the ratio at 1.18 to 1.29, and the profiled captures reported 1.638 to 2.377 ms per step against an unwatched median of 1.420 to 1.619. Proportions survive that inflation, absolute numbers do not.',
          },
          {
            q: 'A backward row dominates your trace and you cannot tell which layer it came from. What do you look up?',
            a: 'The Sequence number in the event args. The forward op and its backward node share it, so sequence 29 pairs aten::addmm with AddmmBackward0 here. It is the same _sequence_nr the autograd lesson reads off a graph node, stamped onto both trace events.',
          },
        ],
        work: [
          { id: 'name-your-gap', label: 'capture five steady steps of your own loop, then write down the annotation self time as a share of the step and name three things inside it', href: '#the-largest-row-in-the-table-is-not-an-op' },
          { id: 'largest-allocation', label: 'group your own capture by input shape and identify the single largest allocation, then say from its shapes which tensor it is', href: '#the-biggest-allocation-in-the-step-has-a-shape' },
        ],
      },
      {
        id: 'instruments-that-answer-nothing-here',
        num: 3,
        title: 'The instruments that answer nothing here',
        lede: 'Ask this machine how much CUDA memory it is using and it says zero bytes. It has no CUDA. Half the performance API of PyTorch answers politely on hardware it cannot see, and knowing which half is the difference between a measurement and a decoration.',
        goal: 'Predict what each CUDA-only instrument does on a machine with none, state what an overlapped host-to-device copy actually requires, say which dtype autocast picks per op and why the CPU table is not the CUDA one, and read allocated bytes against reserved bytes.',
        sections: [
          {
            h: 'three ways an absent device answers',
            ps: [
              "Run every CUDA performance call on this laptop and the failures sort into three kinds. `torch.cuda.memory_allocated()` returns 0. `torch.cuda.synchronize()` raises. `torch.cuda.amp.GradScaler()` warns once and disables itself, then behaves like an object that works.",
              "The first kind is the dangerous one. A memory report that prints 0 bytes allocated and 0 reserved looks exactly like a program with nothing on the device, and `memory_stats()` returning an empty dictionary is what makes `memory_summary()` raise a `KeyError` a line later rather than say anything useful. Nothing in that sequence tells you the answer was never about your program.",
              "So gate on `torch.cuda.is_available()` yourself, in the reporting code, and print the gate's result next to the numbers. A performance report from a CPU-only box that quietly contains a CUDA memory section is not wrong in its arithmetic. It is answering a question nobody asked.",
              ">> An instrument that cannot see your device will still print a number.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): every CUDA-only instrument this arc names, probed on a machine with no CUDA',
              lang: 'text',
              text: 'torch 2.2.2 | cuda available: False\nmemory_allocated()                   0\nmax_memory_allocated()               0\nmemory_reserved()                    0\nmemory_stats()                       OrderedDict()\nmemory_summary()                     KeyError: \'allocated_bytes.all.current\'\nsynchronize()                        AssertionError: Torch not compiled with CUDA enabled\nStream()                             RuntimeError: Tried to instantiate dummy base class Stream\nEvent(enable_timing=True)            RuntimeError: Tried to instantiate dummy base class Event\nGradScaler().is_enabled()            False  [warned: torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling.]\nautocast(cuda) on a matmul           torch.float32  [warned: User provided device_type of \'cuda\', but CUDA is not available. Disabling]\nprofile(CPU and CUDA)                \'ran, no CUDA columns\'  [warned: CUDA is not available, disabling CUDA profiling]',
              full: {
                text: 'import warnings\n\nimport torch\nfrom torch.profiler import ProfilerActivity, profile\n\ndef probe(label, f):\n    with warnings.catch_warnings(record=True) as w:\n        warnings.simplefilter("always")\n        try:\n            out = repr(f())\n        except Exception as err:\n            out = f"{type(err).__name__}: {err}"\n    note = f"  [warned: {w[0].message}]" if w else ""\n    print(f"{label:36s} {out}{note}")\n\ndef cuda_activity():\n    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:\n        (torch.randn(64, 64) @ torch.randn(64, 64)).sum()\n    return "CUDA columns" if "Self CUDA" in p.key_averages().table(row_limit=1) else "ran, no CUDA columns"\n\ndef autocast_matmul():\n    with torch.autocast("cuda", dtype=torch.float16):\n        return (torch.ones(4, 4) @ torch.ones(4, 4)).dtype\n\nprint("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())\nprobe("memory_allocated()", torch.cuda.memory_allocated)\nprobe("max_memory_allocated()", torch.cuda.max_memory_allocated)\nprobe("memory_reserved()", torch.cuda.memory_reserved)\nprobe("memory_stats()", torch.cuda.memory_stats)\nprobe("memory_summary()", lambda: torch.cuda.memory_summary()[:40])\nprobe("synchronize()", torch.cuda.synchronize)\nprobe("Stream()", torch.cuda.Stream)\nprobe("Event(enable_timing=True)", lambda: torch.cuda.Event(enable_timing=True))\nprobe("GradScaler().is_enabled()", lambda: torch.cuda.amp.GradScaler().is_enabled())\nprobe("autocast(cuda) on a matmul", autocast_matmul)\nprobe("profile(CPU and CUDA)", cuda_activity)',
                label: 'the probe script the table above came from',
              },
            },
          },
          {
            h: 'the stream is the order, the event is the clock',
            ps: [
              "A CUDA stream is a queue with one rule: operations inside it run in the order they were issued. Every device has a default stream, and if you never make another one, everything you have ever written is already ordered against itself, which is why the async dispatch the chapter describes is safe by default rather than a race.",
              "Two streams have no such rule between them. The docs' own broken example queues a `normal_()` on the default stream and a `sum` on a side stream, and the sum may start before the fill finishes. Fixing it takes two calls: `wait_stream` makes the side stream wait for what the default stream already queued, and `record_stream` keeps the source tensor alive until the side stream is done with it. The second one is easy to forget because the failure it prevents is the caching allocator handing your buffer to somebody else while a kernel is still reading it.",
              "For timing, the same asynchrony means the host clock is measuring the wrong thing, and CUDA events are the device-side answer. Record an event before and after, synchronize once, then ask for `elapsed_time` between them, and the interval you get was measured by the device rather than by Python. Lesson one's Timer does the coarser version of this for you by synchronizing before each read.",
              "None of this ran here. Both `torch.cuda.Stream()` and `torch.cuda.Event()` raise on this machine, so the block below is the pinned source rather than a capture. No lab on this site measures stream overlap either, because the torch labs run on a plain CPU or on a TPU and a stream is a CUDA object. That leaves the measurement to you, on the first CUDA device you get hold of; the exercises below say what to record.",
            ],
            code: {
              caption: 'verbatim, docs/source/notes/cuda.rst at the pytorch v2.2.2 tag: the event timing recipe, the broken two-stream example, and the fixed one',
              lang: 'python',
              text: '# lines 229-237: timing with events instead of the host clock\nstart_event = torch.cuda.Event(enable_timing=True)\nend_event = torch.cuda.Event(enable_timing=True)\nstart_event.record()\n\n# Run some things here\n\nend_event.record()\ntorch.cuda.synchronize()  # Wait for the events to be recorded!\nelapsed_time_ms = start_event.elapsed_time(end_event)\n\n# lines 257-262: the docs call this one incorrect\ncuda = torch.device("cuda")\ns = torch.cuda.Stream()  # Create a new stream.\nA = torch.empty((100, 100), device=cuda).normal_(0.0, 1.0)\nwith torch.cuda.stream(s):\n    # sum() may start execution before normal_() finishes!\n    B = torch.sum(A)\n\n# lines 270-275: the fixed version\ncuda = torch.device("cuda")\ns = torch.cuda.Stream()  # Create a new stream.\nA = torch.empty((100, 100), device=cuda).normal_(0.0, 1.0)\ns.wait_stream(torch.cuda.default_stream(cuda))  # NEW!\nwith torch.cuda.stream(s):\n    B = torch.sum(A)\nA.record_stream(s)  # NEW!',
            },
          },
          {
            h: 'what an overlapped copy actually needs',
            ps: [
              "The reason to own a second stream at all is usually a copy. A host-to-device transfer can run while the device computes, but only if three things are true at once: the host buffer is page-locked, the copy is issued with `non_blocking=True`, and the copy is not queued behind the compute it was supposed to overlap with.",
              "Page-locking is the loader's job and it has its own home. What `pin_memory=True` does, what it silently does not do on a machine with no accelerator, and the thread it starts when it does apply are told at /pytorch/data/same-order-different-numbers. The half that belongs here is that pinning alone buys nothing: an unpinned copy with `non_blocking=True` is a synchronous copy wearing an argument.",
              "You can watch the argument mean nothing right now. A host-to-host copy accepts `non_blocking=True`, returns an unpinned tensor, and warns about none of it, because there is no device for the transfer to be asynchronous with respect to. Every flag in this section behaves that way here, accepted and inert, which is what the first section of this lesson sorted into three kinds.",
              "One asymmetry is worth carrying forward for when you do have a device. Backward ops run on the stream their forward op ran on, so a forward that splits work across streams gets a backward that splits the same way without you arranging it, and gradients produced inside a stream context must be consumed inside it or after an explicit wait.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the copy flag that means nothing here, quoted with the pinned rule it belongs to',
              lang: 'python',
              text: 'import torch\n\nx = torch.ones(1024)\ny = x.to("cpu", non_blocking=True)\nprint("cpu to cpu, non_blocking:", tuple(y.shape), y.dtype, "| is_pinned:", y.is_pinned())\n\n# ---- stdout ----\n# cpu to cpu, non_blocking: (1024,) torch.float32 | is_pinned: False\n\n# and the rule it is half of, verbatim from docs/source/notes/cuda.rst at v2.2.2:\n#\n#   Host to GPU copies are much faster when they originate from pinned\n#   (page-locked) memory. [...] Once you pin a tensor or storage, you can use\n#   asynchronous GPU copies. Just pass an additional non_blocking=True argument\n#   to a to() or a cuda() call.',
            },
          },
          {
            h: 'autocast is two lists and a pass-through rule',
            ps: [
              "Under `torch.autocast`, an op is not asked what dtype it wants. It is looked up in two lists. One list casts its inputs down to the low-precision dtype, one casts them up to float32, and anything in neither list runs in whatever dtype its inputs already have. That third case is most ops, and it is why a bf16 value flows onward through a chain of unlisted ops without anyone deciding it should.",
              "Run the lookup on this machine with `torch.autocast(\"cpu\", dtype=torch.bfloat16)` and the shape is visible in one table. Matmul, linear and conv come back bfloat16. `mse_loss` and `prod` come back float32. `relu` comes back float32 when handed a float32 input and bfloat16 when handed the output of a matmul, because relu is in neither list.",
              "The CPU lists and the CUDA lists are different documents, and that is where an audit run here stops transferring. `log_softmax`, `softmax`, `sum` and `layer_norm` are all in the CUDA float32 list and in neither CPU list, so the same script under autocast produces float32 for those on a GPU and bfloat16 here. A dtype audit run on CPU is a real audit of the CPU table and tells you nothing certain about the other one.",
              "The weights never change. `lin.weight.dtype` is still float32 during and after the region, because autocast casts at the operator boundary rather than converting your parameters, and it caches that cast so a weight used repeatedly inside one region is converted once. Profile five calls of one Linear under autocast and there are seven `aten::_to_copy` calls: five inputs, plus the weight and bias converted once between them.",
            ],
            table: {
              caption: 'measured under torch.autocast("cpu", dtype=torch.bfloat16) on this machine (verified, torch 2.2.2 CPU); the CUDA column is the op lists published in the torch.amp reference for 2.2, not a measurement',
              cols: ['call', 'dtype out, CPU autocast', 'listed for CUDA autocast'],
              rows: [
                ['x @ w', 'bfloat16', 'float16 list'],
                ['F.conv2d', 'bfloat16', 'float16 list'],
                ['F.linear', 'bfloat16', 'float16 list'],
                ['relu, float32 in', 'float32', 'unlisted, passes through'],
                ['relu, bfloat16 in', 'bfloat16', 'unlisted, passes through'],
                ['sum', 'bfloat16', 'float32 list'],
                ['softmax', 'bfloat16', 'float32 list'],
                ['log_softmax', 'bfloat16', 'float32 list'],
                ['layer_norm', 'bfloat16', 'float32 list'],
                ['mse_loss', 'float32', 'float32 list'],
                ['prod', 'float32', 'float32 list'],
              ],
            },
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the lookup, and the cast count that proves the weight is converted once per region',
              lang: 'python',
              text: 'import torch\nimport torch.nn as nn\nfrom torch.profiler import ProfilerActivity, profile\n\nlin = nn.Linear(512, 512)\nx = torch.randn(64, 512)\n\nwith torch.autocast("cpu", dtype=torch.bfloat16):\n    lin(x)                       # warm the cast cache outside the capture\n\nwith profile(activities=[ProfilerActivity.CPU]) as prof:\n    with torch.autocast("cpu", dtype=torch.bfloat16):\n        for _ in range(5):\n            lin(x)\n\ncounts = {e.key: e.count for e in prof.key_averages()}\nprint("aten::addmm     ", counts["aten::addmm"])\nprint("aten::_to_copy  ", counts["aten::_to_copy"])\nprint("weight dtype    ", lin.weight.dtype)\n\n# ---- stdout ----\n# aten::addmm      5\n# aten::_to_copy   7\n# weight dtype     torch.float32',
              full: {
                text: 'import torch\nimport torch.nn.functional as F\n\nx = torch.randn(8, 64)\nw = torch.randn(64, 64)\nimg = torch.randn(2, 3, 16, 16)\nk = torch.randn(4, 3, 3, 3)\n\ncases = {\n    "x @ w": lambda: x @ w,\n    "F.conv2d": lambda: F.conv2d(img, k),\n    "F.linear": lambda: F.linear(x, w),\n    "relu, float32 in": lambda: torch.relu(x),\n    "relu, bfloat16 in": lambda: torch.relu(x @ w),\n    "sum": lambda: (x @ w).sum(),\n    "softmax": lambda: torch.softmax(x @ w, -1),\n    "log_softmax": lambda: torch.log_softmax(x @ w, -1),\n    "layer_norm": lambda: F.layer_norm(x @ w, (64,)),\n    "mse_loss": lambda: F.mse_loss(x @ w, torch.randn(8, 64)),\n    "prod": lambda: (x @ w).prod(),\n}\nwith torch.autocast("cpu", dtype=torch.bfloat16):\n    for name, f in cases.items():\n        print(f"{name:20s} {f().dtype}")\n\n# ---- stdout ----\n# x @ w                torch.bfloat16\n# F.conv2d             torch.bfloat16\n# F.linear             torch.bfloat16\n# relu, float32 in     torch.float32\n# relu, bfloat16 in    torch.bfloat16\n# sum                  torch.bfloat16\n# softmax              torch.bfloat16\n# log_softmax          torch.bfloat16\n# layer_norm           torch.bfloat16\n# mse_loss             torch.float32\n# prod                 torch.float32',
                label: 'the lookup script the table above came from',
              },
            },
          },
          {
            h: 'a dtype is only fast where the arithmetic is',
            ps: [
              "Autocast is a dtype policy, not a speedup. Whether the policy pays depends entirely on whether the hardware has arithmetic for the dtype it picked, and this machine is a clean demonstration of the failure case: an Intel CPU whose top vector capability is AVX2, with no bfloat16 instructions underneath.",
              "The same Linear that runs in 0.192 milliseconds in float32 takes 8.198 under `autocast(\"cpu\", dtype=torch.bfloat16)`. Casting the module and the input to bfloat16 by hand, with no autocast anywhere, takes 7.783, so the casts are a small part of it and the arithmetic path is the rest. Between thirty and forty times slower, on a change that a GPU or TPU benchmark would report as a speedup.",
              "Nothing about that contradicts the chapter. It is the same claim from the other side: the dtype is fast where the tensor cores or MXUs implement it, and this box implements none of them. State the chip whenever you quote a dtype result, because a bf16 number without a chip beside it is not interpretable at all.",
              "The scaler behaves the same way. `GradScaler` exists for float16's narrow exponent range, and bfloat16 does not need it. On this machine it also does not run: constructing one warns and disables it, `get_scale()` reads 1.0, and `scaler.scale(loss).backward()` produces exactly the gradients an unscaled backward would. Code written that way is not wrong here, it is inert here, and a CPU run proves nothing about whether your scaler is doing its job.",
            ],
            table: {
              caption: 'one nn.Linear(512, 512) on a batch of 64, three dtype paths, Timer medians at 8 threads (verified, torch 2.2.2 CPU, AVX2, no bf16 arithmetic); two consecutive runs',
              cols: ['path', 'run 1 median', 'run 2 median'],
              rows: [
                ['float32', '0.192 ms', '0.289 ms'],
                ['autocast bfloat16', '8.198 ms', '9.264 ms'],
                ['module and input cast to bfloat16', '7.783 ms', '8.460 ms'],
              ],
            },
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the scaler on a machine with no CUDA, warning quoted as printed',
              lang: 'python',
              text: 'import warnings\n\nimport torch\n\nwith warnings.catch_warnings(record=True) as w:\n    warnings.simplefilter("always")\n    scaler = torch.cuda.amp.GradScaler()\n    print("warnings:", [str(x.message) for x in w])\nprint("enabled:", scaler.is_enabled(), "| scale:", scaler.get_scale())\n\nlin = torch.nn.Linear(4, 4)\nopt = torch.optim.SGD(lin.parameters(), lr=0.1)\nloss = lin(torch.randn(2, 4)).sum()\nscaler.scale(loss).backward()\nprint("bias grad:", lin.bias.grad)\nscaler.step(opt)\nscaler.update()\nprint("new scale:", scaler.get_scale())\n\n# ---- stdout ----\n# warnings: [torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling.]\n# enabled: False | scale: 1.0\n# bias grad: tensor([2., 2., 2., 2.])\n# new scale: 1.0',
            },
          },
          {
            h: 'allocated, reserved, and the number the driver shows',
            ps: [
              "PyTorch does not hand memory back to CUDA when your tensor dies. It keeps it, in a caching allocator, so the next allocation of that size costs nothing and no device synchronization is needed to free anything. Two numbers follow from that, and confusing them is the most common way a memory report becomes fiction.",
              "`memory_allocated` is what your live tensors hold. `memory_reserved` is what the allocator has taken from the driver, cached blocks included, and it is the number `nvidia-smi` shows. A run whose reserved figure is far above its allocated figure is not leaking; it is holding a cache. `empty_cache()` returns the unused part of that cache to the driver and changes nothing about what your tensors hold.",
              "`memory_stats()` is where the diagnosis lives, and four of its keys are worth knowing before you need them. `requested_bytes` against `allocated_bytes` shows what allocation rounding costs you. `num_alloc_retries` counts the times a `cudaMalloc` failed and forced a cache flush, which is fragmentation showing itself before the crash does. `num_ooms` counts the crashes. Each core statistic carries current, peak, allocated and freed, so the peak is available without you polling for it.",
              "For the question of which line allocated what, the snapshot tools are the instrument: `_record_memory_history` turns on stack capture per allocation, `_dump_snapshot` writes a pickle, and the viewer at pytorch.org/memory_viz renders it locally. On this machine every one of those calls answers about a device that does not exist, so that workflow waits for a CUDA device too. The CPU substitute you do have is the profiler's memory columns from lesson two, which answer the same shape of question about the host allocator.",
            ],
            table: {
              caption: 'which instrument answers which memory question, and what has to be true first; the CUDA rows are cited from torch/cuda/memory.py:165-232 and docs/source/notes/cuda.rst at v2.2.2, not measured here',
              cols: ['the question', 'the instrument', 'available on this machine'],
              rows: [
                ['what did this op allocate on the host', 'profile(profile_memory=True), grouped by shape', 'yes, lesson two'],
                ['what do my live device tensors hold', 'torch.cuda.memory_allocated()', 'no, answers 0'],
                ['what has the allocator taken from the driver', 'torch.cuda.memory_reserved()', 'no, answers 0'],
                ['what was the peak, without polling', 'memory_stats()["allocated_bytes.all.peak"]', 'no, empty dict'],
                ['is rounding wasting memory', 'requested_bytes against allocated_bytes', 'no, empty dict'],
                ['is fragmentation about to kill the run', 'num_alloc_retries, num_ooms', 'no, empty dict'],
                ['which line allocated this block', '_record_memory_history plus _dump_snapshot', 'no, needs a CUDA device'],
              ],
            },
            code: {
              caption: 'verbatim, torch/cuda/memory.py:173-181 and :210-212 and :230-232 at torch 2.2.2, byte-identical to the v2.2.2 tag',
              lang: 'text',
              text: '- ``"allocated.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:\n  number of allocation requests received by the memory allocator.\n- ``"allocated_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:\n  amount of allocated memory.\n- ``"segment.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:\n  number of reserved segments from ``cudaMalloc()``.\n- ``"reserved_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:\n  amount of reserved memory.\n\n- ``"num_alloc_retries"``: number of failed ``cudaMalloc`` calls that\n  result in a cache flush and retry.\n- ``"num_ooms"``: number of out-of-memory errors thrown.\n\n- ``"requested_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:\n  memory requested by client code, compare this with allocated_bytes to check if\n  allocation rounding adds too much overhead.',
            },
          },
        ],
        readings: [
          { label: 'CUDA semantics', url: 'https://docs.pytorch.org/docs/2.2/notes/cuda.html', note: 'streams, events, pinned copies and the caching allocator, all in one page; the source of every quote in this lesson' },
          { label: 'torch.amp', url: 'https://docs.pytorch.org/docs/2.2/amp.html', note: 'the four op lists, per device, and the sentence that unlisted ops run in the type of their inputs' },
          { label: 'memory.py at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/cuda/memory.py', note: 'every memory_stats key with its meaning, and what reset_peak_memory_stats resets' },
          { label: 'Understanding CUDA memory usage', url: 'https://docs.pytorch.org/docs/2.2/torch_cuda_memory.html', note: 'the snapshot workflow to run when you have a device: record history, dump, open the viewer locally' },
        ],
        check: [
          {
            q: 'Your memory report says 0 bytes allocated and 0 reserved. What are the two possible readings?',
            a: 'Either the program holds nothing on the device, or there is no device. torch.cuda.memory_allocated() and memory_reserved() both return 0 with no CUDA present, and memory_stats() returns an empty dict, which is what makes memory_summary() raise a KeyError. Print torch.cuda.is_available() beside the numbers.',
          },
          {
            q: 'You pinned the host buffer and passed non_blocking=True, and nothing overlapped. What else is required?',
            a: 'That the copy is not queued behind the work it should overlap with. A stream orders everything inside it, so the copy needs its own stream, with wait_stream to order it against what came before and record_stream to keep the source alive until the copy finishes.',
          },
          {
            q: 'A dtype audit under autocast on CPU says sum returns bfloat16. What does that tell you about CUDA?',
            a: 'Nothing directly. The op lists are per device: sum, softmax, log_softmax and layer_norm are in the CUDA float32 list and in neither CPU list, so the same script yields float32 for them on a GPU and bfloat16 here. The audit is valid for the table it ran against.',
          },
        ],
        work: [
          { id: 'gate-your-report', label: 'take any performance report you have written and add the gate: print torch.cuda.is_available() beside every device number it quotes', href: '#three-ways-an-absent-device-answers' },
          { id: 'overlap-conditions', label: 'on a CUDA device, copy a pinned buffer and an unpinned one with non_blocking=True on a side stream, time both with CUDA events, and name which of the three conditions the slow copy missed', href: '#what-an-overlapped-copy-actually-needs' },
          { id: 'dtype-audit', label: 'run one training step of your own under autocast, list every op whose output dtype changed, and mark which entries would differ on CUDA', href: '#autocast-is-two-lists-and-a-pass-through-rule' },
          { id: 'reserved-ledger', label: 'on the first device you get hold of, print allocated against reserved either side of one training step and say what the gap is', href: '#allocated-reserved-and-the-number-the-driver-shows' },
        ],
      },
    ],
  },
]
