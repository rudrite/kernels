// New file: site/src/data/lessons/torch-data.ts
// The input pipeline below the survey depth chapter 5 teaches: the fetch path
// and what collate does to types, the process boundary and what crossing it
// costs, and what workers change about seeds, threads and pinned memory. Every
// printed value ran on the course machine (torch 2.2.2, CPU, 2026-08-14); every
// source excerpt is verbatim from that install, whose four data-loading files
// are byte-identical to the pytorch v2.2.2 tag. No accelerator was involved,
// and the pinned-memory section says what that costs the claim.
import type { UnitLessons } from './index'

export const TORCH_DATA_LESSONS: UnitLessons[] = [
  {
    unit: 'pt:data',
    lessons: [
      {
        id: 'one-index-at-a-time',
        num: 1,
        title: 'One index at a time',
        lede: 'Your Dataset is never handed a batch. It is handed one integer, then another, and a function most people have never opened turns the pile of results into the tensors the step receives.',
        goal: 'Given a DataLoader and a Dataset that returns a tuple, name the three objects between the for-loop and __getitem__, say how many __getitem__ calls one batch costs, and predict the dtype of every field of the collated batch before running it.',
        sections: [
          {
            h: 'three objects between the loop and getitem',
            ps: [
              "The chapter above this lesson names two objects, a Dataset and a DataLoader, which is the right size for a survey. Build the loader and a third one is sitting on it in plain sight. `dl.sampler` yields single indices. `dl.batch_sampler` wraps that sampler and yields lists of indices. `dl._index_sampler` is whichever of the two the loader will actually iterate, and the choice between them is made once, at construction.",
              "Watch what a Dataset receives and the division of labour is unambiguous. Four indices go out, four separate `__getitem__` calls come back, and the stacking happens somewhere else entirely. Your dataset never learns what `batch_size` is.",
              "That sounds like a detail until you try to make a dataset faster. Caching per batch, reading a contiguous slab off disk once, batching a database query: none of it fits, because the only question your dataset is asked is about one index.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the sampler chain, and the four calls one batch costs',
              lang: 'python',
              text: 'import torch\nfrom torch.utils.data import DataLoader, Dataset\n\nclass Counted(Dataset):\n    def __init__(self, n): self.n, self.asked = n, []\n    def __len__(self): return self.n\n    def __getitem__(self, i):\n        self.asked.append(i)\n        return torch.full((3,), float(i))\n\nds = Counted(10)\ndl = DataLoader(ds, batch_size=4)\nit = iter(dl)\nprint("sampler        ", type(dl.sampler).__name__)\nprint("batch_sampler  ", type(dl.batch_sampler).__name__)\nprint("_index_sampler ", type(dl._index_sampler).__name__)\nprint("auto_collation ", dl._auto_collation)\nprint("collate_fn     ", dl.collate_fn.__name__)\nb = next(it)\nprint("asked for      ", ds.asked)\nprint("batch shape    ", tuple(b.shape))\n\n# ---- stdout ----\n# sampler         SequentialSampler\n# batch_sampler   BatchSampler\n# _index_sampler  BatchSampler\n# auto_collation  True\n# collate_fn      default_collate\n# asked for       [0, 1, 2, 3]\n# batch shape     (4, 3)',
            },
          },
          {
            h: 'the fetcher is nine lines, and one branch matters',
            ps: [
              "Everything the last section showed lives in one small class. `_MapDatasetFetcher.fetch` takes the list of indices, runs the list comprehension, and hands the result to `collate_fn`. There is no buffering, no reordering and no threading in it. Read it once and the single-process path stops being mysterious.",
              "The branch above the comprehension is the escape hatch for exactly the problem the last section named. If your dataset defines `__getitems__`, plural, the fetcher calls it once with the whole index list instead of calling `__getitem__` per index. That is the hook a database-backed or memory-mapped dataset wants, and `Subset` already implements it, which is why wrapping a dataset in `Subset` does not silently lose the fast path.",
              "The other subclass in the same file handles iterable-style datasets and does something different enough to be worth naming now. It ignores the indices entirely and calls `next()` on a stored iterator that many times. The third lesson in this arc is about what that costs once workers exist.",
            ],
            code: {
              caption: 'verbatim, torch/utils/data/_utils/fetch.py:45-54 at torch 2.2.2; the batched-fetch run underneath it is from this machine',
              lang: 'python',
              text: 'class _MapDatasetFetcher(_BaseDatasetFetcher):\n    def fetch(self, possibly_batched_index):\n        if self.auto_collation:\n            if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:\n                data = self.dataset.__getitems__(possibly_batched_index)\n            else:\n                data = [self.dataset[idx] for idx in possibly_batched_index]\n        else:\n            data = self.dataset[possibly_batched_index]\n        return self.collate_fn(data)\n\n# a Dataset that also defines __getitems__, batch_size=4, stdout:\n# calls: [(\'getitems\', [0, 1, 2, 3])] -> batch [0, 1, 2, 3]\n# Subset has __getitems__: True',
            },
          },
          {
            h: 'turning auto_collation off moves the batching into your dataset',
            ps: [
              "`auto_collation` is a property, not an argument, and it is true exactly when the loader has a batch sampler. Pass `batch_size=None` and it goes false, the batch sampler disappears, and `_index_sampler` becomes the plain sampler again.",
              "The consequence lands inside `fetch`, on the else branch. With auto-collation off, the fetcher calls `self.dataset[idx]` with a single index and passes the result straight to `collate_fn`, which now defaults to `default_convert` rather than `default_collate`. Batching has not been turned off; it has been handed to you. This is the mode an iterable dataset that already yields whole batches wants, and it is also the mode people land in by accident when they set `batch_size=None` expecting the loader to guess.",
              "One measured detail from the same run is worth carrying into the next section. A Dataset returning `(tensor, float)` comes back as a float32 input and a float64 label, and nothing in the loader warned about the mismatch.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): batch_size=None, and the label dtype nobody asked for',
              lang: 'python',
              text: 'import torch\nfrom torch.utils.data import DataLoader, Dataset\n\nclass Pairs(Dataset):\n    def __len__(self): return 6\n    def __getitem__(self, i): return torch.full((2,), float(i)), float(i)\n\ndl = DataLoader(Pairs(), batch_size=3)\nxb, yb = next(iter(dl))\nprint("label dtype    ", yb.dtype, "| input dtype", xb.dtype)\n\nclass Asked(Dataset):\n    def __init__(self): self.asked = []\n    def __len__(self): return 6\n    def __getitem__(self, i):\n        self.asked.append(i); return torch.tensor([float(i)])\n\nds = Asked()\ndl0 = DataLoader(ds, batch_size=None)\nprint("batch_size None: auto_collation", dl0._auto_collation, "| index_sampler", type(dl0._index_sampler).__name__)\nb = next(iter(dl0))\nprint("asked          ", ds.asked, "| got shape", tuple(b.shape))\n\n# ---- stdout ----\n# label dtype     torch.float64 | input dtype torch.float32\n# batch_size None: auto_collation False | index_sampler SequentialSampler\n# asked           [0] | got shape (1,)',
            },
          },
          {
            h: 'collate walks types, it does not stack',
            ps: [
              "`default_collate` is not a stacking function with special cases bolted on. It is a recursive type walk over a registry, and the registry is a public dictionary you can read at runtime. Look up the type of the first element; if the registry has a handler, call it; otherwise fall through to the structural cases, which recurse.",
              "The structural cases are where the shape of your batch is decided. A mapping is rebuilt as the same mapping type with each key collated across the batch. A namedtuple keeps its class. A plain sequence gets transposed by `zip(*batch)` before recursing, so a dataset returning `(x, y)` gives you a two-element result whose halves are batched separately.",
              "One of those cases surprises people who read the type annotation and stopped. A tuple sample does not come back as a tuple. The source returns a list, with a comment saying `Backwards compatibility`, so `type(batch)` is `list` however carefully your `__getitem__` built its tuple.",
              "What collate refuses is as informative as what it accepts. Uneven tensor shapes reach `torch.stack` and die there, with the stack error rather than a collate error, which is why ragged-length data needs a `collate_fn` of your own. Dictionaries with different key sets die on a `KeyError` naming the missing key, because the walk indexes every sample by the first sample's keys.",
            ],
            table: {
              caption: 'what default_collate returns per element type; every row run on this machine (torch 2.2.2 CPU)',
              cols: ['a sample field of this type', 'comes back as', 'measured'],
              rows: [
                ['torch.Tensor', 'torch.stack over a new leading axis', 'shape (4, 3) from four (3,) samples'],
                ['int', 'torch.tensor(batch), dtype inferred', 'torch.int64'],
                ['float', 'torch.tensor(batch, dtype=torch.float64)', 'torch.float64'],
                ['bool', 'handled by the int path', 'torch.bool'],
                ['str, bytes', 'returned untouched, still a list', "['a', 'b']"],
                ['dict', 'same mapping type, each key collated', "{'x': (2, 2) float32, 'y': (2,) int64}"],
                ['tuple', 'a list, transposed, each position collated', 'list of [(2, 2), (2,)]'],
                ['list', 'same list type, transposed', '[[1, 3, 5], [2, 4, 6]] from [[1,2],[3,4],[5,6]]'],
              ],
            },
          },
          {
            h: 'the float rule has a name and a line number',
            ps: [
              "The float64 label from two sections ago is not inference and not a bug. It is a two-line function in the registry, and it says the dtype outright rather than letting `torch.tensor` guess.",
              "Trace the consequence forward. A Dataset that returns a Python float per sample produces a float64 target, which meets a float32 prediction at the loss. On this machine `mse_loss` accepted the pair and returned a float64 loss, so the promotion travels silently into the backward pass. The museum's class-targets-as-floats exhibit is the loud version of the same category of mistake; this is the quiet one, and the fix is upstream of both, in what `__getitem__` returns.",
              "The registry is also the extension point. `default_collate_fn_map` is a module-level dict, and the docstring in `collate.py` shows updating it in place to change how a type batches everywhere. Reach for a custom `collate_fn` when the batching logic is local; reach for the map when a type should batch the same way across a whole codebase.",
            ],
            code: {
              caption: 'verbatim, torch/utils/data/_utils/collate.py at torch 2.2.2: lines 190-195, then the registry at 202-213, joined here with a blank line between',
              lang: 'python',
              text: 'def collate_float_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):\n    return torch.tensor(batch, dtype=torch.float64)\n\n\ndef collate_int_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):\n    return torch.tensor(batch)\n\ndefault_collate_fn_map: Dict[Union[Type, Tuple[Type, ...]], Callable] = {torch.Tensor: collate_tensor_fn}\nwith contextlib.suppress(ImportError):\n    import numpy as np\n    # For both ndarray and memmap (subclass of ndarray)\n    default_collate_fn_map[np.ndarray] = collate_numpy_array_fn\n    # See scalars hierarchy: https://numpy.org/doc/stable/reference/arrays.scalars.html\n    # Skip string scalars\n    default_collate_fn_map[(np.bool_, np.number, np.object_)] = collate_numpy_scalar_fn\ndefault_collate_fn_map[float] = collate_float_fn\ndefault_collate_fn_map[int] = collate_int_fn\ndefault_collate_fn_map[str] = collate_str_fn\ndefault_collate_fn_map[bytes] = collate_str_fn',
            },
          },
        ],
        readings: [
          { label: 'fetch.py at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/utils/data/_utils/fetch.py', note: 'the whole single-process fetch path, map-style and iterable-style, in 54 lines' },
          { label: 'collate.py at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/utils/data/_utils/collate.py', note: 'the type walk, the registry, and the docstring table of input type to output type' },
          { label: 'torch.utils.data, the 2.2 reference', url: 'https://docs.pytorch.org/docs/2.2/data.html', note: 'sampler, batch_sampler and the automatic-batching rules, from the side that documents them' },
        ],
        check: [
          {
            q: 'A batch of 64 came out of the loader. How many times was __getitem__ called, and by what?',
            a: 'Sixty-four times, unless the dataset defines __getitems__, in which case once. The caller is _MapDatasetFetcher.fetch, running the list comprehension over the index list the batch sampler produced.',
          },
          {
            q: 'A Dataset returns (tensor, label) where label is a Python float. What dtype is the batched label?',
            a: 'torch.float64. collate_float_fn in the default registry builds it with dtype=torch.float64 explicitly, so a float32 model meets a float64 target and the promotion happens silently at the loss.',
          },
          {
            q: 'What changes inside fetch when you pass batch_size=None?',
            a: "auto_collation goes false, so the fetcher takes the else branch and calls dataset[idx] with a single index rather than a list, and collate_fn defaults to default_convert. Batching becomes your job rather than the loader's.",
          },
        ],
        work: [
          { id: 'index-trace', label: 'take one of your own datasets and trace a single batch from dl._index_sampler through fetch to collate, naming the object at each hop', href: '#three-objects-between-the-loop-and-getitem' },
          { id: 'collate-table', label: 'predict the collated type and dtype of every field your own __getitem__ returns, then check the prediction by printing the batch', href: '#collate-walks-types-it-does-not-stack' },
        ],
      },
      {
        id: 'what-actually-forks',
        num: 2,
        title: 'What actually forks',
        lede: 'A worker is a process, which the chapter says. Which kind of process, started how, holding what, is decided by a platform default you did not set, and almost everything people find surprising about workers follows from it.',
        goal: 'Name the start method your platform uses and what it implies about pickling and startup cost, describe the queue topology between the main process and N workers, and explain how batches arrive in order when workers finish out of order.',
        sections: [
          {
            h: 'the start method decides what has to pickle',
            ps: [
              "The chapter states that a worker has to be able to unpickle the dataset it was handed. That is true here and it is not true everywhere, and the difference is one platform default.",
              "On this machine `multiprocessing.get_start_method()` returns `spawn`, which is the Python default on macOS and Windows. A spawned worker is a fresh interpreter. It imports your module again, unpickles the arguments it was sent, and starts from nothing, so an unpicklable dataset fails before a sample loads and a script without an `if __name__` guard re-runs its own top level.",
              "Hand the same loader a fork context and the failure goes away. A forked worker inherits the parent's memory, so the dataset is never serialized at all; it is simply already there. The run below is the same lambda-carrying dataset twice, once under each start method, and the two results are the honest scope of the rule: what breaks is the start method, not the DataLoader.",
              "Linux defaults to fork, which is why a dataset that runs fine on a training box can fail on a laptop, and why the reverse trap exists too. A forked worker inherits open file handles, database connections and library state that were never designed to be shared, and those bugs disappear the moment someone switches to spawn.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU, macOS): the same unpicklable dataset under both start methods',
              lang: 'python',
              text: 'import torch, torch.multiprocessing as mp\nfrom torch.utils.data import DataLoader, Dataset\n\nclass Lam(Dataset):\n    def __init__(self, f): self.f = f\n    def __len__(self): return 4\n    def __getitem__(self, i): return self.f(torch.tensor([float(i)]))\n\nif __name__ == "__main__":\n    print("default start method:", mp.get_start_method())\n    ds = Lam(lambda t: t * 2)\n    try:\n        print(next(iter(DataLoader(ds, batch_size=2, num_workers=1))))\n    except Exception as e:\n        print(type(e).__name__ + ":", str(e).strip().split("\\n")[-1])\n    ctx = mp.get_context("fork")\n    dl = DataLoader(ds, batch_size=2, num_workers=1, multiprocessing_context=ctx)\n    print("fork + lambda ->", next(iter(dl)).flatten().tolist())\n\n# ---- stdout, the two runs joined ----\n# default start method: spawn\n# PicklingError: Can\'t pickle <function <lambda> at 0x10acc0fe0>: attribute lookup <lambda> on __main__ failed\n# fork + lambda -> [0.0, 2.0]',
            },
          },
          {
            h: 'starting a worker is not free, and the bill is the start method',
            ps: [
              "Overlap is worth paying for, and the payment has a number. Time the first batch out of a loader over a tiny in-memory dataset, so that nothing but process startup is being measured, and the spread across start methods is three orders of magnitude.",
              "A spawned worker pays for a full torch import, which is most of that three seconds. A forked worker pays for a page-table copy. The single-process loader pays for a list comprehension.",
              "Two things follow. Short runs on a spawn platform can spend more time starting workers than loading data, and `persistent_workers=True` is how you stop paying the bill once per epoch instead of once per job. On this machine the same two worker pids served both epochs with the flag on, and four different pids appeared without it.",
              "One caveat about the table, since the course publishes provenance rather than adjectives. This is a laptop that was doing other work while the numbers were taken, and the spread was wide, up to 17.6 seconds for one spawn-with-four-workers run. The minimums are quoted because the ratio between rows is the durable part; the absolute values are not.",
            ],
            table: {
              caption: 'time to the first batch of a 256-sample TensorDataset, 5 runs each, minimum quoted (torch 2.2.2 CPU, Intel i9-9880H, macOS, shared machine)',
              cols: ['start method', 'num_workers', 'first batch, min of 5', 'median of 5'],
              rows: [
                ['none, single process', '0', '0.4 ms', '0.4 ms'],
                ['fork', '2', '95 ms', '113 ms'],
                ['fork', '4', '109 ms', '149 ms'],
                ['spawn', '2', '2899 ms', '3839 ms'],
                ['spawn', '4', '4138 ms', '4385 ms'],
              ],
            },
          },
          {
            h: 'one queue per worker in, one queue back',
            ps: [
              "The topology is asymmetric, and the asymmetry is the scheduling policy. Each worker gets its own index queue; all workers share one result queue. Indices go out addressed to a specific worker, and results come back into a common pile.",
              "`_try_put_index` is where a batch is assigned. It pulls the next index list from the sampler in the main process, walks a `cycle` over worker ids until it finds an active one, and puts `(send_idx, index)` on that worker's queue. Plain round robin, so with all workers alive, batch `i` goes to worker `i % num_workers`.",
              "Two consequences are visible from the run underneath. Batches 0 and 2 and 4 came from one pid and 1, 3, 5 from the other, exactly as the modulo predicts. And every worker reported `torch.get_num_threads()` of 1, because `_worker_loop` calls `torch.set_num_threads(1)` before it touches your dataset. The third lesson takes that apart.",
              "Round robin also explains a stall that looks like a bug. Assignment happens before anyone knows how long a sample will take, so one slow index does not get rerouted to an idle worker. It sits in the queue of the worker it was addressed to, and the batches behind it in that queue wait.",
            ],
            code: {
              caption: 'verbatim, torch/utils/data/dataloader.py:1348-1366 at torch 2.2.2, with the measured assignment underneath',
              lang: 'python',
              text: '    def _try_put_index(self):\n        assert self._tasks_outstanding < self._prefetch_factor * self._num_workers\n\n        try:\n            index = self._next_index()\n        except StopIteration:\n            return\n        for _ in range(self._num_workers):  # find the next active worker, if any\n            worker_queue_idx = next(self._worker_queue_idx_cycle)\n            if self._workers_status[worker_queue_idx]:\n                break\n        else:\n            # not found (i.e., didn\'t break)\n            return\n\n        self._index_queues[worker_queue_idx].put((self._send_idx, index))\n        self._task_info[self._send_idx] = (worker_queue_idx,)\n        self._tasks_outstanding += 1\n        self._send_idx += 1\n\n# batch_size=4, num_workers=2, a dataset reporting its own pid; stdout:\n# batch 0: idx [0, 1, 2, 3] pid 79276 worker 0 threads 1 shared True\n# batch 1: idx [4, 5, 6, 7] pid 79277 worker 1 threads 1 shared True\n# batch 2: idx [8, 9, 10, 11] pid 79276 worker 0 threads 1 shared True\n# batch 3: idx [12, 13, 14, 15] pid 79277 worker 1 threads 1 shared True',
            },
          },
          {
            h: 'prefetch is a count of tasks in flight',
            ps: [
              "`prefetch_factor` is not a queue size in batches of memory and not a lookahead in time. It is a cap on outstanding tasks, and the loader primes itself to that cap the moment you call `iter()`, before you have asked for anything.",
              "The priming loop runs `prefetch_factor * num_workers` times. With the default factor of 2 and two workers, four batches are already assigned and in flight before the first `next()`. Reading the iterator's own counters right after `iter()` shows exactly that: `send_idx` at 4, `rcvd_idx` at 0, four entries in the task table.",
              "The cap is also a memory statement nobody writes down. Four batches in flight means four batches of tensors alive at once, plus whatever is sitting in the result queue, and raising `prefetch_factor` to smooth a jittery loader multiplies that. The assert at the top of `_try_put_index` is the invariant being enforced.",
              "As batches are consumed, `_process_data` calls `_try_put_index` again, one task out for one task in, so the pipeline stays exactly that full until the sampler is exhausted.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the loader is already four batches deep before the first next()',
              lang: 'python',
              text: 'import os, time, torch\nfrom torch.utils.data import DataLoader, Dataset\n\nclass Slow(Dataset):\n    def __len__(self): return 12\n    def __getitem__(self, i):\n        time.sleep(0.30 if i < 4 else 0.01)\n        return torch.tensor([float(i), float(os.getpid())])\n\nif __name__ == "__main__":\n    dl = DataLoader(Slow(), batch_size=2, num_workers=2)\n    it = iter(dl)\n    time.sleep(1.0)\n    print("prefetch_factor", dl.prefetch_factor, "| tasks outstanding before first next():", it._tasks_outstanding)\n    print("send_idx", it._send_idx, "rcvd_idx", it._rcvd_idx, "| task_info keys", sorted(it._task_info))\n\n# ---- stdout ----\n# prefetch_factor 2 | tasks outstanding before first next(): 4\n# send_idx 4 rcvd_idx 0 | task_info keys [0, 1, 2, 3]',
            },
          },
          {
            h: 'order is repaired on the receiving end',
            ps: [
              "Workers finish whenever they finish, and the result queue is shared, so results arrive in whatever order they completed. Your loop still sees batch 0, then batch 1, then batch 2, and the machinery that guarantees it is four lines long.",
              "Every result carries the `send_idx` it was dispatched with. `_next_data` wants `_rcvd_idx` and nothing else. A result whose index does not match gets appended to its entry in `_task_info` and left there; a result that matches is returned, and `_rcvd_idx` advances.",
              "That reorder buffer is measurable. Make the first batch slow and the rest fast, iterate with four workers, and the first `next()` blocks while three completed batches pile up in the table behind it. The three that follow then return in microseconds, because they were finished before the first one was.",
              "Which means a single slow sample delays every batch after it, even though the work for those batches is already done. Current torch offers a way out that 2.2.2 does not have: an `in_order` argument, defaulting to true, whose docstring says the loader will not enforce first-in-first-out order when it is false.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): three finished batches waiting behind one slow one. The first figure is mostly the cost of spawning four workers and moves with machine load; a repeat run gave 15.14 s. The six source lines under the output are verbatim from dataloader.py:1341-1346 at 2.2.2, commented so both fit one panel',
              lang: 'python',
              text: 'import time, torch\nfrom torch.utils.data import DataLoader, Dataset\n\nclass Uneven(Dataset):\n    def __len__(self): return 8\n    def __getitem__(self, i):\n        time.sleep(1.0 if i < 2 else 0.01)   # only the first batch is slow\n        return torch.tensor([float(i)])\n\nif __name__ == "__main__":\n    dl = DataLoader(Uneven(), batch_size=2, num_workers=4)\n    it = iter(dl)\n    t0 = time.perf_counter(); first = next(it); t1 = time.perf_counter()\n    print(f"batch 0 = {[int(v) for v in first.flatten()]} after {t1-t0:.2f}s")\n    print("already finished and buffered:", sorted(k for k, v in it._task_info.items() if len(v) == 2))\n    for n in range(3):\n        t = time.perf_counter(); b = next(it)\n        print(f"batch {n+1} = {[int(v) for v in b.flatten()]} after {time.perf_counter()-t:.4f}s")\n\n# ---- stdout (the 5.21 s includes spawning four workers) ----\n# batch 0 = [0, 1] after 5.21s\n# already finished and buffered: [1, 2, 3]\n# batch 1 = [2, 3] after 0.0002s\n# batch 2 = [4, 5] after 0.0001s\n# batch 3 = [6, 7] after 0.0001s\n\n# ---- dataloader.py:1341-1346, the branch that holds them ----\n#            if idx != self._rcvd_idx:\n#                # store out-of-order samples\n#                self._task_info[idx] += (data,)\n#            else:\n#                del self._task_info[idx]\n#                return self._process_data(data)',
            },
          },
          {
            h: 'the batch is built in shared memory, not sent through the queue',
            ps: [
              "Passing a whole batch of tensors through a multiprocessing queue would mean pickling every byte, and collate quietly avoids it. `collate_tensor_fn` asks whether it is running inside a worker, and if it is, allocates the output in shared memory before stacking into it.",
              "So the queue carries a handle, not the data. Check `is_shared()` on a batch and the answer tracks the process boundary exactly: true with workers, false without. The comment in the source states the intent in one line, and it is about avoiding an extra copy.",
              "How that handle travels is a platform question again. `torch.multiprocessing` supports two sharing strategies, and on this machine `get_all_sharing_strategies()` returns only `file_system`. Linux defaults to `file_descriptor`, which passes an open fd per tensor over the socket, and that is the origin of the `Too many open files` failure that a large `num_workers` with a small `ulimit -n` produces. A long comment block in `dataloader.py` documents the failure and includes a standalone script to reproduce it outside torch.",
            ],
            code: {
              caption: 'verbatim, torch/utils/data/_utils/collate.py:168-174 at torch 2.2.2, with the measured flag underneath',
              lang: 'python',
              text: '    if torch.utils.data.get_worker_info() is not None:\n        # If we\'re in a background process, concatenate directly into a\n        # shared memory tensor to avoid an extra copy\n        numel = sum(x.numel() for x in batch)\n        storage = elem._typed_storage()._new_shared(numel, device=elem.device)\n        out = elem.new(storage).resize_(len(batch), *list(elem.size()))\n    return torch.stack(batch, 0, out=out)\n\n# stdout on this machine:\n# num_workers=2: shared True\n# num_workers=0: shared False\n# sharing strategy: file_system | available: {\'file_system\'}',
            },
          },
        ],
        readings: [
          { label: 'dataloader.py at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/utils/data/dataloader.py', note: 'the iterator, the queues, and 200 lines of comments on shutdown logic and the file-descriptor failure' },
          { label: 'multiprocessing best practices', url: 'https://docs.pytorch.org/docs/2.2/notes/multiprocessing.html', note: 'file_descriptor against file_system, and the open-file limit that decides between them' },
          { label: 'multi-process data loading', url: 'https://docs.pytorch.org/docs/2.2/data.html#multi-process-data-loading', note: 'the same machinery from the reference side, including the platform notes on spawn' },
        ],
        check: [
          {
            q: 'Why does the same lambda-carrying dataset load fine on a Linux box and fail on a mac?',
            a: 'Because Linux defaults to the fork start method, where the worker inherits the dataset in memory and nothing is serialized, while macOS defaults to spawn, where the arguments are pickled into a fresh interpreter and a lambda cannot be pickled.',
          },
          {
            q: 'With num_workers=2 and the default prefetch_factor, how many batches are in flight before you ask for the first one?',
            a: 'Four. The reset path primes the pipeline with prefetch_factor * num_workers calls to _try_put_index, so send_idx reads 4 while rcvd_idx is still 0 and four entries sit in the task table.',
          },
          {
            q: 'Four workers, and the batch you are waiting on is slow. What are the other three doing with their finished results?',
            a: 'Sitting in _task_info as out-of-order entries. Results carry the send_idx they were dispatched with, and _next_data returns only the one matching _rcvd_idx, so finished later batches wait until the earlier one arrives.',
          },
        ],
        work: [
          { id: 'start-method', label: 'print multiprocessing.get_start_method() on every machine you train on, and write down which of your dataset attributes would survive a pickle', href: '#the-start-method-decides-what-has-to-pickle' },
          { id: 'prefetch-memory', label: 'work out how many bytes prefetch_factor * num_workers batches of your own data occupy, and compare it against the memory you thought the loader used', href: '#prefetch-is-a-count-of-tasks-in-flight' },
        ],
      },
      {
        id: 'same-order-different-numbers',
        num: 3,
        title: 'Same order, different numbers',
        lede: 'Seed everything, set num_workers to two, and the samples arrive in the same order they did before. The random numbers drawn inside them do not, and the reason is a single addition in the worker startup path.',
        goal: 'Say which parts of a seeded pipeline are invariant to num_workers and which are not, derive each worker seed from the loader generator, explain why an iterable dataset duplicates itself across workers, and state what pin_memory does on a machine with no accelerator.',
        sections: [
          {
            h: 'the order is decided before any worker sees it',
            ps: [
              "Pass a `torch.Generator` to `DataLoader` and two different things read from it. `dl.sampler.generator is g` comes back true, so the `RandomSampler` holds the object you handed over rather than a copy. The iterator draws a single 64-bit integer off that same object at construction.",
              "Chapter four's arc already puts the first of those two to work. Its third lesson, 'The resume that matches', saves the generator's state at an epoch boundary to replay the next epoch's order, and names the one piece of a resume that has no state to save at all. None of that is repeated here. The question this lesson needs answered is a different one: where the draw runs.",
              "It runs in the main process. `self._next_index()` sits inside the `_try_put_index` you read in the last lesson, and only the resulting list of indices goes onto a worker's queue, so no worker ever runs the sampler. Shuffle eight samples with zero, two and three workers and the order comes back `[5, 3, 1, 7, 2, 6, 4, 0]` all three times.",
              "The second reader of that generator is one line of `_BaseDataLoaderIter.__init__`, and it is what makes the other half of reproducibility hard. `self._base_seed = torch.empty((), dtype=torch.int64).random_(generator=loader.generator).item()` is where the single integer comes from, and every worker's random state descends from it.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): one generator the sampler holds by reference, three worker counts, one order and three different noise streams',
              lang: 'python',
              text: 'import torch\nfrom torch.utils.data import DataLoader, Dataset\n\nclass Noisy(Dataset):\n    def __len__(self): return 8\n    def __getitem__(self, i):\n        return torch.tensor([float(i), torch.rand(()).item()])\n\ndef run(nw, seed=0):\n    g = torch.Generator(); g.manual_seed(seed)\n    torch.manual_seed(seed)\n    dl = DataLoader(Noisy(), batch_size=2, shuffle=True, num_workers=nw, generator=g)\n    return dl, g, [[round(v, 4) for v in row] for b in dl for row in b.tolist()]\n\nif __name__ == "__main__":\n    dl, g, _ = run(0)\n    print("sampler:", type(dl.sampler).__name__, "| sampler.generator is g:", dl.sampler.generator is g)\n    for nw in (0, 2, 3):\n        _, _, r = run(nw)\n        print(f"num_workers={nw}")\n        print("  order    ", [int(x[0]) for x in r])\n        print("  torch.rand", [x[1] for x in r])\n\n# ---- stdout ----\n# sampler: RandomSampler | sampler.generator is g: True\n# num_workers=0\n#   order     [5, 3, 1, 7, 2, 6, 4, 0]\n#   torch.rand [0.4963, 0.7682, 0.0885, 0.132, 0.3074, 0.6341, 0.4901, 0.8964]\n# num_workers=2\n#   order     [5, 3, 1, 7, 2, 6, 4, 0]\n#   torch.rand [0.7821, 0.0536, 0.6938, 0.298, 0.9888, 0.1949, 0.1669, 0.2847]\n# num_workers=3\n#   order     [5, 3, 1, 7, 2, 6, 4, 0]\n#   torch.rand [0.7821, 0.0536, 0.6938, 0.298, 0.654, 0.2994, 0.9888, 0.1949]',
            },
          },
          {
            h: 'base seed plus worker id, and nothing else',
            ps: [
              "Three lines at the top of `_worker_loop` set the whole random state of a worker, and the arithmetic is as simple as it looks. The worker adds its own id to the base seed, then seeds Python's `random`, then seeds torch. NumPy gets a derived state instead, computed by a local reimplementation of `SeedSequence`, because seeding two generators with the same integer is a known way to correlate them.",
              "Reading `WorkerInfo.seed` back out of three workers confirms the addition. With the loader generator seeded at 0, the three workers reported seeds ending 455, 456 and 457, and `torch.initial_seed()` inside each worker matched its own.",
              "Now put that beside round robin from the last lesson and the divergence in the run above stops being mysterious. Batch `i` goes to worker `i % num_workers`, and each worker draws from a stream seeded by `base_seed + worker_id`, so changing the worker count changes which stream a given batch draws from. The last four values of the `num_workers=3` row are the proof: `0.9888, 0.1949` appear there as the fourth batch and appear in the `num_workers=2` row as the third, because in both cases that is worker 0's second batch.",
              "The practical rule is short. Sample order is reproducible across worker counts; anything random inside `__getitem__` is reproducible only for a fixed `num_workers`. If a result has to survive a change in worker count, the randomness has to be derived from the sample index rather than drawn from ambient state, and `worker_init_fn` plus `get_worker_info().seed` is where you would put that derivation.",
            ],
            code: {
              caption: 'verbatim, torch/utils/data/_utils/worker.py:222-229 at torch 2.2.2, with the measured seeds underneath',
              lang: 'python',
              text: '        torch.set_num_threads(1)\n        seed = base_seed + worker_id\n        random.seed(seed)\n        torch.manual_seed(seed)\n        if HAS_NUMPY:\n            np_seed = _generate_state(base_seed, worker_id)\n            import numpy as np\n            np.random.seed(np_seed)\n\n# an IterableDataset reporting its own WorkerInfo, generator seeded at 0, stdout:\n# worker id, WorkerInfo.seed mod 1000, torch.initial_seed() mod 1000:\n#   [[0, 455, 455], [1, 456, 456], [2, 457, 457]]',
            },
          },
          {
            h: 'a worker fetches with one thread',
            ps: [
              "The first line of that excerpt is not about randomness at all, and it is the one with performance consequences. `torch.set_num_threads(1)` runs before your dataset is touched, so every worker does its tensor work single-threaded.",
              "The reason is defensive. Without it, N workers each opening a thread pool sized to the machine would oversubscribe every core and fight the training step for them. The cost is that a transform torch would have parallelized in the main process no longer is.",
              "That cost is measurable and the measurement needs an honest caveat. A dataset whose `__getitem__` does one 768 by 768 matmul reported `torch.get_num_threads()` of 8 when fetched in the main process and 1 inside a worker, every run. The per-sample time for that matmul came out at 6.00, 24.11 and 23.75 ms in the main process across three runs, against 29.26, 79.20 and 92.49 ms in a worker. The absolute numbers move with whatever else this laptop is doing and should not be read as anything; the ratio, roughly five to eight times slower per sample, held across all three.",
              "Aggregate throughput is the number that would actually decide a configuration, and this machine could not produce a trustworthy one. Repeat runs of the same epoch swung between 18 and 42 samples per second at `num_workers=2`, so no table of it appears here. Chapter nine's harness discipline is what a claim like that would have to pass first.",
            ],
          },
          {
            h: 'an iterable dataset is copied, not divided',
            ps: [
              "Map-style datasets are safe under workers because the main process owns the indices and each worker is told which ones to fetch. An iterable-style dataset has no indices to hand out. Each worker gets its own copy of the object and calls `iter()` on it, which means each worker produces the entire stream.",
              "The result is duplication with no error and no warning. An eight-item stream read with two workers yielded sixteen items, every sample twice, in an order that interleaves the two copies.",
              "The fix is to shard inside `__iter__`, and `get_worker_info()` is the only information you need to do it. Stride the stream by `num_workers` starting at `id`, treat a `None` return as the single-process case, and the same dataset yields each sample exactly once at any worker count.",
              "This is also the reason `_InfiniteConstantSampler` shows up on an iterable-style loader. There is nothing to sample, so the loader emits a constant to drive the fetch loop and lets the dataset decide when the epoch ends.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the duplication, then the four-line shard that removes it',
              lang: 'python',
              text: 'import torch\nfrom torch.utils.data import DataLoader, IterableDataset, get_worker_info\n\nclass Stream(IterableDataset):\n    def __iter__(self):\n        for i in range(8): yield torch.tensor([float(i)])\n\nclass Sharded(IterableDataset):\n    def __iter__(self):\n        wi = get_worker_info()\n        lo, step = (0, 1) if wi is None else (wi.id, wi.num_workers)\n        for i in range(lo, 8, step): yield torch.tensor([float(i)])\n\nif __name__ == "__main__":\n    for nw in (0, 2):\n        got = [int(v) for b in DataLoader(Stream(), batch_size=2, num_workers=nw) for v in b.flatten()]\n        print(f"IterableDataset, num_workers={nw}: {got}")\n    print("sampler:", type(DataLoader(Stream(), batch_size=2, num_workers=2).sampler).__name__)\n    for nw in (0, 2, 3):\n        got = [int(v) for b in DataLoader(Sharded(), batch_size=2, num_workers=nw) for v in b.flatten()]\n        print(f"sharded, num_workers={nw}: {sorted(got)} (n={len(got)})")\n\n# ---- stdout, the two runs joined ----\n# IterableDataset, num_workers=0: [0, 1, 2, 3, 4, 5, 6, 7]\n# IterableDataset, num_workers=2: [0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7]\n# sampler: _InfiniteConstantSampler\n# sharded, num_workers=0: [0, 1, 2, 3, 4, 5, 6, 7] (n=8)\n# sharded, num_workers=2: [0, 1, 2, 3, 4, 5, 6, 7] (n=8)\n# sharded, num_workers=3: [0, 1, 2, 3, 4, 5, 6, 7] (n=8)',
            },
          },
          {
            h: 'pin_memory on a machine with nothing to pin for',
            ps: [
              "Page-locked host memory is what `pin_memory=True` is asking for, and it only pays off against a device that can DMA out of it while the CPU does something else. This course was written on a machine with no such device, and the loader's behaviour there is worth stating exactly rather than skipping.",
              "The flag is accepted and then dropped. `DataLoader.pin_memory` still reads `True`, while the iterator's `_pin_memory` reads `False`, because the iterator recomputes it as `loader.pin_memory and torch.cuda.is_available()`. No pin-memory thread starts, batches come back with `is_pinned()` false, and torch 2.2.2 issues no warning at all: catching warnings around a full iteration returned an empty list.",
              "Current torch fixed the silence. In pytorch at the v2.9.0 tag the same two lines warn first, with the text `'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.`, and the availability check is `torch.accelerator.is_available()` rather than a CUDA-specific one. Same outcome, louder.",
              "What the flag buys where it does apply is a thread, not a process. `_pin_memory_loop` runs in the main process, pulling finished batches off the worker result queue, calling `.pin_memory()` on every tensor in them, and putting them on a second queue for your loop to read. Only then can a device copy be issued with `non_blocking=True` and actually overlap with compute, which is the pairing the CUDA notes describe and the reason the flag is nearly always set together with that argument.",
              "Worth knowing what the failure looks like if you call it by hand instead. `torch.zeros(4).pin_memory()` on this machine raises `NotImplementedError: Could not run 'aten::_pin_memory' with arguments from the 'CUDA' backend`, which names CUDA on a machine that has none because the pinning op dispatches to the accelerator backend by default.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU, no accelerator): the flag set, the flag dropped, no warning; the three source lines at the end are verbatim from dataloader.py:588-590, commented so both fit one panel',
              lang: 'python',
              text: 'import warnings, torch\nfrom torch.utils.data import DataLoader, TensorDataset\n\nds = TensorDataset(torch.randn(64, 4))\nprint("cuda available:", torch.cuda.is_available())\ndl = DataLoader(ds, batch_size=8, pin_memory=True)\nit = iter(dl)\nprint("loader.pin_memory:", dl.pin_memory, "| iterator _pin_memory:", it._pin_memory)\nb, = next(it)\nprint("batch is_pinned:", b.is_pinned())\n\nwith warnings.catch_warnings(record=True) as w:\n    warnings.simplefilter("always")\n    next(iter(DataLoader(ds, batch_size=8, pin_memory=True)))\n    print("warnings raised:", [str(x.message) for x in w])\n\n# ---- stdout ----\n# cuda available: False\n# loader.pin_memory: True | iterator _pin_memory: False\n# batch is_pinned: False\n# warnings raised: []\n\n# and the line that drops it, verbatim from dataloader.py:588-590 at 2.2.2:\n#        if (len(loader.pin_memory_device) == 0):\n#            self._pin_memory = loader.pin_memory and torch.cuda.is_available()\n#            self._pin_memory_device = None',
            },
          },
        ],
        readings: [
          { label: 'worker.py at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/utils/data/_utils/worker.py', note: 'the worker loop end to end, the seeding block, and the SeedSequence port that feeds numpy' },
          { label: 'randomness in multi-process data loading', url: 'https://docs.pytorch.org/docs/2.2/data.html#randomness-in-multi-process-data-loading', note: 'the same rules from the reference side, plus the worker_init_fn recipe' },
          { label: 'dataloader.py at v2.9.0', url: 'https://github.com/pytorch/pytorch/blob/v2.9.0/torch/utils/data/dataloader.py', note: 'the pin_memory warning and the in_order argument that a current torch has and 2.2.2 does not' },
        ],
        check: [
          {
            q: 'Everything is seeded and you change num_workers from 2 to 4. What is guaranteed to be identical, and what is not?',
            a: 'Sample order is identical, because the sampler runs in the main process off the loader generator. Anything drawn inside __getitem__ is not, because each worker seeds from base_seed plus its own id and round robin sends a given batch to a different worker.',
          },
          {
            q: 'An IterableDataset with two workers returned twice as many samples as it should. Why?',
            a: 'Because there are no indices to divide up, so each worker receives its own copy of the dataset and iterates the whole stream. Sharding inside __iter__ using get_worker_info().id and num_workers is what splits it.',
          },
          {
            q: 'pin_memory=True on a CPU-only box: what does the DataLoader actually do with it?',
            a: 'Accepts it on the loader and drops it on the iterator, which recomputes _pin_memory as pin_memory and torch.cuda.is_available(). No pinning thread starts and batches are not pinned; torch 2.2.2 warns about none of this, while current torch warns and checks torch.accelerator.is_available() instead.',
          },
        ],
        work: [
          { id: 'seed-derivation', label: 'make one augmentation in your own dataset reproducible across worker counts by deriving its seed from the sample index, then prove it with num_workers at 0, 2 and 3', href: '#base-seed-plus-worker-id-and-nothing-else' },
          { id: 'thread-budget', label: 'count the threads your loader actually creates: one per worker at one thread each, plus the main process pool, and compare the total against your core count', href: '#a-worker-fetches-with-one-thread' },
        ],
      },
    ],
  },
]
