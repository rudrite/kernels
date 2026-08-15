// New file: site/src/data/lessons/torch-distributed.ts
// What holds a PyTorch distributed job together underneath the two strategies
// chapter 8 names: the c10d handshake and the store beneath it, the reducer
// DDP installs and the buckets it rebuilds, the flat parameter one FSDP unit
// owns, and the placement a DTensor carries so redistribute knows which
// collective to insert. Every printout is a local gloo run on torch 2.2.2 CPU,
// Python 3.12, with 2 or 4 processes started by torch.multiprocessing.spawn;
// the source lines are from the v2.2.2 tag, and the CUDA-only and multi-node
// behaviour is cited, never simulated.
import type { UnitLessons } from './index'

export const TORCH_DISTRIBUTED_LESSONS: UnitLessons[] = [
  {
    unit: 'pt:distributed',
    lessons: [
      {
        id: 'one-world-four-processes',
        num: 1,
        title: 'One world, four processes',
        lede: 'Nothing launches a distributed job. Four processes agree on an address, publish their own addresses into a key-value store, and from that point the only thing holding them together is that every one of them calls the same collective in the same order.',
        goal: 'Given a torch.distributed script, name what init_process_group blocks on, say which backend serves which device type on the build in front of you, and predict for a given mismatch whether the job raises, hangs until a timeout, or aborts the process outright.',
        sections: [
          {
            h: 'four processes, one script, one meeting point',
            ps: [
              "Chapter 8 runs a collective on a world of one, which proves the call path and nothing about coordination. Four is where the ideas start having consequences, and you do not need four machines to get there. `torch.multiprocessing.spawn` starts four fresh Python processes on this laptop, hands each one its index, and waits for all of them to finish.",
              "Every one of those processes runs the same worker function. The only thing that differs is the integer it was handed, and that integer is what it passes to `init_process_group` as its rank. Nobody hands out assignments. Rank 2 knows it is rank 2 because it was told at startup, and it works out its own share of the data from that number and the world size.",
              "Read the output ordering before anything else. Rank 1 printed first, then 2, then 3, and rank 0 landed last even though rank 0 is the process that printed the three header lines. Four processes writing into one terminal have no ordering discipline between them, and a collective imposes none either. All four came out of the all_reduce holding the same `[10.0, 10.0]`, which is 1 + 2 + 3 + 4, and they arrived there in whatever order the operating system felt like.",
              'That number is the first thing worth checking on any new machine: gloo is available here, nccl and mpi are not, because this is a CPU-only build. Every run in these four lessons is gloo, and every one of them is real.',
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU, gloo, 4 spawned processes): a world of four on one machine',
              lang: 'python',
              text: 'import os\nimport torch\nimport torch.distributed as dist\nimport torch.multiprocessing as mp\n\ndef worker(rank, world):\n    os.environ["MASTER_ADDR"] = "127.0.0.1"\n    os.environ["MASTER_PORT"] = "29531"\n    dist.init_process_group("gloo", rank=rank, world_size=world)\n    x = torch.full((2,), float(rank + 1))\n    dist.all_reduce(x)\n    if rank == 0:\n        print("backend:", dist.get_backend(), "| world:", dist.get_world_size())\n        print("available: gloo", dist.is_gloo_available(),\n              "nccl", dist.is_nccl_available(), "mpi", dist.is_mpi_available())\n    print(f"rank {rank}: {x.tolist()}")\n    dist.barrier()\n    dist.destroy_process_group()\n\nif __name__ == "__main__":\n    mp.spawn(worker, args=(4,), nprocs=4, join=True)',
              full: {
                label: 'the script and its verbatim output',
                text: 'import os\nimport torch\nimport torch.distributed as dist\nimport torch.multiprocessing as mp\n\ndef worker(rank, world):\n    os.environ["MASTER_ADDR"] = "127.0.0.1"\n    os.environ["MASTER_PORT"] = "29531"\n    dist.init_process_group("gloo", rank=rank, world_size=world)\n    x = torch.full((2,), float(rank + 1))\n    dist.all_reduce(x)\n    if rank == 0:\n        print("backend:", dist.get_backend(), "| world:", dist.get_world_size())\n        print("available: gloo", dist.is_gloo_available(),\n              "nccl", dist.is_nccl_available(), "mpi", dist.is_mpi_available())\n    print(f"rank {rank}: {x.tolist()}")\n    dist.barrier()\n    dist.destroy_process_group()\n\nif __name__ == "__main__":\n    mp.spawn(worker, args=(4,), nprocs=4, join=True)\n\n# rank 1: [10.0, 10.0]\n# rank 2: [10.0, 10.0]\n# rank 3: [10.0, 10.0]\n# backend: gloo | world: 4\n# available: gloo True nccl False mpi False\n# rank 0: [10.0, 10.0]',
              },
            },
          },
          {
            h: 'the handshake runs through a key-value store',
            ps: [
              "`MASTER_ADDR` and `MASTER_PORT` are not the address the ranks talk to each other on. They are the address of a rendezvous, and underneath every init method c10d supports there is one object doing the work: a Store, a small key-value service that rank 0 hosts and every other rank connects to. Ranks write their own listening addresses into it, read back everybody else's, and only then does gloo build the pairwise connections that carry the actual tensor traffic.",
              "The store is a public class, not an implementation detail you have to infer. `dist.TCPStore` takes a host, a port, a world size, and a flag saying whether this process is the server, and it gives you `set`, `get`, `add`, `wait`, and `num_keys`. `add` is atomic, which is what makes it usable as the counter a barrier needs. Standing one up by hand takes four lines and no distributed job at all.",
              "After `init_process_group` returns, the store your process group holds is not the raw TCPStore. Print its type and it is a `PrefixStore`, a wrapper that prepends a namespace to every key. That wrapper is why a second process group, a subgroup, or a later library that wants its own coordination channel can all share one TCP server without their keys colliding.",
              ">> The rendezvous is a key-value store. The connections are built from what the ranks read out of it.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the rendezvous primitive, standalone, no ranks involved',
              lang: 'python',
              text: 'from datetime import timedelta\nimport torch.distributed as dist\n\ns = dist.TCPStore("127.0.0.1", 29550, 1, True, timedelta(seconds=5))\ns.set("rank0_addr", "tcp://127.0.0.1:1234")\nprint("get :", s.get("rank0_addr"))    # b"tcp://127.0.0.1:1234"\nprint("keys:", s.num_keys())           # 2\nprint("add :", s.add("counter", 1), s.add("counter", 1))   # 1 2',
            },
          },
          {
            h: 'a backend is an answer for one device type',
            ps: [
              "The string you hand `init_process_group` picks the library that moves the bytes, and the choice is a device question rather than a preference. gloo is the CPU answer and works on any machine. nccl is the CUDA answer and is what a GPU job wants. mpi only exists if torch was built against an MPI installation, which the wheels are not.",
              "A single job can use two of them at once. The backend string accepts a device map, `\"cpu:gloo,cuda:nccl\"`, so a process group can route CPU tensors through gloo and CUDA tensors through nccl without you building two groups by hand. The distributed reference documents the mapping form and the per-backend support matrix.",
              "What gloo can carry is worth measuring rather than assuming, because the reputation is older than the code. Five operations, tried on a world of two on this build, all returned: all_reduce, all_gather_into_tensor, reduce_scatter_tensor, all_to_all_single, and barrier. That last pair matters for the FSDP lesson, which needs reduce-scatter to exist before it can shard a gradient.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU, gloo, 2 spawned processes): what this gloo build actually accepts',
              lang: 'text',
              text: 'all_reduce               ok\nall_gather_into_tensor   ok\nreduce_scatter_tensor    ok\nall_to_all_single        ok\nbarrier                  ok',
              full: {
                label: 'the probe that produced it',
                text: 'import os\nimport torch\nimport torch.distributed as dist\nimport torch.multiprocessing as mp\n\ndef try_op(name, fn, out):\n    try:\n        fn()\n        out.append((name, "ok"))\n    except Exception as e:\n        out.append((name, f"{type(e).__name__}: {str(e).splitlines()[0][:70]}"))\n\ndef worker(rank, world):\n    os.environ["MASTER_ADDR"] = "127.0.0.1"\n    os.environ["MASTER_PORT"] = "29560"\n    dist.init_process_group("gloo", rank=rank, world_size=world)\n    out = []\n    try_op("all_reduce", lambda: dist.all_reduce(torch.ones(4)), out)\n    try_op("all_gather_into_tensor",\n           lambda: dist.all_gather_into_tensor(torch.zeros(8), torch.ones(4)), out)\n    try_op("reduce_scatter_tensor",\n           lambda: dist.reduce_scatter_tensor(torch.zeros(4), torch.ones(8)), out)\n    try_op("all_to_all_single",\n           lambda: dist.all_to_all_single(torch.zeros(4), torch.ones(4)), out)\n    try_op("barrier", lambda: dist.barrier(), out)\n    if rank == 0:\n        for n, s in out:\n            print(f"{n:24s} {s}")\n    dist.destroy_process_group()\n\nif __name__ == "__main__":\n    mp.spawn(worker, args=(2,), nprocs=2, join=True)\n\n# all_reduce               ok\n# all_gather_into_tensor   ok\n# reduce_scatter_tensor    ok\n# all_to_all_single        ok\n# barrier                  ok',
              },
            },
            table: {
              caption: 'the three backends, and what this machine could check',
              cols: ['backend', 'device it serves', 'on this build', 'how the claim is grounded'],
              rows: [
                ['gloo', 'CPU', 'available, five collectives probed', 'the run above, torch 2.2.2 CPU'],
                ['nccl', 'CUDA', 'not available', 'is_nccl_available() returned False; behaviour cited from the distributed reference'],
                ['mpi', 'CPU, when torch was built against MPI', 'not available', 'is_mpi_available() returned False; the wheels are not MPI builds'],
              ],
            },
          },
          {
            h: 'a subgroup is created by everybody, used by some',
            ps: [
              "Once a job has more than one axis of parallelism it needs process groups smaller than the world, and `dist.new_group([0, 1])` builds one. The group is a first-class object: pass it as `group=` to any collective and the reduction happens only among its members, and `dist.get_rank(group)` gives you the position inside it, which is not the same integer as the global rank.",
              "`new_group` is itself a collective, and that is the part people get wrong. Every rank in the world has to call it, including ranks that will never use the resulting group, because building the group requires agreement across all of them. Skip the call on rank 3 because rank 3 is not in the group, and the job hangs on group creation rather than on anything that looks like communication.",
              "The run below builds two groups on a world of four and each rank all_reduces inside its own. Ranks 0 and 1 come out with 3.0, which is 1 + 2. Ranks 2 and 3 come out with 7.0, which is 3 + 4. Rank 2 reports global rank 2 and local rank 0, and that second number is the one a sharding calculation should be using.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU, gloo, 4 spawned processes): two subgroups, two reductions',
              lang: 'python',
              text: 'def worker(rank, world):\n    os.environ["MASTER_ADDR"] = "127.0.0.1"\n    os.environ["MASTER_PORT"] = "29532"\n    dist.init_process_group("gloo", rank=rank, world_size=world)\n    pair = dist.new_group([0, 1])      # every rank calls both,\n    odd = dist.new_group([2, 3])       # including the ranks not in them\n    mine = pair if rank < 2 else odd\n    x = torch.full((2,), float(rank + 1))\n    dist.all_reduce(x, group=mine)\n    print(f"rank {rank}: global {dist.get_rank()} local {dist.get_rank(mine)} -> {x.tolist()}")\n    dist.destroy_process_group()\n\n# rank 2: global 2 local 0 -> [7.0, 7.0]\n# rank 1: global 1 local 1 -> [3.0, 3.0]\n# rank 3: global 3 local 1 -> [7.0, 7.0]\n# rank 0: global 0 local 0 -> [3.0, 3.0]',
            },
          },
          {
            h: 'the two ways the contract breaks',
            ps: [
              "The xla path states the rule at the HLO level in its collectives chapter: every participant issues the matching call, in the same order, or nothing completes. c10d is where you get to feel that rule, because the failure arrives as a specific error rather than as a paragraph.",
              "Break it the first way, by having rank 1 skip an all_reduce that rank 0 issues, and rank 1 sails past its own code while rank 0 sits in gloo waiting for a message that is never sent. The default timeout is thirty minutes, which is why a real hang looks like a job doing nothing rather than a job failing. Set the timeout to five seconds and the error text is immediate and exact: a recv operation that timed out waiting.",
              "Break it the second way, by having the two ranks call all_reduce with different tensor sizes, and there is no exception at all. gloo reads a preamble that does not match the bytes it was handed, raises a C++ enforcement failure that nothing catches, and the process aborts. Python sees `ProcessExitedException: process 1 terminated with signal SIGABRT`, which is a different debugging situation from a traceback: no rank tells you what the mismatch was, and your `try` block never runs.",
              ">> A wrong shape does not raise. It aborts the process.",
            ],
            code: {
              caption: 'both failures, verbatim (verified, torch 2.2.2 CPU, gloo, 2 spawned processes; the bracketed build path is trimmed)',
              lang: 'text',
              text: 'rank 1 skips the collective, five-second timeout:\n  rank 1: returned\n  rank 0: RuntimeError: [... gloo/transport/uv/unbound_buffer.cc:67]\n          Timed out waiting 5000ms for recv operation to complete\n\nranks call all_reduce with tensors of 4 and 8 elements:\n  libc++abi: terminating due to uncaught exception of type gloo::EnforceNotMet:\n  [enforce fail at ... gloo/transport/uv/pair.cc:248] op.nread == op.preamble.nbytes.\n  torch.multiprocessing.spawn.ProcessExitedException: process 1 terminated with signal SIGABRT',
              full: {
                label: 'the script that produced both',
                text: 'import os\nimport sys\nfrom datetime import timedelta\nimport torch\nimport torch.distributed as dist\nimport torch.multiprocessing as mp\n\ndef worker(rank, world, mode):\n    os.environ["MASTER_ADDR"] = "127.0.0.1"\n    os.environ["MASTER_PORT"] = "29543" if mode == "skip" else "29544"\n    dist.init_process_group("gloo", rank=rank, world_size=world,\n                            timeout=timedelta(seconds=5))\n    try:\n        if mode == "skip":\n            if rank == 0:\n                dist.all_reduce(torch.ones(4))\n        else:\n            n = 4 if rank == 0 else 8\n            dist.all_reduce(torch.ones(n))\n        print(f"rank {rank}: returned")\n    except Exception as e:\n        print(f"rank {rank}: {type(e).__name__}: {str(e).splitlines()[0]}")\n    dist.destroy_process_group()\n\nif __name__ == "__main__":\n    mp.spawn(worker, args=(2, sys.argv[1]), nprocs=2, join=True)\n\n# python mismatch.py skip   -> the timeout above\n# python mismatch.py shape  -> the abort above',
              },
            },
          },
        ],
        readings: [
          { label: 'torch.distributed reference', url: 'https://docs.pytorch.org/docs/stable/distributed.html', note: 'the backend support matrix, the device-map backend string, and every collective signature' },
          { label: 'distributed_c10d.py at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/distributed/distributed_c10d.py', note: 'init_process_group, the rendezvous handlers, and where the PrefixStore wrapping happens' },
          { label: 'torchrun (elastic launch)', url: 'https://docs.pytorch.org/docs/stable/elastic/run.html', note: 'what you use instead of mp.spawn once the world spans machines, and the env vars it sets' },
          { label: 'gloo', url: 'https://github.com/facebookincubator/gloo', note: 'the collective library behind every run in these lessons' },
        ],
        check: [
          {
            q: 'Rank 3 is not a member of the subgroup your code is building. Does it call new_group?',
            a: 'Yes. new_group is a collective over the whole world, so every rank has to call it, members and non-members alike. A rank that skips it leaves the others waiting inside group creation.',
          },
          {
            q: 'Two ranks call all_reduce on tensors of different sizes. What do you see in the logs?',
            a: 'No Python traceback from the collective. gloo raises a C++ enforcement failure that nothing catches, the process aborts, and the launcher reports a SIGABRT exit rather than an exception you could have handled.',
          },
          {
            q: 'What is actually listening at MASTER_ADDR and MASTER_PORT?',
            a: 'A TCPStore hosted by rank 0. It is a key-value service used for rendezvous only: ranks publish their own addresses into it and read the others back, and the tensor traffic afterwards runs over connections built from what they read.',
          },
        ],
        work: [
          { id: 'world-of-four', label: 'run the world of four and record the order the four ranks printed in, then run it again and see whether the order held', href: '#four-processes-one-script-one-meeting-point' },
          { id: 'break-it-twice', label: 'break the contract both ways on your own machine and write down which failure you would rather debug at 3am', href: '#the-two-ways-the-contract-breaks' },
          { id: 'subgroup-arithmetic', label: 'build a world of four with two subgroups and predict every local rank before you print one', href: '#a-subgroup-is-created-by-everybody-used-by-some' },
        ],
      },
      {
        id: 'buckets-and-the-reducer',
        num: 2,
        title: 'Buckets and the reducer',
        lede: 'Chapter 8 says DDP overlaps its all-reduces with the backward pass. On the first iteration it does not, by design, and the buckets it uses from the second iteration onward are not the ones it was built with.',
        goal: 'Given a model and a bucket cap, compute the bucket assignment DDP will start from, explain why the first iteration runs a single all-reduce regardless, and read the rebuilt bucket layout out of DDP logging data.',
        sections: [
          {
            h: 'construction is already a collective',
            ps: [
              "Wrapping a module in DDP does something before any training happens: it broadcasts rank 0's parameters and buffers to every other rank. That is not documentation, it is observable. Seed each rank differently so the models genuinely differ, print the weights, wrap, and print again.",
              "Rank 1 walks in holding `[0.364, -0.312]` and walks out holding rank 0's `[-0.005, 0.379]`. Nothing in the training script asked for that. It happens in the constructor because every later step of DDP assumes the replicas started identical, and a gradient all-reduce keeps replicas in sync only if they were in sync to begin with.",
              "The same assumption is why the DDP docstring warns that parameters must be registered in the same order on every rank. Ranks are matched up by position in the parameter list, not by name, so a model whose module construction order varies by rank will happily reduce one layer's gradient into another's.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU, gloo, 2 spawned processes): the constructor broadcast',
              lang: 'python',
              text: 'def worker(rank, world):\n    os.environ["MASTER_ADDR"] = "127.0.0.1"\n    os.environ["MASTER_PORT"] = "29551"\n    dist.init_process_group("gloo", rank=rank, world_size=world)\n    torch.manual_seed(rank)                    # deliberately different models\n    net = nn.Linear(2, 1, bias=False)\n    print(f"rank {rank}: before ddp {net.weight.detach().flatten().tolist()}")\n    DDP(net)                                   # the wrap is the broadcast\n    print(f"rank {rank}: after ddp  {net.weight.detach().flatten().tolist()}")\n    dist.destroy_process_group()\n\n# rank 1: before ddp [0.36434608697891235, -0.3121015429496765]\n# rank 1: after ddp  [-0.0052939653396606445, 0.37932294607162476]\n# rank 0: before ddp [-0.0052939653396606445, 0.37932294607162476]\n# rank 0: after ddp  [-0.0052939653396606445, 0.37932294607162476]',
            },
          },
          {
            h: 'what one backward leaves in .grad',
            ps: [
              "The clearest way to see what DDP does to a gradient is to compute the same gradient twice, once outside the wrapper and once inside, on ranks holding different data. A `Linear(3, 1)` with no bias, fed a row of ones on rank 0 and a row of twos on rank 1, has an analytic gradient: the input row itself.",
              "Alone, rank 0 gets `[1.0, 1.0, 1.0]` and rank 1 gets `[2.0, 2.0, 2.0]`, which is exactly what a per-rank backward should produce. Inside DDP both ranks get `[1.5, 1.5, 1.5]`. The all-reduce summed the two gradients and the reducer divided by the world size, so what lands in `.grad` is the mean over the global batch rather than the sum.",
              "That division is why a DDP step and a single-process step on the concatenated batch agree, and it is also the reason a learning rate tuned on one process usually survives the move. What changes is the effective batch size, which is now the per-rank batch times the world size.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU, gloo, 2 spawned processes): the same gradient alone and inside DDP',
              lang: 'python',
              text: 'def worker(rank, world):\n    os.environ["MASTER_ADDR"] = "127.0.0.1"\n    os.environ["MASTER_PORT"] = "29536"\n    dist.init_process_group("gloo", rank=rank, world_size=world)\n    torch.manual_seed(0)\n    net = nn.Linear(3, 1, bias=False)\n    x = torch.full((1, 3), float(rank + 1))\n\n    solo = nn.Linear(3, 1, bias=False)\n    solo.load_state_dict(net.state_dict())\n    solo(x).sum().backward()\n    print(f"rank {rank}: alone  {solo.weight.grad.tolist()}")\n\n    ddp = DDP(net)\n    ddp(x).sum().backward()\n    print(f"rank {rank}: in ddp {net.weight.grad.tolist()}")\n\n# rank 1: alone  [[2.0, 2.0, 2.0]]\n# rank 1: in ddp [[1.5, 1.5, 1.5]]\n# rank 0: alone  [[1.0, 1.0, 1.0]]\n# rank 0: in ddp [[1.5, 1.5, 1.5]]',
            },
          },
          {
            h: 'bucketing is a function you can call yourself',
            ps: [
              "DDP does not invent its grouping. It calls `dist._compute_bucket_assignment_by_size`, hands it the parameter list and a list of size limits, and gets back a list of index lists. That function is importable, needs no process group, and runs in a single process, which makes the bucketing rule something you can check instead of infer.",
              "Run it on three `Linear(512, 512)` layers with a 1MB cap and the answer is `[[0], [1, 2], [3, 4], [5]]`. Parameter 0 is a weight of exactly 1048576 bytes, which fills the first bucket on its own. Parameters 1 and 2 are a 2048-byte bias followed by the next weight, so that bucket ends up at 1050624 bytes, a little over the cap: the fill is greedy, and a parameter that pushes a bucket past the limit still goes in before the next bucket opens.",
              "Two size limits go in, not one. The first is `dist._DEFAULT_FIRST_BUCKET_BYTES`, 1048576 on this build, and it exists so that the parameters defined first, whose gradients arrive last in the backward, get a small bucket rather than spilling into a large one that then has to wait. Every later bucket uses `bucket_cap_mb`, which defaults to 25MB.",
              "DDP then reverses the list before handing it to the reducer. The comment on that line says why in one clause: the reversal approximates the order gradients are produced in, on the assumption that layers are used in the forward pass in the order they were defined.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU, single process, no process group needed): the bucketing rule, called directly',
              lang: 'python',
              text: 'import sys\nimport torch\nimport torch.distributed as dist\nimport torch.nn as nn\n\nm = nn.Sequential(nn.Linear(512, 512), nn.ReLU(),\n                  nn.Linear(512, 512), nn.ReLU(),\n                  nn.Linear(512, 512))\nps = list(m.parameters())\nsparse = [False] * len(ps)\n\nidx, lim = dist._compute_bucket_assignment_by_size(\n    ps, [dist._DEFAULT_FIRST_BUCKET_BYTES, 1024 * 1024], sparse)\nprint("two-limit :", idx)\nidx2, _ = dist._compute_bucket_assignment_by_size(ps, [sys.maxsize], sparse)\nprint("one-limit :", idx2)\nprint("bytes     :", [p.numel() * p.element_size() for p in ps])\n\n# two-limit : [[0], [1, 2], [3, 4], [5]]\n# one-limit : [[0, 1, 2, 3, 4, 5]]\n# bytes     : [1048576, 2048, 1048576, 2048, 1048576, 2048]',
            },
            table: {
              caption: 'the two size-limit regimes, from the run above',
              cols: ['limits passed', 'buckets produced', 'who passes it', 'consequence'],
              rows: [
                ['[first_bucket_bytes, bucket_cap]', '[[0], [1, 2], [3, 4], [5]]', 'DDP when find_unused_parameters is on', 'four all-reduces, overlappable with the backward'],
                ['[sys.maxsize]', '[[0, 1, 2, 3, 4, 5]]', 'DDP on the first iteration in the default configuration', 'one all-reduce after every gradient exists'],
              ],
            },
          },
          {
            h: 'the first iteration does not overlap',
            ps: [
              "Chapter 8 describes bucketed all-reduce firing as each bucket fills, which is what a steady-state step does. The first iteration is the exception, and the reason is written into `_ddp_init_helper` at v2.2.2: when `static_graph` is true or `find_unused_parameters` is false, the bucket size limit list is `[sys.maxsize]`, which produces exactly one bucket.",
              "The comment above that branch explains the hazard it is avoiding. Before the first backward has run, DDP only knows the order parameters were registered in, which for a model with control flow can be nothing like the order gradients actually become ready. Bucketing on a guess can fire a bucket's all-reduce early on one rank and late on another, and two ranks issuing collectives in different orders is the deadlock from lesson one.",
              "After the first backward, DDP has evidence instead of a guess. It records the order gradients arrived in and rebuilds the buckets against that order. Run twelve steps on the same three-layer model and the logging data says so directly: `has_rebuilt_buckets = 1`, gradients became ready in the order 5, 4, 3, 2, 1, 0, and the rebuilt buckets are `5 4`, `3 2`, `1 0`.",
              "Read those index pairs against the parameter list and the rebuild is legible. Index 5 is the last layer's bias and index 4 is its weight, so each rebuilt bucket is one layer's bias and weight together, in the order the backward produced them, at 1050624 bytes each. The initial assignment had split those same six parameters as `[[0], [1, 2], [3, 4], [5]]`, which pairs each weight with the next layer's bias, a grouping that only made sense before anything had been measured.",
              ">> Iteration one buys correctness with one big all-reduce. Iteration two onward buys overlap with the order it just measured.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU, gloo, 2 spawned processes, 12 steps): the rebuild, read out of DDP logging data',
              lang: 'python',
              text: 'def worker(rank, world):\n    os.environ["MASTER_ADDR"] = "127.0.0.1"\n    os.environ["MASTER_PORT"] = "29535"\n    dist.init_process_group("gloo", rank=rank, world_size=world)\n    torch.manual_seed(0)\n    model = nn.Sequential(nn.Linear(512, 512), nn.ReLU(),\n                          nn.Linear(512, 512), nn.ReLU(),\n                          nn.Linear(512, 512))\n    ddp = DDP(model, bucket_cap_mb=1)\n    x = torch.full((8, 512), float(rank + 1))\n    for _ in range(12):\n        ddp.zero_grad()\n        ddp(x).sum().backward()\n    if rank == 0:\n        d = ddp._get_ddp_logging_data()\n        for k in sorted(d):\n            if "bucket" in k or "rebuilt" in k or k == "iteration":\n                print(k, "=", d[k])\n\n# bucket_cap_bytes = 1048576\n# bucket_sizes = 3151872\n# gradient_as_bucket_view = 0\n# has_rebuilt_buckets = 1\n# iteration = 10\n# prev_iteration_grad_ready_order_indices = 5, 4, 3, 2, 1, 0\n# rebuilt_bucket_sizes = 1050624, 1050624, 1050624\n# rebuilt_per_bucket_param_indices = 5 4, 3 2, 1 0',
            },
          },
          {
            h: 'the knobs, and what each one costs',
            ps: [
              "Five constructor arguments change the ledger above, and each one trades a different resource. `bucket_cap_mb` sets how coarse the overlap is: smaller buckets start communicating sooner and send more, larger buckets send more efficiently and start later. `find_unused_parameters` makes DDP traverse the autograd graph every iteration to discover which parameters got no gradient, which costs a graph walk per step and is what you need when a model skips branches.",
              "`gradient_as_bucket_view` points each parameter's `.grad` at a slice of the bucket rather than at its own tensor, which removes one full copy of the gradients from memory. `static_graph` promises the model's graph never changes, which lets DDP keep the first iteration's analysis forever. `broadcast_buffers` decides whether buffers, batch-norm running statistics being the usual case, get re-broadcast from rank 0 at every forward.",
              "The communication itself is replaceable. DDP comm hooks let you install a function that runs instead of the plain all-reduce for each bucket, which is how bf16 gradient compression and PowerSGD are implemented. That is a mechanism this machine cannot demonstrate honestly at any useful scale, so it stays a pointer to the hook reference rather than a measurement.",
              "One knob lives in the training loop rather than the constructor, and chapter 4's accumulation lesson already owns it: `no_sync` suppresses the reduction for every micro-batch but the last. Read it there; it belongs to the accumulation story, not this one.",
            ],
            table: {
              caption: 'DDP constructor arguments, with the default this build reported',
              cols: ['argument', 'default here', 'what it changes', 'grounding'],
              rows: [
                ['bucket_cap_mb', '25 (26214400 bytes)', 'how many all-reduces a backward issues, and how early the first one starts', 'logging data from the run above'],
                ['find_unused_parameters', 'False (0)', 'off: bucketing from registration order on iteration one. On: a graph traversal every step', '_ddp_init_helper at v2.2.2'],
                ['gradient_as_bucket_view', 'False (0)', 'True points .grad at bucket storage, saving one copy of the gradients', 'logging data from the run above'],
                ['broadcast_buffers', 'True (1)', 'buffers re-broadcast from rank 0 at every forward', 'logging data from the run above'],
                ['static_graph', 'False', 'promises an unchanging graph, so the first iteration analysis is kept', 'distributed.py at v2.2.2'],
              ],
            },
          },
        ],
        readings: [
          { label: 'distributed.py at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/nn/parallel/distributed.py', note: '_ddp_init_helper carries the one-bucket first iteration and the reversal comment, near line 1078' },
          { label: 'reducer.cpp at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/csrc/distributed/c10d/reducer.cpp', note: 'the autograd hooks, mark_variable_ready, and the rebuild that the logging data reports' },
          { label: 'DDP design note', url: 'https://docs.pytorch.org/docs/stable/notes/ddp.html', note: 'the official account of construction, forward, backward, and optimizer step' },
          { label: 'DDP communication hooks', url: 'https://docs.pytorch.org/docs/stable/ddp_comm_hooks.html', note: 'replacing the per-bucket all-reduce, including the bf16 and PowerSGD hooks' },
        ],
        check: [
          {
            q: 'Your model has three big layers and you set bucket_cap_mb=1. How many all-reduces does the first backward issue?',
            a: 'One. With find_unused_parameters left False, DDP passes [sys.maxsize] as the only size limit on the first iteration, so every parameter lands in a single bucket and no overlap happens until the buckets are rebuilt.',
          },
          {
            q: 'Why does DDP reverse the bucket list before handing it to the reducer?',
            a: 'Because gradients arrive in roughly the reverse of the order parameters were defined in. Reversing makes bucket 0 hold the last layers, whose gradients are ready first, so its all-reduce can start while the rest of the backward is still running.',
          },
          {
            q: 'Each rank computed a different gradient. What ends up in .grad after DDP?',
            a: 'The mean across ranks. The reducer all-reduces the sum and divides by the world size, which is why two ranks holding gradients of 1.0 and 2.0 both end the step with 1.5.',
          },
        ],
        work: [
          { id: 'predict-buckets', label: 'write down the bucket assignment for one of your own models before running _compute_bucket_assignment_by_size on it', href: '#bucketing-is-a-function-you-can-call-yourself' },
          { id: 'watch-the-rebuild', label: 'run twelve steps under DDP and read the rebuilt bucket indices out of the logging data', href: '#the-first-iteration-does-not-overlap' },
          { id: 'grad-is-a-mean', label: 'prove the averaging on a model where you can compute the gradient by hand, on two ranks with different data', href: '#what-one-backward-leaves-in-grad' },
        ],
      },
      {
        id: 'the-unit-you-wrapped',
        num: 3,
        title: 'The unit you wrapped',
        lede: 'FSDP shards one flat tensor per unit rather than one parameter at a time, and the wrapping policy is what decides where the units fall. That one argument sets both the peak memory of a step and the number of collectives it issues.',
        goal: 'Given a model and an auto-wrap policy, count the FSDP units it produces, compute the size of one rank\u2019s shard including padding, and predict how many all-gathers and reduce-scatters a forward and backward will issue under each sharding strategy.',
        sections: [
          {
            h: 'one flat parameter per unit',
            ps: [
              "Wrap a two-layer model in FSDP with a policy that wraps each layer, print the parameters, and the original names are gone. What `named_parameters` returns is `_fsdp_wrapped_module.0._fsdp_wrapped_module._flat_param`, one tensor per unit. The `FlatParameter` docstring at v2.2.2 states the construction plainly: it is comprised of one or more original parameters, flattened and concatenated.",
              "The shard is a contiguous slice of that concatenation, and you can see exactly which slice. A `Linear(4, 4)` has 16 parameters, which flatten to 16 numbers in row-major order. On a world of two, rank 0's shard holds the first 8, which are rows 0 and 1 of the weight matrix, and rank 1's holds the second 8. Print rank 0's shard next to the original matrix and the numbers line up value for value.",
              "Nothing about that slicing respects the shape of the original parameter, which is the point worth carrying forward. A shard boundary can fall in the middle of a row, in the middle of a tensor, or between two different parameters that happened to be flattened next to each other. The layer's shape only comes back when the unit is gathered.",
              "`FSDP.summon_full_params` is the context manager that does the gathering on request. Inside it, the wrapped module's `weight` has shape `(4, 4)` again on every rank, which is the same all-gather a forward pass would have done, run because you asked rather than because a layer was about to execute.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU, gloo, 2 spawned processes): the shard, the original, and the summoned view',
              lang: 'text',
              text: 'rank 0: shard of layer0 [-0.0037434101, 0.2682217956, -0.4115225673, -0.3679695129,\n                        -0.1925771832, 0.1340786815, -0.0099065900, 0.3964447379]\nlayer0 full, row 0 and 1: [-0.0037434101, 0.2682217956, -0.4115225673, -0.3679695129]\n                          [-0.1925771832, 0.1340786815, -0.0099065900, 0.3964447379]\nrank 0: summoned layer0 shape (4, 4)\nrank 1: summoned layer0 shape (4, 4)\nrank 0: grad shard of layer0 [0.0, 0.0, 0.0, 0.0, 1.2830595970, 1.2830595970, 1.2830595970, 1.2830595970]\nrank 1: grad shard of layer0 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]',
              full: {
                label: 'the script that printed it (values shown above are the printed floats, trimmed to ten decimals)',
                text: 'import functools\nimport os\nimport torch\nimport torch.distributed as dist\nimport torch.multiprocessing as mp\nimport torch.nn as nn\nfrom torch.distributed.fsdp import FullyShardedDataParallel as FSDP\nfrom torch.distributed.fsdp.wrap import size_based_auto_wrap_policy\n\ndef worker(rank, world):\n    os.environ["MASTER_ADDR"] = "127.0.0.1"\n    os.environ["MASTER_PORT"] = "29540"\n    dist.init_process_group("gloo", rank=rank, world_size=world)\n    torch.manual_seed(0)\n    model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.ReLU(),\n                          nn.Linear(4, 4, bias=False))\n    ref = [p.detach().clone() for p in model.parameters()]\n    policy = functools.partial(size_based_auto_wrap_policy, min_num_params=8)\n    f = FSDP(model, auto_wrap_policy=policy, device_id=torch.device("cpu"))\n    flat = dict(f.named_parameters())\n    k0 = "_fsdp_wrapped_module.0._fsdp_wrapped_module._flat_param"\n    print(f"rank {rank}: shard of layer0 {flat[k0].detach().tolist()}")\n    if rank == 0:\n        print("layer0 full, row 0 and 1:", ref[0][0].tolist(), ref[0][1].tolist())\n    with FSDP.summon_full_params(f):\n        print(f"rank {rank}: summoned layer0 shape "\n              f"{tuple(f._fsdp_wrapped_module[0].weight.shape)}")\n    f(torch.full((2, 4), 1.0)).sum().backward()\n    print(f"rank {rank}: grad shard of layer0 {flat[k0].grad.tolist()}")\n    dist.destroy_process_group()\n\nif __name__ == "__main__":\n    mp.spawn(worker, args=(2,), nprocs=2, join=True)',
              },
            },
          },
          {
            h: 'the policy decides how many units exist',
            ps: [
              "An `auto_wrap_policy` is a predicate FSDP runs over the module tree, and every module it says yes to becomes its own unit with its own flat parameter. Three policies on the same model, two blocks of two `Linear(6, 6)` layers each, 144 parameters in total, produce three different shapes of job on a world of four.",
              "With no policy at all there is one unit and one flat parameter of 144, so each rank holds 36 numbers and the whole model is gathered at once. `ModuleWrapPolicy({Block})` produces three FSDP modules, the root plus one per block, and two flat parameters of 18 per rank. `size_based_auto_wrap_policy` with a 30-parameter floor wraps each `Linear` separately: five FSDP modules and four flat parameters of 9.",
              "Notice the count mismatch in the middle row, because it explains the structure. Three FSDP modules, two flat parameters. The root unit owns no parameters of its own once its children have claimed them all, so it gathers nothing and issues no collective. Counting `isinstance(m, FSDP)` and counting flat parameters answer two different questions.",
              "The tradeoff runs in one direction. One large unit means one all-gather and a peak that holds the entire model unsharded. Many small units mean many all-gathers and a peak that holds one unit unsharded. Wrapping per transformer block is the usual middle, which is what `ModuleWrapPolicy` was built to express.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU, gloo, 4 spawned processes): three policies, one model',
              lang: 'text',
              text: 'rank 0 | no policy: fsdp units 1 flat params [(36,)]\nrank 0 | ModuleWrapPolicy({Block}): fsdp units 3 flat params [(18,), (18,)]\nrank 0 | size_based(min=30): fsdp units 5 flat params [(9,), (9,), (9,), (9,)]',
              full: {
                label: 'the script, and the identical lines every rank printed',
                text: 'class Block(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.a = nn.Linear(6, 6, bias=False)\n        self.b = nn.Linear(6, 6, bias=False)\n    def forward(self, x):\n        return self.b(torch.relu(self.a(x)))\n\ndef build():\n    torch.manual_seed(0)\n    return nn.Sequential(Block(), Block())\n\ndef report(tag, f, rank):\n    units = sum(1 for m in f.modules() if isinstance(m, FSDP))\n    shapes = [tuple(p.shape) for _, p in f.named_parameters()]\n    print(f"rank {rank} | {tag}: fsdp units {units} flat params {shapes}")\n\ndef worker(rank, world):\n    os.environ["MASTER_ADDR"] = "127.0.0.1"\n    os.environ["MASTER_PORT"] = "29552"\n    dist.init_process_group("gloo", rank=rank, world_size=world)\n    cpu = torch.device("cpu")\n    report("no policy", FSDP(build(), device_id=cpu), rank)\n    report("ModuleWrapPolicy({Block})",\n           FSDP(build(), auto_wrap_policy=ModuleWrapPolicy({Block}), device_id=cpu), rank)\n    pol = functools.partial(size_based_auto_wrap_policy, min_num_params=30)\n    report("size_based(min=30)", FSDP(build(), auto_wrap_policy=pol, device_id=cpu), rank)\n    dist.destroy_process_group()\n\nif __name__ == "__main__":\n    mp.spawn(worker, args=(4,), nprocs=4, join=True)\n\n# rank 0 | no policy: fsdp units 1 flat params [(36,)]\n# rank 0 | ModuleWrapPolicy({Block}): fsdp units 3 flat params [(18,), (18,)]\n# rank 0 | size_based(min=30): fsdp units 5 flat params [(9,), (9,), (9,), (9,)]\n# (ranks 1, 2 and 3 printed the same three lines)',
              },
            },
            table: {
              caption: 'the same 144-parameter model, three policies, world of four',
              cols: ['policy', 'FSDP modules', 'flat parameters per rank', 'numbers held per rank'],
              rows: [
                ['none', '1', '1 of size 36', '36'],
                ['ModuleWrapPolicy({Block})', '3', '2 of size 18', '36'],
                ['size_based_auto_wrap_policy(min_num_params=30)', '5', '4 of size 9', '36'],
              ],
            },
          },
          {
            h: 'counting what a step actually sends',
            ps: [
              "The collectives FSDP issues are countable without a profiler. Wrap `dist.all_gather`, `dist.all_gather_into_tensor` and `dist.reduce_scatter_tensor` in counters before the forward, run one step, and read the counts off. On three blocks wrapped one unit each, the forward issues three all-gathers and the backward issues three more plus three reduce-scatters.",
              "Each of those numbers has an obvious owner. Three all-gathers in the forward is one per unit, gathering that unit's parameters just before the layer runs and freeing them again after. Three more in the backward is the same gathering repeated, because the parameters were discarded after the forward and the backward needs them again. Three reduce-scatters is the gradient path: each unit's gradient is reduced across ranks and scattered so every rank keeps only its own shard.",
              "That last one is the mechanical difference from DDP in a single word. DDP all-reduces, so every rank ends up with the whole gradient. FSDP reduce-scatters, so every rank ends up with the slice matching the parameters it owns, which is exactly what its shard of the optimizer state needs.",
              "One honest note on the printout: the gathers went through the list-form `dist.all_gather` here rather than `all_gather_into_tensor`, which is the path this build took on gloo. The count is the fact being taught, not the choice of entry point.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU, gloo, 2 spawned processes): counted collectives, three wrapped blocks',
              lang: 'python',
              text: 'seen = Counter()\nfor n in ("all_gather_into_tensor", "reduce_scatter_tensor", "all_gather", "all_reduce"):\n    fn = getattr(dist, n)\n    def wrapped(*a, _n=n, _fn=fn, **k):\n        seen[_n] += 1\n        return _fn(*a, **k)\n    setattr(dist, n, wrapped)\n\nmodel = nn.Sequential(Block(), Block(), Block())\nf = FSDP(model, auto_wrap_policy=ModuleWrapPolicy({Block}),\n         device_id=torch.device("cpu"))\nout = f(torch.ones(2, 6))\nprint("after forward :", dict(seen))\nout.sum().backward()\nprint("after backward:", dict(seen))\nprint("fsdp units    :", sum(1 for m in f.modules() if isinstance(m, FSDP)))\n\n# after forward : {"all_gather": 3}\n# after backward: {"all_gather": 6, "reduce_scatter_tensor": 3}\n# fsdp units    : 4',
            },
          },
          {
            h: 'the strategy dial, measured',
            ps: [
              "Swap `sharding_strategy=ShardingStrategy.SHARD_GRAD_OP` into the same script and the backward all-gathers disappear: three in the forward, three in total, three reduce-scatters unchanged. The `ShardingStrategy` docstring at v2.2.2 predicts exactly that, saying SHARD_GRAD_OP unshards before the forward, does not reshard after the forward, and only reshards after the backward.",
              "So the dial is a memory-against-bandwidth choice, and the measured counts put a number on it. FULL_SHARD gathers twice per unit per step and holds one unit unsharded at a time. SHARD_GRAD_OP gathers once per unit per step and holds the whole model unsharded between forward and backward. NO_SHARD replicates and all-reduces, which is DDP's deal wearing FSDP's interface.",
              "The two hybrid strategies are the ones a single machine cannot show you. HYBRID_SHARD applies FULL_SHARD inside a node and replicates across nodes, so the expensive collectives stay on the fast intra-node links. That is a claim about a network this course has no second node to measure, and it stays a citation.",
            ],
            table: {
              cols: ['strategy', 'all-gathers per unit per step', 'gradient collective', 'measured here'],
              caption: 'sharding strategies, counts from the two runs above and the v2.2.2 docstring',
              rows: [
                ['FULL_SHARD', '2 (forward, then backward)', 'reduce-scatter', 'yes: 6 all-gathers, 3 reduce-scatters, 3 units'],
                ['SHARD_GRAD_OP', '1 (forward only)', 'reduce-scatter', 'yes: 3 all-gathers, 3 reduce-scatters, 3 units'],
                ['NO_SHARD', '0', 'all-reduce', 'no: docstring at v2.2.2'],
                ['HYBRID_SHARD', 'FULL_SHARD within a node', 'reduce-scatter in node, all-reduce across', 'no: needs two nodes, docstring at v2.2.2'],
              ],
            },
          },
          {
            h: 'padding, and the arithmetic of one shard',
            ps: [
              "A flat parameter has to divide evenly by the world size, and models do not cooperate. Wrap a `Linear(3, 3)` with 9 parameters on a world of four and each rank reports a shard of 3, which is 12 numbers spread over four ranks. Three of those numbers are padding that exists only so the division works.",
              "The `FlatParameter` docstring names both sizes for this reason, `_unpadded_unsharded_size` and `_padded_unsharded_size`, the second being the first with right-hand-side padding for divisibility by the world size. On a small model that overhead is a third of the tensor. On a real one it is a rounding error, which is why nobody notices until they wrap something tiny and the arithmetic stops matching.",
              "This is the same discipline the sharding lessons on the jax path apply from the other side of the fence: a shard is a number you can compute before you run anything, and if the number you computed does not match the number the framework reports, one of your assumptions about the layout is wrong.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU, gloo, 4 spawned processes): nine parameters over four ranks',
              lang: 'python',
              text: 'odd = nn.Linear(3, 3, bias=False)          # 9 parameters, world of 4\nfo = FSDP(odd, device_id=torch.device("cpu"))\np = next(fo.parameters())\nprint(f"rank {rank} | 9 params over world {world}: shard {tuple(p.shape)}")\n\n# rank 0 | 9 params over world 4: shard (3,)\n# rank 1 | 9 params over world 4: shard (3,)\n# rank 2 | 9 params over world 4: shard (3,)\n# rank 3 | 9 params over world 4: shard (3,)\n# 4 ranks x 3 = 12, so 3 of the 12 are padding',
            },
          },
          {
            h: 'what this machine cannot show you',
            ps: [
              "FSDP on a CPU-only build needs one argument that a GPU job never passes. Construct it without `device_id` and `_init_device_handle` falls through to `torch.device(\"cuda\", torch.cuda.current_device())`, which on this build raises `AssertionError: Torch not compiled with CUDA enabled` from inside the FSDP constructor. Passing `device_id=torch.device(\"cpu\")` is what made every run in this lesson possible, and it is the first thing to try when FSDP refuses to build on a machine with no GPU.",
              "Three features stay cited rather than measured. Mixed precision through `MixedPrecision` is a real memory and bandwidth win on hardware with fast reduced-precision arithmetic, and measuring it on a CPU would produce a number that means nothing. `cpu_offload` moves shards to host memory between uses, which only has a point when the shards were somewhere else. `limit_all_gathers` throttles prefetching against the allocator, and the allocator it is protecting is CUDA's.",
              "The other absence is a version boundary rather than a hardware one. Everything in this lesson is the FSDP that ships in 2.2.2, wrapper classes and flat parameters. The rewrite that arrived later, built on DTensor with per-parameter sharding and a `fully_shard` function instead of a wrapper class, is the next lesson's machinery applied to this lesson's problem. Read it in the current docs and hold the two apart by version, because the printouts here are 2.2.2 and will not match.",
            ],
          },
        ],
        readings: [
          { label: 'FSDP API reference', url: 'https://docs.pytorch.org/docs/stable/fsdp.html', note: 'the constructor arguments, the sharding strategies, and the wrapping policies' },
          { label: '_flat_param.py at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/distributed/fsdp/_flat_param.py', note: 'the FlatParameter docstring: flattened and concatenated originals, and the padded and unpadded sizes' },
          { label: '_init_utils.py at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/distributed/fsdp/_init_utils.py', note: '_init_device_handle, which is why a CPU-only build needs device_id passed explicitly' },
          { label: 'PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel', url: 'https://arxiv.org/abs/2304.11277', note: 'the paper behind the design, including why the unit rather than the parameter is the sharding granularity' },
        ],
        check: [
          {
            q: 'A policy wraps every transformer block and your model has 24 of them. How many all-gathers does one FULL_SHARD step issue?',
            a: 'Forty-eight, two per block: one in the forward when the block is about to run, and one in the backward because the parameters were freed after the forward. SHARD_GRAD_OP would issue twenty-four by keeping them resident.',
          },
          {
            q: 'Your model has 10 parameters and the world size is 4. How large is one rank\u2019s shard?',
            a: 'Three. The flat parameter is padded up to 12 so the world size divides it evenly, and two of those twelve numbers are padding that no original parameter maps to.',
          },
          {
            q: 'Why does FSDP reduce-scatter gradients where DDP all-reduces them?',
            a: 'Because each rank only owns a shard of the parameters and a shard of the optimizer state, so it only needs the matching shard of the gradient. Handing it the whole gradient would be traffic it has no use for.',
          },
        ],
        work: [
          { id: 'count-the-units', label: 'run all three policies on one of your own models and predict the flat parameter sizes before printing them', href: '#the-policy-decides-how-many-units-exist' },
          { id: 'count-the-collectives', label: 'count the collectives under FULL_SHARD and SHARD_GRAD_OP on the same model, and write down what you paid for the difference', href: '#counting-what-a-step-actually-sends' },
          { id: 'shard-arithmetic', label: 'pick a model whose parameter count does not divide by your world size and compute the padding before FSDP tells you', href: '#padding-and-the-arithmetic-of-one-shard' },
        ],
      },
      {
        id: 'mesh-and-dtensor',
        num: 4,
        title: 'Mesh, placement, and the collective it implies',
        lede: 'A DTensor carries its own layout: a mesh of devices, and one placement per mesh axis. Ask it to change layout and the collective needed to get there is derived rather than written, which makes the question of which collective runs into something you can check.',
        goal: 'Given a device mesh and a pair of placements, name the collective a redistribute inserts, explain what a Partial placement is promising, and say which collective a colwise-then-rowwise tensor-parallel pair costs per forward.',
        sections: [
          {
            h: 'a mesh is a grid with names on its axes',
            ps: [
              "`init_device_mesh(\"cpu\", (2, 2), mesh_dim_names=(\"dp\", \"tp\"))` on a world of four builds a 2 by 2 grid and gives its axes names. Print it and you get `DeviceMesh([[0, 1], [2, 3]])`: ranks 0 and 1 sit along the tp axis of the first dp row, ranks 2 and 3 along the second.",
              "The names are the working part. Once an axis is called `dp` and another `tp`, a layout stops being a tuple of integers you have to keep straight and becomes a statement about which parallelism runs where. A subgroup for either axis comes out of the mesh rather than out of a hand-built `new_group` call, which is the same coordination lesson one showed, one level up.",
              "This grid is the same object the jax path's sharding chapter builds with a mesh and a PartitionSpec, and the same grid the xla path's SPMD chapter says the partitioner rewrites a program against. Three courses, one idea, arrived at from three directions. The reason to state that here rather than teach it three times is that the mechanism below is where PyTorch's version differs.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU, gloo, 4 spawned processes): a named 2 by 2 mesh',
              lang: 'python',
              text: 'from torch.distributed.device_mesh import init_device_mesh\n\nmesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))\nif rank == 0:\n    print("mesh:", mesh)\n    print("mesh shape:", mesh.shape, "dim names:", mesh.mesh_dim_names)\n\n# mesh: DeviceMesh([[0, 1], [2, 3]])\n# mesh shape: (2, 2) dim names: ("dp", "tp")',
            },
          },
          {
            h: 'one placement per mesh axis',
            ps: [
              "A DTensor is a global tensor plus a list of placements, one entry per mesh axis, each saying how that axis divides the tensor. `Shard(0)` on the first axis and `Shard(1)` on the second means: split rows across the dp axis, split columns across the tp axis.",
              "Distribute a 4 by 4 tensor of 0 through 15 that way and the block each rank holds is exactly the one the two placements imply. Rank 0 gets `[0, 1, 4, 5]`, the top-left 2 by 2. Rank 3 gets `[10, 11, 14, 15]`, the bottom-right. The DTensor's `.shape` still reports `(4, 4)` on every rank while `.to_local().shape` reports `(2, 2)`, which is the difference between what the program is computing with and what this process is holding.",
              "The third placement type has no equivalent in a shape. `Replicate()` says an axis does not divide the tensor at all, so every rank along it holds the same values. `_Partial` says something stranger, and the section after next is where it earns its own paragraph.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU, gloo, 4 spawned processes): a 4 by 4 tensor split both ways at once',
              lang: 'python',
              text: 'from torch.distributed._tensor import Shard, distribute_tensor\n\nbig = torch.arange(16.).reshape(4, 4)\nd = distribute_tensor(big, mesh, [Shard(0), Shard(1)])\nprint(f"rank {rank}: global {tuple(d.shape)} local {tuple(d.to_local().shape)} "\n      f"{d.to_local().flatten().tolist()}")\n\n# rank 0: global (4, 4) local (2, 2) [0.0, 1.0, 4.0, 5.0]\n# rank 1: global (4, 4) local (2, 2) [2.0, 3.0, 6.0, 7.0]\n# rank 2: global (4, 4) local (2, 2) [8.0, 9.0, 12.0, 13.0]\n# rank 3: global (4, 4) local (2, 2) [10.0, 11.0, 14.0, 15.0]\n# placements: (Shard(dim=0), Shard(dim=1))',
            },
          },
          {
            h: 'redistribute is where a collective gets chosen',
            ps: [
              "Ask a DTensor for a different layout and it works out what has to move. `CommDebugMode` is the instrument that makes the answer visible: a dispatch mode that counts functional collectives inside its context, shipped in `torch.distributed._tensor.debug` at 2.2.2 for exactly this purpose.",
              "Four transitions, on a one-dimensional mesh of four, and the counts are not symmetric. Going from `Shard(0)` to `Replicate()` costs one all-gather, because every rank has to receive what the others hold. Going the other way, `Replicate()` to `Shard(0)`, costs nothing at all: every rank already has every value, so it just keeps its slice and drops the rest.",
              "That asymmetry is the whole reason to think in placements rather than in calls. Sharding an already-replicated tensor is free. Replicating a sharded one is a collective. The gym's naming drill at GYM\u00b711 works the same skill from the other end, giving you the before-and-after tensors and asking which collective produced them; this section gives you the placements and asks which collective they imply.",
              ">> Sharding a replicated tensor is free. Replicating a sharded one is an all-gather.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU, gloo, 4 spawned processes, 1-D mesh): counted collectives per transition',
              lang: 'text',
              text: 'Shard(0) -> Replicate : {"all_gather_into_tensor": 1}\nReplicate -> Shard(0) : {}\nShard(0) -> Shard(1)  : {"all_gather_into_tensor": 1}\nShard(1) @ Shard(0)   : {} -> placements (_Partial(reduce_op=RedOpType.SUM),)\nReplicate @ Shard(1)  : {} -> placements (Shard(dim=1),)\nfull_tensor of the matmul: {"all_reduce": 1}\nresult row 0: [6.0, 6.0, 6.0, 6.0]',
              full: {
                label: 'the script that produced every line',
                text: 'import os\nimport torch\nimport torch.distributed as dist\nimport torch.multiprocessing as mp\nfrom torch.distributed._tensor import Replicate, Shard, distribute_tensor\nfrom torch.distributed._tensor.debug import CommDebugMode\nfrom torch.distributed.device_mesh import init_device_mesh\n\ndef counts(mode):\n    return {str(k).replace("c10d_functional.", ""): v\n            for k, v in mode.get_comm_counts().items()}\n\ndef worker(rank, world):\n    os.environ["MASTER_ADDR"] = "127.0.0.1"\n    os.environ["MASTER_PORT"] = "29542"\n    dist.init_process_group("gloo", rank=rank, world_size=world)\n    mesh = init_device_mesh("cpu", (4,))\n    x = torch.arange(16.).reshape(4, 4)\n    w = torch.ones(4, 4)\n\n    sharded = distribute_tensor(x, mesh, [Shard(0)])\n    m = CommDebugMode()\n    with m:\n        r = sharded.redistribute(mesh, [Replicate()])\n    if rank == 0:\n        print("Shard(0) -> Replicate :", counts(m))\n\n    m2 = CommDebugMode()\n    with m2:\n        s = r.redistribute(mesh, [Shard(0)])\n    if rank == 0:\n        print("Replicate -> Shard(0) :", counts(m2))\n\n    m3 = CommDebugMode()\n    with m3:\n        s.redistribute(mesh, [Shard(1)])\n    if rank == 0:\n        print("Shard(0) -> Shard(1)  :", counts(m3))\n\n    xs = distribute_tensor(x, mesh, [Shard(1)])\n    ws = distribute_tensor(w, mesh, [Shard(0)])\n    m4 = CommDebugMode()\n    with m4:\n        out = torch.matmul(xs, ws)\n    if rank == 0:\n        print("Shard(1) @ Shard(0)   :", counts(m4), "-> placements", out.placements)\n\n    m6 = CommDebugMode()\n    with m6:\n        full = out.full_tensor()\n    if rank == 0:\n        print("full_tensor of the matmul:", counts(m6))\n        print("result row 0:", full[0].tolist())\n    dist.destroy_process_group()\n\nif __name__ == "__main__":\n    mp.spawn(worker, args=(4,), nprocs=4, join=True)',
              },
            },
            table: {
              caption: 'placement transitions and the collectives they cost, counted on a 1-D mesh of four',
              cols: ['from', 'to', 'collectives counted', 'why'],
              rows: [
                ['Shard(0)', 'Replicate()', 'all_gather_into_tensor 1', 'every rank needs what every other rank holds'],
                ['Replicate()', 'Shard(0)', 'none', 'each rank already holds the values and keeps its slice'],
                ['Shard(0)', 'Shard(1)', 'all_gather_into_tensor 1', 'this build routes the reshuffle through a gather'],
                ['_Partial', 'Replicate()', 'all_reduce 1', 'the deferred sum finally has to happen'],
              ],
            },
          },
          {
            h: 'partial is a promise to reduce later',
            ps: [
              "Multiply two DTensors whose sharding does not line up and something unexpected happens: no collective runs. A tensor sharded on its columns times a weight sharded on its rows is a valid local matmul on every rank, and each rank's answer is a partial sum, correct in shape and incomplete in value. The result comes back with placement `_Partial(reduce_op=RedOpType.SUM)`, and the count is zero.",
              "The reduction is not skipped, it is owed. Call `full_tensor()` on that result and one all_reduce fires, and the value comes out right: row 0 of the answer is `[6.0, 6.0, 6.0, 6.0]`, which is 0 + 1 + 2 + 3 against a weight of ones. Deferring the reduction is what lets a chain of operations run without paying for a collective between every pair of them.",
              "The same tracking runs the other way. A replicated input times a column-sharded weight needs no communication and the answer is already `Shard(1)`, ready for a row-sharded layer to consume. Placement algebra is doing the bookkeeping a hand-written distributed program would do in comments.",
              "Chapter 8 sets the goal of predicting which collective a given call inserts. This is the mechanism that makes it predictable: the placements of the inputs determine the placement of the output, and a collective appears only where a placement has to change.",
            ],
          },
          {
            h: 'parallelize_module writes the placements for you',
            ps: [
              "The tensor-parallel API is a thin layer over everything above. Hand `parallelize_module` a mesh and a dictionary saying `ColwiseParallel` for the up-projection and `RowwiseParallel` for the down-projection, and it replaces both weights with DTensors carrying the placements those names describe.",
              "Print them and the naming makes sense: the colwise layer's weight is `Shard(0)` on a `(8, 4)` weight, which splits the output features, and the rowwise layer's is `Shard(1)` on a `(4, 8)` weight, which splits the input features. Both ranks hold a local `(4, 4)`. The two shardings are chosen to fit together, so the intermediate activation stays sharded across the ReLU and never needs gathering.",
              "One collective runs in the whole forward: a single all_reduce, at the end, where the row-sharded output's partial sums have to be combined. That is the classic tensor-parallel MLP, and here it is a count rather than a diagram.",
              "The output comes back as an `AsyncCollectiveTensor` rather than a plain tensor, which is the wrapper that lets the all_reduce be issued now and waited on at first use. The xla path's collectives chapter describes the same split as an async start and done pair in the compiled schedule. Both are the same idea: launch the communication, keep computing, wait only when the value is genuinely needed.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU, gloo, 2 spawned processes): a tensor-parallel MLP, and its one collective',
              lang: 'python',
              text: 'from torch.distributed.tensor.parallel import (\n    ColwiseParallel, RowwiseParallel, parallelize_module)\n\nmesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))\ntp = parallelize_module(MLP(), mesh,\n                        {"up": ColwiseParallel(), "down": RowwiseParallel()})\nmode = CommDebugMode()\nwith mode:\n    out = tp(torch.ones(2, 4))\n\n# up.weight  : DTensor (8, 4) (Shard(dim=0),)\n# down.weight: DTensor (4, 8) (Shard(dim=1),)\n# local shapes: (4, 4) (4, 4)\n# forward collectives: {"all_reduce": 1}\n# out: AsyncCollectiveTensor (2, 4)',
            },
          },
          {
            h: 'where this lands, and what to hold loosely',
            ps: [
              "Two version facts are worth carrying out of this lesson. At 2.2.2 the tensor type lives at `torch.distributed._tensor`, with the leading underscore that means the API is not promised to hold still, and `CommDebugMode` says in its own docstring that it counts functional collectives only, so the FSDP counts from the previous lesson had to be taken by wrapping `dist` functions instead. Both of those have moved since. Check the import path against your own build before copying a line.",
              "What has not moved is the shape of the idea. A named grid of devices, a per-axis statement of how each tensor sits on it, and a rule that derives the communication from a change in that statement. The jax path teaches it as a mesh plus a PartitionSpec handed to a compiler, and the xla path teaches what the compiler then does to the module. PyTorch does it eagerly, one operation at a time, which is why you can count the collectives with a dispatch mode instead of reading a compiled schedule.",
              "The gym station at GYM\u00b711 is the practice half of this. It shows four ranks' tensors before and after one collective, taken from a real four-process gloo run, and asks which collective ran. Do a streak of five there once the placement transitions above feel obvious, because naming a collective from its effect and predicting one from a placement change are the same knowledge tested from two sides.",
            ],
          },
        ],
        readings: [
          { label: 'DeviceMesh recipe', url: 'https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html', note: 'init_device_mesh, named axes, and getting a subgroup out of a mesh instead of new_group' },
          { label: 'redistribute.py at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/distributed/_tensor/redistribute.py', note: 'the transition table: which placement change lowers to which collective' },
          { label: 'placement_types.py at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/distributed/_tensor/placement_types.py', note: 'Shard, Replicate, and the Partial placement that defers a reduction' },
          { label: 'Tensor parallel API', url: 'https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html', note: 'ColwiseParallel, RowwiseParallel, and the input and output layouts each one assumes' },
        ],
        check: [
          {
            q: 'A DTensor is Replicate() and you redistribute it to Shard(0). Which collective runs?',
            a: 'None. Every rank already holds every value, so each one keeps the slice it owns under the new placement and discards the rest. The reverse direction, Shard(0) to Replicate(), is the one that costs an all-gather.',
          },
          {
            q: 'A matmul returns a DTensor with a Partial placement and no collective ran. Is the result wrong?',
            a: 'No, it is incomplete on purpose. Each rank holds a partial sum, and the all-reduce that combines them is deferred until something asks for the full value, for example full_tensor or a redistribute to Replicate.',
          },
          {
            q: 'How many collectives does a colwise-then-rowwise tensor-parallel MLP issue per forward?',
            a: 'One all-reduce, at the end. The colwise layer leaves its output sharded on the feature axis, the rowwise layer consumes that sharding directly, and only its output needs the partial sums combined.',
          },
        ],
        work: [
          { id: 'predict-the-transitions', label: 'write down the collective for six placement transitions of your own, then count them with CommDebugMode', href: '#redistribute-is-where-a-collective-gets-chosen' },
          { id: 'block-by-hand', label: 'compute which block of a 4 by 4 tensor each rank of a 2 by 2 mesh holds before you print one', href: '#one-placement-per-mesh-axis' },
          { id: 'placement-to-effect', label: 'take one before-and-after pair from the gym drill and write the placement change that would have produced it', href: '/gym/pytorch#collectives' },
        ],
      },
    ],
  },
]
