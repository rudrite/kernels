// New file: site/src/data/lessons/torch-training-run.ts
// Below chapter 12's capstone: the checkpoint as an artifact rather than a
// dictionary, resume failures sorted by how many steps they stay quiet, and
// what a preemptible multi-rank run adds to both. Every snippet here and every
// number it prints ran on this machine, torch 2.2.2 on CPU, 2026-08-15; the
// distributed captures are gloo with two spawned processes. The source excerpt
// is verbatim from pytorch at the v2.2.2 tag. The TPU numbers this arc points
// at are LAB·P4's, quoted and never re-derived, and the sharded-checkpoint
// section is documentation rather than a capture.
import type { UnitLessons } from './index'

export const TORCH_TRAINING_RUN_LESSONS: UnitLessons[] = [
  {
    unit: 'pt:training-run',
    lessons: [
      {
        id: 'the-checkpoint-is-a-file',
        num: 1,
        title: 'The checkpoint is a file',
        lede: 'Chapter four weighed a checkpoint piece by piece and measured what each missing piece costs. This lesson takes the same list and asks what the artifact carrying it is actually made of: which random streams a saved block reaches, what a state dict does to a tensor that is a view, and what is left on disk when the write does not finish.',
        goal: 'Given a training loop, name every random stream a step can touch and say which one a saved RNG block restores, predict the size of the file a state dict produces when its tensors are views of something larger, and write a save that a crash halfway through cannot destroy.',
        sections: [
          {
            h: 'what the next process needs from this one',
            ps: [
              'A checkpoint is a message to a process that does not exist yet. That process will build a model, an optimizer and a scheduler from scratch, overwrite their state with what it reads, and carry on as though nothing happened. Everything it needs has to be in the message, and nothing in the message is checked against the run that wrote it.',
              'The contents of that message were settled in chapter four\'s arc. Five things go in, four of them serialize, and the third lesson there measured what each omission costs over the six steps after a restore. None of that is repeated here, and the itemized table is worth having open beside this page.',
              'What is left is the artifact. A file has a size, a write has a duration, a process has a set of random streams that a single saved block does not cover, and none of those facts appear in a dictionary of tensors. They appear later, as a checkpoint that is a hundred times too large, as a resume that diverges on step one, or as a directory holding a file that loads and is wrong.',
            ],
          },
          {
            h: 'one restore, one stream',
            ps: [
              'A step reads randomness from more places than a loop makes obvious. The batch index comes from the global generator. A `DataLoader` built with `shuffle=True` and a generator of your own reads that one instead. An augmentation written with the standard library reads Python\'s `random`. On an accelerator, an op that draws numbers reads that device\'s generator, not the CPU\'s.',
              '`torch.get_rng_state()` returns one of them. Save it, restore it, and the two draws that did not come from the default CPU generator carry straight on from where they were: the private generator gives -0.293429 where it first gave 1.540996, and `random.random()` gives 0.757954 where it first gave 0.844422. Neither raises, and neither is visible in a loss until the numbers have already gone somewhere else.',
              'The count that matters for your own loop is the number of streams, not the number of lines. Chapter five\'s arc derives worker seeding from a base seed the main process draws out of whichever generator the loader was handed, so a stream you restore is one the workers inherit. Everything else on the list is yours to save by hand or to design out of the loop.',
              '>> A checkpoint with one RNG entry in it is a claim that the loop has one random stream.',
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): three streams, one of them restored',
              lang: 'python',
              text: 'import random\nimport torch\n\ntorch.manual_seed(0)\nloader_gen = torch.Generator().manual_seed(0)\nrandom.seed(0)\nsaved = torch.get_rng_state()          # the whole "rng" entry of a typical checkpoint\n\nfirst = (torch.randn(1).item(), torch.randn(1, generator=loader_gen).item(), random.random())\ntorch.set_rng_state(saved)             # the whole restore\nagain = (torch.randn(1).item(), torch.randn(1, generator=loader_gen).item(), random.random())\n\nprint([round(v, 6) for v in first])\nprint([round(v, 6) for v in again])\nprint([a == b for a, b in zip(first, again)])\nprint(torch.cuda.is_available(), torch.cuda.get_rng_state_all())\n\n# [1.540996, 1.540996, 0.844422]\n# [1.540996, -0.293429, 0.757954]\n# [True, False, False]\n# False []',
            },
            table: {
              caption: 'the random streams a training step can read, and what a saved CPU block reaches (verified, torch 2.2.2 CPU)',
              cols: ['stream', 'read by', 'restored by torch.set_rng_state'],
              rows: [
                ['the default CPU generator', 'randint, randn, dropout, and any op you did not hand a generator', 'yes, and this is the one the 5056-byte block holds'],
                ['a torch.Generator you constructed', 'a DataLoader\'s shuffle, and anything you passed it to', 'no; save its own get_state beside the block'],
                ['Python\'s random module', 'transforms and samplers written without torch', 'no; random.getstate is a separate save'],
                ['a device generator', 'the same ops, once the tensors live on an accelerator', 'no; the CPU setter is documented as CPU-only'],
              ],
            },
          },
          {
            h: 'the device the CPU block does not reach',
            ps: [
              'The docstring says it outright, in four lines that have survived every release since. `set_rng_state` works for CPU. For CUDA the note points you at `manual_seed`, which is a reseed rather than a restore, and a reseed puts the stream somewhere defined rather than somewhere continuous.',
              'The body of `manual_seed` explains why the note has to exist. One call fans out to every device family torch knows about, cuda first, then mps, then xpu, then whatever a custom backend registered, before it finally seeds the default CPU generator on the last line. Seeding is device-wide; state capture is not.',
              'So a bitwise resume on an accelerator needs the device\'s own state in the dictionary, `torch.cuda.get_rng_state_all()` for a multi-device host, and its matching setter on the way back. This machine has no such device, `torch.cuda.get_rng_state_all()` returns an empty list above, and that empty list is exactly what a CPU-authored checkpoint quietly carries onto a GPU run.',
            ],
            code: {
              caption: 'verbatim, torch/random.py at the v2.2.2 tag: set_rng_state at 9-18 and the body of manual_seed at 36-51, with the middle of the file elided',
              lang: 'python',
              text: 'def set_rng_state(new_state: torch.Tensor) -> None:\n    r"""Sets the random number generator state.\n\n    .. note: This function only works for CPU. For CUDA, please use\n             torch.manual_seed(seed), which works for both CPU and CUDA.\n    ...\n    """\n    default_generator.set_state(new_state)\n\n...\n\n    seed = int(seed)\n    import torch.cuda\n\n    if not torch.cuda._is_in_bad_fork():\n        torch.cuda.manual_seed_all(seed)\n\n    import torch.mps\n    if not torch.mps._is_in_bad_fork():\n        torch.mps.manual_seed(seed)\n\n    if hasattr(torch, \'xpu\') and not torch.xpu._is_in_bad_fork():\n        torch.xpu.manual_seed_all(seed)\n\n    _seed_custom_device(seed)\n\n    return default_generator.manual_seed(seed)',
            },
          },
          {
            h: 'the flag that is not in the dictionary',
            ps: [
              'A state dict holds parameters and buffers. Put a model in `eval()`, print the keys, and the batch-norm running statistics are there along with `num_batches_tracked`, while the thing that decides how those statistics get used is not.',
              '`model.training` is a plain Python attribute on the module and every submodule, and it never enters serialization in either direction. Load the eval-mode dictionary above into a freshly built model and that model reports `training` as True, because construction set it and the load never touched it.',
              'The direction that costs you is the other one. A resume script that runs a validation pass before continuing leaves every module in eval mode, dropout stops firing, batch norm switches to its running estimates, and the loop trains on. Lesson two measures what that does to a hundred steps of loss.',
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the keys, and the flag that is not among them',
              lang: 'python',
              text: 'import torch\nfrom torch import nn\n\nmodel = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5), nn.BatchNorm1d(4))\nmodel.eval()\nprint(sorted(model.state_dict()))\n# [\'0.bias\', \'0.weight\', \'2.bias\', \'2.num_batches_tracked\', \'2.running_mean\',\n#  \'2.running_var\', \'2.weight\']\nprint(model.training, any("training" in k for k in model.state_dict()))   # False False\n\nfresh = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5), nn.BatchNorm1d(4))\nfresh.load_state_dict(model.state_dict())\nprint(fresh.training)                # True: the flag did not travel',
            },
          },
          {
            h: 'a ten-element tensor and a four-megabyte file',
            ps: [
              'Serialization keeps storage sharing, which is a good property with an expensive corner. Save a ten-element slice of a million-element buffer and the file is 4,001,101 bytes. Clone it first and the same ten elements cost 1,170.',
              'The zip listing shows where the size went. A torch checkpoint is a zip archive, and the record named `view/data/0` holds 4,000,000 bytes, the whole storage the slice was reading through. The slice itself is 40 bytes of that, and the archive has no way to express a partial storage.',
              'Any parameter carved out of one flat buffer has this shape, and so does any metric tensor sliced from a longer log. The symptom is a checkpoint many times larger than the parameter count justifies, and the fix is a clone on the way into the dictionary. The serialization note upstream documents the same behaviour with a nine-element example.',
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the same ten numbers, saved two ways',
              lang: 'python',
              text: 'import os\nimport zipfile\n\nimport torch\n\nflat = torch.zeros(1_000_000)          # one buffer the parameters were carved from\nhead = flat[:10]\n\ntorch.save({"head": head}, "view.pt")\ntorch.save({"head": head.clone()}, "clone.pt")\nprint(os.path.getsize("view.pt"), os.path.getsize("clone.pt"))   # 4001101 1170\nprint(head.numel() * head.element_size(), head.untyped_storage().nbytes())\n# 40 4000000\nwith zipfile.ZipFile("view.pt") as z:\n    print([(i.filename, i.file_size) for i in z.infolist() if i.file_size > 100])\n# [(\'view/data.pkl\', 168), (\'view/data/0\', 4000000)]',
            },
          },
          {
            h: 'the save that destroyed the last good checkpoint',
            ps: [
              '`torch.save` opens its destination for writing before it knows whether the write will succeed, so the previous checkpoint stops existing the moment the new save begins. Hand it a file object that fails on the fourth write, the way a full disk does, and the 3,120-byte checkpoint holding step 10 is a 64-byte fragment by the time the exception arrives.',
              'Reading it back raises, which is the merciful half. `PytorchStreamReader failed reading zip archive: failed finding central directory` is a torn file announcing itself, because the zip central directory is written last and a partial archive has none. What you have lost is not detectable from this process at all: the run that could have resumed from step 10 no longer can.',
              'Writing to a temporary name and renaming afterwards is the whole fix, and `os.replace` is the call that swaps the two names in one step within a filesystem. The same failing writer now wrecks `safe.pt.partial` and leaves `safe.pt` at 3,152 bytes and step 10, still loadable. Keeping the last two files rather than one is the other half, for the case where the failure is in the run rather than in the disk.',
              '>> A checkpoint that exists is not a checkpoint that loads, and a save in progress is a checkpoint you do not have.',
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): a write that dies partway, and the checkpoint it takes with it',
              lang: 'python',
              text: 'import os\n\nimport torch\nfrom torch import nn\n\n\nclass Dying:\n    """a disk that fills up on the fourth write"""\n\n    def __init__(self, path):\n        self.f = open(path, "wb")\n        self.n = 0\n\n    def write(self, chunk):\n        self.n += 1\n        if self.n > 3:\n            raise OSError(28, "No space left on device")\n        return self.f.write(chunk)\n\n    def flush(self):\n        self.f.flush()\n\n    def close(self):\n        self.f.close()\n\n\ntorch.manual_seed(0)\nmodel = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1))\n\n\ndef ckpt(step):\n    return {"model": model.state_dict(), "step": step}\n\n\ntorch.save(ckpt(10), "run.pt")                       # the last good checkpoint\nprint(os.path.getsize("run.pt"), torch.load("run.pt")["step"])   # 3120 10\n\ntry:\n    torch.save(ckpt(20), Dying("run.pt"))            # the save that does not finish\nexcept Exception as err:\n    print(type(err).__name__, err)\nprint(os.path.getsize("run.pt"))\ntry:\n    torch.load("run.pt")\nexcept Exception as err:\n    print(type(err).__name__, str(err)[:64])\n\n# RuntimeError [enforce fail at inline_container.cc:595] . unexpected pos 64 vs 0\n# 64\n# RuntimeError PytorchStreamReader failed reading zip archive: failed finding c',
            },
          },
          {
            h: 'and the same write, made atomic',
            ps: [
              'Six lines, and the ordering inside them is what earns the promise. Write into the temporary file, flush Python\'s buffer, `fsync` so the bytes have left the page cache for the disk, close, and only then rename. Skip the fsync and a machine that loses power shortly after the rename can come back with the new name pointing at data that never landed.',
              'The rename is the moment the checkpoint becomes real, and no reader can observe a half-renamed file. That is why the pattern works across processes as well as across crashes, and why the ranks in the next lesson can be made to agree with it.',
            ],
            code: {
              caption: 'continuing the block above (verified, torch 2.2.2 CPU): the same failing writer, against a temp-and-rename save',
              lang: 'python',
              text: 'def save_atomic(obj, path, opener=lambda p: open(p, "wb")):\n    tmp = path + ".partial"\n    f = opener(tmp)\n    try:\n        torch.save(obj, f)\n        f.flush()\n        os.fsync(f.fileno())\n    finally:\n        f.close()\n    os.replace(tmp, path)          # one syscall: it either happened or it did not\n\n\nsave_atomic(ckpt(10), "safe.pt")\ntry:\n    save_atomic(ckpt(20), "safe.pt", Dying)\nexcept Exception as err:\n    print(type(err).__name__, str(err)[:48])\nprint(torch.load("safe.pt")["step"], os.path.getsize("safe.pt"),\n      os.path.getsize("safe.pt.partial"))\n\n# RuntimeError [enforce fail at inline_container.cc:595] . unex\n# 10 3152 64',
            },
          },
          {
            h: 'the epoch number is state too',
            ps: [
              'Chapter four\'s arc showed the one piece of a loop with no state dict at all, and where a saved generator does and does not recover an order. A distributed run adds a number to that same gap, and it is easy to miss because nothing about it looks like state.',
              '`DistributedSampler` shuffles from a seed plus an epoch counter it holds internally, and the counter only moves when you call `set_epoch`. Iterate the sampler twice without calling it and both epochs hand back `[4, 7, 2, 1]`, the identical order. Call `set_epoch(1)` and the order becomes `[5, 2, 7, 1]`. Two ranks at the same epoch partition the dataset between them without overlap, which is the property that makes the shuffle correct in the first place.',
              'So the epoch belongs in the checkpoint next to the step, and every rank has to restore the same one. A resume that restarts the counter at zero replays the first epoch\'s order for the rest of the run, and a resume where the ranks disagree about the epoch hands the same rows to two ranks and never shows the rest.',
              'The mid-epoch position remains unsolved by anything in `torch.utils.data` itself. `StatefulDataLoader`, in the torchdata package, is the answer upstream has been building: a drop-in loader with `state_dict` and `load_state_dict` that records how far into an epoch the iteration got. It is a separate install and it is not on this machine, so it is named here rather than measured.',
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): a sampler that repeats itself, and two ranks that do not overlap',
              lang: 'python',
              text: 'import torch\nfrom torch.utils.data import DistributedSampler, TensorDataset\n\nds = TensorDataset(torch.arange(8.))\nsampler = DistributedSampler(ds, num_replicas=2, rank=0, shuffle=True)\n\nprint(list(sampler), list(sampler))       # two epochs, no set_epoch call\nsampler.set_epoch(1)\nprint(list(sampler))\n\nr0 = DistributedSampler(ds, num_replicas=2, rank=0)\nr1 = DistributedSampler(ds, num_replicas=2, rank=1)\nr0.set_epoch(3)\nr1.set_epoch(3)\nprint(list(r0), list(r1))\n\n# [4, 7, 2, 1] [4, 7, 2, 1]\n# [5, 2, 7, 1]\n# [2, 3, 1, 6] [4, 5, 0, 7]',
            },
          },
        ],
        readings: [
          { label: 'Reproducibility', url: 'https://docs.pytorch.org/docs/stable/notes/randomness.html', note: 'every generator a step can touch, and the ones a single saved block does not reach' },
          { label: 'Serialization semantics', url: 'https://docs.pytorch.org/docs/stable/notes/serialization.html', note: 'the preserved-views section, where saving a slice of a large storage is documented with its own file-size example' },
          { label: 'random.py at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/random.py', note: 'set_rng_state at 9 and manual_seed at 26: the CPU-only note and the device fan-out quoted above' },
          { label: 'StatefulDataLoader', url: 'https://meta-pytorch.org/data/main/stateful_dataloader_tutorial.html', note: 'the mid-epoch position, saved and restored; a separate package, named here rather than run' },
        ],
        check: [
          {
            q: 'A resume restored torch.get_rng_state() and the first step after it still differs from the uninterrupted run. What are the three places to look?',
            a: 'A torch.Generator you constructed and passed to the DataLoader, Python\'s random module, and the device generator if the tensors are not on CPU. The saved block covers the default CPU generator and nothing else, and set_rng_state is documented as CPU-only.',
          },
          {
            q: 'Why can a checkpoint be a hundred times larger than the parameters it holds?',
            a: 'Because saving keeps storage sharing, so a tensor that is a view writes the whole storage it reads through. A ten-element slice of a million-element buffer came to 4,001,101 bytes here, against 1,170 for the same values cloned first.',
          },
          {
            q: 'What does torch.save do to the previous checkpoint at the moment the new save begins, and what fixes it?',
            a: 'It truncates it, so a save that fails leaves neither the old checkpoint nor a loadable new one; the 3,120-byte file above became a 64-byte fragment. Writing to a temporary name, fsyncing, and renaming with os.replace makes the swap atomic, and the failed write lands on the temporary file instead.',
          },
        ],
        work: [
          { id: 'count-your-streams', label: 'list every random stream one step of your own loop reads, then check which of them your checkpoint actually restores', href: '#one-restore-one-stream' },
          { id: 'make-it-atomic', label: 'convert one save of your own to temp-and-rename, then prove it by failing the write and reloading the previous checkpoint', href: '#and-the-same-write-made-atomic' },
        ],
      },
      {
        id: 'how-long-a-mistake-stays-quiet',
        num: 2,
        title: 'How long a mistake stays quiet',
        lede: 'Sort resume failures by the thing that actually decides whether you find them: how many steps pass before the numbers say so. The loud ones are the museum. The interesting ones are the resumes that are wrong from the first step and look better than the run they replaced.',
        goal: 'Given a broken resume, place it on the scale from raises-immediately to never-raises, say why a converged run and a matching final loss are not evidence, and write the comparison that separates a real resume from a plausible one.',
        sections: [
          {
            h: 'sort the failures by how long they stay quiet',
            ps: [
              'The failures that get fixed are the ones that raise. Six of the pytorch museum\'s exhibits are step-time exceptions, and every one of them names its own cause in the message: a shape that does not fit the layer, class labels arriving as floats, a backward through a graph that was already freed. They cost minutes.',
              'One step further out are the failures that produce a number and the wrong number. An optimizer state dict from a differently shaped model loads without complaint and fails inside the update, which chapter four\'s first lesson takes apart. A missing RNG entry gives a first resumed loss that is simply different, measured in the same arc.',
              'Past that the signal gets thin. A missing scheduler is exact for two steps and then bends. A missing optimizer state is exact for one. A mid-epoch loader position never announces itself at all. The rest of this lesson lives at that end of the scale, because a failure that takes a hundred steps to show is a failure people attribute to the data.',
            ],
            table: {
              caption: 'resume failures by the number of steps before the numbers disagree; the measured column names where the number was taken',
              cols: ['failure', 'first sign', 'measured'],
              rows: [
                ['a shape, a dtype, a freed graph', 'an exception, this step', 'the pytorch museum: six loop-time exhibits with their errors verbatim'],
                ['optimizer state from another model', 'an exception one step later, from inside the update', 'chapter four\'s first lesson'],
                ['the RNG entry left out', 'step one after the restore, as a different loss', 'chapter four\'s third lesson'],
                ['the optimizer state left out', 'step two; step one is identical', 'this lesson, 200 steps'],
                ['the model left in eval mode', 'step one, as a lower loss', 'this lesson, 200 steps'],
                ['the scheduler left out', 'step three', 'chapter four\'s third lesson'],
                ['the loader position', 'never, on its own', 'chapter four\'s third lesson'],
              ],
            },
          },
          {
            h: 'the resume that trains better than the run it replaced',
            ps: [
              'Twenty steps of an Adam loop with dropout in it, killed at ten, restored with the model, the optimizer and the RNG. The resumed ten losses equal the last ten of the run that was never interrupted, as a list comparison. That is the shape of the proof, and it is the same shape LAB\u00b7P4 runs over a TPU bridge with its own numbers.',
              'Now leave the model in eval mode, which is what a validation pass before the resume does if nobody calls `train()` afterwards. The first resumed loss reads 6.386841 where the correct run reads 6.504373. It is lower, it is plausible, and the run continues descending from there.',
              'Two things happened at once and both are silent. Dropout stopped zeroing activations, so the model in front of the loss is a different model. And dropout stopped drawing from the generator, so every batch index after the first is a different batch. The loop has no way to notice either.',
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): a resume proved by equality, and the same resume in eval mode',
              lang: 'python',
              text: 'import torch\nfrom torch import nn\n\ntorch.manual_seed(1)\nDATA = torch.randn(512, 16)\nTARGET = DATA.sum(dim=1, keepdim=True)\n\n\ndef build():\n    torch.manual_seed(0)\n    model = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))\n    return model, torch.optim.AdamW(model.parameters(), lr=1e-2)\n\n\ndef run(model, opt, n):\n    out = []\n    for _ in range(n):\n        idx = torch.randint(0, 512, (32,))\n        loss = nn.functional.mse_loss(model(DATA[idx]), TARGET[idx])\n        opt.zero_grad()\n        loss.backward()\n        opt.step()\n        out.append(round(loss.item(), 6))\n    return out\n\n\nmodel, opt = build()                       # the run that is never interrupted\ntorch.manual_seed(7)\nstraight = run(model, opt, 20)\n\nmodel, opt = build()                       # the same run, killed at step 10\ntorch.manual_seed(7)\nrun(model, opt, 10)\ntorch.save({"model": model.state_dict(), "opt": opt.state_dict(),\n            "step": 10, "rng": torch.get_rng_state()}, "ck.pt")\n\n\ndef resume(train_mode=True, restore_opt=True):\n    state = torch.load("ck.pt")\n    model, opt = build()\n    model.load_state_dict(state["model"])\n    if restore_opt:\n        opt.load_state_dict(state["opt"])\n    torch.set_rng_state(state["rng"])\n    if not train_mode:\n        model.eval()                       # the validation pass nobody turned off\n    return run(model, opt, 10)\n\n\nprint(resume() == straight[10:])\nprint(straight[10:13])\nprint(resume(train_mode=False)[:3])\nprint(resume(restore_opt=False)[:3])\n\n# True\n# [6.504373, 9.049624, 2.990426]\n# [6.386841, 9.498335, 2.94874]\n# [6.504373, 9.063147, 3.139434]',
            },
          },
          {
            h: 'converged is not resumed',
            ps: [
              'Stretch the same experiment to two hundred steps with the kill at a hundred and the three resumes end up in places that all look like success. The whole-state resume equals the uninterrupted tail exactly. The eval-mode resume ends twelve times lower, at a last-twenty mean of 0.011072 against 0.137139. The resume that dropped the optimizer state ends at 0.139627, within two percent of the correct run.',
              'Read the third row before the second. Dropping Adam\'s moments produced a first resumed loss of 0.262158, identical to the correct one, because the loss is measured before the update and a missing moment cannot reach it until the step after. By the end of a hundred steps the two runs are indistinguishable by any summary you would put in a report, while the worst individual step sits 1.5058 times its own value away from the correct one.',
              'The second row is the more uncomfortable one. A broken resume that converges better than the correct run will survive every review that asks whether the loss went down, and this is the ordinary case rather than a contrived one: turning dropout off improves a training loss on a small model almost by definition.',
              '>> Every one of these runs converged. Only one of them resumed.',
            ],
            code: {
              caption: 'continuing the block above (verified, torch 2.2.2 CPU): two hundred steps, killed at a hundred, three resumes',
              lang: 'python',
              text: 'from statistics import mean\n\nmodel, opt = build()\ntorch.manual_seed(7)\nlong_run = run(model, opt, 200)\n\nmodel, opt = build()\ntorch.manual_seed(7)\nrun(model, opt, 100)\ntorch.save({"model": model.state_dict(), "opt": opt.state_dict(),\n            "step": 100, "rng": torch.get_rng_state()}, "ck.pt")\n\ntail = long_run[100:]                      # resume() above, with 100 in place of 10\nfor name, got in [("whole state", resume()),\n                  ("eval mode", resume(train_mode=False)),\n                  ("no optimizer", resume(restore_opt=False))]:\n    a, b = torch.tensor(got), torch.tensor(tail)\n    print(f"{name:14} equal={got == tail} step101={got[0]} "\n          f"last20={round(mean(got[-20:]), 6)} max_rel={((a - b).abs() / b.abs()).max():.4f}")\n\n# whole state    equal=True step101=0.262158 last20=0.137139 max_rel=0.0000\n# eval mode      equal=False step101=0.09149 last20=0.011072 max_rel=0.9674\n# no optimizer   equal=False step101=0.262158 last20=0.139627 max_rel=1.5058',
            },
            table: {
              caption: 'the same checkpoint restored three ways, 100 resumed steps each (verified, torch 2.2.2 CPU)',
              cols: ['restore', 'step 101', 'mean of last 20', 'largest relative gap', 'equal to the uninterrupted run'],
              rows: [
                ['model, optimizer, RNG', '0.262158', '0.137139', '0.0000', 'yes'],
                ['the same, left in eval mode', '0.09149', '0.011072', '0.9674', 'no, from step 101'],
                ['model and RNG, no optimizer', '0.262158', '0.139627', '1.5058', 'no, from step 102'],
              ],
            },
          },
          {
            h: 'equality is a test, a curve is not',
            ps: [
              'The bar the chapter sets is an overlay: the resumed curve on top of the uninterrupted one, the same curve rather than two curves that end up close. Turning that into something a machine can fail is a one-line change, and worth making, because a human looking at a plot cannot express a tolerance and will accept anything with the right shape.',
              'Compared point by point, both broken resumes above are easy to catch. The correct one passes `allclose` down to a relative tolerance of one part in a million. The eval-mode resume needs a tolerance of 1.0 to pass, and the optimizer-free one fails at every tolerance in the list. So the numbers are not subtle. What is subtle is that nobody compares a hundred pairs of numbers by hand, and every summary of those same hundred numbers agrees.',
              'The comparison worth writing down is therefore a list against a list, taken from a run you kept for the purpose. Save the uninterrupted losses once, resume, compare with `==`, and record the index of the first disagreement rather than a verdict. On a run long enough that keeping the whole list is awkward, compare a fixed window after the restore and say in the report how many steps it covered.',
              'LAB\u00b7P4 closes the TPU version of this on a v6e-1: a kill at step 99 with a loss of 0.01456 and a first resumed loss of 0.01437, published as the lab\'s reference block. Those two numbers are the lab\'s, not this page\'s, and the reason they are quoted rather than re-derived is that they come from hardware this machine does not have.',
            ],
            code: {
              caption: 'continuing the block above (verified, torch 2.2.2 CPU): which relative tolerances each resume survives',
              lang: 'python',
              text: 'def survives(got, rtol):\n    return torch.allclose(torch.tensor(got), torch.tensor(tail), rtol=rtol, atol=0)\n\n\nfor name, got in [("whole state", resume()),\n                  ("eval mode", resume(train_mode=False)),\n                  ("no optimizer", resume(restore_opt=False))]:\n    print(name, [r for r in (1e-6, 1e-3, 1e-2, 1e-1, 1.0) if survives(got, r)])\n\n# whole state [1e-06, 0.001, 0.01, 0.1, 1.0]\n# eval mode [1.0]\n# no optimizer []',
            },
          },
          {
            h: 'the checkpoint that carries the poison',
            ps: [
              'One infinite target is enough. The loss comes back `inf`, the backward fills every gradient with `inf` or `nan`, and one optimizer step turns all 1,153 parameters of this model into `nan`. Nothing raises at any point.',
              'Checkpoint at that moment and the file is a permanent record of a dead run. Restoring it gives `[nan, nan, nan]` for the next three losses, and it will give `nan` forever, because there is no arithmetic that brings a `nan` weight back. A resume from the checkpoint before it is the only recovery, which is the argument for keeping more than one.',
              'A guard costs one line at the save site: refuse to write a checkpoint whose loss is not finite. Chapter four\'s second lesson covers the other half of this, the clip that returns `inf` and quietly zeroes every gradient except the poisoned one, and the flag that turns it into an error. Between the two of them a run stops rather than saving over its own last good state.',
            ],
            code: {
              caption: 'run it, continuing the definitions above (verified, torch 2.2.2 CPU): one bad batch, and the checkpoint it produces',
              lang: 'python',
              text: 'import math\n\nmodel, opt = build()\ntorch.manual_seed(7)\nrun(model, opt, 5)\n\nbad = nn.functional.mse_loss(model(DATA[:32]), TARGET[:32] * float("inf"))\nopt.zero_grad()\nbad.backward()\nopt.step()\nprint(bad.item(), math.isfinite(bad.item()))          # inf False\n\nweights = model.state_dict()\nprint(sum(int(v.isnan().sum()) for v in weights.values()),\n      sum(v.numel() for v in weights.values()))       # 1153 1153\n\ntorch.save({"model": weights, "opt": opt.state_dict(), "step": 6}, "poison.pt")\nstate = torch.load("poison.pt")\nmodel, opt = build()\nmodel.load_state_dict(state["model"])\nopt.load_state_dict(state["opt"])\ntorch.manual_seed(7)\nprint(run(model, opt, 3))                             # [nan, nan, nan]',
            },
          },
        ],
        readings: [
          { label: 'torch.allclose reference', url: 'https://docs.pytorch.org/docs/stable/generated/torch.allclose.html', note: 'rtol and atol, and how the two combine; the snippet above passes atol=0 so the tolerances read as pure relative ones' },
          { label: 'Automatic mixed precision', url: 'https://docs.pytorch.org/docs/stable/amp.html', note: 'the grad scaler skips a step whose gradients are not finite, which is the same guard the poison section argues for at the save site' },
          { label: 'Saving and loading a general checkpoint', url: 'https://docs.pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html', note: 'the official recipe, worth rereading against the scale above for what it does not carry' },
        ],
        check: [
          {
            q: 'A resume dropped the optimizer state and the first resumed loss was exactly right. Why, and where does it show?',
            a: 'The loss is measured before the update, so a missing moment cannot reach it until the following step. Here step 101 read 0.262158 on both runs and the lists diverged from step 102; by the end of 100 steps the means were 0.139627 against 0.137139.',
          },
          {
            q: 'Why is a resumed run that converges lower than the uninterrupted one a reason to look harder, not to relax?',
            a: 'Because the ordinary way to get a lower training loss by accident is to lose dropout, which is exactly what a model left in eval mode after a validation pass does. That resume ended at a last-twenty mean of 0.011072 against 0.137139 and was wrong from the first step.',
          },
          {
            q: 'What comparison separates a real resume from a plausible one?',
            a: 'The resumed loss list against the uninterrupted loss list, element by element, with the index of the first disagreement recorded. A plot cannot express a tolerance, and every summary of these runs agreed while the runs did not.',
          },
        ],
        work: [
          { id: 'place-your-failure', label: 'take one resume bug you have hit and place it on the scale: how many steps passed before the numbers disagreed, and what told you', href: '#sort-the-failures-by-how-long-they-stay-quiet' },
          { id: 'list-against-list', label: 'replace the plot in one of your own resume checks with a list comparison that reports the index of the first disagreement', href: '#equality-is-a-test-a-curve-is-not' },
        ],
      },
      {
        id: 'when-the-machine-takes-it-back',
        num: 3,
        title: 'When the machine takes it back',
        lede: 'A preemptible run is killed on somebody else\u2019s schedule, restarted with a different world size, and expected to carry on. Everything the first two lessons proved for one process has to hold across a group of them that cannot see each other\u2019s files, each other\u2019s steps, or each other\u2019s failures.',
        goal: 'Given a multi-rank run, say what a collective checks and what it does not, name the two moments a checkpoint has to be guarded by a barrier, and describe what torchrun restarts and what it leaves you responsible for.',
        sections: [
          {
            h: 'two ranks, two answers, no error',
            ps: [
              'Every rank runs the same script, which chapter eight establishes, and every rank therefore reaches the load line on its own clock. Rank zero is usually the one that writes. If rank one reads before that write has landed, and the previous checkpoint is still sitting on disk, it loads the previous checkpoint.',
              'Two ranks, two step numbers, no exception anywhere. Rank one resumes at step 10 and rank zero at step 20, and from that point they are training different runs while believing they are training one. Delete the old file and the failure changes shape rather than going away: rank one gets a `FileNotFoundError` on the first attempt, which is louder but no more correct.',
              '`dist.barrier()` after the save is the fix, and the second half of the capture shows both ranks at step 20. The same barrier belongs on the other side of a save as well, before rank zero starts writing, if any rank might still be reading the file it is about to replace. Combine that with the temp-and-rename from lesson one and a reader can never observe a partial file at all.',
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU, gloo, two spawned processes): the same read, without a barrier and with one',
              lang: 'python',
              text: 'import time\n\nimport torch\nimport torch.distributed as dist\nimport torch.multiprocessing as mp\n\n\ndef worker(rank, world, wait):\n    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29530",\n                            rank=rank, world_size=world)\n    if rank == 0:\n        time.sleep(0.5)                      # writing a real checkpoint is not instant\n        torch.save({"step": 20}, "shared.pt")\n    if wait:\n        dist.barrier()\n    print(f"rank {rank} resumes at step {torch.load(\'shared.pt\')[\'step\']}")\n    dist.destroy_process_group()\n\n\nif __name__ == "__main__":\n    for wait in (False, True):\n        torch.save({"step": 10}, "shared.pt")   # the previous checkpoint, still on disk\n        mp.spawn(worker, args=(2, wait), nprocs=2, join=True)\n\n# rank 1 resumes at step 10\n# rank 0 resumes at step 20\n# rank 1 resumes at step 20\n# rank 0 resumes at step 20',
            },
          },
          {
            h: 'what the collective does not check',
            ps: [
              'A collective is agreement about shapes and dtypes, not about meaning. Two ranks that resumed from different checkpoints all-reduce their step numbers, 10 and 20, and get 30 back. The operation is correct. Nothing in c10d has any opinion about whether the two numbers should have been the same.',
              'The check has to be yours, and it costs two collectives at startup. Reduce the restored step with `MIN`, reduce it again with `MAX`, and refuse to run when the two differ; here they come back 10 and 20 on both ranks, which is the disagreement made visible on the line before the training loop. Do the same for anything else the ranks have to share, the epoch counter among them.',
              'One mechanism unifies part of the state on its own, which is worth knowing because of what it leaves behind. Constructing `DistributedDataParallel` broadcasts module state from rank zero, and the distributed unit measures that constructor broadcast directly. For a resume it means the weights end up consistent whatever each rank loaded, while the optimizer state, which DDP never touches, stays exactly as inconsistent as the files were.',
              '>> The weights agree because DDP made them agree. The step counters and the optimizer moments agree only if you checked.',
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU, gloo, two spawned processes): a reduction over mismatched steps, and the two that catch it',
              lang: 'python',
              text: 'import torch\nimport torch.distributed as dist\nimport torch.multiprocessing as mp\n\n\ndef worker(rank, world):\n    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29532",\n                            rank=rank, world_size=world)\n    step = torch.tensor([10.0 if rank == 0 else 20.0])   # two ranks, two checkpoints\n\n    total = step.clone()\n    dist.all_reduce(total)                               # the collective is happy\n    print(f"rank {rank}: all_reduce returned {total.item()}")\n\n    lo, hi = step.clone(), step.clone()                  # the check that is not free\n    dist.all_reduce(lo, op=dist.ReduceOp.MIN)\n    dist.all_reduce(hi, op=dist.ReduceOp.MAX)\n    print(f"rank {rank}: step min {lo.item()} max {hi.item()} agreed {lo.item() == hi.item()}")\n    dist.destroy_process_group()\n\n\nif __name__ == "__main__":\n    mp.spawn(worker, args=(2,), nprocs=2, join=True)\n\n# rank 1: all_reduce returned 30.0\n# rank 1: step min 10.0 max 20.0 agreed False\n# rank 0: all_reduce returned 30.0\n# rank 0: step min 10.0 max 20.0 agreed False',
            },
          },
          {
            h: 'a rank that stops answering',
            ps: [
              'Preemption is not a clean exit. A rank disappears in the middle of a collective the others have already entered, and those others do not fail, they wait. The distributed unit captures what that looks like when the timeout finally fires, a transport-layer error naming your own operation and the milliseconds it waited, and it is the same error whether the peer was preempted, killed by the memory manager, or stuck.',
              'What belongs to a resumable loop is the number in front of it. `default_pg_timeout` is thirty minutes, so an untuned job spends half an hour dead before anything says so, and that half hour is charged. Pass a `timeout` to `init_process_group` that matches how long you are willing to lose, and treat the error it eventually raises as the signal to restart rather than as a bug to fix.',
              'The training script is the wrong place to handle it either way. The error names your operation, not the missing peer, and the surviving ranks cannot form a new group by themselves. Restarting is the launcher\'s job, which is what the next section is about.',
            ],
          },
          {
            h: 'what torchrun actually restarts',
            ps: [
              'The elastic launcher\'s contract is short and blunt, and reading it in its own words settles most design questions about a preemptible loop. For a job with n workers, if k of them fail, all workers are stopped and restarted, up to `max_restarts`. Not the failed ones. All of them.',
              'The training-script guidance follows from that. Workers are restarted with the same program arguments, so progress is lost back to the most recent checkpoint, and the script is expected to carry `load_checkpoint` and `save_checkpoint` of its own. Nothing in the launcher preserves anything held in a process\'s memory across a restart, because there is no process left to preserve it in.',
              'Elastic membership makes the shape of the job part of the state. `--nnodes=1:4` gives a minimum and a maximum, and when a node departs or arrives the docs describe the same sequence: existing workers stopped, a new worker group formed, all workers started with a new RANK and WORLD_SIZE. The launcher\'s own warning is that RANK is not stable between restarts and that no assumption about the correlation between RANK and LOCAL_RANK should be hard-coded.',
              'For a checkpoint that means two rules. Nothing in the file may be keyed by rank position if the run can reshard, and anything that scales with world size, a per-rank batch count or a data shard assignment, has to be recomputed after the restart rather than restored from the file.',
            ],
            table: {
              caption: 'what survives a torchrun restart, from the elastic launcher docs (docs.pytorch.org, read 2026-08-15)',
              cols: ['thing', 'across a restart', 'source'],
              rows: [
                ['every worker process', 'stopped and restarted, however many failed', 'run.html: if k<=n workers fail all workers are stopped and restarted up to max_restarts'],
                ['in-process state', 'gone; progress falls back to the last checkpoint', 'train_script.html: you will lose progress up to the most recent checkpoint'],
                ['RANK and WORLD_SIZE', 'reassigned; a new worker group is formed', 'run.html: RANK is NOT stable between restarts'],
                ['the checkpoint', 'yours to write and yours to load', 'train_script.html: make sure you have load_checkpoint and save_checkpoint logic in your script'],
              ],
            },
          },
          {
            h: 'one file per rank',
            ps: [
              'A single-file `torch.save` from rank zero works while the whole model fits in one process. Shard the parameters across ranks, with FSDP or a tensor-parallel mesh, and there is no rank holding the whole thing to write it.',
              '`torch.distributed.checkpoint` is the answer in the box. It writes a directory rather than a file, with at least one file per rank, saving and loading from every rank in parallel; the load is in-place, so the model allocates its own storage and the checkpoint reads into it. The property that matters for a preemptible job is load-time resharding: a checkpoint saved from one cluster topology can be loaded into another, which is what an elastic restart with a different world size needs.',
              'The pairing that makes it usable across strategies is `get_state_dict` and `set_state_dict`, which normalize a model and optimizer state dict into fully qualified names rather than the positional ids chapter four\'s first lesson takes apart. That normalization is what lets a checkpoint written under one parallelism be read under another.',
              'None of this is measured here. The environment on this machine cannot run the coordinating collective the saver needs, so the sharded save and the resharding load are Colab-pending rather than captured, and everything above is the upstream documentation\'s claim rather than this page\'s measurement.',
            ],
          },
          {
            h: 'the bar, and the report that carries it',
            ps: [
              'The capstone above this arc asks for a run on real hardware, a checkpoint holding the whole state, a deliberate kill, a resume, and an overlay proving the curve continued rather than restarted. Every piece of that has now been taken apart on a machine with no accelerator in it, which is the point: none of the difficulty was ever in the chip.',
              'What the chip adds is the device RNG from lesson one, the bridge\'s own save path, and a scheduler that can take the machine back mid-epoch. LAB\u00b7P4 runs the whole sequence over torch_xla and publishes its reference numbers; this arc is what those numbers mean and how to know when yours are lying.',
              'The half-page provenance report is the last piece and the one people skip. Which chip, which dtype, which shapes, which bridge, and the comparison you actually ran, stated as the comparison rather than as a claim about it. A resumed loss list equal to the uninterrupted one is a result. A curve that looks right is a screenshot.',
            ],
          },
        ],
        readings: [
          { label: 'torchrun (elastic launch)', url: 'https://docs.pytorch.org/docs/stable/elastic/run.html', note: 'max_restarts, the MIN:MAX node range, and the warning that RANK is not stable between restarts' },
          { label: 'Train script requirements', url: 'https://docs.pytorch.org/docs/stable/elastic/train_script.html', note: 'what an elastic script owes the launcher: checkpoint logic of its own, and no assumption that anything survives in memory' },
          { label: 'Distributed checkpoint', url: 'https://docs.pytorch.org/docs/stable/distributed.checkpoint.html', note: 'a directory with a file per rank, parallel save and load, and load-time resharding; the section above is its claim, not a capture' },
          { label: 'torch.distributed reference', url: 'https://docs.pytorch.org/docs/stable/distributed.html', note: 'barrier, the timeout argument on init_process_group, and every collective the checks above are built from' },
        ],
        check: [
          {
            q: 'Rank zero writes the checkpoint and every rank loads it. What goes wrong without a barrier, and what does the collective afterwards tell you?',
            a: 'A rank that reads before the write lands gets the previous checkpoint, so the ranks resume at different steps with no error at all: rank 1 at step 10 while rank 0 was at step 20. The collective tells you nothing; all_reduce over 10 and 20 returns 30, because c10d checks shapes and dtypes rather than meaning.',
          },
          {
            q: 'Two ranks loaded different checkpoints and then wrapped their models in DDP. Which parts of the state end up consistent, and which do not?',
            a: 'The module state, because constructing DDP broadcasts it from rank zero, so the weights look fine whatever each rank read. The optimizer state and the step counters are never touched by that broadcast, so they stay as inconsistent as the files were, which is why the restored step is worth reducing with MIN and MAX before the loop starts.',
          },
          {
            q: 'Why can a checkpoint for an elastic job not be keyed by rank position?',
            a: 'Because a restart re-forms the worker group with a new RANK and WORLD_SIZE, and the launcher docs warn that RANK is not stable between restarts. Anything scaled to world size has to be recomputed after the restart, and a sharded checkpoint needs the resharding load that distributed checkpoint provides.',
          },
        ],
        work: [
          { id: 'agree-on-the-step', label: 'add a step-agreement check to one distributed loop of your own: all-reduce the restored step with MIN and MAX and refuse to start when they differ', href: '#what-the-collective-does-not-check' },
          { id: 'restart-budget', label: 'write down your run\'s restart budget and set the process-group timeout to match it, then confirm the error a dead rank produces', href: '#a-rank-that-stops-answering' },
        ],
      },
    ],
  },
]
