// The PyTorch path: twelve chapters through the eager world, landing on
// TPU. Part i snippets ran on this machine (torch 2.2.2, CPU, 2026-07-27;
// Python 3.11 for the dynamo and export snippets) and printed values are
// real outputs; the TPU bridge chapters state their pending-hardware status
// in prose. Mastery is manual work plus streak items served by the pytorch
// gym floor (docs/plans/pytorch-mastery-path.md).
import type { WorkItem } from '../mastery'

export interface PtCode {
  caption: string
  text: string
  lang: 'python' | 'text'
}
export interface PtSection {
  h: string
  /** Paragraphs; a string starting '>> ' renders as a pull quote. */
  ps: string[]
  code?: PtCode
}
export interface PtReading {
  label: string
  url: string
  note: string
}
export interface PtChapter {
  id: string
  num: number
  part: 'i' | 'ii'
  title: string
  lede: string
  goal: string
  sections: PtSection[]
  readings: PtReading[]
  /** A path diagram pinned after the section at this index. */
  diagram?: { id: string; after: number }
}

export const PT_CHAPTERS: PtChapter[] = [
  {
    "id": "tensors",
    "num": 1,
    "part": "i",
    "title": "Tensors",
    "lede": "A tensor is a view onto a block of storage, not the numbers themselves, and the strides are the map that reads it.",
    "goal": "Given a tensor and a chain of view operations, predict its shape, stride, and contiguity before you run a line of it.",
    "sections": [
      {
        "h": "a view, not the numbers",
        "ps": [
          "Take a tensor, call `.t()` on it, and nothing gets copied. What you're holding afterward still points at the same block of memory the original tensor pointed at; only the map describing how to walk that memory changed. A PyTorch tensor is that pair made explicit: a storage, one flat run of numbers, and metadata, a shape, a stride, and an offset, that says how to read positions out of it. Slicing, `t()`, `permute`, `expand`: none of them touch the storage. Each one just builds a new tensor object with new metadata pointed at the same bytes.",
          "The stride is the part that does the actual reading. It says how many elements to skip in storage for each step along a given axis, so a `(3, 4)` tensor with strides `(4, 1)` means: move one whole row, four elements, to advance the first index, move one element to advance the second. Transpose it and the strides swap without a single number in storage moving; the shape says `(4, 3)` now, but you are reading the exact same numbers through a different map.",
          ">> A tensor is a view of a storage, and the strides are the map."
        ],
        "code": {
          "caption": "run it (verified, torch 2.2.2 CPU): a view, a copy, and a mutation through the view",
          "text": "import torch\n\nx = torch.arange(12.).reshape(3, 4)\nv = x.t()                    # a view: same storage, remapped\nprint(v.stride(), v.is_contiguous())   # (1, 4) False\nc = v.contiguous()           # a copy that owns a row-major layout\nprint(c.stride())            # (4, 1)\n\nrow = x[0]                   # also a view\nrow += 100                   # in-place: x changes too\nprint(x[0])                  # tensor([100., 101., 102., 103.])",
          "lang": "python"
        }
      },
      {
        "h": "contiguous is a copy, not a promise",
        "ps": [
          "Call `.contiguous()` on that transposed view and something different happens: PyTorch allocates a fresh block of storage, walks the view in row-major order, and writes those values out linearly. The result is a new tensor with new storage, strides that read `(4, 1)` for this shape, the same layout you'd get building the tensor fresh. That's the boundary worth holding in your head: everything before `.contiguous()` reads the same bytes through different maps; `.contiguous()` is the one operation here that pays for an actual copy.",
          "This is also why `expand` can be free while `reshape` sometimes can't be. `expand` works by setting a dimension's stride to zero, repeating one row or column without writing any new bytes, so it never needs the tensor to be contiguous first. `reshape` wants to reinterpret existing strides as a new shape, and when the current strides don't support that reinterpretation, there's no map that satisfies the request. `reshape` falls back to copying, quietly, exactly where a plain `view` would have refused outright.",
          "That refusal is deliberate, and it's worth knowing which call you're making. `.view()` either finds a stride pattern that reinterprets the existing storage, or it raises. `.reshape()` promises the shape you asked for, however it has to get there, copy included."
        ]
      },
      {
        "h": "the shared storage is a live wire",
        "ps": [
          "Slice a row out of `x` and mutate it in place, and `x` changes too. `row = x[0]` is a view, so `row += 100` isn't writing to some private copy of the first row; it's writing into the exact storage `x` already owns. Print `x[0]` afterward and the new values are sitting right there, because there was never a second copy to keep them separate.",
          "That's a feature, not an accident. Views exist so that slicing, transposing, and reshaping stay cheap, and an in-place update through a view is how PyTorch lets you edit a tensor's data without allocating a new one. The cost shows up later: the next chapter's autograd tape assumes a tensor's value hasn't shifted under it once an operation has recorded it, and an in-place write through a view is exactly the kind of edit that can pull the floor out from under a gradient computation that hasn't run yet."
        ]
      },
      {
        "h": "dtype and device don't get inferred for you",
        "ps": [
          "Every tensor carries its own dtype and its own device, and neither one is shared or inherited from context. Build a tensor from Python floats with no dtype specified and you get `float32`; that's the default floating type PyTorch reaches for, and it stays the working assumption unless you ask for `float64` or one of the lower-precision types explicitly.",
          "Try to add a CPU tensor to a CUDA tensor and PyTorch does not move one of them for you. It refuses, naming the mismatch, rather than guessing which device you meant and copying silently. An operation that insists you say `.to(device)` yourself is an operation that never lets a slow transfer over a real bus hide inside an innocent-looking `+`."
        ]
      },
      {
        "h": "the stride oracle",
        "ps": [
          "Everything in this chapter reduces to one skill: given a chain of view operations, read off the resulting shape, stride, and whether the result is contiguous, without running anything. The stride oracle at /gym/pytorch#drills drills exactly that, feeding you a chain of code and asking for the three numbers before you're allowed to check. It's a narrow skill, and it's the one that makes every stack trace involving `contiguous()` or a `view()` refusal legible on sight instead of mysterious."
        ]
      }
    ],
    "readings": [
      {
        "label": "PyTorch tensors",
        "url": "https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html",
        "note": "the official tensor tutorial this chapter supersets"
      },
      {
        "label": "PyTorch internals",
        "url": "https://blog.ezyang.com/2019/05/pytorch-internals/",
        "note": "the internals essay; storage and strides drawn properly"
      },
      {
        "label": "Learn the Basics",
        "url": "https://docs.pytorch.org/tutorials/beginner/basics/intro.html",
        "note": "the series this path covers end to end"
      }
    ],
    "diagram": {
      "id": "pt-tensor",
      "after": 0
    }
  },
  {
    "id": "autograd",
    "num": 2,
    "part": "i",
    "title": "Autograd",
    "lede": "The autograd tape isn't traced once ahead of time; it's built fresh, node by node, every time your Python actually runs.",
    "goal": "Given a short sequence of tensor operations, draw the graph autograd records, predict what backward frees and accumulates, and know when to replace the automatic rule with your own.",
    "sections": [
      {
        "h": "the tape gets built as you go",
        "ps": [
          "Multiply a tensor by itself and sum it, with `requires_grad=True` set on the input, and PyTorch doesn't wait for you to ask for a gradient before it starts working. Each operation you run appends a node to a graph right then, live, as the Python executes: `y.grad_fn` is that graph, and it exists the moment `y` does, not after some separate tracing pass discovers it.",
          "This is define-by-run, and it's worth naming the alternative it's not, because the contrast is the whole personality of the framework. A trace-once system runs your function one time to build a static program, then replays that same program on every later call. PyTorch's tape is rebuilt from scratch on every forward pass: run the same Python twice, with a branch that goes a different way the second time, and you get two different graphs, both entirely valid, because nothing was ever fixed in advance.",
          ">> The tape is built by running, and consumed by backward."
        ],
        "code": {
          "caption": "run it (verified, torch 2.2.2 CPU): the tape appears as you run, and backward consumes it",
          "text": "import torch\n\nx = torch.ones(3, requires_grad=True)\ny = (x * x).sum()\nprint(y.grad_fn)     # <SumBackward0 object ...>: the tape, built as you ran\ny.backward()\nprint(x.grad)        # tensor([2., 2., 2.])",
          "lang": "python"
        }
      },
      {
        "h": "backward walks it once, then throws it away",
        "ps": [
          "Call `y.backward()` and PyTorch walks the graph from `y` back toward every leaf tensor, applying the chain rule at each node and accumulating the result into that leaf's `.grad`. By default, once that walk finishes, the graph is gone: the intermediate values backward needed are freed, because keeping them around for a second walk nobody asked for would just be wasted memory.",
          "Call `.backward()` a second time on the same `y` without telling PyTorch to keep the graph, and it raises, because the very thing the second call needs to walk has already been discarded. That error isn't a bug to route around quietly; it's the direct, honest consequence of a design that frees what it no longer needs. Passing `retain_graph=True` on the first call is how you tell PyTorch you meant to walk it again."
        ]
      },
      {
        "h": "no_grad, leaves, and where .grad accumulates",
        "ps": [
          "`torch.no_grad()` and `torch.inference_mode()` both suspend the recording itself: operations inside either context don't append anything to a graph at all, which is exactly what you want once you're only running a model forward and never planning to call `backward()` on the result.",
          "Whether `.grad` lands anywhere depends on whether a tensor is a leaf. A leaf is a tensor you created directly, with `requires_grad=True` set, rather than one produced by an operation on other tensors; only leaves accumulate gradients in `.grad` by default. And accumulate is literal: call `backward()` twice in a row without clearing gradients between calls, and the second call's gradients get added on top of the first's, not swapped in for them. That's exactly why every training loop in the next chapter opens with `zero_grad()`: skip it, and your gradients are the sum of every step you've ever run, not just the current one."
        ]
      },
      {
        "h": "when the automatic rule is wrong",
        "ps": [
          "Every operation's backward behavior comes from a rule PyTorch already knows, registered once for that primitive. Sometimes that rule is the wrong one to use, numerically unstable, or the operation itself is something autograd has no way to see through: a call out to code the tracer can't inspect. `torch.autograd.Function` is the escape hatch. Subclass it, define a `forward` and a `backward` by hand, and you've supplied the derivative yourself instead of trusting the automatic table.",
          "The role this plays is the same one the JAX path's `custom_vjp` plays on the other side of the two philosophies: a place to stand between the framework not knowing how to differentiate something and training still working. Different tape, same fix."
        ]
      }
    ],
    "readings": [
      {
        "label": "Automatic differentiation with torch.autograd",
        "url": "https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html",
        "note": "the official autograd tutorial this chapter supersets"
      },
      {
        "label": "Autograd mechanics",
        "url": "https://docs.pytorch.org/docs/stable/notes/autograd.html",
        "note": "the mechanics note, the real contract"
      },
      {
        "label": "Extending PyTorch",
        "url": "https://docs.pytorch.org/docs/stable/notes/extending.html",
        "note": "custom Functions, when you own the derivative"
      }
    ],
    "diagram": {
      "id": "pt-tape",
      "after": 0
    }
  },
  {
    "id": "modules",
    "num": 3,
    "part": "i",
    "title": "Modules",
    "lede": "An nn.Module isn't a function you call with parameters; it's a Python object that owns its parameters and remembers them between calls.",
    "goal": "Given an nn.Module, predict exactly what parameters(), state_dict(), and train()/eval() each expose, and explain why PyTorch keeps that state in the object instead of threading it through pure functions.",
    "sections": [
      {
        "h": "state lives on the object",
        "ps": [
          "Assign a `nn.Linear` to `self.up` inside a module's `__init__` and you haven't just stored a reference the way a plain Python attribute would. `nn.Module` overrides attribute assignment to notice what you handed it, a parameter, a buffer, or another module, and registers that object into the tracking structures the module keeps for exactly this purpose.",
          "Two categories cover everything a module can hold. Parameters are leaf tensors with `requires_grad=True`, the numbers a training loop is meant to update. Buffers are state without gradients: a running mean in batch normalization, something the module needs to carry forward but never wants a gradient computed for. `parameters()` walks every registered parameter across a module and its submodules; `state_dict()` does the same walk but returns names mapped to tensors instead of a bare iterator, covering both parameters and buffers together.",
          ">> In PyTorch, state is the model."
        ],
        "code": {
          "caption": "run it (verified, torch 2.2.2 CPU): the parameter count and the state_dict keys are both real",
          "text": "import torch\nfrom torch import nn\n\nclass Mlp(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.up = nn.Linear(8, 32)\n        self.down = nn.Linear(32, 1)\n\n    def forward(self, x):\n        return self.down(torch.relu(self.up(x)))\n\nmodel = Mlp()\nprint(sum(p.numel() for p in model.parameters()))   # 321\nprint(list(model.state_dict())[:2])                 # ['up.weight', 'up.bias']",
          "lang": "python"
        }
      },
      {
        "h": "the exact opposite of threading state through functions",
        "ps": [
          "This is the philosophical opposite of the other path's answer to the same question. Over there, state is never something an object holds; it's an argument a function takes and a new value that function returns, and nothing is ever mutated in place. Here, a module's parameters live on `self`, get mutated by the optimizer step in the next chapter, and never appear in the function signature of `forward` at all.",
          "Neither answer is a workaround for the other's limitation; each is chosen on purpose, for a reason that shows up two chapters from now. `torch.compile` and `torch.export` exist specifically to pull a static graph back out of this mutable, stateful world, one snapshot at a time. Recovering a graph from mutation is real work; the other path never has to do that work, because it never left graph form to begin with."
        ]
      },
      {
        "h": "state_dict is the contract, not a convenience",
        "ps": [
          "`state_dict()` is not a debugging tool you reach for occasionally. It's the actual serialization contract PyTorch checkpoints on: a flat mapping from dotted names, `\"up.weight\"`, `\"down.bias\"`, to tensors, and `load_state_dict()` reads that same mapping back into an existing module by matching names. Save a checkpoint, change nothing about the architecture, and `state_dict()` round-trips exactly; change a layer's name and loading fails loudly, on a name it can't find, rather than silently loading the wrong tensor into the wrong place.",
          "Chapter 4's checkpoint is built entirely on top of this: the model's half of a saved checkpoint is just this dictionary, written to disk and read back."
        ]
      },
      {
        "h": "train() and eval() are a mode, not an argument",
        "ps": [
          "Call `model.train()` or `model.eval()` and you haven't passed anything into `forward`. You've flipped a flag on the module itself, and every submodule that cares, dropout, batch normalization, checks that flag to decide how it behaves: dropout zeroes activations during training and passes everything through during eval; batch norm uses batch statistics while training and the running statistics it accumulated once it's in eval.",
          "That's the same pattern as the first section, playing out at a smaller scale. The behavior a layer shows for the exact same input depends on state sitting on the object, not on anything you supplied to the call. Move a model from train to eval without noticing, and every dropout layer in it silently starts doing something different than it was a line before, for a reason nothing in your function call reveals."
        ]
      }
    ],
    "readings": [
      {
        "label": "Build the Neural Network",
        "url": "https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html",
        "note": "the official build-model tutorial this chapter supersets"
      },
      {
        "label": "PyTorch internals",
        "url": "https://blog.ezyang.com/2019/05/pytorch-internals/",
        "note": "the internals essay again, now for how modules hold their tensors"
      }
    ]
  },
  {
    "id": "the-loop",
    "num": 4,
    "part": "i",
    "title": "Own the state, own the loop",
    "lede": "The loop that trains a PyTorch model is five lines you write by hand, and every one of them exists because some piece of state needed managing on purpose.",
    "goal": "Given the canonical training loop, name every place state lives in it and prove that a checkpoint captures all of that state, not part of it.",
    "sections": [
      {
        "h": "the five lines, and why each one is there",
        "ps": [
          "Forward, loss, zero the gradients, backward, step. Written out, a PyTorch training loop is barely longer than its own description, and every line in it earns its place. The forward pass computes a prediction; the loss compares it to a target; then three more lines manage state that nothing in the math above required you to think about.",
          "`opt.zero_grad()` exists because of a fact chapter two already put on the table: gradients accumulate into `.grad` across calls to `backward()`, they never clear themselves. Skip that line and the second step's gradient is the first step's gradient plus the second, and the third step's is worse than that. The loop has to clear the accumulator by hand, every time, because the tape has no opinion about when one step ends and the next one starts.",
          "`opt.step()` is where the parameters actually move, and it makes that move under `torch.no_grad()` internally, mutating tensors in place without asking autograd to track the mutation. Do that write by hand instead, in place on a leaf tensor that still requires grad, and you've built exactly the kind of mistake this site's mistakes gallery keeps a specimen of. The optimizer's whole job is doing that unsafe-looking write safely, once, in the one place it belongs.",
          ">> The loop is explicit because the state is yours."
        ]
      },
      {
        "h": "training it, to a real number",
        "ps": [
          "None of this is theoretical. The loop below fits a small MLP to a synthetic regression target, two hundred steps, Adam at a fixed learning rate, and it runs to a specific loss on this machine: 0.00372, not an order-of-magnitude guess. Read the five lines inside the `for` loop against the previous section and each one should already have a job attached to it before you see what it prints.",
          "Adam itself is state, though the loop above never shows it directly. Every parameter gets a running estimate of its gradient's mean and its squared magnitude, both updated on `opt.step()`, both invisible unless you go looking for them in `opt.state_dict()`. Nothing about that memory is optional. It's what makes Adam behave like Adam instead of like plain gradient descent wearing Adam's name."
        ],
        "code": {
          "caption": "the canonical loop, run to convergence (verified, torch 2.2.2 CPU)",
          "text": "import torch\nfrom torch import nn\n\ntorch.manual_seed(0)\nx = torch.randn(256, 8)\ny = x.sum(dim=1, keepdim=True)\n\nmodel = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1))\nopt = torch.optim.Adam(model.parameters(), lr=1e-2)\n\nfor step in range(200):\n    loss = nn.functional.mse_loss(model(x), y)\n    opt.zero_grad()\n    loss.backward()\n    opt.step()\nprint(f\"final loss {loss.item():.5f}\")   # final loss 0.00372",
          "lang": "python"
        }
      },
      {
        "h": "a checkpoint is the whole state, or it is not a checkpoint",
        "ps": [
          "Save `model.state_dict()` alone and reload it later, and the weights come back correct. What doesn't come back is Adam's memory: the running mean and variance it had built up for every parameter, gone, replaced by whatever a fresh `opt.init` would produce. Training resumes, technically, but from weights that used to belong to a well-adapted optimizer and now belong to a naive one. The loss curve after resume will not match the curve that never stopped.",
          "The fix is to save the whole state as one unit: `model.state_dict()`, `opt.state_dict()`, and the step count, together, in one dictionary, one file. If bitwise resume matters, meaning the run after restore should reproduce identical numbers rather than merely plausible ones, the RNG state belongs in that same dictionary too. This is the exact rule the jax path's twelfth chapter states for its own training loop; only the serialization surface differs, `state_dict` and `torch.save` here instead of a pytree and Orbax there.",
          "Nothing about `torch.load` or `load_state_dict` is unusual once the save side is right. The real discipline was deciding, before the first checkpoint ever got written, what counts as the state in the first place."
        ],
        "code": {
          "caption": "the whole state, saved and reloaded (verified)",
          "text": "import torch\n\ntorch.save({\"model\": model.state_dict(), \"opt\": opt.state_dict(), \"step\": step}, \"ckpt.pt\")\nstate = torch.load(\"ckpt.pt\")\nmodel.load_state_dict(state[\"model\"])\nopt.load_state_dict(state[\"opt\"])",
          "lang": "python"
        }
      },
      {
        "h": "schedulers: more state, same pattern",
        "ps": [
          "A learning-rate schedule is not a separate mechanism bolted onto training. `torch.optim.lr_scheduler` objects wrap an existing optimizer, and calling `scheduler.step()` once per epoch or once per step mutates that optimizer's `lr` in place: same optimizer, same state dict, one more field changing over time.",
          "Nothing here contradicts the chapter's opening claim. A scheduler is state you own, wired through the same loop you already wrote, mutated by an explicit call in exactly the shape of `zero_grad` and `step`. PyTorch never hides this behind a config file; it hands you the object and expects you to call it yourself."
        ]
      }
    ],
    "readings": [
      {
        "label": "Optimizing model parameters",
        "url": "https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html",
        "note": "the official loop tutorial, zero_grad and step in context"
      },
      {
        "label": "Save, load, and run model predictions",
        "url": "https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html",
        "note": "the state_dict contract this chapter's checkpoint follows"
      }
    ]
  },
  {
    "id": "data",
    "num": 5,
    "part": "i",
    "title": "The input pipeline",
    "lede": "A DataLoader is not part of the model. It's the concurrent program standing between your disk and your device, and it runs by its own rules.",
    "goal": "Given a training loop that seems slow, decide whether the bottleneck is the model or the pipeline feeding it, and fix the side that's actually starved.",
    "sections": [
      {
        "h": "what a Dataset and a DataLoader each do",
        "ps": [
          "Implement `__len__` and `__getitem__` on a class and PyTorch already knows what to call it: a `Dataset`, the smallest contract in the pipeline, one item in, one item back, nothing else required. Everything that makes training practical sits one layer up, in `DataLoader`, which composes batching, shuffling, and prefetching on top of that bare contract, along with the option to hand the work off to separate worker processes entirely.",
          ">> Loading data is concurrent Python, not tensor math.",
          "None of this is model code, and treating it as an afterthought is the single most common way a training run ends up slower than the model itself would predict. The loader is a small concurrent program running alongside whatever device is doing the actual math, and like any concurrent program it has failure modes that have nothing to do with tensors."
        ]
      },
      {
        "h": "workers are processes, not threads",
        "ps": [
          "Set `num_workers` above zero and `DataLoader` doesn't spin up threads inside the same process. It forks or spawns real OS processes, each running its own copy of the dataset, each handing finished batches back to the main process through a queue. That has a consequence worth stating plainly: everything a worker needs, the dataset object itself included, has to survive being pickled across that process boundary.",
          "This is exactly where a `lambda` inside a dataset's transform pipeline stops working: a worker process reconstructs the dataset it was handed by unpickling it, and a `lambda` can't be pickled, so the transform fails before a single sample loads. The failure looks like it's about tensors or shapes; it's actually a multiprocessing constraint, and the fix is the same as it would be anywhere a lambda meets pickling: a named function, or a small callable class, in place of the anonymous one.",
          "The reason to pay workers this way at all, instead of loading everything on the main process, is overlap. While the device is busy computing on batch `n`, a worker process is already reading and transforming batch `n+1` off disk, so the next batch is often waiting by the time the device asks for it. It's the same overlap argument this site's kernel path makes at the level of a single Pallas grid, one level up: hide the slow transfer behind the fast compute, wherever the boundary between them happens to sit."
        ],
        "code": {
          "caption": "run it: one batch out of the loader, shapes intact (verified)",
          "text": "import torch\nfrom torch.utils.data import DataLoader, TensorDataset\n\nds = TensorDataset(torch.randn(100, 8), torch.randn(100, 1))\ndl = DataLoader(ds, batch_size=32, shuffle=True)\nxb, yb = next(iter(dl))\nprint(xb.shape, yb.shape)   # torch.Size([32, 8]) torch.Size([32, 1])",
          "lang": "python"
        }
      },
      {
        "h": "where the heavy work belongs",
        "ps": [
          "Transforms, resizing an image, tokenizing a string, augmenting a sample, run inside the worker process, per sample, before the batch is ever assembled. That placement is deliberate, not an accident of the API: heavy per-sample work belongs in a worker, where it can run concurrently with the device, not inside the training step, where it would simply add to the time every step already takes.",
          "Get this placement backward, put the heavy work in the step instead of the loader, or starve the loader with `num_workers=0` and a transform that actually costs something, and the symptom looks identical from the outside: the device sits idle waiting on data that isn't ready yet. Chapter nine closes this loop with the tool that tells the two apart for certain, a profiler trace, rather than a guess about where the time went."
        ]
      }
    ],
    "readings": [
      {
        "label": "Datasets and DataLoaders",
        "url": "https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html",
        "note": "the official tutorial this chapter builds on"
      },
      {
        "label": "PyTorch Profiler recipe",
        "url": "https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html",
        "note": "how you will prove the loader is or is not the bottleneck, closed out in chapter nine"
      }
    ]
  },
  {
    "id": "dynamo",
    "num": 6,
    "part": "i",
    "title": "Guarded capture",
    "lede": "torch.compile doesn't trace your program once and lock it in. It captures your tensor math behind a set of guards and lets everything else keep running as plain Python.",
    "goal": "Given a torch.compile call, predict whether a given call hits the cache or triggers a new capture, and explain what happens to a side effect or a data-dependent branch either way.",
    "sections": [
      {
        "h": "guards, not a single trace",
        "ps": [
          "Decorate a function with `@torch.compile` and dynamo attaches itself to it. On a call, dynamo walks the function's Python bytecode and captures every tensor operation it finds into an FX graph, the same kind of flat op list chapter seven reads directly. Alongside that graph it installs a set of guards: conditions on the shapes, dtypes, types, and closed-over values that were true when it captured. A later call that satisfies every guard reuses the cached graph outright. A call that breaks even one guard, a new shape, a different dtype, triggers a fresh capture and adds another entry to the cache.",
          ">> dynamo compiles your tensor math and replays the rest.",
          "Set this next to the jax path's model and the disagreement is exact, not a matter of degree. jax's `jit` traces a function once for a given signature, and the resulting jaxpr is the only thing that ever runs again; whatever Python ran during that one trace is gone from every later call. dynamo captures per guard set instead, as many times as your inputs actually vary, and it does something jax's tracer structurally cannot: the Python around the compiled tensor math keeps running on every single call, side effects included, not only once at capture time."
        ]
      },
      {
        "h": "a side effect that keeps firing",
        "ps": [
          "That last claim is worth proving rather than taking on faith, and the proof is small enough to run yourself. `dict(counters[\"stats\"])` is dynamo's own bookkeeping, a counter that tracks how many calls it has captured and how many distinct graphs exist. Call a compiled function `f` twice with the same shape and `calls_captured` reads 2 while `unique_graphs` stays at 1: two calls went through, one graph served both. Call it a third time with a new shape and both numbers move, to 4 and 2, because the guard on the first shape no longer matches and a second graph gets captured.",
          "Notice what that sequence shows. `calls_captured` climbing on the second call, the one that hit the cache, means real work happened on that call beyond a lookup of a cached executable: enough of the wrapped function ran for dynamo's own instrumentation to fire again. Under jax's `jit`, a Python-level side effect like this would fire exactly once, during the single trace, and every call after that would be invisible to it, running only the recorded jaxpr with none of the original Python left around it. Under `torch.compile`, the side effect survives every call, cached graph or not, because the surrounding Python was never thrown away in the first place."
        ],
        "code": {
          "caption": "the counters, real (verified, torch 2.2.2, Python 3.11): two same-shape calls share a graph, a new shape starts another",
          "text": "import torch\nfrom torch._dynamo.utils import counters\n\n@torch.compile(backend=\"eager\")\ndef f(x):\n    return torch.sin(x) + 1\n\nf(torch.randn(4)); f(torch.randn(4))\nprint(dict(counters[\"stats\"]))   # {'calls_captured': 2, 'unique_graphs': 1}\nf(torch.randn(5))\nprint(dict(counters[\"stats\"]))   # {'calls_captured': 4, 'unique_graphs': 2}",
          "lang": "python"
        }
      },
      {
        "h": "when the trace can't decide, it splits",
        "ps": [
          "A jitted jax function facing a data-dependent branch raises outright, `TracerBoolConversionError`, because a tracer with no concrete value has nothing an `if` statement can test. dynamo has no sibling to that error. Write `if x.sum() > 0:` inside a compiled function with `x` traced, and dynamo does not fail. It stops capturing at that point, runs the branch as ordinary, uncompiled Python, and resumes capturing on the other side, stitching the compiled pieces around the gap.",
          "That gap has a name: a graph break. Nothing about it raises or warns by default, which is exactly what makes it worth checking for on purpose rather than assuming it away. `torch._dynamo.explain(g)(*args)` runs the function and reports, among other things, a `graph_break_count`. Run it on the branch below and it comes back 1, one break, for the one condition that depends on a traced value.",
          "The cost of a graph break is real but silent: instead of one large compiled graph, you get several smaller ones with an eager-mode gap between them, and every one of those boundaries is a place fusion and codegen can't see across. A single break rarely matters. A compiled model with many of them starts looking, performance-wise, a lot like the eager code it was meant to replace, and `explain` is how you find that out before a slow benchmark tells you instead."
        ],
        "code": {
          "caption": "the graph break, counted (verified)",
          "text": "import torch\nimport torch._dynamo as dynamo\n\ndef g(x):\n    if x.sum() > 0:        # data-dependent: a graph break, not an error\n        return x + 1\n    return x - 1\n\nprint(dynamo.explain(g)(torch.ones(3)).graph_break_count)   # 1",
          "lang": "python"
        }
      },
      {
        "h": "capture and codegen are two different jobs",
        "ps": [
          "Everything in this chapter so far, the FX graph, the guards, the cache, the break, is dynamo's job: capture. What happens to a captured graph afterward, how it actually gets compiled into fast code, is a separate job, handled by a backend. Inductor is the default, and it's the one doing real codegen, choosing fusions and generating device code from the graph dynamo handed it.",
          "`backend=\"eager\"` is the setting used in this chapter's own examples, and it exists to isolate capture from codegen on purpose: with it, dynamo still captures, still installs guards, still counts calls and graphs exactly as shown above, but the captured graph just runs through PyTorch's ordinary eager dispatch instead of being compiled further. It's the right tool for studying dynamo's behavior in isolation, which is what this chapter needed, and the wrong one for measuring speed, which chapter nine's harness is built for instead."
        ]
      }
    ],
    "readings": [
      {
        "label": "Introduction to torch.compile",
        "url": "https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html",
        "note": "the official tutorial this chapter reasons underneath"
      },
      {
        "label": "Dynamo overview",
        "url": "https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_overview.html",
        "note": "how capture actually works, guard by guard"
      }
    ],
    "diagram": {
      "id": "pt-compile",
      "after": 0
    }
  },
  {
    "id": "graphs",
    "num": 7,
    "part": "i",
    "title": "Graphs",
    "lede": "Whatever captures your model, dynamo's guarded graph or torch.export's whole program, what's underneath is the same ATen vocabulary eager PyTorch already speaks.",
    "goal": "Given a torch.export call, read the resulting graph line by line, and predict which programs export accepts outright and which ones it refuses instead of silently splitting them.",
    "sections": [
      {
        "h": "below dynamo, ATen",
        "ps": [
          "Dynamo captures your tensor math into a graph, chapter 6's whole subject, but the graph itself is not exotic. It's FX, and every node in it calls an ATen op: the same operator library eager PyTorch calls under the hood, whether or not dynamo ever ran. Compile a function and print its captured ops, and you're reading the same vocabulary a plain, uncompiled forward pass already spoke.",
          "That graph doesn't stop at the forward pass either. aot_autograd traces the backward through the same machinery, so what dynamo hands the compiler already covers a full training step, forward and backward both, one FX graph each, not something bolted on after the fact. This is why torch.compile speeds up training loops, not only inference calls.",
          "One more pass runs before the graph settles: decompositions rewrite the larger ATen surface down toward a smaller core set of ops, fewer signatures for a compiler backend to implement, more of them expressible as combinations of the ones that remain. A backend author who supports the core set gets most of the rest for free.",
          ">> Below dynamo, everything is ATen."
        ]
      },
      {
        "h": "torch.export: the whole graph, no Python left",
        "ps": [
          "Dynamo's graphs are stitched: wherever a guard might miss or Python control flow resists capture, a graph break splits the function into pieces and Python fills the gap between them, chapter 6's whole story. torch.export refuses that compromise. Call torch.export.export on a module and you get back an ExportedProgram: one standalone graph for the whole function, with no Python left in it at all, and no seam where a break could have happened.",
          "That completeness is what makes an ExportedProgram portable. It's the artifact you hand to a serving stack running with no Python interpreter alongside it, and it's the same artifact chapter 10 picks up and carries across the bridge to StableHLO, the wire format a compiler on the other side of that bridge actually reads. Where dynamo optimizes the training loop you're already running, export produces something you can hand off entirely."
        ],
        "code": {
          "caption": "run it: exporting a two-op module (torch 2.2.2, Python 3.11)",
          "text": "import torch\n\nclass Sin(torch.nn.Module):\n    def forward(self, x):\n        return torch.sin(x).sum()\n\nep = torch.export.export(Sin(), (torch.randn(4),))\nprint(ep.graph)",
          "lang": "python"
        }
      },
      {
        "h": "reading the graph",
        "ps": [
          "The call above prints exactly the text below, and reading it takes a habit the jax path already built for jaxprs: find the placeholders, the inputs coming in, then follow one operation per line down to the outputs returned at the end. The IR is different, ATen calls here, JAX primitives there, but the fluency is the same skill wearing a different vocabulary.",
          "Read it line by line and there's nothing left to guess at. %l_x_ is the placeholder, the function's one input. %sin calls torch.ops.aten.sin.default on it, and %sum_1 calls torch.ops.aten.sum.default on whatever %sin produced. The graph returns sum_1, and that's the whole function: two ATen calls, one input, one output, exactly what Sin.forward wrote in Python, with nothing hidden and nothing implicit left standing between them."
        ],
        "code": {
          "caption": "the verbatim output of the export call above",
          "text": "graph():\n    %l_x_ : [num_users=1] = placeholder[target=l_x_]\n    %sin : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%l_x_,), kwargs = {})\n    %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%sin,), kwargs = {})\n    return (sum_1,)",
          "lang": "text"
        }
      }
    ],
    "readings": [
      {
        "label": "torch.export tutorial",
        "url": "https://docs.pytorch.org/tutorials/intermediate/torch_export_tutorial.html",
        "note": "the official export tutorial this chapter reads past"
      },
      {
        "label": "torch.export reference",
        "url": "https://docs.pytorch.org/docs/stable/export.html",
        "note": "the ExportedProgram contract"
      }
    ]
  },
  {
    "id": "distributed",
    "num": 8,
    "part": "ii",
    "title": "Distributed",
    "lede": "A PyTorch distributed job has no separate orchestrator: every process runs the same script, and a collective is the one moment those processes actually talk to each other.",
    "goal": "Given a distributed training setup, name exactly what DDP replicates versus what FSDP shards, and predict which collective a given call inserts.",
    "sections": [
      {
        "h": "every rank runs your script",
        "ps": [
          "Distributed PyTorch starts from one idea, worth stating before any API: every rank runs the exact same script. There's no separate driver process orchestrating workers from outside; process 0 runs the training script, and so does process 7, and what tells each one who it is and how many others exist is init_process_group, backed by c10d, PyTorch's collective communication layer. Call it once per process and every rank gets a rank number and a world size, the two facts everything else in this chapter is built on.",
          "That shape, the same script running once per participant rather than one controller dispatching to many workers, is called multi-controller, and it isn't unique to PyTorch. The xla path's McJAX chapter describes the identical architecture from the other framework's side: PyTorch's default distributed world is multi-controller too, arrived at independently because the same problem, many devices, one program, tends to produce the same answer.",
          ">> Every rank runs your script; the collectives are the meeting points."
        ]
      },
      {
        "h": "DDP and FSDP: the same tradeoff, both directions",
        "ps": [
          "DistributedDataParallel is the simpler of the two strategies to reason about: every rank holds a full copy of the model, runs its own forward and backward on its own shard of data, and then all-reduces the gradients so every replica ends the step with the identical update. DDP doesn't wait for the whole backward pass to finish before it starts communicating either. Gradients are bucketed, and each bucket's all-reduce fires the moment that bucket's gradients are ready, overlapping communication with the backward computation still running behind it.",
          "FSDP takes the opposite trade. Instead of replicating the full model on every rank, it shards parameters, gradients, and optimizer state across the group, and gathers a parameter back to its full size only when a layer is about to use it. DDP spends memory, a full copy everywhere, to keep the programming model simple. FSDP spends complexity and extra communication to keep memory down. Neither is strictly better: the choice comes down to which resource you have less of."
        ]
      },
      {
        "h": "meshes, named the same way twice",
        "ps": [
          "Once a job needs more than one axis of parallelism, data-parallel replicas along one dimension, tensor-parallel shards along another, PyTorch names the layout with DeviceMesh, and the tensor parallel APIs built on top of it place a module's weights across that mesh by axis. The idea underneath, a named grid of devices with a spec saying which axis shards which dimension, is the same mesh the jax path's chapter 10 teaches. PyTorch arrives at it from the mutation side of the fence; the destination is the same."
        ]
      },
      {
        "h": "the collective, run for real",
        "ps": [
          "None of this needs multiple machines to observe. The snippet below ran on this machine as a single process standing in for a world of one, over gloo, PyTorch's CPU-friendly collective backend. all_reduce on a world of one can't change any values, there's nobody else to reduce with, but the collective genuinely executes: the process group stands up, the operation dispatches, and the tensor comes back unchanged because unchanged is the correct answer for a group of size one, not because nothing happened."
        ],
        "code": {
          "caption": "run it: a real all_reduce, gloo backend, world of one (verified, torch 2.2.2 CPU)",
          "text": "import torch\nimport torch.distributed as dist\n\ndist.init_process_group(\"gloo\", init_method=\"tcp://127.0.0.1:29511\", rank=0, world_size=1)\nx = torch.ones(4)\ndist.all_reduce(x)     # a world of one: unchanged, but the collective ran\nprint(x)               # tensor([1., 1., 1., 1.])\ndist.destroy_process_group()",
          "lang": "python"
        }
      }
    ],
    "readings": [
      {
        "label": "DDP tutorial",
        "url": "https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html",
        "note": "the official walkthrough of replication and bucketed all-reduce"
      },
      {
        "label": "FSDP tutorial",
        "url": "https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html",
        "note": "sharded parameters, gradients, and optimizer state, from the source"
      },
      {
        "label": "Tensor parallel tutorial",
        "url": "https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html",
        "note": "DeviceMesh and the tensor-parallel APIs built on it"
      },
      {
        "label": "torch.distributed reference",
        "url": "https://docs.pytorch.org/docs/stable/distributed.html",
        "note": "the c10d layer: process groups, ranks, every collective"
      }
    ]
  },
  {
    "id": "performance",
    "num": 9,
    "part": "ii",
    "title": "Performance",
    "lede": "A benchmark number is only as honest as the clock behind it, and PyTorch gives you three different ways for that clock to lie.",
    "goal": "Given a PyTorch function on any backend, build a benchmark harness that measures compute rather than dispatch, and read a profiler trace well enough to name its largest host-side gap.",
    "sections": [
      {
        "h": "three timing traps, one discipline",
        "ps": [
          "Time a PyTorch op the naive way, wrap it in time.perf_counter() and stop the clock a line later, and CPU tensors mostly tell the truth: CPU dispatch is synchronous, so the line doesn't return until the op has actually finished. CUDA breaks that assumption completely. A CUDA call returns to Python the moment it's queued, not the moment it runs, so an unsynchronized clock measures how fast you can enqueue work, not how fast the device does it. torch.cuda.synchronize() is what makes the clock wait for the device to actually catch up.",
          "The TPU bridges chapter 10 covers add a third trap on top of the first two: those tensors are lazy, recording operations instead of running them until a sync point forces materialization, so a naive timer there measures almost nothing at all. Three backends, three different reasons the same lazy mistake produces a fast-looking, wrong number. One discipline covers all three: never trust a clock you haven't forced to wait.",
          "This site's rule for every number it publishes carries over here without modification: state the chip, the dtype, the shapes, and the method, every time a latency claim gets made. A latency number with no provenance attached isn't a measurement. State the chip, the dtype, the shapes, and the method, or the number doesn't count.",
          ">> The profiler is the referee, on every device."
        ]
      },
      {
        "h": "the harness",
        "ps": [
          "The harness below is built to survive exactly the traps the last section named. A warmup call runs first and forces whatever compilation or first-time cost exists to happen once, off the clock. The timed loop then calls the function n times in a row, back to back, and only synchronizes once, at the very end, so the wait happens after every call has been dispatched rather than after each individual one.",
          "Dividing the total by n gives you the per-call time with dispatch overhead spread thin across every iteration instead of dominating a single one. On this machine, on the CPU backend, that number for a 512-by-512 linear layer comes out to 0.142 milliseconds. On CUDA the same shape of harness needs torch.cuda.synchronize() added exactly where the comment marks it, or the number it reports is a lie about dispatch, not a fact about compute."
        ],
        "code": {
          "caption": "run it: a harness that only trusts the clock once it has waited (verified, torch 2.2.2 CPU)",
          "text": "import time\nimport torch\n\ndef bench(f, *args, n=50):\n    out = f(*args)                     # warmup: capture + compile if any\n    t0 = time.perf_counter()\n    for _ in range(n):\n        out = f(*args)\n    if out.device.type == \"cuda\":\n        torch.cuda.synchronize()       # async dispatch settles before the clock\n    return (time.perf_counter() - t0) / n\n\nlin = torch.nn.Linear(512, 512)\nprint(f\"{bench(lin, torch.randn(64, 512)) * 1e3:.3f} ms\")   # 0.142 ms (this machine, CPU)",
          "lang": "python"
        }
      },
      {
        "h": "torch.profiler: schedule and steady state",
        "ps": [
          "Wrapping a benchmark loop tells you the average cost of a call. It doesn't tell you where inside that call the time actually went, and for that torch.profiler is the tool: it records both host-side Python overhead and device-side kernel execution on the same timeline, so a single trace shows you the CPU launching work and the accelerator running it, side by side.",
          "Profiling every single step of a long run would be wasteful and would skew the very thing you're measuring, so the profiler takes a schedule: a wait phase where nothing gets recorded, a warmup phase that primes caches and compilation without keeping the trace, and an active phase that's the only part actually captured. What lands in the trace is steady-state behavior, not the noisy first few steps every training loop pays once and never again."
        ]
      },
      {
        "h": "AMP: autocast and the scaler",
        "ps": [
          "Automatic mixed precision changes the dtype decision from something you hardcode per op into something torch.autocast picks for you, per operation, based on which dtype each op tends to be numerically safe and fast in. Matmuls and convolutions typically autocast to a lower-precision dtype; reductions and other numerically sensitive ops stay in float32 underneath the same context manager, without you naming either choice by hand.",
          "fp16's narrow dynamic range means small gradients can underflow to exactly zero before they ever reach the optimizer, and GradScaler exists to prevent that: it scales the loss up before backward, so gradients land in a range fp16 can represent, then unscales them back down before the optimizer step. bf16 doesn't need any of that machinery. Its exponent range matches float32's, just with less mantissa precision, so no scaler is required, and bf16 is also the dtype the kernel path measures with natively on TPU, the hardware this whole track eventually lands on."
        ]
      }
    ],
    "readings": [
      {
        "label": "PyTorch profiler recipe",
        "url": "https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html",
        "note": "the schedule and trace-capture workflow, official"
      },
      {
        "label": "Automatic mixed precision recipe",
        "url": "https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html",
        "note": "autocast and GradScaler, from the source"
      }
    ]
  },
  {
    "id": "bridges",
    "num": 10,
    "part": "ii",
    "title": "Bridges",
    "lede": "PyTorch never runs on a TPU directly; it crosses one of two bridges first, and by the time either one lands, the framework you wrote in is no longer the point.",
    "goal": "Given a PyTorch module, name which bridge would carry it to TPU, predict how it executes once it crosses, and state what the compiler actually sees once both bridges converge.",
    "sections": [
      {
        "h": "two bridges, and an honest gap",
        "ps": [
          "Everything in this path so far runs on whatever device happens to sit under Python: a laptop CPU, the Intel chip this course was written on, maybe a GPU somewhere. None of it runs on a TPU without crossing first, and PyTorch itself does not build that crossing. Two separate projects do, and this chapter is about both of them at once, because knowing only one leaves you unable to read half of what you will find once you go looking.",
          "Say this plainly, once, because it matters more here than it has anywhere earlier in the path: neither this chapter nor the next ran on a TPU. The machine that wrote this course has no TPU attached to it. Every fact below is checked against the current pytorch/xla repository and its official docs rather than printed from a real run, and the Colab lab pass later in this path is where these same facts turn into your own numbers, measured on real hardware.",
          "That gap does not make the facts soft, but it does change what fluency looks like for these two chapters. Instead of predicting a printed value, the goal is reading two bridges well enough to know which one a given module would cross, and what is waiting for it on the other side."
        ]
      },
      {
        "h": "the lazy bridge: torch_xla",
        "ps": [
          "The first bridge is torch_xla, and its central idea is laziness. Move a model and its tensors onto the `xla` device and every operation you run against them stops executing right away; it gets recorded instead, the way the tape in chapter 2 records an op, except this tape waits for a sync point before anything reaches the chip. `torch_xla.step()` marks that boundary inside a training loop, and `torch_xla.sync()` marks it directly: only there does the recorded graph actually materialize, compiled through XLA and run through PJRT, with `PJRT_DEVICE=TPU` telling the runtime which hardware to reach for.",
          "The classic cost of this bridge is recompilation. Change the shape feeding a lazy graph and the sync point has to compile a new graph for it, the same cache-key discipline chapter 6 taught for dynamo, paid again at a different layer. A training loop whose batch shapes drift, a ragged last batch above all, pays this bill on every drift instead of once. The xla path's chapters 4 through 6 on this site describe exactly what runs underneath that sync point, StableHLO in and XLA's own decisions out; this chapter only needs you fluent in when the sync happens and why it costs what it costs."
        ]
      },
      {
        "h": "the jax bridge: torchax",
        "ps": [
          "The second bridge, torchax, began inside the pytorch/xla repository and now lives in its own (google/torchax), and it takes an entirely different route. Instead of recording torch ops for a later sync, it maps them onto jax primitives directly: a torch tensor under torchax is backed by a real jax array underneath, not a lazy recording of one. Because that array genuinely is a jax array, everything the jax path spent nine chapters building, `jit`, `grad`, sharding over a named mesh, applies to a torch module exactly as it would to a jax function, with no separate torch-side reimplementation waiting to be written.",
          "The interop runs in both directions, and that detail is worth holding onto. A jax program can call a torch module the way it calls any other jax function, and torch code can reach into jax the same way back. Neither side has to pretend the other framework does not exist; torchax's whole job is making that pretending unnecessary."
        ]
      },
      {
        "h": "one destination: StableHLO",
        "ps": [
          "Follow either bridge far enough and they stop looking like two things. torch_xla's sync point and torchax's jax-backed tensors both bottom out at StableHLO, the same portable tensor IR the xla path's chapter 2 teaches from the jax side, and the kernel path's own account of the stack, at /l/source, describes from below. By the time the compiler is looking at the program, it has no memory of which framework wrote it. The framework identity that felt so load-bearing through chapters 1 through 9, mutation against threading, guard-and-recapture against trace-once, is gone by this layer, and that convergence is exactly why two different bridges can lead to the same chip.",
          ">> Two bridges, one destination: StableHLO.",
          "`torch.export`, chapter 7's whole-graph artifact, is exactly the shape both bridges want hold of: no Python left in it, decomposed toward a core set of ATen ops, nothing left implicit for a bridge to guess at. Export a module before you hand it to either bridge and you have already done the part of the crossing that depends on you; what happens after that is the bridge's job, not yours."
        ]
      },
      {
        "h": "what fluency looks like here",
        "ps": [
          "Until a real chip is measuring these facts back at you, the work this chapter asks for is legibility, not a benchmark. Given a module, you should be able to say which bridge would carry it and why, what its execution model looks like once it crosses, and what the compiler is left holding once both bridges converge. The torchax README is short enough to read end to end in one sitting, and it is worth doing exactly that before chapter 11, where this same convergence stops being theory and starts being the ground you stand on."
        ]
      }
    ],
    "readings": [
      {
        "label": "pytorch/xla",
        "url": "https://github.com/pytorch/xla",
        "note": "the repo both bridges live in"
      },
      {
        "label": "torchax",
        "url": "https://github.com/google/torchax",
        "note": "the torch-on-jax bridge, in its own repo since late 2025"
      },
      {
        "label": "PyTorch/XLA docs",
        "url": "https://docs.pytorch.org/xla/",
        "note": "the official torch_xla docs"
      },
      {
        "label": "StableHLO spec",
        "url": "https://openxla.org/stablehlo/spec",
        "note": "the convergence point"
      }
    ],
    "diagram": {
      "id": "pt-bridges",
      "after": 0
    }
  },
  {
    "id": "tpu-practice",
    "num": 11,
    "part": "ii",
    "title": "TPU practice",
    "lede": "Once a torch model crosses either bridge, the TPU treats it exactly the way it treats jax: same mesh, same profiler, same kernels underneath.",
    "goal": "Given a torch model and a device mesh, mark shardings the way the jax path does, name the collectives GSPMD inserts, and place every layer of a production serving stack on the chapter that actually teaches it.",
    "sections": [
      {
        "h": "the same gap, briefly",
        "ps": [
          "The disclosure from chapter 10 holds here too, so say it once and move on: nothing in this chapter ran on a TPU either, for the same reason, and the same Colab lab pass is where it turns from a checked fact into a measured one.",
          "What does change once you actually cross is which machine is underneath you, and that machine does not know which framework it is running. Mark shardings on a torch tensor over a named mesh through torchax, the same `NamedSharding` and `PartitionSpec` objects the jax path's chapter 10 built, and GSPMD partitions the torch program exactly the way it partitions a jax one: same collectives inserted, same propagation, same cost model. Nothing about SPMD had to be reinvented for torch; it only had to be reached through a bridge that already speaks jax underneath.",
          ">> The TPU does not care which framework you came from."
        ]
      },
      {
        "h": "the profiler does not care either",
        "ps": [
          "Profile a torch model running on a TPU and you land in the same XProf world the kernel path's gym reads live at /gym/kernels#timeline: a capture, a device plane, hardware time accounted for op by op. The profiler was built once, for one machine, and it has no separate torch-shaped version of itself. Whatever discipline chapter 9 taught for reading a trace on CPU or GPU carries over here unchanged; only the device plane you are looking at is TPU instead of CPU."
        ]
      },
      {
        "h": "kernels are a compiler question, not a framework one",
        "ps": [
          "Pallas kernels, the whole subject of the kernel path on this site, are callable directly from torch code running on TPU. That fact is worth sitting with, because it means a custom kernel was never really a PyTorch question or a JAX question. It is a question about which side of the compiler boundary you are willing to write by hand, and once you are willing, the framework you started in stops mattering."
        ]
      },
      {
        "h": "serving: the bridge, in production",
        "ps": [
          "The clearest proof that this bridge story is real architecture and not a demo already sits in production. vLLM's TPU backend, tpu-inference, runs PyTorch model definitions through a jax lowering path, the same convergence chapter 10 described, serving real models at real scale without asking the model's author to rewrite anything. If the bridge were a toy, nobody would be routing inference traffic through it."
        ]
      },
      {
        "h": "mapping the stack",
        "ps": [
          "The work this chapter asks for is a map, not a run. Take tpu-inference's stack apart in writing and place every piece of it on the chapter of this site that actually teaches the layer underneath: the SPMD machinery on the jax path's chapter 10, the kernels on the kernel path, the profiler on this chapter. Shard one of your own matmuls over a mesh on paper first, name every collective GSPMD would have to insert, and check your answer against the jax path's chapter 7 before you trust it."
        ]
      }
    ],
    "readings": [
      {
        "label": "Run a calculation on Cloud TPU",
        "url": "https://cloud.google.com/tpu/docs/run-calculation-pytorch",
        "note": "first contact, official"
      },
      {
        "label": "tpu-inference",
        "url": "https://github.com/vllm-project/tpu-inference",
        "note": "the serving stack in production"
      },
      {
        "label": "PyTorch/XLA docs",
        "url": "https://docs.pytorch.org/xla/",
        "note": "the SPMD and profiling guides live here"
      }
    ]
  },
  {
    "id": "training-run",
    "num": 12,
    "part": "ii",
    "title": "The training run",
    "lede": "Nothing in this chapter is new. The training run is what the other eleven chapters were for.",
    "goal": "Given the loop, the pipeline, compilation, distribution, and measurement you already own, compose all five over a TPU bridge, prove a mid-run kill and resume, and report the result with provenance.",
    "sections": [
      {
        "h": "composition, not invention",
        "ps": [
          "Every piece of this capstone already exists somewhere earlier in this path. Chapter 4 gave you the loop: forward, loss, zero the gradients, backward, step. Chapter 5 gave you the pipeline that keeps a device fed instead of starved. Chapter 6 gave you compilation, and chapter 8 gave you distribution across ranks. Chapter 9 gave you the discipline of measuring instead of feeling. Chapter 10 gave you the bridge that carries all of it onto a TPU. The capstone asks you to run all six at once, over real hardware, and that is the assignment in full.",
          ">> The loop you own, on the machine this site teaches.",
          "This is deliberate, not a shortcut taken because nothing new was left to teach. The jax path and the xla path both close the same way, with a capstone that composes rather than introduces, because composing correctly under real constraints, real shapes, a real chip, a real interruption, is a harder test than any single new concept would be."
        ]
      },
      {
        "h": "the bar",
        "ps": [
          "The bar is specific enough to fail cleanly. A small model, trained end to end on a TPU, through either bridge, torch_xla or torchax, your choice. Checkpointed with its full state, not just the parameters: the model's state_dict, the optimizer's state_dict, the step count, and RNG state if a bitwise-identical resume matters to you, the same rule chapter 4 taught, aimed now at a different serialization surface. Killed on purpose partway through. Resumed from that checkpoint.",
          "And the resume proven, not asserted: overlay the loss curve from before the kill against the curve after resume and show they are the same curve, not two curves that happen to end up close."
        ]
      },
      {
        "h": "criteria plus starting points",
        "ps": [
          "Until the Colab labs for this path land, the capstone ships the way the kernel path's own capstone ships: as criteria and starting points rather than a gate this site measures for you. Three things you already have are everything the assignment requires: the loop from chapter 4, the bridge chapter you just read, and a Colab notebook with a TPU attached to it. Nothing else on this site needs to exist first.",
          "That is a genuine constraint, not an excuse. The criteria are exact: the run has to actually train on a TPU, the checkpoint has to actually hold the full state, and the resume has to actually prove itself in an overlaid curve, not a claim in a comment. A capstone that skips any one of those three is not a smaller version of this assignment; it is a different, easier assignment that happens to look similar."
        ]
      },
      {
        "h": "the report is part of the work",
        "ps": [
          "Write a half-page provenance report when the run is done, the same habit this whole path has been building one measured number at a time: which chip, which dtype, the shapes you trained at, which bridge and which method you used, and the overlaid curve itself as the proof. A number without that context is not a result. It is a claim nobody else can check, and this path has spent eleven chapters teaching you not to trust those either."
        ]
      }
    ],
    "readings": [
      {
        "label": "PyTorch basics, the official series",
        "url": "https://docs.pytorch.org/tutorials/beginner/basics/intro.html",
        "note": "the series this path started by superseding, now a subset of what you own"
      },
      {
        "label": "Run a calculation on Cloud TPU",
        "url": "https://cloud.google.com/tpu/docs/run-calculation-pytorch",
        "note": "first contact with the machine, one more time, now for the run that counts"
      }
    ]
  }
]

export const PT_MASTERY: Record<string, WorkItem[]> = {
  "pt:tensors": [
    {
      "id": "read",
      "label": "Read the chapter, the tensor tutorial, and the internals essay"
    },
    {
      "id": "predict-chains",
      "label": "Predict shape, stride, and contiguity for five view chains in writing, then run them to check"
    },
    {
      "id": "explain-expand-reshape",
      "label": "Explain in writing why expand can be free while reshape sometimes has to copy"
    },
    {
      "id": "stride-oracle",
      "label": "stride oracle: streak of 5",
      "href": "/gym/pytorch#drills",
      "auto": {
        "type": "streak",
        "key": "gym.stride.streak",
        "goal": 5
      }
    },
    {
      "id": "labs",
      "label": "run LAB·P1",
      "href": "#labs",
      "auto": {
        "type": "labs",
        "ids": [
          "LAB·P1"
        ]
      }
    }
  ],
  "pt:autograd": [
    {
      "id": "read",
      "label": "Read the chapter, the autograd tutorial, and the mechanics note"
    },
    {
      "id": "draw-tape",
      "label": "Draw the tape for a three-op program in writing and mark what each node saves"
    },
    {
      "id": "museum-exhibits",
      "label": "Visit the autograd exhibits and predict each error before revealing it",
      "href": "/mistakes/pytorch"
    },
    {
      "id": "custom-function",
      "label": "Write one custom Function with a checked backward"
    },
    {
      "id": "labs",
      "label": "run LAB·P2",
      "href": "#labs",
      "auto": {
        "type": "labs",
        "ids": [
          "LAB·P2"
        ]
      }
    }
  ],
  "pt:modules": [
    {
      "id": "read",
      "label": "Read the chapter and the build-model tutorial"
    },
    {
      "id": "param-ledger",
      "label": "Write the parameter-count ledger for one of your own models by hand"
    },
    {
      "id": "check-numel",
      "label": "Check the ledger against sum(p.numel() for p in model.parameters())"
    },
    {
      "id": "train-eval",
      "label": "Move a model between train() and eval() and name every behavior that changed, in writing"
    }
  ],
  "pt:the-loop": [
    {
      "id": "read",
      "label": "read the chapter: the five lines, why zero_grad exists, and what a checkpoint has to contain"
    },
    {
      "id": "train-the-loop",
      "label": "train the loop above end to end and confirm the final loss"
    },
    {
      "id": "break-it-three-ways",
      "label": "break it three ways on purpose: drop zero_grad, checkpoint the model only, resume without the optimizer state"
    },
    {
      "id": "write-the-breakage",
      "label": "write down what each of the three breakages does to the loss curve"
    }
  ],
  "pt:data": [
    {
      "id": "read",
      "label": "read the chapter: Dataset and DataLoader, why workers must pickle, and where the overlap argument comes from"
    },
    {
      "id": "starve-the-loader",
      "label": "starve a training loop on purpose, num_workers=0 and a heavy transform, and find the resulting gap in a profile"
    },
    {
      "id": "state-the-assembly",
      "label": "state in writing where batch tensors are assembled and when they actually reach the device"
    }
  ],
  "pt:dynamo": [
    {
      "id": "read",
      "label": "read the chapter: guards, the cache, graph breaks, and the contrast with tracing once"
    },
    {
      "id": "count-graphs-and-breaks",
      "label": "count graphs and breaks for three of your own functions with torch._dynamo.explain, in writing"
    },
    {
      "id": "name-every-guard",
      "label": "name every guard on one compiled function and trigger each miss deliberately"
    },
    {
      "id": "labs",
      "label": "run LAB·P3",
      "href": "#labs",
      "auto": {
        "type": "labs",
        "ids": [
          "LAB·P3"
        ]
      }
    },
    {
      "id": "guard",
      "label": "guard or break: streak of 5",
      "href": "/gym/pytorch#drills",
      "auto": {
        "type": "streak",
        "key": "gym.guard.streak",
        "goal": 5
      }
    }
  ],
  "pt:graphs": [
    {
      "id": "read",
      "label": "read the chapter: FX under dynamo, aot_autograd's backward trace, decompositions, and the ExportedProgram torch.export produces"
    },
    {
      "id": "annotate-exports",
      "label": "export two of your own modules and annotate every graph line in writing"
    },
    {
      "id": "find-export-refusal",
      "label": "find one program dynamo splits with a graph break but export refuses outright"
    },
    {
      "id": "explain-contract-gap",
      "label": "explain in writing the difference in contract between a dynamo graph break and an export refusal"
    }
  ],
  "pt:distributed": [
    {
      "id": "read",
      "label": "read the chapter: c10d and ranks, DDP versus FSDP, DeviceMesh, and the collective that actually ran"
    },
    {
      "id": "ddp-fsdp-ledger",
      "label": "write the DDP-vs-FSDP ledger from memory, what's replicated, what's sharded, what moves when, then check it against the chapter"
    },
    {
      "id": "world-of-one-collective",
      "label": "run the world-of-one collective above on your own machine"
    },
    {
      "id": "world-of-two-collective",
      "label": "run the same collective again as a world of two processes on one machine, and compare"
    },
    {
      "id": "collective",
      "label": "name the collective: streak of 5",
      "href": "/gym/pytorch#collectives",
      "auto": {
        "type": "streak",
        "key": "gym.collective.streak",
        "goal": 5
      }
    }
  ],
  "pt:performance": [
    {
      "id": "read",
      "label": "read the chapter: the three timing traps, the honest harness, the profiler's schedule, and the AMP dtype decisions"
    },
    {
      "id": "benchmark-three-lies",
      "label": "benchmark one module three wrong ways, no warmup, no sync, a single call, and once right, and write down the size of each lie"
    },
    {
      "id": "profile-the-loop",
      "label": "capture one profiler trace of the training loop from chapter 4 and name its largest host-side gap"
    },
    {
      "id": "amp-dtype-audit",
      "label": "run one training step under autocast and list every op that changed dtype, in writing"
    }
  ],
  "pt:bridges": [
    {
      "id": "read",
      "label": "read the chapter: two bridges, one convergence"
    },
    {
      "id": "ledger",
      "label": "write the two-bridge ledger in writing: capture mechanism, execution model, and sharding story, for both bridges side by side"
    },
    {
      "id": "export",
      "label": "export one of your own modules and state in writing which bridge would consume it, and how"
    },
    {
      "id": "torchax-readme",
      "label": "read the torchax README end to end"
    },
    {
      "id": "labs",
      "label": "run LAB·P4",
      "href": "#labs",
      "auto": {
        "type": "labs",
        "ids": [
          "LAB·P4"
        ]
      }
    }
  ],
  "pt:tpu-practice": [
    {
      "id": "read",
      "label": "read the chapter: SPMD, profiling, kernels, and serving, all on the other side of the bridge"
    },
    {
      "id": "shard-matmul",
      "label": "shard one torch matmul over a mesh on paper and name every collective GSPMD must insert; check it against the jax path's chapter 7"
    },
    {
      "id": "map-stack",
      "label": "map each piece of tpu-inference's stack, in writing, to the path on this site that actually teaches it"
    }
  ],
  "pt:training-run": [
    {
      "id": "read",
      "label": "read the chapter"
    },
    {
      "id": "run-capstone",
      "label": "run the capstone: train the small model end to end on a TPU, through either bridge"
    },
    {
      "id": "prove-resume",
      "label": "kill the run mid-training, resume from the checkpoint, and prove the resume with an overlaid loss curve"
    },
    {
      "id": "provenance-report",
      "label": "write the half-page provenance report: chip, dtype, shapes, method, curve"
    }
  ]
}

export interface PtPathChapter {
  key: string
  href: string
  num: number
  title: string
  part: 'i' | 'ii'
}

/** The path's order; every next/previous control on /xla derives from this. */
export const PT_PATH: PtPathChapter[] = PT_CHAPTERS.map((c) => ({
  key: `pt:${c.id}`,
  href: `/pytorch/${c.id}`,
  num: c.num,
  title: c.title,
  part: c.part,
}))

export const ptChapterAt = (
  key: string,
): { current: PtPathChapter; prev?: PtPathChapter; next?: PtPathChapter } => {
  const idx = PT_PATH.findIndex((c) => c.key === key)
  if (idx === -1) throw new Error(`unknown pytorch path chapter: ${key}`)
  return { current: PT_PATH[idx]!, prev: PT_PATH[idx - 1], next: PT_PATH[idx + 1] }
}
