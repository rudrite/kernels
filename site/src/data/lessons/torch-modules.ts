// New file: site/src/data/lessons/torch-modules.ts
// Module state as a discipline, below the survey chapter 3 teaches: what
// registration does, what the state_dict round trip copies and what it only
// points at, and what a device or dtype move does to the storage underneath.
// Every run in these lessons was executed on this machine (torch 2.2.2, CPU,
// Python 3.12); every source excerpt is verbatim from the installed
// torch/nn/modules/module.py, which is byte-identical to the v2.2.2 tag.
import type { UnitLessons } from './index'

export const TORCH_MODULES_LESSONS: UnitLessons[] = [
  {
    unit: 'pt:modules',
    lessons: [
      {
        id: 'what-the-object-holds',
        num: 1,
        title: 'What the object holds',
        lede: 'A bare nn.Module has seventeen slots on it before you assign anything. Three of them hold your tensors, eleven are tables for code you attach later, and which slot a line of __init__ lands in is decided by one overridden method.',
        goal: 'Given a module class, say for every attribute whether it is registered state, and predict what parameters(), state_dict() and .to() will each do with it; then say where a hook lives and what survives when the model is rebuilt from a checkpoint.',
        sections: [
          {
            h: 'seventeen slots and only three of them hold tensors',
            ps: [
              "Construct `nn.Module()` with no subclass, no layers, nothing, and its `__dict__` already has seventeen keys. Print them and the shape of the class shows itself: `_parameters`, `_buffers` and `_modules` are the three registries, `_non_persistent_buffers_set` marks which buffers stay out of a checkpoint, `training` is the flag chapter 3 explains, and the remaining eleven are hook tables plus one flag about which kind of backward hook is in use.",
              "The way those slots get created is worth reading, because the comment above them explains the whole design in four lines. Every one is written with `super().__setattr__`, deliberately bypassing the module's own attribute assignment, since that assignment is the thing being set up here and calling it before `_parameters` exists would fail.",
              "That failure is real and you can trigger it. Assign an `nn.Parameter` to `self` before calling `super().__init__()` and you get `AttributeError: cannot assign parameters before Module.__init__() call`, raised because `self.__dict__.get('_parameters')` came back `None`.",
            ],
            code: {
              caption: "verbatim, torch/nn/modules/module.py:464-486 in torch 2.2.2 (byte-identical to the v2.2.2 tag), the whole of what a bare module is",
              lang: 'python',
              text: '        """\n        Calls super().__setattr__(\'a\', a) instead of the typical self.a = a\n        to avoid Module.__setattr__ overhead. Module\'s __setattr__ has special\n        handling for parameters, submodules, and buffers but simply calls into\n        super().__setattr__ for all other attributes.\n        """\n        super().__setattr__(\'training\', True)\n        super().__setattr__(\'_parameters\', OrderedDict())\n        super().__setattr__(\'_buffers\', OrderedDict())\n        super().__setattr__(\'_non_persistent_buffers_set\', set())\n        super().__setattr__(\'_backward_pre_hooks\', OrderedDict())\n        super().__setattr__(\'_backward_hooks\', OrderedDict())\n        super().__setattr__(\'_is_full_backward_hook\', None)\n        super().__setattr__(\'_forward_hooks\', OrderedDict())\n        super().__setattr__(\'_forward_hooks_with_kwargs\', OrderedDict())\n        super().__setattr__(\'_forward_hooks_always_called\', OrderedDict())\n        super().__setattr__(\'_forward_pre_hooks\', OrderedDict())\n        super().__setattr__(\'_forward_pre_hooks_with_kwargs\', OrderedDict())\n        super().__setattr__(\'_state_dict_hooks\', OrderedDict())\n        super().__setattr__(\'_state_dict_pre_hooks\', OrderedDict())\n        super().__setattr__(\'_load_state_dict_pre_hooks\', OrderedDict())\n        super().__setattr__(\'_load_state_dict_post_hooks\', OrderedDict())\n        super().__setattr__(\'_modules\', OrderedDict())',
            },
          },
          {
            h: 'assignment is a dispatch',
            ps: [
              "`self.up = nn.Linear(8, 32)` does not store a reference. It calls `Module.__setattr__`, which runs a chain of type tests and routes the value into one of four places, and the order of those tests is the rule you actually need.",
              "A `Parameter` goes first: it gets removed from the plain `__dict__` and from the other two registries, then registered. Next comes a guard: if the name is already a parameter and the new value is not a Parameter, you get a `TypeError` rather than a silent downgrade to a plain attribute. Modules and buffers follow the same two-step shape, each with its own guard.",
              "The last branch is where most bugs live, because it does nothing special at all. Anything that is not a Parameter, not a Module, and not landing on an existing buffer name falls through to `super().__setattr__(name, value)`, which is ordinary Python attribute storage. A tensor put there is not part of the model in any sense the rest of the library recognises.",
            ],
            code: {
              caption: "verbatim, torch/nn/modules/module.py:1699-1711 in torch 2.2.2: the parameter branch and the guard under it",
              lang: 'python',
              text: "        params = self.__dict__.get('_parameters')\n        if isinstance(value, Parameter):\n            if params is None:\n                raise AttributeError(\n                    \"cannot assign parameters before Module.__init__() call\")\n            remove_from(self.__dict__, self._buffers, self._modules, self._non_persistent_buffers_set)\n            self.register_parameter(name, value)\n        elif params is not None and name in params:\n            if value is not None:\n                raise TypeError(f\"cannot assign '{torch.typename(value)}' as parameter '{name}' \"\n                                \"(torch.nn.Parameter or None expected)\"\n                                )\n            self.register_parameter(name, value)",
            },
          },
          {
            h: 'the tensor that is not part of the model',
            ps: [
              "Five assignments, five different fates. The run below registers a parameter, a persistent buffer, a non-persistent buffer, a plain tensor and a submodule, then asks the three questions that matter: does it train, does it get saved, does it move.",
              "The plain tensor answers no to all three. It never appears in `named_parameters()` or `state_dict()`, and after `m.double()` it is still `float32` while the parameter beside it is `float64`, because the walker that performs a conversion visits `_parameters` and `_buffers` and nothing else. A constant you keep on `self` and use inside `forward` will therefore be on the wrong device the first time you run on an accelerator.",
              "The fix is not to remember harder. Register it, or build it inside `forward` from something that is registered, so the answer stops depending on anyone noticing.",
              '>> Registration is the only thing that makes a tensor part of the model. Assignment on its own is just Python.',
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): five assignments, and what each one is worth',
              lang: 'python',
              text: 'import torch\nfrom torch import nn\n\n\nclass Holder(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.w = nn.Parameter(torch.zeros(2))\n        self.register_buffer("steps", torch.zeros(1))\n        self.register_buffer("cache", torch.zeros(2), persistent=False)\n        self.raw = torch.ones(2)                 # a plain attribute\n        self.inner = nn.Linear(2, 2)\n\n\nm = Holder()\nprint(list(dict(m.named_parameters())))   # [\'w\', \'inner.weight\', \'inner.bias\']\nprint(list(dict(m.named_buffers())))      # [\'steps\', \'cache\']\nprint(list(m.state_dict()))               # [\'w\', \'steps\', \'inner.weight\', \'inner.bias\']\nm.double()\nprint(m.w.dtype, m.raw.dtype)             # torch.float64 torch.float32',
            },
            table: {
              caption: 'what each assignment form buys, measured on torch 2.2.2 CPU',
              cols: ['written as', 'in parameters()', 'in state_dict()', 'converted by .to()'],
              rows: [
                ['self.w = nn.Parameter(t)', 'yes', 'yes', 'yes'],
                ['self.register_buffer("b", t)', 'no', 'yes', 'yes'],
                ['self.register_buffer("b", t, persistent=False)', 'no', 'no', 'yes'],
                ['self.raw = t', 'no', 'no', 'no'],
                ['self.inner = nn.Linear(...)', 'yes, through the child', 'yes, prefixed', 'yes'],
                ['self.layers = [nn.Linear(...), ...]', 'no', 'no', 'no'],
                ['self.layers = nn.ModuleList([...])', 'yes', 'yes', 'yes'],
              ],
            },
          },
          {
            h: 'the buffer your checkpoint has never heard of',
            ps: [
              "`persistent=False` is the one registration that says yes to training-time behaviour and no to serialization. The buffer moves with the module, shows up in `buffers()`, and is simply absent from `state_dict()`, which is what you want for a mask or a cached table you can rebuild from the shapes.",
              "Flipping that flag on an existing model has a consequence people meet as a mystery. A checkpoint written before the flip still carries the key, and loading it into the new class reports the key as unexpected rather than ignoring it, because the load walks the module's own persistent names and treats everything else in the dictionary as surplus.",
              "Under `strict=True` that surplus is a `RuntimeError` naming the key. It is the same machinery that catches a genuine typo, which is why it fires on a change you made on purpose.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the same buffer, one flag apart',
              lang: 'python',
              text: 'import torch\nfrom torch import nn\n\n\nclass Old(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.register_buffer("mask", torch.ones(3))\n\n\nclass New(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.register_buffer("mask", torch.ones(3), persistent=False)\n\n\nck = Old().state_dict()\nprint(list(New().buffers()))                       # [tensor([1., 1., 1.])]\nprint(New().load_state_dict(ck, strict=False))\n# _IncompatibleKeys(missing_keys=[], unexpected_keys=[\'mask\'])\nNew().load_state_dict(ck)\n# RuntimeError: Error(s) in loading state_dict for New:\n#     Unexpected key(s) in state_dict: "mask".',
            },
          },
          {
            h: 'eleven tables for code you attach later',
            ps: [
              "Back to the eleven hook dictionaries. A hook is a callable you register on one module instance; the registration returns a handle whose `remove()` deletes that one entry. Forward pre-hooks run before `forward`, forward hooks run after it and may replace the output by returning a value, and backward hooks run when the gradient reaches the module.",
              "Two of those tables share a name for a reason. `register_backward_hook` and `register_full_backward_hook` both write into `_backward_hooks`, with `_is_full_backward_hook` recording which kind you chose, and asking for both on one module raises rather than guessing.",
              "The part that matters for checkpointing is what a hook is not. It is not in `state_dict()` and never was; `deepcopy` carries hooks along, and a freshly constructed module loaded from a checkpoint has none. So a debugging hook you registered in a notebook survives every copy you make in that session and vanishes the moment the model is rebuilt in a training script, silently, with the model still running.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): where hooks live, what order they fire in, and what carries them',
              lang: 'python',
              text: 'import copy\nimport torch\nfrom torch import nn\n\nm = nn.Linear(2, 2)\nlog = []\nm.register_forward_pre_hook(lambda mod, args: log.append("pre"))\nm.register_forward_hook(lambda mod, args, out: log.append("fwd"))\nm.register_full_backward_hook(lambda mod, gi, go: log.append("bwd"))\nm(torch.randn(1, 2, requires_grad=True)).sum().backward()\n\nprint(log)                                                              # [\'pre\', \'fwd\', \'bwd\']\nprint(len(m._forward_hooks), len(m._backward_hooks), m._is_full_backward_hook)   # 1 1 True\nprint(list(m.state_dict()))                                             # [\'weight\', \'bias\']\nprint(len(copy.deepcopy(m)._forward_hooks))                             # 1\n\nfresh = nn.Linear(2, 2)\nfresh.load_state_dict(m.state_dict())\nprint(len(fresh._forward_hooks))                                        # 0',
            },
          },
        ],
        readings: [
          { label: 'module.py at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/nn/modules/module.py', note: 'the 2541 lines this arc reads; __init__ at 456, __setattr__ at 1690' },
          { label: 'the Modules note', url: 'https://docs.pytorch.org/docs/2.2/notes/modules.html', note: "the maintainers' own tour: module state, initialization, and the hook catalogue" },
          { label: 'global module hooks', url: 'https://docs.pytorch.org/docs/2.2/generated/torch.nn.modules.module.register_module_forward_hook.html', note: 'the process-wide version of the same tables, for when you cannot touch the model code' },
        ],
        check: [
          {
            q: 'A tensor you assigned in __init__ is missing from state_dict and stayed float32 after model.double(). What happened to it?',
            a: 'It was a plain tensor, not an nn.Parameter and not a registered buffer, so __setattr__ fell through to ordinary Python attribute storage. Conversions walk _parameters and _buffers only, so nothing ever visits it.',
          },
          {
            q: 'Why does a module holding its children in a Python list hand the optimizer nothing to train?',
            a: 'Because __setattr__ only intercepts values that are themselves Modules. A list is stored plainly, so its contents never enter _modules and parameters() never reaches them. nn.ModuleList registers each child; the measured difference is 0 parameters against 18.',
          },
          {
            q: 'You rebuild a model in a new process and load its checkpoint. What happened to the forward hooks you had registered?',
            a: 'They are gone. Hooks live in the instance hook dictionaries, and state_dict carries only parameters and persistent buffers. A deepcopy would have kept them; a fresh construction plus load_state_dict does not, and nothing warns.',
          },
        ],
        work: [
          { id: 'slot-audit', label: 'take one module of your own and label every line of its __init__ with which of the four slots the value lands in', href: '#assignment-is-a-dispatch' },
          { id: 'plain-tensor-hunt', label: 'search your code for tensors assigned to self that are not Parameters or buffers, and decide for each whether it should be registered or built in forward', href: '#the-tensor-that-is-not-part-of-the-model' },
        ],
      },
      {
        id: 'the-round-trip',
        num: 2,
        title: 'The round trip',
        lede: 'state_dict does not copy your weights and load_state_dict does not install the ones you hand it. One returns references into the live model, the other writes through the tensors that are already there, and four familiar checkpoint bugs follow from those two sentences.',
        goal: 'Given a checkpoint and a module, predict whether the load copies or assigns, which dtype and device the result carries, which mismatches raise and which are only reported, and what the version stamp beside the tensors is for.',
        sections: [
          {
            h: 'detach is not copy',
            ps: [
              "The whole of what `state_dict()` puts in the dictionary is two lines of `_save_to_state_dict`, one for parameters and one for buffers, and both read the same way. Each entry is the live tensor, or `.detach()` of it, which shares the same storage and only drops the autograd history.",
              "Chapter 3 calls this the serialization contract, and it is, once `torch.save` has written the bytes out. Before that write, the dictionary is a set of pointers into the model you are still training.",
              "Sometimes the live tensors are the point, and `keep_vars=True` hands them over ungutted, gradients and all, for inspection rather than saving. Weight tying shows the same identity question from the other side: `parameters()` deduplicates by object identity while `state_dict()` does not, so a tied weight is one parameter under two keys, both pointing at one storage.",
            ],
            code: {
              caption: 'verbatim, torch/nn/modules/module.py:1800-1805 in torch 2.2.2: everything state_dict puts in the dictionary',
              lang: 'python',
              text: '        for name, param in self._parameters.items():\n            if param is not None:\n                destination[prefix + name] = param if keep_vars else param.detach()\n        for name, buf in self._buffers.items():\n            if buf is not None and name not in self._non_persistent_buffers_set:\n                destination[prefix + name] = buf if keep_vars else buf.detach()',
            },
          },
          {
            h: 'so the best weights were never a snapshot',
            ps: [
              "Validation improves, so you keep `best = model.state_dict()` in a variable, training runs on, and at the end you restore from `best`. What you restore is the final model, because `best` was never holding numbers of its own.",
              "The run below shows it in four lines: the dictionary's tensor and the parameter report the same `data_ptr`, and one in-place update through the parameter is visible through the dictionary immediately.",
              "`copy.deepcopy(model.state_dict())` is the fix for an in-memory snapshot, and `torch.save` is the fix for a durable one, since writing to disk materializes the bytes. Chapter 4 covers what else belongs in that file next to the model.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the snapshot that moves with the model, and the tied weight that is two keys',
              lang: 'python',
              text: 'import torch\nfrom torch import nn\n\n\nclass Tied(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.a = nn.Linear(2, 2, bias=False)\n        self.b = nn.Linear(2, 2, bias=False)\n        self.b.weight = self.a.weight\n\n\nm = nn.Linear(2, 2)\nbest = m.state_dict()                                    # the "snapshot"\nprint(best["weight"].data_ptr() == m.weight.data_ptr())  # True\nwith torch.no_grad():\n    m.weight.add_(1.0)                                   # one more optimizer step\nprint(torch.equal(best["weight"], m.weight))             # True\nprint(best["weight"].requires_grad,\n      m.state_dict(keep_vars=True)["weight"].requires_grad)   # False True\n\nt = Tied()\nprint(len(list(t.parameters())), list(t.state_dict()))   # 1 [\'a.weight\', \'b.weight\']',
            },
          },
          {
            h: 'the load writes through what is already there',
            ps: [
              "Loading runs the mirror image. `_load_from_state_dict` finds the module's own parameter, checks the shape, and then calls `param.copy_(input_param)` under `no_grad`. The tensor object in your model is the destination and it keeps its own dtype, its own device and its own identity; only the values arrive.",
              "That is why a float64 checkpoint loads into a float32 model without a word. The copy casts, the model stays float32, and the numbers you get are the checkpoint's rounded to single precision. Nothing in the return value mentions it.",
              "`assign=True` switches the whole thing to the other semantics. Instead of copying into the existing tensor, it calls `setattr` with the checkpoint's tensor, so the module adopts that object whole: its dtype, its device, its storage. The measured `data_ptr` is the same one the checkpoint dictionary holds.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the same checkpoint, copied and assigned',
              lang: 'python',
              text: 'import torch\nfrom torch import nn\n\nck = {"weight": torch.ones(2, 2, dtype=torch.float64),\n      "bias": torch.ones(2, dtype=torch.float64)}\n\ndst = nn.Linear(2, 2)\ndst.load_state_dict(ck)\nprint(dst.weight.dtype, dst.weight[0, 0].item())   # torch.float32 1.0\n\ndst2 = nn.Linear(2, 2)\ndst2.load_state_dict(ck, assign=True)\nprint(dst2.weight.dtype,\n      dst2.weight.data_ptr() == ck["weight"].data_ptr())   # torch.float64 True',
            },
          },
          {
            h: 'strict picks the error message, not the detection',
            ps: [
              "Read `load_state_dict` and one argument in the recursive call settles a common misreading. The `strict` you passed is not forwarded; the literal `True` is, so missing and unexpected keys are collected on every load. What `strict` decides is whether those two lists get turned into an exception at the end.",
              "So `load_state_dict(sd, strict=False)` still knows exactly which keys were missing and which were surplus. It returns them, as `_IncompatibleKeys(missing_keys, unexpected_keys)`, and almost every call site in the wild throws that return value away. Assign it and print it and half of the load-time mysteries stop being mysteries.",
              "One class of problem ignores `strict` entirely. A shape mismatch appends to `error_msgs`, a third list, and `error_msgs` raises whether or not you asked for strictness, because there is no sensible way to partially copy a differently shaped tensor. The message names the key, the checkpoint's shape and the model's, in that order.",
              '>> strict=False does not make the load quieter about what it found. It makes the load stop raising about it.',
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the report you get back, and the one mismatch that raises anyway',
              lang: 'python',
              text: 'import torch\nfrom torch import nn\n\n\nclass Wider(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.lin = nn.Linear(2, 2)\n        self.extra = nn.Linear(2, 2)\n\n\nck = {"lin.weight": torch.zeros(2, 2), "lin.bias": torch.zeros(2), "ghost": torch.zeros(1)}\nreport = Wider().load_state_dict(ck, strict=False)\nprint(report.missing_keys)      # [\'extra.weight\', \'extra.bias\']\nprint(report.unexpected_keys)   # [\'ghost\']\n\nWider().load_state_dict({"lin.weight": torch.zeros(3, 3)}, strict=False)\n# RuntimeError: Error(s) in loading state_dict for Wider:\n#     size mismatch for lin.weight: copying a param with shape torch.Size([3, 3])\n#     from checkpoint, the shape in current model is torch.Size([2, 2]).',
            },
            table: {
              caption: 'four mismatches, two settings; measured on torch 2.2.2 CPU',
              cols: ['mismatch', 'strict=True', 'strict=False'],
              rows: [
                ['a key the model wants and the checkpoint lacks', 'RuntimeError, key named', 'returned in missing_keys'],
                ['a key the checkpoint has and the model does not', 'RuntimeError, key named', 'returned in unexpected_keys'],
                ['same key, different shape', 'RuntimeError', 'RuntimeError, identical message'],
                ['same key, different dtype', 'loads, cast into the model dtype', 'loads, cast into the model dtype'],
              ],
            },
          },
          {
            h: 'the version stamp travelling beside the tensors',
            ps: [
              "A state dict carries one thing that is not a tensor. `state_dict()` attaches a `_metadata` attribute mapping each submodule prefix to a small dict, and the entry it always writes is `version=self._version`, a class attribute that a layer bumps when its saved state changes shape.",
              "`BatchNorm` is the worked example. Its `_version` is 2, because version 2 added the `num_batches_tracked` buffer, and its own `_load_from_state_dict` looks at the incoming version and synthesizes that key when an older checkpoint does not have it. A version-1 checkpoint therefore loads clean under `strict=True`.",
              "Stamp the same incomplete dictionary as version 2 and it fails with a missing key. The tensors did not change; the claim about what those tensors mean did. That is the mechanism available to your own modules too: bump `_version`, override `_load_from_state_dict`, migrate the old layout forward before calling `super()`.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the same tensors, two version stamps',
              lang: 'python',
              text: 'import torch\nfrom torch import nn\n\nsd = nn.BatchNorm1d(3).state_dict()\nprint(list(sd))\n# [\'weight\', \'bias\', \'running_mean\', \'running_var\', \'num_batches_tracked\']\nprint(dict(sd._metadata))                  # {\'\': {\'version\': 2}}\n\nold = type(sd)({k: v for k, v in sd.items() if k != "num_batches_tracked"})\nold._metadata = {"": {"version": 1}}\nprint(nn.BatchNorm1d(3).load_state_dict(old, strict=True))\n# <All keys matched successfully>\n\nold._metadata = {"": {"version": 2}}\nnn.BatchNorm1d(3).load_state_dict(old, strict=True)\n# RuntimeError: Error(s) in loading state_dict for BatchNorm1d:\n#     Missing key(s) in state_dict: "num_batches_tracked".',
            },
          },
          {
            h: 'one flag that stays in the dictionary',
            ps: [
              "The `assign` argument reaches `_load_from_state_dict` through that same metadata, and the line that puts it there writes into the dictionary the caller owns rather than into a copy. `local_metadata` is the per-prefix dict stored inside `state_dict._metadata`, so `local_metadata['assign_to_params_buffers'] = assign` is a permanent edit to the checkpoint object.",
              "Load once with `assign=True` and every later load from that same dictionary assigns, whether or not you ask. Two models loaded from it afterwards do not get copies; they get the same tensors, sharing one storage, so a training step on one moves the other.",
              "The line is unchanged upstream at v2.8.0, at module.py:2571-2575, so this is a property of the API rather than a fixed bug in an old release. The safe habit is one dictionary per load when `assign` is in play, or a fresh `state_dict()` call, which builds a fresh `_metadata` every time.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): one assign load, and two models that now share storage',
              lang: 'python',
              text: 'import torch\nfrom torch import nn\n\nck = nn.Linear(4, 4).state_dict()\nprint(dict(ck._metadata))                  # {\'\': {\'version\': 1}}\n\nnn.Linear(4, 4, device="meta").load_state_dict(ck, assign=True)\nprint(dict(ck._metadata))\n# {\'\': {\'version\': 1, \'assign_to_params_buffers\': True}}\n\na, b = nn.Linear(4, 4), nn.Linear(4, 4)\na.load_state_dict(ck)                      # no assign asked for\nb.load_state_dict(ck)\nprint(a.weight.data_ptr() == b.weight.data_ptr())   # True\nwith torch.no_grad():\n    a.weight.add_(1.0)\nprint(torch.equal(a.weight, b.weight))              # True',
            },
          },
        ],
        readings: [
          { label: 'load_state_dict at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/nn/modules/module.py#L1953', note: '_load_from_state_dict at 1953 and load_state_dict at 2067: shape check, copy_, and the three lists' },
          { label: 'Saving and loading models', url: 'https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html', note: 'the official tour of the same contract, including partial loads and warmstarting' },
          { label: 'serialization semantics', url: 'https://docs.pytorch.org/docs/2.2/notes/serialization.html', note: 'what torch.save actually writes, and why saving the state dict beats pickling the module' },
        ],
        check: [
          {
            q: 'You loaded a float64 checkpoint into a freshly built float32 model and nothing complained. What dtype are the parameters now, and why?',
            a: 'float32. The load calls param.copy_(input_param) into the tensor the model already holds, so the destination keeps its dtype and the values are cast down. Passing assign=True instead adopts the checkpoint tensor whole, dtype included.',
          },
          {
            q: 'What did you throw away by ignoring the return value of load_state_dict(sd, strict=False)?',
            a: 'The _IncompatibleKeys pair. Missing and unexpected keys are collected on every load, because the recursive call passes a literal True rather than your strict argument; strict only decides whether those lists become a RuntimeError.',
          },
          {
            q: 'Why does a size mismatch raise even when you asked for strict=False?',
            a: 'Because it lands in error_msgs, a third list, and error_msgs is raised unconditionally at the end of load_state_dict. Only the missing and unexpected lists are gated on strict.',
          },
        ],
        work: [
          { id: 'snapshot-proof', label: 'prove on your own model that a kept state_dict tracks later training, then fix it with deepcopy and prove that too', href: '#so-the-best-weights-were-never-a-snapshot' },
          { id: 'report-the-report', label: 'add the _IncompatibleKeys return value to every strict=False load in your code and print it', href: '#strict-picks-the-error-message-not-the-detection' },
          { id: 'version-migration', label: 'write a _load_from_state_dict override for a module of yours that renames one buffer, and bump its _version' },
        ],
      },
      {
        id: 'what-a-move-does',
        num: 3,
        title: 'What a move does',
        lede: 'model.to() is not the tensor call you already know. It edits the module in place, and depending on what you asked for it either swaps the storage under your parameters or replaces the parameter objects entirely, which decides whether the optimizer you built five lines earlier is still pointing at your model.',
        goal: 'For a given .to() call, say whether the Parameter objects survive, what happens to their storage and their grads, which buffers and attributes are skipped, and what state elsewhere in your program has just gone stale.',
        sections: [
          {
            h: 'one walker, two outcomes',
            ps: [
              "Every conversion a module can perform goes through `_apply`: `.to()`, `.cuda()`, `.float()`, `.half()`, `to_empty()`. It recurses into children, then walks `_parameters` and `_buffers`, applies the function under `no_grad`, and returns `self`. That last word is the difference from tensors: `t.to(...)` hands you a new tensor and leaves `t` alone, while `m.to(...)` rewrites `m` and hands it back.",
              "Inside the parameter loop there is a fork, and it decides more than it looks like it does. `compute_should_use_set_data` asks `torch._has_compatible_shallow_copy_type(tensor, tensor_applied)`, and on a yes it writes `param.data = param_applied`, keeping the Parameter object and giving it different storage. On a no it constructs a new `Parameter` and puts that in `_parameters` instead.",
              "A dtype change on the same device takes the first branch. A device change takes the second. Everything in this lesson follows from which branch ran.",
            ],
            code: {
              caption: 'verbatim, torch/nn/modules/module.py:818-834 in torch 2.2.2: the parameter half of _apply',
              lang: 'python',
              text: '        for key, param in self._parameters.items():\n            if param is None:\n                continue\n            # Tensors stored in modules are graph leaves, and we don\'t want to\n            # track autograd history of `param_applied`, so we have to use\n            # `with torch.no_grad():`\n            with torch.no_grad():\n                param_applied = fn(param)\n            should_use_set_data = compute_should_use_set_data(param, param_applied)\n            if should_use_set_data:\n                param.data = param_applied\n                out_param = param\n            else:\n                assert isinstance(param, Parameter)\n                assert param.is_leaf\n                out_param = Parameter(param_applied, param.requires_grad)\n                self._parameters[key] = out_param',
            },
          },
          {
            h: 'a dtype move keeps the object and swaps the bytes',
            ps: [
              "Ask for the dtype a module already has and nothing happens at all: `t.to(torch.float32)` on a float32 tensor returns the same tensor, so `param.data` is reassigned to itself and the storage address never changes. That is why calling `.float()` defensively costs nothing.",
              "Ask for a different dtype and the object identity survives while the storage does not. The measured result is the same `Parameter` at a new address, with `requires_grad` carried over by the `set_data` path and grads converted alongside in the loop just below.",
              "So anything holding a reference to a parameter keeps working across a dtype change. That includes the optimizer, which is the reason the next failure is confusing when you meet it.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the same parameter, three conversions',
              lang: 'python',
              text: 'import torch\nfrom torch import nn\n\nm = nn.Linear(4, 4)\np, ptr = m.weight, m.weight.data_ptr()\n\nm.to(torch.float32)                       # already float32\nprint(m.weight is p, m.weight.data_ptr() == ptr)          # True True\n\nm.to(torch.float64)                       # a real conversion, same device\nprint(m.weight is p, m.weight.data_ptr() == ptr,\n      m.weight.requires_grad)                             # True False True\n\nn = nn.Linear(4, 4)\nq = n.weight\nn.to("meta")                              # a device change\nprint(n.weight is q,\n      torch._has_compatible_shallow_copy_type(q, n.weight))   # False False',
            },
          },
          {
            h: 'a device move replaces the object',
            ps: [
              "Cross a device boundary and `_has_compatible_shallow_copy_type` says no, so `_apply` installs brand new `Parameter` objects in `_parameters`. The old ones are still alive, still holding their storage, and still sitting in whatever list you handed them to earlier.",
              "An optimizer built before the move is still holding the originals. The arc under chapter 4 establishes why: `model.parameters()` is drained into a flat list of tensor references at construction, and nothing re-reads it. So the optimizer keeps stepping, it raises nothing, and it updates tensors the module no longer refers to. The run below does it on this machine using the meta device, which is the second device available here: after the move, the optimizer's parameter is the old object, that old object has moved, and the module's parameter has not.",
              "This is the mechanism under the usual advice to construct the optimizer after moving the model. The advice is right; the reason is not that PyTorch loses track of your parameters, but that a device move is defined as replacement rather than mutation.",
              '>> A dtype change edits your parameters. A device change replaces them, and every reference you kept still points at the originals.',
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU; meta stands in for a second device, and the branch taken is the same one a cuda move takes)',
              lang: 'python',
              text: 'import torch\nfrom torch import nn\n\ntorch.manual_seed(0)\nm = nn.Linear(4, 4)\nopt = torch.optim.SGD(m.parameters(), lr=1.0)\nm(torch.randn(8, 4)).sum().backward()\n\norphan = m.weight\nbefore = orphan.detach().clone()\nm.to("meta")                               # a device change: new Parameter objects\nopt.step()\n\nprint(opt.param_groups[0]["params"][0] is orphan)   # True\nprint(torch.equal(orphan.detach(), before))         # False: the step moved the orphan\nprint(m.weight.is_meta, m.weight.data_ptr())        # True 0',
            },
          },
          {
            h: 'what the optimizer is left holding',
            ps: [
              "The dtype case fails differently, and later. Adam's per-parameter state is allocated on that parameter's first step and in that parameter's dtype, which the arc under chapter 4 measures from the optimizer side. `_apply` never touches the optimizer, so convert the model afterwards and the parameter is float64 while `exp_avg` is still float32.",
              "The lookup still works, because the state is keyed by the Parameter object and that object survived. The next `opt.step()` is where it lands: `RuntimeError: expected dtype float for` `end` `but got dtype double`, raised from `exp_avg.lerp_(grad, 1 - beta1)` inside Adam.",
              "Ordering is what removes the whole class. Decide device and dtype before the optimizer exists, and treat a mid-run conversion as a rebuild: new optimizer, then reload its state dict if the run has to continue from where it was.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): Adam converted out from under, error text verbatim',
              lang: 'python',
              text: 'import torch\nfrom torch import nn\n\ntorch.manual_seed(0)\nm = nn.Linear(4, 4)\nopt = torch.optim.Adam(m.parameters(), lr=1e-2)\nm(torch.randn(8, 4)).sum().backward()\nopt.step()\nprint(opt.state[m.weight]["exp_avg"].dtype)          # torch.float32\n\nm.to(torch.float64)                        # same device, so the objects survive\nprint(m.weight.dtype, m.weight in opt.state,\n      opt.state[m.weight]["exp_avg"].dtype)          # torch.float64 True torch.float32\n\nopt.zero_grad()\nm(torch.randn(8, 4).double()).sum().backward()\nopt.step()\n# RuntimeError: expected dtype float for `end` but got dtype double',
            },
          },
          {
            h: 'what convert refuses to touch',
            ps: [
              "`Module.to` does not hand `_apply` your arguments directly. It builds a closure called `convert` at module.py:1146, and the decisive expression inside it reads `dtype if t.is_floating_point() or t.is_complex() else None`. The dtype reaches a tensor only when that test passes, so an integer buffer keeps its own dtype while the parameters around it change.",
              "That rule is what keeps a step counter, a token id table or a boolean mask intact through `model.half()`. It also means a dtype argument is not a promise about every tensor in the module, which matters when you are reasoning about what a mixed-dtype forward pass will do.",
              "The plain attribute from lesson one is skipped for the older reason: `_apply` walks two registries, and it was never in either. A module converted to float64 can therefore be carrying an int64 buffer and a float32 tensor on `self` at the same time, all three correct by the rules and only one of them what you meant.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): one .to(), three dtypes afterwards',
              lang: 'python',
              text: 'import torch\nfrom torch import nn\n\n\nclass Step(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.lin = nn.Linear(2, 2)\n        self.register_buffer("count", torch.zeros(1, dtype=torch.int64))\n        self.scale = torch.ones(1)              # a plain attribute\n\n\nm = Step().to(torch.float64)\nprint(m.lin.weight.dtype, m.count.dtype, m.scale.dtype)\n# torch.float64 torch.int64 torch.float32',
            },
          },
          {
            h: 'starting with no storage at all',
            ps: [
              "One device allocates nothing. Build a module under `torch.device('meta')` and its parameters have real shapes and dtypes with a `data_ptr` of zero, which is how you get the structure of a model too large to allocate twice, or how you inspect shapes without paying for them.",
              "From there, two ways forward. `to_empty(device='cpu')` allocates storage of the right shape and leaves it uninitialized, ready for a normal load; the values in between are whatever the allocator handed back, so reading them before the load is a bug. Or load with `assign=True`, which adopts the checkpoint's tensors and allocates nothing of its own.",
              "The path that does not work is the ordinary copy load onto a meta module. `copy_` into a tensor with no storage writes nowhere, so the load warns per key that it is a no-op and suggests `assign=True`, and the module stays meta. Read the warning rather than the missing exception; there is no error here, only a model that never got its weights.",
              "The same storage question runs through initialization. `reset_parameters` writes into the storage the module already has, in place, so it keeps the address and the parameter identity, which is why re-initializing a model does not invalidate an optimizer while moving it does.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): three ways to fill a model that started with nothing',
              lang: 'python',
              text: 'import torch\nfrom torch import nn\n\ntorch.manual_seed(0)\ntrained = nn.Linear(4, 4)\n\nskeleton = nn.Linear(4, 4, device="meta")\nprint(skeleton.weight.device, skeleton.weight.data_ptr())   # meta 0\n\na = nn.Linear(4, 4, device="meta")\na.to_empty(device="cpu")                   # allocate, uninitialized\na.load_state_dict(trained.state_dict())\nprint(a.weight.data_ptr() != 0, torch.equal(a.weight, trained.weight))   # True True\n\nck = trained.state_dict()\nb = nn.Linear(4, 4, device="meta")\nb.load_state_dict(ck, assign=True)         # adopt the checkpoint\'s tensors\nprint(b.weight.device, b.weight.data_ptr() == ck["weight"].data_ptr())   # cpu True\n\nc = nn.Linear(4, 4, device="meta")\nc.load_state_dict(trained.state_dict())    # the copy path, onto no storage\n# UserWarning: for weight: copying from a non-meta parameter in the checkpoint\n# to a meta parameter in the current model, which is a no-op.\nprint(c.weight.is_meta)                    # True',
            },
          },
        ],
        readings: [
          { label: '_apply at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/nn/modules/module.py#L799', note: 'the walker every conversion goes through, and the set_data fork at 804' },
          { label: 'nn.Module.to_empty', url: 'https://docs.pytorch.org/docs/2.2/generated/torch.nn.Module.html#torch.nn.Module.to_empty', note: 'the documented way to give a meta module storage, uninitialized' },
          { label: 'torch.nn.utils.skip_init', url: 'https://docs.pytorch.org/docs/2.2/generated/torch.nn.utils.skip_init.html', note: 'the wrapper around the same idea for modules whose __init__ would otherwise initialize' },
        ],
        check: [
          {
            q: 'After model.to(torch.float64), is the optimizer you built beforehand still pointing at the right tensors?',
            a: 'Yes. A same-device dtype change takes the set_data branch, so the Parameter objects survive and the optimizer still finds its state. Its state tensors are still float32 though, and the next step raises: expected dtype float for `end` but got dtype double.',
          },
          {
            q: 'After a device change, why does opt.step() move tensors the model no longer holds?',
            a: 'Because _has_compatible_shallow_copy_type is false across devices, so _apply builds new Parameter objects and installs them in _parameters. The optimizer param_groups still reference the originals, which are alive and now orphaned.',
          },
          {
            q: 'Your module has an int64 step counter as a registered buffer. What does model.to(torch.float64) do to it?',
            a: 'Nothing. Module.to builds a convert closure that passes the dtype through only when the tensor is floating point or complex, so integer and boolean buffers keep their dtype while the parameters around them convert.',
          },
        ],
        work: [
          { id: 'branch-call', label: 'for every .to(), .cuda(), .half() and to_empty() in your code, say which _apply branch it takes and what references it invalidates', href: '#one-walker-two-outcomes' },
          { id: 'orphan-repro', label: 'reproduce the orphaned-optimizer step yourself, then fix it by ordering the move before the optimizer', href: '#a-device-move-replaces-the-object' },
          { id: 'meta-build', label: 'build one model of your own under torch.device("meta") and fill it both ways, to_empty plus load and load with assign=True', href: '#starting-with-no-storage-at-all' },
        ],
      },
    ],
  },
]
