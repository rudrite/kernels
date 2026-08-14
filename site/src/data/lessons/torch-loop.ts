// New file: site/src/data/lessons/torch-loop.ts
// Below chapter 4's five lines: what an optimizer actually holds, the two
// lines that commonly sit between backward and step, and what a checkpoint
// has to carry for a resume to match digit for digit. Every snippet here and
// every number it prints ran on this machine, torch 2.2.2 on CPU, 2026-08-14.
// The two source excerpts are verbatim from pytorch at the v2.2.2 tag.
import type { UnitLessons } from './index'

export const TORCH_LOOP_LESSONS: UnitLessons[] = [
  {
    unit: 'pt:the-loop',
    lessons: [
      {
        id: 'what-the-optimizer-holds',
        num: 1,
        title: 'What the optimizer holds',
        lede: 'An optimizer looks like a box you construct once and call twice per step. Open it and there are two containers: a list of parameter groups carrying every hyperparameter, and a dictionary of per-parameter state that does not exist at all until the first step runs.',
        goal: 'Given any optimizer, say what lives in a param group and what lives in state, predict exactly what load_state_dict checks before it accepts a checkpoint, and explain what a gradient of None does that a gradient of zeros does not.',
        sections: [
          {
            h: 'params is a list, captured once',
            ps: [
              'The chapter above this lesson calls `opt.step()` the line where parameters actually move. Which parameters is a question answered once, at construction: `model.parameters()` is a generator, and the optimizer drains it into a flat list of tensor references that nothing ever re-reads.',
              'That sentence is easy to nod along to, so run the version that bites. Build the optimizer, then replace a layer on the model. The new weight is in the graph, so backward fills its `.grad` exactly as you would expect. It is not in the optimizer\'s list, so `step()` walks past it. Nothing raises, and one layer of the model quietly never learns.',
              'Two operations exist for the case where the parameter set legitimately changes. `add_param_group` appends to the list, which is how a fine-tuning script unfreezes a backbone partway through a run. Rebuilding the optimizer is the other, and it throws away the state that the third section of this lesson is about.',
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): a layer swapped in after construction gets a gradient and never moves',
              lang: 'python',
              text: 'import torch\nfrom torch import nn\n\ntorch.manual_seed(0)\nmodel = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 1))\nopt = torch.optim.SGD(model.parameters(), lr=0.1)\n\nmodel[2] = nn.Linear(4, 1)        # a layer replaced after the optimizer was built\nx = torch.randn(16, 8)\nbefore = model[2].weight.detach().clone()\n\nloss = nn.functional.mse_loss(model(x), x.sum(dim=1, keepdim=True))\nopt.zero_grad()\nloss.backward()\nopt.step()\n\nprint(model[2].weight.grad is not None)               # True: it did get a gradient\nprint(torch.equal(before, model[2].weight.detach()))  # True: it never moved',
            },
          },
          {
            h: 'one group, ten hyperparameters',
            ps: [
              '`param_groups` is a list of plain dictionaries. Each one holds its own `params` list plus every hyperparameter the optimizer takes, filled in from the constructor defaults. An AdamW built with nothing but a learning rate still carries ten of them, `fused` and `capturable` included.',
              'So the learning rate is not one number on the optimizer. It is one number per group, and anything that changes it changes it group by group. The third lesson in this arc lives on that fact.',
              'Groups exist so one optimizer can apply different hyperparameters to different parameters, and the common use is weight decay: decay the weight matrices, leave the biases alone. Hand the constructor a list of dicts instead of a bare iterable and each dict becomes a group, inheriting whatever it does not override.',
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): one group, four parameters, ten hyperparameters',
              lang: 'python',
              text: 'import torch\nfrom torch import nn\n\ntorch.manual_seed(0)\nmodel = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1))\nopt = torch.optim.AdamW(model.parameters(), lr=1e-2)\n\nprint(len(opt.param_groups), len(opt.param_groups[0]["params"]))    # 1 4\nprint(sorted(k for k in opt.param_groups[0] if k != "params"))\n# [\'amsgrad\', \'betas\', \'capturable\', \'differentiable\', \'eps\', \'foreach\',\n#  \'fused\', \'lr\', \'maximize\', \'weight_decay\']',
            },
            table: {
              caption: 'the two containers on any torch.optim.Optimizer, read off AdamW on torch 2.2.2 CPU',
              cols: ['container', 'keyed by', 'holds', 'exists from'],
              rows: [
                ['param_groups[i]', 'position in the list', 'a params list plus every hyperparameter, lr among them', 'construction'],
                ['state[p]', 'the Parameter object itself', "the optimizer's memory for that one parameter: exp_avg, exp_avg_sq and step for Adam", "that parameter's first step"],
              ],
            },
          },
          {
            h: 'state arrives on the first step',
            ps: [
              'Construct an AdamW and `len(opt.state)` reads zero. The running estimates are allocated lazily, on the first step that sees a gradient for a given parameter, which is why an optimizer you built but never stepped serializes to an empty state.',
              'The dictionary is keyed by the `Parameter` object, not by a name and not by an index. Ask for `opt.state[model[0].weight]` and you get the three tensors Adam keeps for that weight: `exp_avg`, `exp_avg_sq`, and `step`. That last one is a tensor rather than a Python int, which starts to matter the moment the state has to move to a device or be captured inside a graph.',
              'Laziness has a practical edge worth carrying. A parameter that has never received a gradient has no state at all, so a checkpoint taken before the first step and one taken after are different objects, and both of them load without complaint.',
            ],
            code: {
              caption: 'continuing the block above (verified, torch 2.2.2 CPU): state is empty, then it is not',
              lang: 'python',
              text: 'print(len(opt.state))                       # 0: nothing has stepped yet\n\nx = torch.randn(64, 8)\nnn.functional.mse_loss(model(x), x.sum(dim=1, keepdim=True)).backward()\nopt.step()\n\nw = model[0].weight\nprint(len(opt.state), sorted(opt.state[w]))   # 4 [\'exp_avg\', \'exp_avg_sq\', \'step\']\nprint(opt.state[w]["step"])                   # tensor(1.)',
            },
          },
          {
            h: 'the state dict speaks in positions',
            ps: [
              '`opt.state_dict()` cannot serialize tensor keys, so it replaces them with integers, counting through the groups in order. The state comes back keyed 0, 1, 2, 3, and each group\'s `params` entry becomes the list of indices belonging to it.',
              '`load_state_dict` rebuilds the mapping by zipping the saved indices against the live parameters in the same order. Two checks run before that zip, and both are about counts: how many groups, and how many parameters in each group. The twelve lines below are the whole of it.',
              'Read what is not there. No shape is compared. No name exists at this layer to compare. Reorder the parameters you hand the constructor and the checkpoint still loads, mapping Adam\'s memory for layer one onto layer three.',
            ],
            code: {
              caption: 'verbatim, torch/optim/optimizer.py:741-752 at the v2.2.2 tag, inside load_state_dict',
              lang: 'python',
              text: '        if len(groups) != len(saved_groups):\n            raise ValueError("loaded state dict has a different number of "\n                             "parameter groups")\n        param_lens = (len(g[\'params\']) for g in groups)\n        saved_lens = (len(g[\'params\']) for g in saved_groups)\n        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):\n            raise ValueError("loaded state dict contains a parameter group "\n                             "that doesn\'t match the size of optimizer\'s group")\n\n        # Update the state\n        id_map = dict(zip(chain.from_iterable(g[\'params\'] for g in saved_groups),\n                      chain.from_iterable(g[\'params\'] for g in groups)))',
            },
          },
          {
            h: 'what load_state_dict does not check',
            ps: [
              'Take the state dict from a model with a 32-wide hidden layer and load it into a model with a 4-wide one. Same number of groups, same four parameters, so both checks pass and the load returns cleanly. The optimizer now holds an `exp_avg` of shape (32, 8) for a parameter of shape (4, 8).',
              'The failure arrives one step later, from inside AdamW, as an ordinary broadcasting error out of `exp_avg.lerp_(grad, 1 - beta1)`. That is a good error to recognize on sight, because the line it names has nothing to do with the mistake, which happened at load time in a different function.',
              'The other half of a checkpoint behaves the opposite way. `model.state_dict()` is keyed by parameter name and refuses a mismatch at load time, which the modules arc under chapter three demonstrates on a buffer key. Only one of the two halves tells you at the moment you were wrong.',
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): a load that succeeds and a step that does not',
              lang: 'python',
              text: 'import torch\nfrom torch import nn\n\ntorch.manual_seed(0)\nbig = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1))\nopt = torch.optim.AdamW(big.parameters(), lr=1e-2)\nx = torch.randn(64, 8)\nnn.functional.mse_loss(big(x), x.sum(dim=1, keepdim=True)).backward()\nopt.step()\n\nsd = opt.state_dict()\nprint(list(sd), sorted(sd["state"]), sd["param_groups"][0]["params"])\n# [\'state\', \'param_groups\'] [0, 1, 2, 3] [0, 1, 2, 3]\n\nsmall = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 1))\nopt2 = torch.optim.AdamW(small.parameters(), lr=1e-2)\nopt2.load_state_dict(sd)                      # accepted, no complaint\nprint(opt2.state[small[0].weight]["exp_avg"].shape, small[0].weight.shape)\n# torch.Size([32, 8]) torch.Size([4, 8])\n\nnn.functional.mse_loss(small(x), x.sum(dim=1, keepdim=True)).backward()\nopt2.step()\n# RuntimeError: The size of tensor a (32) must match the size of tensor b (4)\n# at non-singleton dimension 0',
            },
          },
          {
            h: 'None and zero are not the same gradient',
            ps: [
              'The autograd arc under chapter two covers what `zero_grad` does to a reference you were holding. The question here is what the optimizer does next, and it is a different one: every optimizer in `torch.optim` skips a parameter whose `.grad` is None outright, so a cleared parameter is not merely left un-updated, it is never visited.',
              'A grad of zeros is a different instruction. The parameter is not skipped, the update runs, and an update with momentum or weight decay in it does not need a gradient to move anything. Momentum keeps discharging the buffer it built; weight decay keeps pulling toward zero.',
              'The arithmetic is small enough to check by hand. SGD at lr 0.1, momentum 0.9, weight decay 0.1, one real step from 1.0 with grad 0.5 lands at 0.94 and leaves a buffer of 0.6. Step again on a zero grad and the effective gradient is `0 + 0.1 * 0.94`, the buffer becomes `0.9 * 0.6 + 0.094`, and the parameter lands at 0.8766. Under the default it would still read 0.94.',
              '>> A parameter with no gradient this step and a parameter with a zero gradient this step are different requests, and only one of them holds still.',
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the same parameter, the same optimizer, two ways of clearing',
              lang: 'python',
              text: 'import torch\n\n\ndef one():\n    p = torch.nn.Parameter(torch.ones(3))\n    opt = torch.optim.SGD([p], lr=0.1, momentum=0.9, weight_decay=0.1)\n    p.grad = torch.full((3,), 0.5)\n    opt.step()                       # one real step, so momentum exists\n    return p, opt\n\n\na, oa = one()\nb, ob = one()\nprint(a[0].item(), b[0].item())      # 0.9399999976158142 0.9399999976158142\n\noa.zero_grad()                       # the default: set_to_none=True\nob.zero_grad(set_to_none=False)\nprint(a.grad, b.grad)                # None tensor([0., 0., 0.])\n\noa.step()\nob.step()\nprint(a[0].item(), b[0].item())      # 0.9399999976158142 0.8766000270843506',
            },
          },
          {
            h: 'the optimizer half aliases too',
            ps: [
              'The modules arc establishes the model side of this: `model.state_dict()` hands back tensors sharing storage with the live parameters, so an in-memory snapshot moves when the model does. The optimizer side is a different mechanism, and it shows up one restore later than you would look for it.',
              '`Optimizer.load_state_dict` casts each saved tensor to the target parameter\'s device and dtype, and when those already match, the cast returns the same tensor. The restored optimizer then writes its updates into the dictionary you loaded. `Module.load_state_dict` copies through `param.copy_` and does not do this, so only one half of a checkpoint is consumed by restoring it.',
              'So a checkpoint dict held in memory serves exactly one restore. A `torch.load` per restore is the fix, or a deepcopy if the dict has to stay in memory. It costs a file read, and it is the difference between three comparable resume experiments and three that quietly share state.',
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the restored optimizer writes into the dict it was restored from',
              lang: 'python',
              text: 'import torch\nfrom torch import nn\n\ntorch.manual_seed(0)\nm = nn.Linear(4, 2)\nopt = torch.optim.AdamW(m.parameters(), lr=1e-2)\nm(torch.randn(3, 4)).sum().backward()\nopt.step()\n\nckpt = opt.state_dict()\nkept = ckpt["state"][0]["exp_avg"].clone()\n\nm2 = nn.Linear(4, 2)\nopt2 = torch.optim.AdamW(m2.parameters(), lr=1e-2)\nopt2.load_state_dict(ckpt)\nprint(opt2.state[m2.weight]["exp_avg"] is ckpt["state"][0]["exp_avg"])   # True\nm2(torch.randn(3, 4)).sum().backward()\nopt2.step()\nprint(torch.equal(kept, ckpt["state"][0]["exp_avg"]))   # False: the restore consumed it',
            },
          },
        ],
        readings: [
          { label: 'torch.optim reference', url: 'https://docs.pytorch.org/docs/stable/optim.html', note: 'param groups, per-parameter options, and the base class every optimizer shares' },
          { label: 'optimizer.py at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/optim/optimizer.py', note: 'state_dict at 530, load_state_dict at 720, zero_grad at 789; all three are short' },
          { label: 'Saving and loading a general checkpoint', url: 'https://docs.pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html', note: 'the official recipe, which saves the optimizer state and stops there' },
        ],
        check: [
          {
            q: 'You replaced a layer after building the optimizer. What happens on the next step?',
            a: 'The new weight gets a gradient, because it is in the graph, and never moves, because the optimizer drained model.parameters() into a flat list of tensor references at construction and still holds the tensor you replaced. Nothing raises.',
          },
          {
            q: 'What does load_state_dict check before it accepts an optimizer checkpoint?',
            a: 'The number of parameter groups, and the number of parameters in each group. Nothing else. Shapes are never compared and names do not exist at this layer, so a state dict from a differently sized model loads cleanly and fails one step later inside exp_avg.lerp_.',
          },
          {
            q: 'Why does zero_grad(set_to_none=False) move a parameter that the default leaves alone?',
            a: 'Because a parameter whose grad is None is skipped by step entirely, while a grad of zeros still runs the update, and momentum and weight decay do not need a gradient to move anything. With SGD at lr 0.1, momentum 0.9 and weight decay 0.1, one such step turns 0.94 into 0.8766.',
          },
        ],
        work: [
          { id: 'two-containers', label: 'print param_groups and state for one optimizer of your own, before the first step and after it, and name every key you find', href: '#one-group-ten-hyperparameters' },
          { id: 'positional-load', label: 'load an optimizer state dict into a model with the same parameter count and different shapes, then find the exact line where it finally fails', href: '#what-load-state-dict-does-not-check' },
        ],
      },
      {
        id: 'between-backward-and-step',
        num: 2,
        title: 'Between backward and step',
        lede: 'Two lines commonly sit between backward and step, and each one makes an arithmetic claim about a batch that was never actually formed. Get either wrong and nothing raises. The loss curve just goes somewhere else.',
        goal: 'Given an accumulation schedule and a clipping call, say which divisor makes the accumulated gradient equal the full-batch gradient, where the clip has to sit and why, and what one non-finite entry does to every other gradient in the model.',
        sections: [
          {
            h: 'accumulation is an arithmetic claim',
            ps: [
              'Gradient accumulation is usually introduced as a way around a memory limit, and the mechanical part is trivial: skip `zero_grad`, let `.grad` sum across several backward passes, step once at the end. The claim underneath is the interesting half, because you are asserting that the sum you built equals the gradient of a batch you never had room for.',
              'It does, on one condition. `mse_loss` and `cross_entropy` average over their own batch by default, so four micro-batches of 16 give you four averages over 16 rows, and summing those is four times the average over 64 rows. Divide each micro-batch loss by four before backward and the identity holds.',
              'Run both and the difference is not subtle. With the divisor, the accumulated gradient and the full-batch gradient agree to 2.4e-07, which is float32 rounding on a sum of 64 terms. Without it, the largest element is off by 4.8, and the gradient norms sit at a ratio of exactly 4.0.',
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): four micro-batches against one batch of 64, with the divisor and without',
              lang: 'python',
              text: 'import torch\nfrom torch import nn\n\n\ndef fresh():\n    torch.manual_seed(0)\n    return nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1))\n\n\ndef grads(m):\n    return [p.grad.detach().clone() for p in m.parameters()]\n\n\ndef gap(a, b):\n    return max((u - v).abs().max().item() for u, v in zip(a, b))\n\n\ntorch.manual_seed(1)\nx = torch.randn(64, 8)\ny = x.sum(dim=1, keepdim=True)\n\nm = fresh()\nnn.functional.mse_loss(m(x), y).backward()\nfull = grads(m)                                  # one batch of 64\n\nm = fresh()\nfor i in range(4):\n    s = slice(i * 16, (i + 1) * 16)\n    (nn.functional.mse_loss(m(x[s]), y[s]) / 4).backward()\nprint(gap(full, grads(m)))                       # 2.384185791015625e-07\n\nm = fresh()\nfor i in range(4):\n    s = slice(i * 16, (i + 1) * 16)\n    nn.functional.mse_loss(m(x[s]), y[s]).backward()\nprint(gap(full, grads(m)))                       # 4.807855606079102',
            },
          },
          {
            h: 'the tail batch breaks the divisor',
            ps: [
              'Dividing by the number of micro-batches is a shortcut that happens to be correct when every micro-batch is the same size. Datasets are rarely so obliging, and the last group in an epoch is the one that catches people.',
              'Split 64 rows as 24, 24 and 16 and divide each loss by three, and the result misses the full-batch gradient by 0.202 in its largest element. Weight each micro-batch by its own sample count over the total instead, 24/64 and 24/64 and 16/64, and the gap drops back to 2.4e-07.',
              'The rule that survives every batch shape: the weight on a micro-batch is its share of the samples, not its share of the micro-batches. A loop that hardcodes `loss / accum_steps` is asserting the two are the same thing.',
            ],
            code: {
              caption: 'continuing the block above (verified, torch 2.2.2 CPU): a 24/24/16 split, weighted two ways',
              lang: 'python',
              text: 'def accumulate(weights):\n    m = fresh()\n    off = 0\n    for n, w in zip([24, 24, 16], weights):\n        s = slice(off, off + n)\n        off += n\n        (nn.functional.mse_loss(m(x[s]), y[s]) * w).backward()\n    return grads(m)\n\n\nprint(gap(full, accumulate([1 / 3, 1 / 3, 1 / 3])))          # 0.20217061042785645\nprint(gap(full, accumulate([24 / 64, 24 / 64, 16 / 64])))    # 2.384185791015625e-07',
            },
          },
          {
            h: 'what accumulation does not reproduce',
            ps: [
              'The identity covers gradients and stops there. Anything in the model that computes a statistic over the rows in front of it sees micro-batches, because micro-batches are what it was given.',
              'Batch normalization is the clear case. One `BatchNorm1d` fed 64 rows at once and another fed the same 64 rows in four passes end up with running means that differ by 0.049, and their `num_batches_tracked` counters read 1 and 4. Nothing about that is a bug in accumulation. The layer normalized over 16 rows because 16 rows is what arrived.',
              'There is a distributed consequence too, and its mechanism is chapter eight\'s. Bucketed all-reduce fires as soon as a bucket\'s gradients are ready, so a plain accumulation loop under DDP pays a reduction per micro-batch rather than one per optimizer step. `DistributedDataParallel.no_sync` is the context manager that suppresses it for every micro-batch but the last.',
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the same 64 rows, one batch against four',
              lang: 'python',
              text: 'import torch\nfrom torch import nn\n\ntorch.manual_seed(1)\nx = torch.randn(64, 8)\n\none = nn.BatchNorm1d(8)\none(x)                                        # a single batch of 64\n\nfour = nn.BatchNorm1d(8)\nfor i in range(4):\n    four(x[i * 16 : (i + 1) * 16])            # the same 64 rows, four at a time\n\nprint((one.running_mean - four.running_mean).abs().max().item())\n# 0.048792045563459396\nprint(int(one.num_batches_tracked), int(four.num_batches_tracked))   # 1 4',
            },
          },
          {
            h: 'one norm over every parameter',
            ps: [
              '`clip_grad_norm_` does not clip tensors. It concatenates every gradient you hand it into one conceptual vector, takes that vector\'s norm, and if the norm exceeds your threshold it multiplies every gradient by the same scalar. Direction is preserved exactly; only magnitude moves.',
              'Two details of the interface are easy to miss. The value it returns is the norm before clipping, not after, which makes it the cheapest gradient-norm logger available and a confusing one if you assume otherwise. And it returns that value whether or not it clipped anything.',
              'The per-tensor norms afterward are the tell that this is a global operation. Clip a model whose total norm is 206.35 down to 1.0 and the four parameters come back at 0.380, 0.087, 0.921 and 0.010: nowhere near 1.0 individually, exactly 1.0 together.',
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the returned value is the norm before the clip',
              lang: 'python',
              text: 'import torch\nfrom torch import nn\n\n\ndef fresh():\n    torch.manual_seed(0)\n    return nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1))\n\n\ndef norm(m):\n    return torch.cat([p.grad.flatten() for p in m.parameters()]).norm().item()\n\n\ntorch.manual_seed(1)\nx = torch.randn(64, 8)\ny = x.sum(dim=1, keepdim=True) * 50.0            # far enough away to make big grads\n\nm = fresh()\nnn.functional.mse_loss(m(x), y).backward()\nprint(norm(m))                                                        # 206.35205078125\nprint(nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0).item())  # 206.35205078125\nprint(norm(m))                                                        # 1.0\nprint([round(p.grad.norm().item(), 6) for p in m.parameters()])\n# [0.380417, 0.086949, 0.920664, 0.010048]',
            },
          },
          {
            h: 'the clip goes after the last micro-batch',
            ps: [
              'Put those two sections together and the placement question answers itself. Clipping rescales a vector, and the vector you meant to clip is the accumulated gradient. Clipping each partial sum rescales four different vectors, and their sum is not the clipped total.',
              'The measurement makes the trap visible. Clip every micro-batch and the final norm reads 1.0. Clip once at the end and it also reads 1.0. The two gradient sets differ by 0.245 in their largest element, so the norm you were watching cannot tell you which loop you wrote.',
              'Mixed precision adds one more ordering constraint on top. Chapter nine covers the scaler itself; what it imposes here is that gradients have to be unscaled before they are clipped, because a norm measured on scaled gradients is not the quantity your threshold was chosen against.',
            ],
            code: {
              caption: 'continuing the block above (verified, torch 2.2.2 CPU): same threshold, same final norm, different gradients',
              lang: 'python',
              text: 'def accumulate(clip_each):\n    m = fresh()\n    for i in range(4):\n        s = slice(i * 16, (i + 1) * 16)\n        (nn.functional.mse_loss(m(x[s]), y[s]) / 4).backward()\n        if clip_each:\n            nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)\n    if not clip_each:\n        nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)\n    return [p.grad.clone() for p in m.parameters()]\n\n\nper_micro = accumulate(True)\nonce = accumulate(False)\nprint(torch.cat([g.flatten() for g in per_micro]).norm().item())   # 1.0\nprint(torch.cat([g.flatten() for g in once]).norm().item())        # 1.0\nprint(max((a - b).abs().max().item() for a, b in zip(per_micro, once)))\n# 0.24483658373355865',
            },
          },
          {
            h: 'an infinity does not stop it',
            ps: [
              'One `inf` in one gradient entry, out of 321 in this small model, and the whole clip goes quiet in a specific way. The total norm comes back `inf`, the coefficient is computed as `max_norm / (total_norm + 1e-6)`, which is 0.0, and `torch.clamp(clip_coef, max=1.0)` leaves it there.',
              'Multiplying by zero then produces 320 zeros and one `nan`, because `inf * 0` is `nan`. Your step runs on a model whose entire gradient is zero except for one poisoned entry, and the only signal is the return value you probably discarded.',
              '`error_if_nonfinite=True` turns this into a `RuntimeError` naming the norm order and telling you the flag exists. It defaults to False on torch 2.2.2, and the current reference still documents the default as False, so the loud version is the one you have to ask for.',
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): one inf entry, and what the other 320 become',
              lang: 'python',
              text: 'import torch\nfrom torch import nn\n\ntorch.manual_seed(0)\nm = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1))\ntorch.manual_seed(1)\nx = torch.randn(64, 8)\nnn.functional.mse_loss(m(x), x.sum(dim=1, keepdim=True) * 50.0).backward()\n\nm[0].weight.grad[0, 0] = float("inf")            # one entry out of 321\nprint(nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0).item())   # inf\n\nflat = torch.cat([p.grad.flatten() for p in m.parameters()])\nprint(flat.numel(), int((flat == 0).sum()), int(flat.isnan().sum()))   # 321 320 1',
            },
          },
        ],
        readings: [
          { label: 'clip_grad_norm_ reference', url: 'https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html', note: 'the return value and the error_if_nonfinite default, stated in two lines' },
          { label: 'clip_grad.py at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/nn/utils/clip_grad.py', note: 'the whole clip at 61-82: one norm, one coefficient, one clamp, one multiply' },
          { label: 'CUDA automatic mixed precision examples', url: 'https://docs.pytorch.org/docs/stable/notes/amp_examples.html', note: 'the accumulation and clipping recipes, with the unscale ordering spelled out' },
        ],
        check: [
          {
            q: 'Four micro-batches of 16 replace one batch of 64. What has to happen to each micro-batch loss?',
            a: 'Each one is divided by four before backward, because a mean loss over 16 rows summed four times is four times the mean over 64. With the divisor the two gradients agree to 2.4e-07 in float32; without it the norms sit at a ratio of exactly 4.0.',
          },
          {
            q: 'Your accumulation group splits 64 rows as 24, 24 and 16. What does dividing each loss by three give you?',
            a: 'A gradient weighted per micro-batch rather than per sample, off the full-batch gradient by 0.202 in its largest element. Weighting each loss by its own sample count over the total, 24/64 and 24/64 and 16/64, brings it back to 2.4e-07.',
          },
          {
            q: 'Clipping ran on every micro-batch and the final norm still reads exactly 1.0. What is wrong?',
            a: 'The direction of the gradient. Clipping rescales one vector, so rescaling four partial sums separately gives a total that is not the clipped full-batch gradient. On this run the two disagree by 0.245 in the largest element while both read norm 1.0.',
          },
        ],
        work: [
          { id: 'accumulation-equality', label: 'prove the accumulation identity on a model of your own: four micro-batches with the divisor against one full batch, and the max element difference between the two gradients', href: '#accumulation-is-an-arithmetic-claim' },
          { id: 'clip-audit', label: 'take one loop that clips and write down where the clip sits relative to the last backward, the unscale, and the step', href: '#the-clip-goes-after-the-last-micro-batch' },
        ],
      },
      {
        id: 'the-resume-that-matches',
        num: 3,
        title: 'The resume that matches',
        lede: 'The chapter above sets the rule: a checkpoint is the whole state or it is not a checkpoint. This lesson weighs the whole state piece by piece, and measures what each missing piece costs over the first six steps after a restore.',
        goal: 'Given a loop with a scheduler and any source of randomness, list every piece a checkpoint has to carry, name the one piece that has no state dict at all, and prove a resume matches the uninterrupted run exactly rather than approximately.',
        sections: [
          {
            h: 'the scheduler writes one field',
            ps: [
              'Constructing a scheduler is not a passive act. Before any step runs, it walks the optimizer\'s param groups and writes an `initial_lr` key into each one, then copies those values into its own `base_lrs`. Every learning rate it ever produces is computed from `base_lrs` and a counter, not from whatever `lr` currently reads.',
              'That is why a scheduler needs a state dict of its own. Its whole memory is `last_epoch` and `base_lrs` plus its own hyperparameters, eight keys for a cosine schedule, and none of that lives in the optimizer.',
              'It also explains a class of confusion around resuming. Set `lr` by hand on a param group, then let a scheduler step, and your value is gone: the scheduler recomputed the rate from `base_lrs` and overwrote it.',
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): what a scheduler writes at construction, and what it remembers',
              lang: 'python',
              text: 'import torch\nfrom torch.optim.lr_scheduler import CosineAnnealingLR\n\np = torch.nn.Parameter(torch.zeros(1))\nopt = torch.optim.SGD([p], lr=0.1)\nprint("initial_lr" in opt.param_groups[0])                            # False\n\nsched = CosineAnnealingLR(opt, T_max=10)\nprint("initial_lr" in opt.param_groups[0], opt.param_groups[0]["initial_lr"])\n# True 0.1\nprint(sched.state_dict())\n# {\'T_max\': 10, \'eta_min\': 0, \'base_lrs\': [0.1], \'last_epoch\': 0,\n#  \'verbose\': False, \'_step_count\': 1, \'_get_lr_called_within_step\': False,\n#  \'_last_lr\': [0.1]}',
            },
          },
          {
            h: 'two schedules, one optimizer',
            ps: [
              'Warmup then decay is two schedules on one optimizer, and `SequentialLR` is the object that hands over between them. Give it the schedulers, give it the step index where the second takes over, and step it once per optimizer step.',
              'The ten learning rates a three-step linear warmup and a seven-step cosine actually produce are worth reading as a sequence rather than a formula. The warmup climbs 0.01, 0.04, 0.07 and hits the base rate of 0.1 on step four. The cosine then descends 0.095, 0.081, 0.061, 0.039, 0.019, 0.005.',
              '`SequentialLR` keeps its own `last_epoch` and a nested list of its children\'s state dicts, so restoring it restores the hand-over point too. Composition here means one object that owns the others, not two objects racing to write the same field.',
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): a three-step warmup into a seven-step cosine',
              lang: 'python',
              text: 'import torch\nfrom torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR\n\np = torch.nn.Parameter(torch.zeros(1))\nopt = torch.optim.SGD([p], lr=0.1)\nsched = SequentialLR(\n    opt,\n    schedulers=[LinearLR(opt, start_factor=0.1, total_iters=3), CosineAnnealingLR(opt, T_max=7)],\n    milestones=[3],\n)\n\nseen = []\nfor _ in range(10):\n    opt.step()\n    seen.append(round(opt.param_groups[0]["lr"], 6))\n    sched.step()\nprint(seen)\n# [0.01, 0.04, 0.07, 0.1, 0.095048, 0.081174, 0.061126, 0.038874, 0.018826,\n#  0.004952]\nprint(sorted(sched.state_dict()), sched.state_dict()["last_epoch"])\n# [\'_last_lr\', \'_milestones\', \'_schedulers\', \'last_epoch\'] 10',
            },
          },
          {
            h: 'the milestone hands over with a reset',
            ps: [
              'The hand-over is six lines, and the interesting one is the branch. On the step where a milestone is reached, the incoming scheduler is stepped with an explicit epoch of 0, which restarts it at its own beginning rather than continuing from wherever its counter happened to be.',
              'That explicit epoch argument is deprecated in torch 2.2.2, so a composed schedule emits the deprecation warning from inside the library, once per hand-over, on a line you never wrote. Current torch calls a private `_update_lr(0)` in the same branch instead, which does the same reset without the warning.',
              'The reason to know this is diagnostic. A warning that names `scheduler.step()` and the epoch parameter, fired by a program that never passes one, is `SequentialLR` reaching a milestone, not a bug in your loop.',
            ],
            code: {
              caption: 'verbatim, torch/optim/lr_scheduler.py:706-715 at the v2.2.2 tag; current torch replaces the scheduler.step(0) line with scheduler._update_lr(0)',
              lang: 'python',
              text: '    def step(self):\n        self.last_epoch += 1\n        idx = bisect_right(self._milestones, self.last_epoch)\n        scheduler = self._schedulers[idx]\n        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:\n            scheduler.step(0)\n        else:\n            scheduler.step()\n\n        self._last_lr = scheduler.get_last_lr()',
            },
          },
          {
            h: 'five thousand and fifty-six bytes',
            ps: [
              'The chapter\'s rule names the RNG as part of the state when bitwise resume matters. Here is what it weighs: `torch.get_rng_state()` returns a uint8 tensor of 5056 elements, the CPU generator\'s Mersenne Twister block, and it goes into the checkpoint dict like anything else.',
              'The proof is worth running rather than reading. Twelve steps with dropout in the model and a `randint` drawing the batch, checkpointed at step six, restored into a fresh model, optimizer and scheduler, with the RNG set back. The six resumed losses equal the last six of the run that never stopped, as a list comparison, not as a curve that looks about right.',
              'Equality rather than similarity is the bar because a bitwise resume is a claim you can check in one line. A curve that overlays plausibly is compatible with a scheduler that reset, a dropout mask that shifted, and an optimizer moment that came back empty.',
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): twelve steps, killed at six, resumed to the same numbers',
              lang: 'python',
              text: 'import torch\nfrom torch import nn\nfrom torch.optim.lr_scheduler import CosineAnnealingLR\n\n\ndef build():\n    torch.manual_seed(0)\n    model = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))\n    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)\n    return model, opt, CosineAnnealingLR(opt, T_max=12)\n\n\ndef step(model, opt, sched):\n    idx = torch.randint(0, 256, (32,))              # this line reads the global RNG\n    loss = nn.functional.mse_loss(model(data[idx]), target[idx])\n    opt.zero_grad()\n    loss.backward()\n    opt.step()\n    sched.step()\n    return round(loss.item(), 6)\n\n\ntorch.manual_seed(1)\ndata = torch.randn(256, 8)\ntarget = data.sum(dim=1, keepdim=True)\n\nmodel, opt, sched = build()                         # the run that never stops\ntorch.manual_seed(7)\nstraight = [step(model, opt, sched) for _ in range(12)]\n\nmodel, opt, sched = build()                         # the same run, killed at step 6\ntorch.manual_seed(7)\nfor _ in range(6):\n    step(model, opt, sched)\ntorch.save(\n    {\n        "model": model.state_dict(),\n        "opt": opt.state_dict(),\n        "sched": sched.state_dict(),\n        "step": 6,\n        "rng": torch.get_rng_state(),\n    },\n    "ckpt.pt",\n)\n\nstate = torch.load("ckpt.pt")\nmodel, opt, sched = build()\nmodel.load_state_dict(state["model"])\nopt.load_state_dict(state["opt"])\nsched.load_state_dict(state["sched"])\ntorch.set_rng_state(state["rng"])\nprint([step(model, opt, sched) for _ in range(6)] == straight[6:])   # True\nprint(state["rng"].dtype, state["rng"].shape)   # torch.uint8 torch.Size([5056])',
            },
          },
          {
            h: 'what each omission costs',
            ps: [
              'Drop one piece at a time from that restore and the damage has a shape. Leave the RNG out and the very first resumed step is already wrong, 8.478 against 7.358, because a different batch got drawn and a different dropout mask fired.',
              'Leave the scheduler out and the first two steps match exactly, then the third drifts. The optimizer\'s own state dict carries the current `lr`, so step seven runs at the correct 0.005 by accident. From step eight the fresh scheduler walks the cosine from its beginning: 0.004915 where the real run had 0.003706, and by step twelve 0.003147 against 0.00017, more than eighteen times the intended rate.',
              'That delay is the part to remember. A resume bug that is invisible for two steps and then bends the curve is a bug people attribute to data, to hardware, to anything except the two lines that restored the wrong object.',
            ],
            code: {
              caption: 'continuing the block above (verified, torch 2.2.2 CPU): the whole state, then the same restore minus one piece',
              lang: 'python',
              text: 'def resume(restore_rng, restore_sched):\n    state = torch.load("ckpt.pt")        # reloaded each time, for lesson one\'s reason\n    model, opt, sched = build()\n    model.load_state_dict(state["model"])\n    opt.load_state_dict(state["opt"])\n    if restore_sched:\n        sched.load_state_dict(state["sched"])\n    if restore_rng:\n        torch.set_rng_state(state["rng"])\n    return [step(model, opt, sched) for _ in range(6)]\n\n\nprint(straight[6:])\nprint(resume(True, True))\n# [7.357611, 6.734547, 9.127593, 8.237515, 8.615776, 5.698226]  on both lines\nprint(resume(False, True))\n# [8.477634, 4.781869, 6.414695, 6.638178, 9.703715, 5.7183]\nprint(resume(True, False))\n# [7.357611, 6.734547, 9.101551, 8.163261, 8.458454, 5.567288]',
            },
            table: {
              caption: 'the learning rate on resumed steps 7 through 12, with the scheduler state and without it (verified, torch 2.2.2 CPU)',
              cols: ['resumed step', 'scheduler restored', 'scheduler left out'],
              rows: [
                ['7', '0.005', '0.005'],
                ['8', '0.003706', '0.004915'],
                ['9', '0.0025', '0.004665'],
                ['10', '0.001464', '0.004268'],
                ['11', '0.00067', '0.00375'],
                ['12', '0.00017', '0.003147'],
              ],
            },
          },
          {
            h: 'the loader keeps no position',
            ps: [
              'Every piece so far has a state dict. The input pipeline does not, and chapter five\'s shuffling is the reason it matters here.',
              'A `DataLoader` with `shuffle=True` draws its permutation from a generator, and you can pass your own. Save that generator\'s state at the end of an epoch and the next epoch\'s batch order is reproducible: the same four batches, in the same order, from a state you saved yourself.',
              'Stopping mid-epoch is the case with no answer. Nothing records how many batches were consumed, so a resume from the same generator state gets a fresh permutation of the whole dataset rather than the batches you had left. Two batches into a four-batch epoch, the resume hands you all twelve samples again in a new order, and some rows get seen twice this epoch while others wait.',
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): an epoch order restored, and a mid-epoch position that cannot be',
              lang: 'python',
              text: 'import torch\nfrom torch.utils.data import DataLoader, TensorDataset\n\nds = TensorDataset(torch.arange(12.).unsqueeze(1))\n\n\ndef epoch(rng_state, stop_after):\n    g = torch.Generator()\n    g.set_state(rng_state)\n    out = []\n    for i, (b,) in enumerate(DataLoader(ds, batch_size=3, shuffle=True, generator=g)):\n        if i == stop_after:\n            break\n        out.append([int(v) for v in b.flatten()])\n    return out, g.get_state()\n\n\nstart = torch.Generator().manual_seed(0).get_state()\nfirst, after_first = epoch(start, 4)\nsecond, _ = epoch(after_first, 4)\nprint(first)    # [[5, 4, 0], [9, 7, 8], [1, 10, 3], [11, 6, 2]]\nprint(second)   # [[1, 11, 0], [3, 5, 4], [2, 7, 6], [9, 10, 8]]\nprint(epoch(after_first, 4)[0] == second)   # True: the order is recoverable\n\nhalf, mid = epoch(start, 2)     # killed two batches into epoch one\nprint(half)                     # [[5, 4, 0], [9, 7, 8]]\nprint(epoch(mid, 4)[0])         # [[8, 6, 10], [0, 5, 11], [4, 3, 2], [1, 9, 7]]',
            },
          },
          {
            h: 'the checkpoint, itemized',
            ps: [
              'Five things, four of which serialize and one of which does not. The table is the whole answer to what a checkpoint has to contain, and the last column is what this lesson measured rather than what it assumed.',
              'One version difference belongs in the same breath, because it changes how the file comes back. On torch 2.2.2 `torch.load` defaults to `weights_only=False`; current torch defaults it to True, documented as restricting the unpickler to tensors, primitive types, dictionaries and anything you explicitly allow. A checkpoint shaped like the one above is tensors, ints and dicts, and it loads under `weights_only=True` on this machine when the flag is passed by hand. A checkpoint that pickled an optimizer object or an argparse namespace does not.',
              'The TPU version of this exercise, kill the run, resume, and overlay the two curves, is the capstone in LAB·P4 and chapter twelve\'s mastery bar. What changes over there is the device and the bridge. What does not change is the list below.',
            ],
            table: {
              caption: 'what a resumable step carries, and what each omission cost on this run (verified, torch 2.2.2 CPU)',
              cols: ['piece', 'where it lives', 'what a resume without it costs'],
              rows: [
                ['model weights', 'model.state_dict(), keyed by parameter name', 'everything, and it fails loudly, which is why nobody forgets it'],
                ['optimizer state', 'opt.state_dict(), keyed by position', "Adam's moments come back empty, and the current lr is silently restored with it"],
                ['scheduler', 'sched.state_dict(), a plain dict of its own fields', 'two correct steps, then a schedule walked from the beginning'],
                ['RNG', 'torch.get_rng_state(), 5056 uint8 bytes', 'the first resumed step already differs: 8.478 against 7.358'],
                ['loader position', 'nowhere', 'a fresh permutation of the whole dataset instead of the tail of an epoch'],
              ],
            },
          },
        ],
        readings: [
          { label: 'Reproducibility', url: 'https://docs.pytorch.org/docs/stable/notes/randomness.html', note: 'every generator a training step can touch, and which of them torch.manual_seed reaches' },
          { label: 'lr_scheduler.py at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/torch/optim/lr_scheduler.py', note: 'SequentialLR.step at 706; the milestone hand-over is the six lines quoted above' },
          { label: 'torch.load reference', url: 'https://docs.pytorch.org/docs/stable/generated/torch.load.html', note: 'the weights_only default, which is not the one torch 2.2 shipped' },
        ],
        check: [
          {
            q: 'What does constructing a scheduler do to the optimizer, before any step runs?',
            a: 'It writes an initial_lr key into every param group and copies those values into its own base_lrs. Every later rate is computed from base_lrs and last_epoch rather than from the current lr, which is why the scheduler needs a state dict of its own to resume.',
          },
          {
            q: 'A resume restored the model and optimizer but not the scheduler, and the first two steps matched exactly. Why does the third not?',
            a: 'Because the optimizer state dict carries the current lr, so those steps are right by accident. The fresh scheduler has last_epoch at zero and walks the cosine from its beginning: 0.004915 where the real run had 0.003706, and 0.003147 against 0.00017 by step twelve.',
          },
          {
            q: 'Which piece of a training loop has no state dict to save at all?',
            a: 'The data loader. Its shuffling generator can be saved and restored by hand, which reproduces an epoch order exactly, but nothing records how far into an epoch the run got, so a mid-epoch resume starts a fresh permutation of the whole dataset.',
          },
        ],
        work: [
          { id: 'digit-for-digit', label: 'run your own loop twelve steps, checkpoint at six, and prove the resumed steps equal the uninterrupted ones exactly rather than approximately', href: '#five-thousand-and-fifty-six-bytes' },
          { id: 'omission-cost', label: 'drop one piece of the checkpoint at a time and write down how many steps pass before the loss sequence diverges', href: '#what-each-omission-costs' },
        ],
      },
    ],
  },
]
