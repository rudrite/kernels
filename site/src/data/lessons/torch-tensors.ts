// New file: site/src/data/lessons/torch-tensors.ts
// Storage and strides below the survey chapter 1 teaches: the TensorImpl and
// StorageImpl split, the address arithmetic, the chunk rule view runs before it
// refuses, and memory formats. Every printed value ran on this machine (torch
// 2.2.2 CPU, Python 3.12, 2026-08-14). Every C++ excerpt is verbatim from a
// header the torch 2.2.2 wheel ships; all seven are byte-identical to the
// v2.2.2 tag, so the quoted line numbers hold against the public tree. Shapes,
// strides and contiguity in the tables are quoted from the stride corpus, not
// recomputed here.
import type { UnitLessons } from './index'

export const TORCH_TENSOR_LESSONS: UnitLessons[] = [
  {
    unit: 'pt:tensors',
    lessons: [
      {
        id: 'two-objects-one-tensor',
        num: 1,
        title: 'Two objects, one tensor',
        lede: 'Nothing you can reach from Python holds both the numbers and the map. Two C++ objects split the job, one owning bytes and knowing nothing else, the other owning everything except the bytes.',
        goal: 'Say which of the two objects each piece of tensor state lives on, compute a view’s address from its storage offset and element size, and explain what a shared storage decides about aliasing and about mutation tracking.',
        sections: [
          {
            h: 'the object that owns bytes owns nothing else',
            ps: [
              "Ask a tensor for its storage and what comes back has seven fields and no opinions. `StorageImpl` holds a data pointer, a size in bytes, three booleans, an allocator, and a slot for the Python object wrapping it. No shape. No stride. No dtype. A storage is a run of bytes with a length and a way to free itself.",
              "The comment above the class states the invariant the rest of the library leans on: two non-null data pointers alias if and only if they came from the same storage. Aliasing is decidable because of that sentence. Instead of comparing address ranges operation by operation, torch compares storage identity.",
              "The same comment names where mutation tracking lives, and it is not on the tensor. Version counting happens at the level of storages, which is why an in-place write through any view is visible to every other view of the same bytes. The museum's version-counter exhibit is where that turns into an error message; the fact to carry out of here is only which object owns the counter.",
            ],
            code: {
              caption: 'verbatim, c10/core/StorageImpl.h at v2.2.2: the class comment at 11-21 and 31-33, then the field list at 214-224, joined here under added path headings',
              lang: 'c',
              text: '// c10/core/StorageImpl.h:11-21\n// A storage represents the underlying backing data buffer for a\n// tensor.  This concept was inherited from the original Torch7\n// codebase; we\'d kind of like to get rid of the concept\n// (see https://github.com/pytorch/pytorch/issues/14797) but\n// it\'s hard work and no one has gotten around to doing it.\n//\n// NB: storage is supposed to uniquely own a data pointer; e.g.,\n// two non-null data pointers alias if and only if they are from\n// the same storage.  Technically you can violate this invariant\n// (e.g., you can create a non-owning StorageImpl with at::from_blob)\n// but a lot of things won\'t work correctly, including:\n\n// c10/core/StorageImpl.h:31-33, the third of those consequences\n// - Version counts won\'t work correctly, because we do all VC tracking at the\n//   level of storages (unless you explicitly disconnect the VC with detach);\n//   mutation because data pointers are the same are totally untracked\n\n// c10/core/StorageImpl.h:214-224, every field the class has\n private:\n  DataPtr data_ptr_;\n  SymInt size_bytes_;\n  bool size_bytes_is_heap_allocated_;\n  bool resizable_;\n  // Identifies that Storage was received from another process and doesn\'t have\n  // local to process cuda memory allocation\n  bool received_cuda_;\n  Allocator* allocator_;\n  impl::PyObjectSlot pyobj_slot_;\n};',
            },
          },
          {
            h: 'everything else rides on the other object',
            ps: [
              "`TensorImpl` is what a Python `Tensor` actually points at, and it owns one `Storage` by value. Everything the storage refused to know sits here: a packed sizes-and-strides container, an integer storage offset, a cached element count, a `TypeMeta` for the dtype, an optional device, and a dispatch key set that decides which kernel table a call lands in.",
              "Two of those are caches rather than truth, and knowing that changes how you read a profile. `numel_` is stored, not derived, so asking a tensor how many elements it has costs a load and not a product over the shape. The contiguity flags work the same way, and they get the last lesson of this arc to themselves.",
              "One line in the header is worth reading twice. Above `data_type_` the comment says the type meta must agree with the type meta in storage. Go back to the field list on `StorageImpl` and there is no type meta on it at all, in this version. The invariant outlived the field it was written about, which is a normal thing to find in a header this old and a good reason to check the fields rather than the comments.",
              "In Python the typed view of a storage is on its way out too. Calling `.storage()` on torch 2.2.2 warns that `TypedStorage is deprecated` and points at `untyped_storage()`, which reports bytes and nothing else. Every storage number in this arc comes from that call.",
            ],
            code: {
              caption: 'verbatim, c10/core/TensorImpl.h at v2.2.2: the storage field at 2803-2804 and the metadata fields at 2839-2850, joined here under added path headings',
              lang: 'c',
              text: '// c10/core/TensorImpl.h:2803-2804\n protected:\n  Storage storage_;\n\n// c10/core/TensorImpl.h:2839-2850\n  c10::impl::SizesAndStrides sizes_and_strides_;\n\n  int64_t storage_offset_ = 0;\n  // If sizes and strides are empty, the numel is 1!!  However, most of the\n  // time, we will immediately set sizes to {0} and reset numel to 0.\n  // (Can\'t do that in the default initializers, because there\'s no way to\n  // spell "allocate a one-element array" for strides_).\n  int64_t numel_ = 1;\n\n  // INVARIANT: When storage is non-null, this type meta must\n  // agree with the type meta in storage\n  caffe2::TypeMeta data_type_;',
            },
            table: {
              caption: 'where each piece of tensor state lives, from the field lists above; line numbers are c10/core/ at v2.2.2',
              cols: ['state', 'object', 'what it decides'],
              rows: [
                ['data_ptr_, allocator_', 'StorageImpl:215, 222', 'the bytes, and who frees them'],
                ['size_bytes_', 'StorageImpl:216', 'how far the allocation runs, in bytes, not elements'],
                ['sizes_and_strides_', 'TensorImpl:2839', 'the shape and the map that reads it'],
                ['storage_offset_', 'TensorImpl:2841', 'where in the storage this tensor starts, in elements'],
                ['numel_', 'TensorImpl:2846', 'the element count, cached rather than derived'],
                ['data_type_, device_opt_, key_set_', 'TensorImpl:2850, 2864, 2987', 'how to read a byte, where it is, and which kernel table answers'],
                ['version_counter_', 'TensorImpl:2835, tracked per storage', 'whether a saved value has been overwritten since'],
              ],
            },
          },
          {
            h: 'offset times element size is the whole connection',
            ps: [
              "Slice three corners off a tensor and the arithmetic connecting the two objects fits in one line. The address of a tensor's first element is the storage's data pointer plus the storage offset times the element size, and the slice below sits 17 elements into a 24-element storage, which on float32 is 68 bytes along.",
              "The second printed line is the part people find surprising. The tensor reports 6 elements while its storage still reports 24, because slicing narrowed the map and freed nothing. A view keeps the entire allocation alive, so a small crop of a large tensor holds the large tensor's memory until both go away.",
              '>> A view narrows the map. It never narrows the allocation.',
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): one slice, and the four numbers that place it in storage',
              lang: 'python',
              text: 'import torch\n\nbase = torch.arange(24.)\nx = base.reshape(2, 3, 4)\ns = x[1:, 1:, 1:]\n\nprint(s.shape, s.stride(), s.storage_offset())\nprint(s.numel(), s.untyped_storage().nbytes() // s.element_size())\nprint(s.data_ptr() - x.data_ptr())\nprint(x.untyped_storage().data_ptr() == s.untyped_storage().data_ptr())\n\n# torch.Size([1, 2, 3]) (12, 4, 1) 17\n# 6 24\n# 68\n# True',
            },
          },
          {
            h: 'the metadata has a byte budget',
            ps: [
              "Five is not an arbitrary number anywhere in this code. `C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE` is 5, so a tensor of five axes or fewer carries its entire shape and stride tuple inside the `TensorImpl` object. Add a sixth axis and the container allocates out of line and keeps a pointer instead.",
              "A comment near the bottom of the header does the arithmetic that explains why anyone cared. With 400 million live tensors in a training run, every 64-bit word added to a tensor costs another 3.2 gigabytes of RAM, and the comment records that they once ran out of memory at 160 bytes per `TensorImpl`.",
              "The budget is enforced, not hoped for. A dummy class exists only to run `static_assert`s across the field sizes, and on a 64-bit build the whole object has to fit in 26 words, with the packed sizes and strides taking exactly 88 of those 208 bytes. Metadata is a memory cost, and somebody is watching it at compile time.",
            ],
            code: {
              caption: 'verbatim, c10/core/TensorImpl.h at v2.2.2: the budget comment at 3010-3021 and four lines of the 64-bit size check at 3184-3203, the nine other field assertions trimmed',
              lang: 'c',
              text: '// Struct size matters.  In some production systems at Facebook, we have\n// 400M live tensors during a training run.  Do the math: every 64-bit\n// word you add to Tensor is an extra 3.2 gigabytes in RAM.\n//\n// If you are a Facebook employee, you can check if the run in question\n// has tipped you over the point using the command here:\n// https://fburl.com/q5enpv98\n//\n// For reference, we OOMed at 160 bytes (20 words) per TensorImpl.\n// This is not counting overhead from strides out-of-line allocation and\n// StorageImpl space and this is from before we inlined sizes and strides\n// directly into TensorImpl as SmallVectors.\n\n  // This is a 64-bit system\n  static constexpr bool check_sizes() {\n    constexpr size_t tsize = 26 * sizeof(int64_t);\n    are_equal<sizeof(sizes_and_strides_), 88,  FieldNameEnum::sizes_and_strides_>();\n    is_le<sizeof(TensorImpl),          tsize,  FieldNameEnum::TOTAL_SIZE>();',
            },
          },
          {
            h: 'a dtype is a reading, not a container',
            ps: [
              "Since the dtype lives on the tensor and the storage is only bytes, you can point a second tensor at the same bytes and read them as something else. `x.view(torch.int32)` does exactly that, and the address does not move.",
              "What prints is the float32 bit pattern shown as an integer. A 1.0 in single precision is 0x3F800000, which is 1065353216 in decimal, and there it is in the second slot. Nothing was converted and nothing was copied. One `TypeMeta` changed, and the same four bytes answered a different question.",
              "This is also the cleanest way to see why the dtype could not have lived on the storage. Two tensors over one storage with two different dtypes is a legal thing to build, so the field has to sit on the object there can be many of.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the same bytes, read as float32 and as int32',
              lang: 'python',
              text: 'import torch\n\nx = torch.arange(12.).reshape(3, 4)\ni = x.view(torch.int32)\nprint(i.dtype, i.data_ptr() == x.data_ptr())\nprint(i[0].tolist())\n\n# torch.int32 True\n# [0, 1065353216, 1073741824, 1077936128]',
            },
          },
          {
            h: 'every view in a chain points at one base',
            ps: [
              "Chain three view operations and you might expect three links back up the chain. There is only one. `_base` on any view in the chain points at the root tensor that owns the storage, not at the immediate parent, so a slice of a transpose of a reshape reports the original 1-D `arange` as its base.",
              "The last line is the detector this arc uses instead of comparing addresses. A tensor that owns fresh storage has `_base` set to `None`, so `_base is None` is a yes-or-no answer to whether a call copied. Lesson three leans on it hard, because `reshape` will not tell you which road it took.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): three views deep, one base',
              lang: 'python',
              text: 'import torch\n\nbase = torch.arange(12.)\nx = base.reshape(3, 4)\nv = x.t()\ns = v[1:]\nprint(x._base is base, v._base is base, s._base is base)\nprint(s._base is v, base._base is None)\nprint(v.contiguous()._base is None)\n\n# True True True\n# False True\n# True',
            },
          },
        ],
        readings: [
          { label: 'TensorImpl.h at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/c10/core/TensorImpl.h', note: 'the field list is at the bottom; read it before the 2800 lines of accessors above it' },
          { label: 'StorageImpl.h at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/c10/core/StorageImpl.h', note: 'the whole class in 237 lines, aliasing invariant included' },
          { label: 'SizesAndStrides.h at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/c10/core/impl/SizesAndStrides.h', note: 'why five axes fit inline and a sixth allocates' },
        ],
        check: [
          {
            q: 'A tensor reports 6 elements and its storage reports 24. What happened, and what does it cost?',
            a: 'It is a view: slicing rewrote the shape, the stride and the storage offset and freed nothing. The whole allocation stays alive as long as any view of it does, so a small crop holds the large tensor’s memory.',
          },
          {
            q: 'Which of the two objects carries the dtype, and why can it not live on the other one?',
            a: 'TensorImpl carries it, as data_type_. Two tensors can point at one storage and read it as different types, as x.view(torch.int32) does, so the dtype has to sit on the object there can be many of.',
          },
          {
            q: 'You slice a transposed reshape of a tensor. What does the slice’s _base point at?',
            a: 'The root of the chain, which is the tensor that owns the storage, not the transpose it came from directly. A tensor with fresh storage of its own reports _base as None, which is how you detect a copy after the fact.',
          },
        ],
        work: [
          { id: 'field-ledger', label: 'write the field ledger from memory: which object holds shape, stride, offset, dtype, device and the data pointer, and which of those are caches', href: '#everything-else-rides-on-the-other-object' },
          { id: 'offset-by-hand', label: 'take one sliced tensor of your own and compute its data_ptr from the storage pointer, the storage offset and the element size before you print it', href: '#offset-times-element-size-is-the-whole-connection' },
        ],
      },
      {
        id: 'the-address-arithmetic',
        num: 2,
        title: 'The address arithmetic',
        lede: 'One multiply-add per axis turns an index into a position in storage. Run it by hand a few times and the stride oracle stops being a memory test, because you can derive every answer it asks for.',
        goal: 'Compute the storage position of any element from shape, stride and offset without running torch, state the stride rule for slicing, unsqueeze and expand, and say which stride patterns make a tensor overlap itself and what torch does about each case.',
        sections: [
          {
            h: 'one multiply-add per axis',
            ps: [
              "Every read a strided tensor performs is the same expression. Start at the storage offset, then for each axis add the index along that axis times that axis's stride. That position is an element index into the storage, and multiplying by the element size turns it into a byte address.",
              "The chain below is one of the corpus rows, and the second printed line does the lookup twice. Once through the tensor's own indexing, and once by evaluating the expression against the flat `arange` the view was built from. Same number, because indexing is arithmetic and nothing more than arithmetic.",
              "Nothing about that expression cares whether the strides are increasing, decreasing or repeated. It also does not care whether two different index tuples produce the same position, which is where the last section of this lesson goes.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): the same element, through the map and by hand',
              lang: 'python',
              text: 'import torch\n\nbase = torch.arange(64.)\ny = base.reshape(2, 4, 8).transpose(0, 2)\nprint(y.shape, y.stride(), y.storage_offset())\n\ni, j, k = 5, 2, 1\nprint(y[i, j, k].item(), base[0 + i * 1 + j * 8 + k * 32].item())\n\n# torch.Size([8, 4, 2]) (1, 8, 32) 0\n# 53.0 53.0',
            },
          },
          {
            h: 'four ops the survey did not spell out',
            ps: [
              "Transposing swaps two entries in both tuples, and the chapter above says so. The four operations below are the ones whose arithmetic is easy to get wrong, and each of them is one line of bookkeeping over shape, stride and offset.",
              "Slicing rows is the only one that moves the offset. `x[k:]` keeps every stride, subtracts k from the first size, and pushes the offset forward by k times the first stride. A strided slice like `x[:, 1::2]` does both jobs at once: it multiplies the stepped axis's stride by the step, ceiling-divides that axis's size, and moves the offset by the start times the original stride.",
              "`unsqueeze` is the one that looks like it should be free of arithmetic and is not. The inserted axis has size 1, so its stride is never stepped and could hold anything, but torch fills it with the stride of the axis it displaced times that axis's size. That is why a size-1 axis often carries a stride larger than any of its neighbours, and why a chain with an `unsqueeze` in it produces stride tuples that look wrong at first glance.",
            ],
            table: {
              caption: 'the four rules, each checked against a run on torch 2.2.2 CPU; d indexes the axis named in the call',
              cols: ['call', 'shape', 'stride', 'offset'],
              rows: [
                ['x[k:]', 'shape[0] -= k', 'unchanged', 'offset += k * stride[0]'],
                ['x[:, a::s]', 'shape[1] = ceil(shape[1] / s)', 'stride[1] *= s', 'offset += a * stride[1]'],
                ['x.unsqueeze(d)', 'a 1 inserted at d', 'stride[d] * shape[d] inserted at d, or 1 past the end', 'unchanged'],
                ['x.expand(...) on a size-1 axis', 'the axis takes the requested size', 'that axis’s stride becomes 0', 'unchanged'],
              ],
            },
          },
          {
            h: 'predict these six before you read the right-hand columns',
            ps: [
              "The six rows below are quoted from the corpus behind the stride oracle, which is 48 chains executed on torch 2.2.2 and stored with what torch printed. All 48 replayed identically on this machine on 14 August 2026, so these are measurements rather than derivations.",
              "Work the third row by hand and the arithmetic from this lesson does all of it. Start at (2, 3, 4) with strides (12, 4, 1). The transpose swaps the outer two entries of both tuples, giving shape (4, 3, 2) and stride (1, 4, 12). Then `[:, ::2]` halves the middle size to 2 and doubles its stride to 8, which is exactly the (4, 2, 2) and (1, 8, 12) in the table.",
              "The fifth row is worth working through twice. Two strided slices in a row leave a size-1 axis whose stride is 12, the same as the axis above it, because the second slice doubled a stride that no longer has anything to step over. A size-1 axis's stride is arbitrary in the sense that nothing reads it, and specific in the sense that torch still computes and stores a number there.",
            ],
            table: {
              caption: 'six chains quoted from the stride corpus behind the stride oracle: 48 chains executed on torch 2.2.2 CPU, shape and stride and contiguity as torch printed them',
              cols: ['base', 'chain', 'shape', 'stride', 'contiguous'],
              rows: [
                ['torch.arange(32.).reshape(4, 8)', 'x.unsqueeze(0).transpose(0, 2)', '(8, 4, 1)', '(1, 8, 32)', 'False'],
                ['torch.arange(24.).reshape(6, 2, 2)', 'x[:, ::2].transpose(0, 2)', '(2, 1, 6)', '(1, 4, 4)', 'False'],
                ['torch.arange(24.).reshape(2, 3, 4)', 'x.transpose(0, 2)[:, ::2]', '(4, 2, 2)', '(1, 8, 12)', 'False'],
                ['torch.arange(32.).reshape(4, 8)', 'x.contiguous()[:, ::2]', '(4, 4)', '(8, 2)', 'False'],
                ['torch.arange(60.).reshape(5, 4, 3)', 'x[:, ::2][:, ::2]', '(5, 1, 3)', '(12, 12, 1)', 'False'],
                ['torch.arange(24.).reshape(2, 3, 4)', 'x.unsqueeze(0)[:, ::2]', '(1, 1, 3, 4)', '(24, 24, 4, 1)', 'True'],
              ],
            },
          },
          {
            h: 'a stride of zero reads the same bytes forever',
            ps: [
              "Set an axis's stride to zero and the multiply-add stops advancing along it. Every index along that axis lands on the same position, which is how broadcasting gets to be free: `expand` writes a zero and changes nothing else.",
              "Two numbers in the run below refuse to fit together at first. The tensor claims 15 elements, and the storage under it holds 3. There is no rule saying a tensor's element count has to fit inside its storage, only that every position the map produces has to land inside it.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): fifteen elements over three elements of storage, and what happens when you write to them',
              lang: 'python',
              text: 'import torch\n\ne = torch.arange(3.).unsqueeze(1).expand(3, 5)\nprint(e.stride(), e.numel(), e.untyped_storage().nbytes() // e.element_size())\nprint(torch._debug_has_internal_overlap(e))\ntry:\n    e.add_(1)\nexcept RuntimeError as err:\n    print(err)\n\n# (1, 0) 15 3\n# 1\n# unsupported operation: more than one element of the written-to tensor refers\n# to a single memory location. Please clone() the tensor before performing the\n# operation.',
            },
          },
          {
            h: 'when two indices land on one address',
            ps: [
              "A tensor whose map sends two different index tuples to one position overlaps itself. Reading such a tensor is fine, because a read that returns the same number twice is still correct. Writing to it is not, because the result depends on the order the kernel happens to visit elements in.",
              "torch has a check for this, and the check has three answers rather than two. `MemOverlap` is `No`, `Yes`, or `TooHard`, and the third one is the one to know about. An expanded tensor answers `Yes` and in-place ops on it raise. A tensor built by `as_strided` into a sliding window answers `TooHard`, so nothing raises, and the write goes ahead and produces a result that depends on the visiting order.",
              "The four-by-three window below reads six values through a stride of one on both axes. Adding one in place should leave `[1, 2, 3, 4, 5, 6]` if every element were written once. What comes back is `[1, 3, 5, 6, 6, 6]`, because the middle positions were visited three times each. No error, no warning, and a wrong answer that looks plausible.",
              '>> No and Yes are the easy answers. TooHard means the check gave up and your write went through anyway.',
            ],
            code: {
              caption: 'verbatim, aten/src/ATen/MemoryOverlap.h:12-19 at v2.2.2, then a run on this machine (torch 2.2.2 CPU) of the case the enum’s third value covers',
              lang: 'c',
              text: '// MemOverlap: Whether or not there is memory overlap\n//\n// No: Absolutely no memory overlap\n// Yes: Absolutely yes memory overlap\n// TooHard: There might be memory overlap, but it was too expensive to compute.\n//\n// NB: Please update the python test for these if you renumber them.\nenum class MemOverlap { No, Yes, TooHard };\n\n// >>> b = torch.arange(6.)\n// >>> w = torch.as_strided(b, (4, 3), (1, 1))\n// >>> torch._debug_has_internal_overlap(w)\n// 2\n// >>> w.add_(1)\n// >>> b.tolist()\n// [1.0, 3.0, 5.0, 6.0, 6.0, 6.0]',
            },
          },
          {
            h: 'and no strides below zero',
            ps: [
              "One shape of map torch will not build is a descending one. `as_strided` with a negative stride raises, and the message says `Negative strides are not supported at the moment`, which is the same reason `torch.flip` allocates instead of returning a view. Flip a tensor and check its stride and it reads like a fresh contiguous tensor, because it is one.",
              "NumPy does allow it, and a reversed slice there is a view with a negative stride: `np.arange(5)[::-1]` reports strides of -8 and owns no data of its own, on numpy 2.2.6. So a habit carried over from NumPy costs a full copy here, every time you reverse.",
            ],
          },
        ],
        readings: [
          { label: 'MemoryOverlap.h at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/aten/src/ATen/MemoryOverlap.h', note: '42 lines: the three-valued answer and the four assertions built on it' },
          { label: 'Tensor Views, the docs list', url: 'https://docs.pytorch.org/docs/2.2/tensor_view.html', note: 'every op that returns a view rather than a copy, in one page' },
          { label: 'torch.Tensor.as_strided', url: 'https://docs.pytorch.org/docs/2.2/generated/torch.Tensor.as_strided.html', note: 'the raw constructor, and the warning about what you can build with it' },
        ],
        check: [
          {
            q: 'A tensor has shape (8, 4, 2), stride (1, 8, 32) and offset 0. Which storage position holds element [5, 2, 1]?',
            a: 'Position 53. The rule is offset plus the sum of index times stride per axis, so 0 + 5*1 + 2*8 + 1*32 = 53, which is what indexing the flat base tensor at 53 returns.',
          },
          {
            q: 'expand produced a 15-element tensor over 12 bytes of storage. Which stride made that possible?',
            a: 'A stride of zero on the broadcast axis, so every index along it lands on the same position. The tensor reads three float32 values fifteen times, and an in-place op on it raises because more than one element refers to one memory location.',
          },
          {
            q: 'The overlap check answered TooHard for a sliding window. What does that mean for an in-place add?',
            a: 'It means no error is raised and the write happens anyway, with each repeated position updated once per visit. A window of (4, 3) with strides (1, 1) over arange(6) leaves [1, 3, 5, 6, 6, 6] instead of every element incremented once.',
          },
        ],
        work: [
          { id: 'positions-by-hand', label: 'take five chains from the stride oracle and, for each, compute the storage position of one element by hand before you check it', href: '#one-multiply-add-per-axis' },
          { id: 'overlap-three', label: 'run torch._debug_has_internal_overlap on three tensors of your own building and explain each of the three answers you get', href: '#when-two-indices-land-on-one-address' },
        ],
      },
      {
        id: 'when-a-view-refuses',
        num: 3,
        title: 'When a view refuses',
        lede: 'The error says one dimension spans across two contiguous subspaces. That sentence is a summary of a short loop, and once you can run the loop in your head you know before you type it which reshape returns a view and which one copies.',
        goal: 'State the chunk rule view runs, apply it to decide whether a given reshape aliases or copies, and name what the copy path guarantees about the tensor it hands back.',
        sections: [
          {
            h: 'the condition, in the docs’ own notation',
            ps: [
              "The `view` docstring states the rule before it states the error, and the notation is worth reading slowly. Each new axis must either be a subspace of an old axis, or span a run of old axes d through d+k where every neighbouring pair satisfies one equation: stride[i] equals stride[i+1] times size[i+1].",
              "Read that equation as a question about adjacency. It asks whether stepping one unit along axis i lands exactly where you would arrive by walking all the way through axis i+1. When that holds, the two axes are laid out end to end in storage and merging them into one axis is just relabelling. When it fails, there is a gap or an overlap between them and no single stride can describe the merged axis.",
              "A tensor that is contiguous everywhere satisfies the equation at every pair, which is why any reshape of a contiguous tensor is a view. The equation is weaker than contiguity, though, and the next section is about the gap between them.",
            ],
            code: {
              caption: 'verbatim, the torch.Tensor.view docstring as printed by the local torch 2.2.2 install; the sphinx math directive rendered as it appears in the string',
              lang: 'text',
              text: 'The returned tensor shares the same data and must have the same number\nof elements, but may have a different size. For a tensor to be viewed, the new\nview size must be compatible with its original size and stride, i.e., each new\nview dimension must either be a subspace of an original dimension, or only span\nacross original dimensions :math:`d, d+1, \\dots, d+k` that satisfy the following\ncontiguity-like condition that :math:`\\forall i = d, \\dots, d+k-1`,\n\n.. math::\n\n  \\text{stride}[i] = \\text{stride}[i+1] \\times \\text{size}[i+1]',
            },
          },
          {
            h: 'chunks, not contiguity',
            ps: [
              "The implementation states the same rule as a two-step procedure, and this phrasing is easier to run by hand. Cut the old shape into chunks of axes that are contiguous within the chunk. Then check that the new shape can be cut into the same number of chunks with matching element counts, chunk for chunk.",
              "The word doing the work is within. A transposed tensor is not contiguous as a whole, and it still has chunks. Ask for a new shape whose cuts fall on the chunk boundaries and the view succeeds; ask for one whose cuts fall inside a chunk boundary that does not exist, and it cannot.",
              "One line in that loop is worth pointing at, because it is where size-1 axes get their exemption. A chunk boundary is not declared when the neighbouring old axis has size 1, so a size-1 axis never breaks a chunk no matter what stride it carries. The other size-1 exemption, the one inside the contiguity test, is a different line in a different function, and the last lesson of this arc has it.",
            ],
            code: {
              caption: 'verbatim, aten/src/ATen/TensorUtils.cpp:317-324 at v2.2.2, the comment above computeStride_impl',
              lang: 'c',
              text: '// On a high level,\n// 1. separate `oldshape` into chunks of dimensions, where the dimensions are\n//    ``contiguous\'\' in each chunk, i.e., oldstride[i] = oldshape[i+1] *\n//     oldstride[i+1]\n// 2. `newshape` must be able to be separated into same number of chunks as\n//    `oldshape` was separated into, where each chunk of newshape has matching\n//    ``numel\'\', i.e., number of subspaces, as the corresponding chunk of\n//    `oldshape`.',
            },
          },
          {
            h: 'three answers from one tensor',
            ps: [
              "The run below asks three questions of one storage. The first tensor is a slice with a size-1 middle axis, strides (12, 4, 1), not contiguous, and `view(2, 4)` succeeds on it: the size-1 axis is exempt, the remaining pair satisfies the equation, and the result aliases the original base.",
              "Notice what the successful view produced. Shape (2, 4) with strides (12, 1), which is itself not contiguous. A view is not a promise about the result's layout, only about whether one existed.",
              "The second tensor is a transpose with strides (4, 12, 1). Merging its first two axes needs stride[0] to equal stride[1] times size[1], which is 12 times 2, and 4 is not 24. The view raises with the message the survey chapter quotes, and `reshape` answers the same request by copying.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): a view that succeeds on a non-contiguous tensor, and one that cannot',
              lang: 'python',
              text: 'import torch\n\nbase = torch.arange(24.)\nx = base.reshape(2, 3, 4)\n\nb = x[:, 1:2, :]\nprint(b.stride(), b.is_contiguous())\nprint(b.view(2, 4).stride(), b.view(2, 4)._base is base)\n\nt = x.transpose(0, 1)\nprint(t.stride(), t.is_contiguous())\ntry:\n    t.view(6, 4)\nexcept RuntimeError as err:\n    print(err)\n\n# (12, 4, 1) False\n# (12, 1) True\n# (4, 12, 1) False\n# view size is not compatible with input tensor\'s size and stride (at least one\n# dimension spans across two contiguous subspaces). Use .reshape(...) instead.',
            },
          },
          {
            h: 'reshape never says which road it took',
            ps: [
              "`reshape` is a short dispatch around the same function. It calls `computeStride`, and if that returns strides it hands back an alias through a private op that skips `view`'s duplicated work. If `computeStride` returns nothing, the last line clones and views the clone.",
              "Read that last line closely, because it decides something about the result that nothing in the call site hints at. The clone is taken with `at::MemoryFormat::Contiguous`, so a reshape that copies always lands on a row-major layout, whatever the input's layout was. A reshape that aliases keeps whatever strides `computeStride` produced.",
              "That gives you two tensors from one call that differ in ownership, in layout, and in whether a later in-place write is visible to the original. `_base is None` is the one-line test that tells them apart after the fact, and it costs nothing to add to a debugging session.",
              '>> The same call returns an alias or an owner, and only the tensor knows which.',
            ],
            code: {
              caption: 'verbatim, aten/src/ATen/native/TensorShape.cpp at v2.2.2: the two decisive lines of reshape at 1700-1702 and 1713-1726, with the twelve-line NB comment between them trimmed',
              lang: 'c',
              text: '  // `computeStride` returns the proper strides to use if this\n  // `reshape` can be just a view.\n  auto stride = at::detail::computeStride(self.sizes(), self.strides(), shape);\n\n  if (stride.has_value()) {\n    // Temporary check to revert to the old behavior/view in cases where the\n    // device is not supported (e.g. for XLA the operation is not supported\n    // so we use `view` instead).\n    //\n    // We need to do the checks here instead of in `native_functions.yaml`\n    // to preserve backwards compatibility.\n    if (!self.is_xla() && !self.is_lazy() && !self.is_ipu()) {\n      return self._reshape_alias(shape, stride.value());\n    } else {\n      return self.view(shape);\n    }\n  }\n  return at::_unsafe_view(self.clone(at::MemoryFormat::Contiguous), shape);',
            },
          },
          {
            h: 'the chains that come back',
            ps: [
              "Two view operations that undo each other leave the strides where they started, and the corpus has several rows that do it. A permute followed by a transpose that reverses it, or `t()` applied twice, both return a contiguous tensor with row-major strides, because the metadata went out and came back.",
              "This matters for a habit rather than for a fact. A long chain of views is not progressively more broken; it is a single stride tuple that got rewritten a few times, and it can land anywhere in the space, contiguous included. The only way to know where it landed is to compute it, which the previous lesson gave you the arithmetic for, or to ask, which the explorer on the chapter page does op by op.",
            ],
            table: {
              caption: 'three round-trip chains quoted from the stride corpus (48 chains executed on torch 2.2.2 CPU)',
              cols: ['base', 'chain', 'shape', 'stride', 'contiguous'],
              rows: [
                ['torch.arange(60.).reshape(5, 4, 3)', 'x.permute(2, 1, 0).transpose(0, 2)', '(5, 4, 3)', '(12, 3, 1)', 'True'],
                ['torch.arange(24.).reshape(2, 3, 4)', 'x.transpose(0, 2).permute(2, 1, 0)', '(2, 3, 4)', '(12, 4, 1)', 'True'],
                ['torch.arange(32.).reshape(4, 8)', 'x.t().transpose(0, 1)', '(4, 8)', '(8, 1)', 'True'],
              ],
            },
          },
        ],
        readings: [
          { label: 'computeStride_impl in TensorUtils.cpp at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/aten/src/ATen/TensorUtils.cpp', note: 'the chunk loop itself, from the comment at 317 to the three overloads at 397' },
          { label: 'reshape in TensorShape.cpp at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/aten/src/ATen/native/TensorShape.cpp', note: 'reshape at 1690 and _reshape_alias right below it' },
          { label: 'torch.Tensor.view', url: 'https://docs.pytorch.org/docs/2.2/generated/torch.Tensor.view.html', note: 'the contiguity-like condition, and the dtype overload underneath it' },
        ],
        check: [
          {
            q: 'A tensor reports is_contiguous() False and view() still returned an alias. How?',
            a: 'Because view needs the chunk rule, not contiguity. Its axes were cuttable into chunks matching the requested shape, with the size-1 axis exempt from breaking a chunk, so a stride tuple for the new shape existed even though the whole tensor is not row-major.',
          },
          {
            q: 'reshape handed you a tensor. How do you tell afterward whether it copied?',
            a: 'Check _base. An alias reports the root tensor of the view chain; a copy owns fresh storage and reports None. Nothing in the call, the shape or the strides announces which path ran.',
          },
          {
            q: 'Why does a reshape that copies always come back row-major?',
            a: 'Because the fallback line clones with at::MemoryFormat::Contiguous before viewing the clone. The copy path does not preserve the input’s layout, so a channels-last input reshaped into a new shape returns a contiguous tensor.',
          },
        ],
        work: [
          { id: 'chunk-rule-by-hand', label: 'run the chunk rule by hand on three reshapes you are about to write, predict alias or copy, then check each with _base is None', href: '#chunks-not-contiguity' },
          { id: 'find-a-copy', label: 'find one reshape in code of your own that copies, and decide whether the copy or the layout that forced it is the thing to change' },
        ],
      },
      {
        id: 'two-layouts-one-shape',
        num: 4,
        title: 'Two layouts, one shape',
        lede: 'Two tensors can hold the same numbers at the same indices and disagree about which bytes sit next to which. torch keeps that disagreement in five cached bits and answers questions about it one format at a time.',
        goal: 'Write the channels-last stride tuple for a given four-axis shape, say what is_contiguous answers for each memory format and when both answer True, and predict which layout comes out of an elementwise op with mixed inputs.',
        sections: [
          {
            h: 'the same tensor, two byte orders',
            ps: [
              "Convert a four-axis tensor to channels-last and nothing about it changes logically. Same shape, same values at the same indices, and `torch.equal` says so. What changed is the stride tuple, from (60, 20, 5, 1) to (60, 1, 15, 3), and the bytes underneath got rewritten to match.",
              "Read the raw storage in both and the difference is plain. The contiguous tensor stores 0, 1, 2, 3, 4, 5, which is one row of one channel. The channels-last tensor stores 0, 20, 40, 1, 21, 41, which is all three channels of one pixel, then all three channels of the next.",
              "Which of those a kernel wants depends on what it loops over innermost. A convolution reading all channels at one pixel gets contiguous memory in the second layout and a strided gather in the first, which is what the format is for.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): one tensor, two layouts, and the first six elements of each storage',
              lang: 'python',
              text: 'import torch\n\nt = torch.arange(120.).reshape(2, 3, 4, 5)\ntc = t.to(memory_format=torch.channels_last)\n\nprint(t.stride(), tc.stride())\nprint(torch.equal(t, tc), tc.data_ptr() == t.data_ptr())\nprint(tc.is_contiguous(), tc.is_contiguous(memory_format=torch.channels_last))\nprint(torch.as_strided(t, (6,), (1,)).tolist())\nprint(torch.as_strided(tc, (6,), (1,)).tolist())\n\n# (60, 20, 5, 1) (60, 1, 15, 3)\n# True False\n# False True\n# [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]\n# [0.0, 20.0, 40.0, 1.0, 21.0, 41.0]',
            },
          },
          {
            h: 'contiguity is a computed bit, and there are five of them',
            ps: [
              "`is_contiguous()` does not compute anything. It reads a bit that was computed the last time the tensor's metadata changed, and there are five of them sitting in one bitfield: plain contiguous, channels-last 2d, channels-last 2d contiguous, and the two 3d equivalents. A sixth bit alongside them records whether the tensor is non-overlapping and dense.",
              "The function behind the first bit is nineteen lines and worth knowing by heart. Walk the axes from the last to the first, carrying a running product that starts at 1. Skip any axis of size 1 entirely. For every other axis, the stride has to equal the running product, and then the product absorbs that axis's size.",
              "The skip is the whole reason a shape with a 1 in it can carry a strange stride and still call itself contiguous. Nothing ever steps along an axis of size 1, so its stride is unconstrained, and the corpus row with shape (1, 1, 3, 4) and strides (24, 24, 4, 1) reports True for exactly that reason.",
            ],
            code: {
              caption: 'verbatim, c10/core/Contiguity.h:11-30 at v2.2.2, the whole of _compute_contiguous',
              lang: 'c',
              text: 'template <typename T>\nbool _compute_contiguous(ArrayRef<T> sizes, ArrayRef<T> strides, T numel) {\n  bool is_contiguous = true;\n  if (numel == 0)\n    return is_contiguous;\n  T z = 1;\n  // NB: make sure we do signed arithmetic\n  for (int64_t d = int64_t(sizes.size()) - 1; d >= 0; d--) {\n    const auto& size_d = sizes[d];\n    if (size_d != 1) {\n      if (strides[d] == z) {\n        z *= size_d;\n      } else {\n        is_contiguous = false;\n        break;\n      }\n    }\n  }\n  return is_contiguous;\n}',
            },
          },
          {
            h: 'channels-last is the same walk in a different order',
            ps: [
              "The channels-last test is the contiguity test with the axis order hard-coded. Instead of walking axes 3, 2, 1, 0, it walks 1, 3, 2, 0: channels innermost, then width, then height, then batch. Same running product, same size-1 skip, same early exit.",
              "The switch around it says something the docs do not. Only rank 4 is handled, with a `TODO` next to rank 3 saying it will be enabled once it is fully tested, and everything else returns false. So `is_contiguous(memory_format=torch.channels_last)` on a two-axis tensor is not an error and not a meaningful answer. It is False because the function had no case for it.",
              "The 3d version is the same code with the order 1, 4, 3, 2, 0, and it handles rank 5 only. Two formats, two hard-coded orders, one algorithm.",
            ],
            code: {
              caption: 'verbatim, c10/core/Contiguity.h:32-59 at v2.2.2, _compute_channels_last_contiguous_2d in full',
              lang: 'c',
              text: 'template <typename T>\nbool _compute_channels_last_contiguous_2d(\n    ArrayRef<T> sizes,\n    ArrayRef<T> strides) {\n  // Please don\'t combine these code, constant array is used here to let\n  // compiler fully unroll the loop to get better performance\n  switch (sizes.size()) {\n    case 4: {\n      T expected = 1;\n      for (auto& d : {1, 3, 2, 0}) {\n        const auto& size_d = sizes[d];\n        if (size_d != 1) {\n          if (strides[d] != expected) {\n            return false;\n          }\n          expected *= size_d;\n        }\n      }\n      return true;\n    }\n      // NOLINTNEXTLINE(bugprone-branch-clone)\n    case 3:\n      // TODO dim == 3 case will be enabled once it is fully tested\n      return false;\n    default:\n      return false;\n  }\n}',
            },
          },
          {
            h: 'with one channel the question stops having two answers',
            ps: [
              "Give a tensor a single channel and both tests skip the same axis, which means both can pass. A (2, 1, 4, 5) tensor converted to channels-last comes back with strides (20, 1, 5, 1) and answers True to both `is_contiguous()` and `is_contiguous(memory_format=torch.channels_last)`.",
              "This is the ambiguity that makes layout bugs hard to reproduce. A test written with one channel, or a batch of one, or a one-by-one spatial size, will pass under either layout and prove nothing about which one your kernel receives. Give the test at least two of everything before you trust what it says about layout.",
            ],
            code: {
              caption: 'run it (verified, torch 2.2.2 CPU): one channel, both answers True, and the non-answer for a two-axis tensor',
              lang: 'python',
              text: 'import torch\n\na = torch.arange(40.).reshape(2, 1, 4, 5).to(memory_format=torch.channels_last)\nprint(a.stride(), a.is_contiguous(), a.is_contiguous(memory_format=torch.channels_last))\nprint(torch.arange(12.).reshape(3, 4).is_contiguous(memory_format=torch.channels_last))\n\n# (20, 1, 5, 1) True True\n# False',
            },
          },
          {
            h: 'the format is an instruction to an operator',
            ps: [
              "The header opens by denying the thing its name suggests. A memory format is not a property of a tensor; it is a way to tell an operator how to organize its result. What a tensor carries is strides, and the five bits summarizing them. `torch.channels_last` is an argument you pass, and `Preserve` is the option that says to follow the inputs.",
              "Follow that through an elementwise op and the consequence is a result whose layout depends on argument order. `tc + t` comes back with the channels-last strides and `t + tc` comes back row-major, with identical values either way, because the iterator picks a traversal order from its inputs, and the two calls hand it the same two tensors in the opposite order.",
              "The header for that iterator says why in one line: `reorder_dimensions() reorders dimensions to improve coalescing`. It is picking the traversal order that makes the inner loop contiguous, and the output gets built to match. Which means a layout you established at the top of a model can survive a long way down it, and a single argument swap can drop it.",
              '>> A layout is not something a tensor is. It is the arrangement an operator was asked to produce, remembered as strides.',
            ],
            code: {
              caption: 'verbatim, c10/core/MemoryFormat.h:9-25 at v2.2.2, then a run on this machine (torch 2.2.2 CPU) showing argument order deciding the output layout',
              lang: 'c',
              text: '// Memory format is not the property of a Tensor. It is the way to tell an\n// operator how the result should be organized in memory and nothing more. That\n// means memory format should never be used as return value for any tensor state\n// interrogation functions (internally and externally).\n//\n// Possible options are:\n//  Preserve:\n//    If any of the input tensors is in channels_last format, operator output\n//    should be in channels_last format\n//\n//  Contiguous:\n//    Regardless of input tensors format, the output should be contiguous\n//    Tensor.\n//\n//  ChannelsLast:\n//    Regardless of input tensors format, the output should be in channels_last\n//    format.\n\n// >>> (tc + t).stride()\n// (60, 1, 15, 3)\n// >>> (t + tc).stride()\n// (60, 20, 5, 1)\n// >>> torch.equal(t + tc, tc + t)\n// True',
            },
            table: {
              caption: 'one shape, two layouts, measured on torch 2.2.2 CPU for shape (2, 3, 4, 5)',
              cols: ['question', 'contiguous tensor', 'channels-last tensor'],
              rows: [
                ['stride()', '(60, 20, 5, 1)', '(60, 1, 15, 3)'],
                ['is_contiguous()', 'True', 'False'],
                ['is_contiguous(memory_format=torch.channels_last)', 'False', 'True'],
                ['first six elements in storage', '0, 1, 2, 3, 4, 5', '0, 20, 40, 1, 21, 41'],
                ['stride of x + other, x first', '(60, 20, 5, 1)', '(60, 1, 15, 3)'],
              ],
            },
          },
        ],
        readings: [
          { label: 'Contiguity.h at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/c10/core/Contiguity.h', note: 'all four compute functions in under 130 lines' },
          { label: 'MemoryFormat.h at v2.2.2', url: 'https://github.com/pytorch/pytorch/blob/v2.2.2/c10/core/MemoryFormat.h', note: 'the enum, the stride builders, and the ambiguity rules for size-1 axes' },
          { label: 'the channels-last tutorial', url: 'https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html', note: 'the same format from the user side, with the conversion and propagation rules' },
        ],
        check: [
          {
            q: 'A four-axis tensor of shape (2, 3, 4, 5) says is_contiguous() False and is_contiguous(memory_format=torch.channels_last) True. What are its strides?',
            a: '(60, 1, 15, 3). Channels innermost with stride 1, then width at 3, then height at 15, then batch at 60, which is the walk order 1, 3, 2, 0 with a running product.',
          },
          {
            q: 'How can one tensor answer True to both memory formats at once?',
            a: 'Because both tests skip axes of size 1. A tensor with one channel, such as (2, 1, 4, 5) with strides (20, 1, 5, 1), satisfies both walks, so a test written with a single channel proves nothing about layout.',
          },
          {
            q: 'Two tensors hold identical values, one contiguous and one channels-last. Why do t + tc and tc + t come back with different strides?',
            a: 'Because the iterator reorders its axes for coalescing based on the inputs, and the two calls give it a different first input. The values are equal either way; only the byte order of the result differs.',
          },
        ],
        work: [
          { id: 'channels-last-by-hand', label: 'write the channels-last stride tuple for three four-axis shapes of your own, then check each against to(memory_format=torch.channels_last)', href: '#channels-last-is-the-same-walk-in-a-different-order' },
          { id: 'layout-audit', label: 'take one input pipeline you have written and name the line where its layout is decided, then the first op that could change it', href: '#the-format-is-an-instruction-to-an-operator' },
        ],
      },
    ],
  },
]
