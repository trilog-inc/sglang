# DeepSeek-V4-Pro Single-Server Inference Strategy

Status: Phase 0/1 implemented; Phase 2+ gated on the target-server audit
Last updated: 2026-08-13

Server setup and validation runbook:
[DEEPSEEK_V4_PRO_SERVER_SETUP.md](DEEPSEEK_V4_PRO_SERVER_SETUP.md)

## Implementation checkpoint

The repository now includes the metadata-only capacity gate and the native
UE8M0 CPU path required before attempting a Pro load:

```bash
python scripts/deepseek_v4_pro_memory_planner.py \
  /mnt/home_extend/models/DeepSeek-V4-Pro \
  --verbose-layers \
  --json-output deepseek-v4-pro-memory-plan.json
```

The command exits with status 0 only when every configured memory reserve is
met and the checkpoint metadata has no unexplained routed-expert warnings. It
does not read tensor payloads. Phase 2 and the helper-GPU phases must be run on
the target inference server only after this command returns `GO`; this follows
the go/no-go requirement at the end of this document.

## Objective

Run the native `deepseek-ai/DeepSeek-V4-Pro` checkpoint on one inference
server with:

- 768 GiB host RAM;
- one NVIDIA RTX PRO 6000 Blackwell Workstation Edition with 96 GiB VRAM;
- one NVIDIA GeForce RTX 4090 with 24 GiB VRAM;
- two NVIDIA GeForce RTX 3090 GPUs with 24 GiB VRAM each;
- an AMX-capable processor; and
- the SGLang, KTransformers, and `kt-kernel` integration maintained in this
  project.

The implementation must preserve the checkpoint's native MXFP4 expert
semantics. It must not convert expert weights to AMXINT4. CPU inference should
use native packed E2M1 weights, native UE8M0 scales, BF16 activations, and AMX
where profitable.

The initial goal is correct, stable, low-concurrency inference. High
throughput and maximum context support are later optimization stages.

## Feasibility conclusion

The model cannot run with the current two-tier target-GPU/CPU implementation.
The existing CPU MXFP4 representation expands every one-byte UE8M0 scale to a
four-byte FP32 value, making the host expert allocation alone larger than the
available 768 GiB RAM.

A native-precision deployment is likely feasible after two fundamental
changes:

1. Retain UE8M0 scales in their native one-byte representation throughout the
   CPU loader and AMX kernels.
2. Use the RTX 4090 and both RTX 3090s as additional target-model expert
   compute tiers rather than reserving a GPU for a speculative draft.

This is a narrow capacity envelope. Every large allocation must be intentional,
and model loading must never materialize a full duplicate of the checkpoint.

## Source model facts

The published DeepSeek-V4-Pro configuration contains:

| Property | Value |
| --- | ---: |
| Total parameters | 1.6T |
| Activated parameters | 49B |
| Hidden layers | 61 |
| Routed experts per layer | 384 |
| Experts selected per token | 6 |
| Hidden size | 7,168 |
| MoE intermediate size | 3,072 |
| Context limit | 1,048,576 tokens |
| Routed-expert precision | Native FP4 E2M1 plus UE8M0 group scales |
| Most non-expert precision | FP8 |
| Sliding window | 128 tokens |

References:

- [DeepSeek-V4-Pro model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [DeepSeek-V4-Pro configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/config.json)
- [DeepSeek-V4-Pro checkpoint files](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main)
- [vLLM DeepSeek-V4-Pro deployment recipe](https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4-Pro.yaml)

The Hugging Face repository currently reports approximately 865 GB decimal.
A local download has been reported as approximately 893 GB. The apparent byte
size of the actual safetensor files must be measured before implementation;
filesystem block usage, cached revisions, hard links, and auxiliary files are
not reliable model-memory measurements.

## Capacity model

### Native routed-expert size

Each routed expert has three projections:

```text
parameters_per_expert = 3 * hidden_size * intermediate_size
                      = 3 * 7,168 * 3,072
                      = 66,060,288 parameters
```

Native MXFP4 uses a packed nibble per weight plus one UE8M0 byte per group of
32 weights:

```text
packed_weights = 66,060,288 / 2
scales         = 66,060,288 / 32
native_size    = 35,094,528 bytes
               = 33.47 MiB per expert
```

There are 61 x 384 = 23,424 routed-expert instances:

```text
native routed-expert bank = 765.6 GiB
```

The remaining dense, attention, embedding, shared-expert, router, and output
weights are approximately 40 GiB, based on the current official checkpoint
size.

### Current KT runtime expansion

The current AMX MXFP4 buffer stores group scales as FP32:

```text
current CPU representation = packed FP4 + four-byte FP32 scales
current routed-expert bank  = approximately 900.7 GiB
```

This exceeds host capacity before accounting for Linux, Python, pinned
buffers, allocator metadata, or model-loading transients.

Changing the final CPU scale representation from FP32 back to native UE8M0
saves approximately 135 GiB for the full routed-expert bank. This conversion
is lossless because the UE8M0 byte is the exponent needed to construct the
corresponding positive FP32 scale.

## Target steady-state placement

The following is an initial conservative placement, not a final tuned result:

| Tier | Proposed placement | Approximate weight use | Remaining capacity |
| --- | ---: | ---: | ---: |
| Host RAM | 339 experts/layer in native packed form | 676 GiB | 92 GiB |
| RTX PRO 6000 | Dense model plus 18 experts/layer | 76 GiB | 20 GiB |
| RTX 4090 | 9 experts/layer using Marlin-compatible MXFP4 | 19 GiB | 5 GiB |
| RTX 3090 #1 | 9 experts/layer using Marlin-compatible MXFP4 | 19 GiB | 5 GiB |
| RTX 3090 #2 | 9 experts/layer using Marlin-compatible MXFP4 | 19 GiB | 5 GiB |

The exact counts must be resolved from a dry-run memory planner because GPU
representations, alignment, Marlin repacking, CUDA contexts, FlashInfer
workspaces, and graph pools consume additional memory.

The RTX PRO 6000 remains the target model's primary device. It owns:

- all dense and attention computation;
- embeddings, router, shared experts, and LM head;
- the primary KV cache;
- sampling and request state; and
- final MoE output combination.

The three 24 GiB GPUs own only compact banks of routed experts and their
execution workspaces. They must not participate in dense tensor parallelism.

## Topology constraints

The installed GPUs do not support CUDA peer access with one another. Transfers
between the RTX PRO 6000 and helper GPUs therefore traverse host memory and PCIe.

Consequences:

- Do not use tensor parallelism across these four heterogeneous GPUs.
- Do not distribute attention or the KV cache merely to consume spare VRAM.
- Use explicit pinned host staging rather than relying on implicit peer copies.
- Transfer compact token activations and outputs, never weights during decode.
- Execute CPU experts and all helper-GPU expert groups concurrently.
- Prefer static placement so routing never triggers weight migration on the
  critical path.

The tensors transferred per routed token are small compared with expert
weights, so helper-GPU execution can still be beneficial despite the lack of
P2P. The principal risk is per-layer synchronization latency across 61 layers.

## Required software changes

### 1. Native UE8M0 CPU scale storage

Create an MXFP4-specific CPU weight buffer rather than reusing the generic
INT4 K-group buffer with FP32 scales.

Required properties:

- Keep E2M1 weights nibble-packed.
- Keep scale arrays as `uint8_t` UE8M0.
- Update `required_size` to account for one scale byte per 32 weights.
- Pass raw scale pointers from the safetensor loader without an intermediate
  BF16 conversion.
- Construct FP32 scale vectors inside the AMX and AVX-512 kernels using exponent
  bit construction.
- Retain the existing native BF16 activation and FP32 accumulation behavior.
- Preserve the `swiglu_limit=10.0` behavior from the checkpoint.
- Add parity tests covering all UE8M0 exponent values used by the checkpoint,
  including zero and boundary encodings.

The implementation should use a dedicated type such as
`BufferBMXFP4UE8M0`. It should not silently change the behavior of generic
INT4 or other K-group kernels.

### 2. Multi-GPU target expert tiers

Extend the current KT routing split from two destinations:

```text
primary GPU | CPU
```

to five destinations:

```text
RTX PRO 6000 | RTX 4090 | RTX 3090 #1 | RTX 3090 #2 | CPU
```

Each layer needs a logical-expert-to-tier map and a compact index within that
tier. A possible CLI shape is:

```text
--kt-gpu-expert-devices 0,1,2,3
--kt-num-gpu-experts-per-device 18,9,9,9
--kt-gpu-expert-backends flashinfer_mxfp4,marlin,marlin,marlin
```

The actual option names should follow existing SGLang argument conventions.

Placement strategies should include:

- `uniform` for initial correctness testing;
- `frequency` using recorded target-model expert distributions; and
- an explicit placement file for reproducible deployments.

The frequency policy should place the hottest experts on the RTX PRO 6000,
the next group on the RTX 4090, then distribute the remaining GPU-resident
experts across the RTX 3090s while accounting for their equal memory but lower
compute throughput.

### 3. Explicit host-staged transport

Allocate persistent pinned buffers for every helper GPU:

- compact input activations;
- remapped expert IDs;
- routing weights;
- token-to-output mappings; and
- compact output activations.

Each helper needs dedicated copy and compute streams with CUDA events. A layer
should execute conceptually as follows:

1. Partition routed assignments on the RTX PRO 6000.
2. Launch the primary GPU expert subset.
3. Copy CPU assignments into the existing KT pinned staging path.
4. Copy each helper subset through its pinned host buffers.
5. Run CPU AMX and the three helper GPUs concurrently.
6. Return compact outputs to the RTX PRO 6000.
7. Combine outputs using original routing weights and token indices.

Buffer sizes should be bucketed and reusable. No per-token `cudaMalloc`, host
allocation, or Python list construction is acceptable in the decode loop.

### 4. Direct compact loading

The loader must stream checkpoint shards and place weights directly into their
final tier.

It must never construct:

- a complete 384-expert layer on the RTX PRO 6000;
- all expert weights in ordinary PyTorch CPU tensors;
- a raw scale copy plus BF16 and FP32 scale copies;
- duplicate complete expert banks for NUMA workers; or
- a full DSpark draft replica.

Loading should remain layerwise. After a layer is copied into owned CPU/GPU
buffers, release all safetensor mappings and advise the kernel that the mapped
pages are no longer required. File-backed page cache must remain reclaimable.

A dry-run loader mode should report predicted and actual bytes by:

- tensor category;
- layer;
- expert tier;
- final representation; and
- device.

### 5. Breakable graph integration

CPU and helper-GPU expert regions must remain explicit breaks in the target
decode CUDA graph.

Initial implementation:

- use the existing breakable backend for the RTX PRO 6000 graph;
- run helper GPU expert kernels eagerly;
- use stable persistent input/output addresses;
- synchronize with CUDA events rather than device-wide synchronization; and
- disable full decode graphs.

Later optimization can capture helper work in independent graphs keyed by
bucketed assignment count. Dynamic expert routing makes one fully static
multi-device graph impractical.

### 6. Long-prefill execution

A complete Pro expert layer is approximately 12.5 GiB. Staging a whole layer
onto the RTX PRO 6000 leaves insufficient workspace.

Long prefill must therefore:

- determine the routed expert subset for the current chunk;
- process experts in bounded batches;
- reuse per-device staging slots;
- distribute batches across all helper GPUs;
- overlap CPU AMX with helper GPU execution; and
- avoid retaining staged weights after the layer completes.

Start with 2-8 staged experts per helper and tune from measured transfer,
repack, and kernel times. Do not assume the Flash model's layerwise prefill
threshold or slot count remains appropriate for Pro.

### 7. KV cache and context policy

Keep the primary KV cache on the RTX PRO 6000. DeepSeek-V4's compressed hybrid
cache is substantially smaller than a conventional full-attention cache.
Based on the current SGLang layout, a single one-million-token context is on
the order of 5 GiB before allocator, state-pool, and request metadata overhead.

Validation should proceed through these context lengths:

1. 4K correctness smoke test;
2. 32K startup and generation;
3. 128K chunked prefill;
4. 384K Think-Max target;
5. 1M single-request stress test.

Concurrency must be introduced only after the 1-request memory envelope is
stable. Explicit `max_total_tokens` and request limits are required; the server
must not automatically consume every byte freed by expert offload.

## Speculative decoding policy

Do not reserve the RTX 4090 for a separate DSpark draft in the first Pro
deployment. Its VRAM is more valuable as a target expert tier, both for
capacity and for reducing CPU memory traffic.

Recommended progression:

1. Run target-only eager decode.
2. Enable the checkpoint's built-in MTP after target correctness is proven.
3. Evaluate whether MTP improves end-to-end latency at the low expected batch
   sizes.
4. Consider DSpark only if a smaller compatible draft can be hosted without
   evicting target experts or reducing KV/workspace headroom.

The DSpark expert-distribution recorder guard must remain in place so draft
routing cannot pollute target placement statistics.

## Implementation phases

### Phase 0: exact memory audit

- Add a safetensor metadata scanner that does not load tensor payloads.
- Categorize native bytes into routed experts, shared experts, attention,
  indexer, embeddings, heads, MTP, and miscellaneous tensors.
- Compare filesystem payload with the official configuration-derived estimate.
- Measure available host RAM after boot and CUDA context overhead on all GPUs.
- Produce a proposed placement and fail early if safety reserves cannot be met.

Exit criterion: predicted placement leaves at least 64 GiB host reserve,
12 GiB primary-GPU reserve, and 3 GiB per helper GPU.

### Phase 1: packed UE8M0 AMX path

- Implement native UE8M0 storage in `kt-kernel`.
- Remove BF16 scale materialization from the MXFP4 loader.
- Add scalar, AVX-512, and AMX parity tests.
- Measure final resident bytes with synthetic Pro-shaped layers.

Exit criterion: the complete theoretical expert bank requires approximately
765.6 GiB, excluding allocator alignment of less than one percent.

### Phase 2: primary GPU plus CPU proof of life

- Load a small fixed number of experts per layer on the RTX PRO 6000.
- Keep all remaining experts in packed host memory.
- Disable speculative decoding and use eager or breakable decode.
- Validate short-prompt output against a reference backend.

This phase may remain too tight for production but isolates the packed-scale
and large-model-loading changes before multi-GPU routing is introduced.

### Phase 3: one helper GPU

- Add one remote target expert tier on the RTX 4090.
- Implement pinned host staging and asynchronous event coordination.
- Verify exact output combination for arbitrary routing patterns.
- Benchmark the latency crossover between CPU AMX and remote Marlin.

Exit criterion: helper execution reduces or matches layer latency without
increasing peak host memory.

### Phase 4: four-device expert routing

- Add both RTX 3090 tiers.
- Implement weighted static placement and per-device capacity planning.
- Run all four expert compute paths concurrently.
- Add failure diagnostics that identify the tier, layer, expert, device, and
  bucket involved.

Exit criterion: the complete native checkpoint loads with required memory
reserves and generates deterministically for a short greedy prompt.

### Phase 5: prefill and long context

- Implement demand-driven expert staging for prefill.
- Tune chunk size, staging slots, and helper batch sizes.
- Validate 32K, 128K, 384K, and 1M contexts sequentially.
- Measure time-to-first-token and peak host/GPU memory at every size.

### Phase 6: graph and placement optimization

- Re-enable target breakable graphs incrementally.
- Add helper execution buckets and optional helper CUDA graphs.
- Record real target expert distributions.
- Move hot experts to GPU tiers and compare hit rate, throughput, and latency.
- Tune CPU thread count, thread pools, NUMA binding, and AMX crossover.

### Phase 7: MTP and concurrency

- Enable built-in MTP.
- Verify acceptance rate and net latency improvement.
- Test concurrency levels 1, 2, 4, and 8 only while memory reserves hold.
- Establish production-safe token and request limits.

## Validation requirements

### Numerical correctness

- Compare UE8M0 scale decoding bit-for-bit with the existing conversion path.
- Compare CPU AMX, CPU scalar reference, RTX PRO FlashInfer, and helper Marlin
  expert outputs across real checkpoint tensors.
- Test all projection orientations and non-contiguous expert selections.
- Validate MoE output combination when one token routes to multiple tiers.
- Compare end-to-end greedy tokens with a trusted reference for short prompts.

### Memory correctness

- Record RSS, anonymous RSS, file-backed RSS, pinned host bytes, and swap use.
- Record allocated and reserved VRAM on every GPU.
- Verify that loader mappings are released after every layer.
- Verify that no scale representation is duplicated after loading.
- Run repeated load/unload cycles to detect retained mappings and allocator
  fragmentation.

### Runtime stability

- Exercise empty tier assignments and maximum tier assignments.
- Exercise one-token decode, chunked prefill, mixed-length batches, and request
  cancellation.
- Test breakable graph capture and replay with dynamic routing.
- Run at least a one-hour generation soak before enabling MTP.
- Run with CUDA launch blocking and device-side assertions during development.

### Performance telemetry

Collect per layer and per tier:

- routed token count;
- expert hit rate;
- H2D and D2H bytes and time;
- kernel time;
- CPU AMX time;
- combine time;
- synchronization wait time;
- achieved host memory bandwidth; and
- helper GPU utilization.

End-to-end metrics must include time to first token, inter-token latency,
accepted tokens per speculative step, prefill throughput, and total output
throughput.

## Initial runtime posture

The first successful launch should use conservative limits:

- tensor parallel size 1;
- one running request;
- no separate draft model;
- prefill CUDA graph disabled;
- decode graph disabled or breakable with small batch capture sizes;
- explicit maximum total tokens;
- chunked prefill of 1,024-4,096 tokens;
- FP8 KV cache;
- small MXFP4 prefill staging slots; and
- frequency recording disabled until basic generation is stable.

After correctness, enable target expert-distribution recording and build a
static placement file from representative requests. Expert redistribution at
runtime is out of scope for the first implementation.

## Expected performance

This design prioritizes feasibility over throughput. Pro activates 49B
parameters per token versus 13B for Flash, and a substantial part of its expert
traffic will still be served from host RAM. It should be expected to run
several times slower than the current Flash deployment until measurements
show otherwise.

The most important performance variables will be:

1. effective host memory bandwidth under AMX execution;
2. fraction of routed assignments served by GPU-resident experts;
3. per-layer host-staged helper latency;
4. long-prefill expert staging efficiency; and
5. MTP acceptance rate at batch size one.

The absence of GPU peer access prevents this machine from matching a
high-bandwidth eight-GPU H200/B300 deployment. The goal is a practical local
deployment, not data-center-class throughput.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Host OOM during model loading | Metadata-only planner, direct compact loads, layerwise mmap release, raw UE8M0 scales |
| Primary GPU OOM during autotune or graph capture | Reserve at least 12-20 GiB, cap graph shapes, disable prefill graph |
| Helper GPUs slower than CPU for tiny assignments | Measure crossover and route small buckets to CPU |
| Host-staged transfers serialize every layer | Persistent pinned buffers, separate streams, concurrent CPU/helper execution |
| Marlin representation exceeds helper capacity | Use 8-9 experts/layer, account for repack and scale bytes before allocation |
| Long prefill needs a full layer staging buffer | Demand-driven expert minibatches across all helpers |
| File cache displaces anonymous expert memory | Release mappings and use advisory cache eviction after each layer |
| Frequency placement overfits one workload | Preserve a uniform baseline and validate on multiple prompt classes |
| MTP consumes memory without improving latency | Keep optional and enable only after measured net benefit |

## Pre-implementation measurements

Run these commands on the inference server and save their output with the
implementation benchmark artifacts:

```bash
MODEL=/mnt/home_extend/models/DeepSeek-V4-Pro

find "$MODEL" -maxdepth 1 -name '*.safetensors' -printf '%s\n' |
awk '{s += $1} END {
  printf "Safetensors: %.0f bytes, %.2f GiB\n", s, s / 1073741824
}'

du --apparent-size -sb "$MODEL"
du -sb "$MODEL"
free -b
numactl --hardware

nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap \
  --format=csv,noheader
nvidia-smi topo -m
nvidia-smi topo -p2p r
nvidia-smi topo -p2p w
```

Also record CPU model, memory channel population, measured STREAM bandwidth,
kernel version, NVIDIA driver, CUDA toolkit, PyTorch, FlashInfer, SGLang,
KTransformers, and `kt-kernel` commit hashes.

## Go/no-go criteria

Proceed to a complete implementation only if the Phase 0 planner shows all of
the following:

- Native checkpoint tensor payload matches the architecture-derived estimate
  closely enough to explain all large tensors.
- Packed host expert placement stays below 704 GiB, or another explicitly
  justified host safety threshold.
- Dense weights, primary experts, 1M single-request KV, and workspaces fit on
  the RTX PRO 6000 with at least 8 GiB post-capture reserve.
- Every helper GPU retains at least 3 GiB after its compact expert bank is
  prepared.
- No required tensor is duplicated across host and GPU tiers after loading.
- The server has sufficient storage for the checkpoint plus temporary
  conversion and test artifacts without relying on swap.

If these criteria cannot be met, the native checkpoint should not be forced
through disk paging. The alternatives are a smaller checkpoint, a lower-bit
derived quantization with separately validated quality, additional host RAM,
or GPUs with more aggregate VRAM and a high-bandwidth interconnect.
