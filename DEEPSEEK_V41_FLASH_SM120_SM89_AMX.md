# DeepSeek-V4.1-Flash: RTX PRO 6000 + RTX 4090 + AMX

This branch ports the complete current DeepSeek-V4.1 change stack from SGLang
PR #38798 onto the optimized KTransformers/SGLang fork. The intended topology
is:

| Role | Device | Backend |
| --- | --- | --- |
| Target attention, dense layers, hot routed experts | RTX PRO 6000 96 GB (SM120) | DSV4 + FlashInfer MXFP4 |
| Bundled DSPARK draft model | RTX 4090 24 GB (SM89) | Marlin MXFP4 |
| Remaining routed experts and Engram tables | 768 GB DDR5 | KT native MXFP4 + AMX-BF16 |

The model checkpoint includes the DSPARK draft weights. Do not supply a
separate `--speculative-draft-model-path`.

## Build KT-Kernel

Use the KTransformers checkout on `codex/glm5-nextn-mtp-4090`. The launcher
builds a native host binary and includes code for both GPU architectures:

```bash
export KT_KERNEL_ROOT=/path/to/ktransformers/kt-kernel
scripts/launch_deepseek_v41_flash_dspark_kt_sm120_sm89.sh build-kt
```

The check fails unless `/proc/cpuinfo` exposes `amx_tile`, `amx_bf16`, and the
required AVX512-BF16 flags; the installed KT package must select its `amx` CPU
variant and expose `AMXFP4_KGroup_MOE`. The server also
sets `KT_MXFP4_BACKEND=amx` and passes `--kt-mxfp4-backend amx`; it does not
silently select AVX2 for the main CPU path. Small per-expert batches may still
use the AVX512-BF16 tail below the configured AMX crossover.

## Launch

Physical GPU 0 defaults to the target and physical GPU 1 to the draft. Change
`TARGET_GPU` and `DRAFT_GPU` if `nvidia-smi` reports a different order:

```bash
export MODEL_PATH=/models/DeepSeek-V4.1-Flash
export KT_WEIGHT_PATH="$MODEL_PATH"
export TARGET_GPU=0
export DRAFT_GPU=1
export KT_NUMA_NODES="0 1"

scripts/launch_deepseek_v41_flash_dspark_kt_sm120_sm89.sh check-kt
scripts/launch_deepseek_v41_flash_dspark_kt_sm120_sm89.sh serve
```

The script exposes only those two devices, so the physical target becomes
logical `cuda:0` and the physical draft becomes logical `cuda:1`. It validates
SM120/SM89 explicitly and prepares FlashInfer sampling for both architectures.
The SM89 draft automatically uses Triton for FP8 GEMMs inside its device-local
runner context while the target retains its SM120 backend.

Starting values are deliberately conservative:

- `DSPARK_BLOCK_SIZE=3`: the checkpoint advertises five proposals, but an open
  SM120 correctness report identifies depth 5 as unsafe. Test depth 4 and 5
  against a deterministic target-only baseline before changing production.
- `KT_NUM_GPU_EXPERTS=96`: lower this if target graph capture or long-context KV
  allocation runs out of VRAM; tune upward only while target VRAM remains safe.
- `CUDA_GRAPH_MAX_BS_DECODE=16` and `MAX_RUNNING_REQUESTS=16`: avoid deriving a
  graph tier too large for the 24 GB draft GPU. Raise both after a stable smoke
  test.
- `CONTEXT_LENGTH=262144`: validate short and medium contexts before increasing
  toward the checkpoint's one-million-token limit.
- `HOST_MEM_MIN_GIB=384`: startup refuses a heavily occupied host. The V4.1
  Engram tables alone require roughly 203 GB of host memory; KT's cold expert
  complement and runtime overhead need additional headroom.

The target uses the FP4 indexer, the Blackwell fused DSPARK candidate/verify
kernels, fused router-to-MoE packing, bounded decoder SWA replay, and breakable
decode graphs. Prefill graphs stay disabled because KT streams cold MXFP4
experts during long prefills.

## Correctness gate

First run a target-only baseline with `DISABLE_DSPARK=1`, then restart without
that variable for the DSPARK leg. Use identical prompts, temperature, seed, and
concurrency for both legs:

```bash
DISABLE_DSPARK=1 PORT=30001 \
  scripts/launch_deepseek_v41_flash_dspark_kt_sm120_sm89.sh serve
```

For the DSPARK leg, verify:

```bash
curl -fsS http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v41-flash",
    "messages": [{"role": "user", "content": "List the integers 1 through 20."}],
    "temperature": 0,
    "max_tokens": 128
  }'
```

Then test image input, tool calls, a long prompt, and concurrent requests. The
server's internal state should report an average speculative acceptance length
greater than 1.0. Do not promote DSPARK block size 5 merely because it starts;
compare output tokens over a representative corpus and run a sustained load
test for device-side assertions.

## What was ported

The branch contains V4.1 text/vision config normalization, multimodal input,
Engram host tables, hybrid compressed attention and unified KV pools, bundled
DSPARK loading, DP-vision handling, long-context DSA top-k fixes, Blackwell
candidate/verify and MXFP4 MoE optimizations, static DSPARK PD protocol support,
the DP-idle guard, and the V4.1 DSML tool-call parser fixes. The fork's SM89
remote-draft and native MXFP4/AMX offload paths are retained around those
changes.
