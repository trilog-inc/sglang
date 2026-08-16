# DeepSeek-V4-Pro single-server setup guide

This runbook prepares the target server for the DeepSeek-V4-Pro capacity audit
and native MXFP4/UE8M0 CPU-kernel validation in a fresh Conda environment. It
is written for the implementation in these branches:

- SGLang: `trilog-inc/sglang`, branch `codex/dsv4`, at a revision containing
  the per-device KT topology flags verified in Section 6;
- KTransformers/KT-Kernel: `trilog-inc/ktransformers`, branch
  `codex/nvfp4-kt-support`, minimum merge commit
  `26f008fbca8ba0f092f668554f40e4b091a054b7`.

## Read this first: current implementation boundary

The current update implements:

- a metadata-only checkpoint and placement planner;
- native packed E2M1 weights plus one-byte UE8M0 scales on CPU;
- adaptive AMX-BF16/AVX-512 and AVX2 UE8M0 decoding paths;
- direct loading into the final CPU expert buffers;
- one remote target-expert tier with compact Marlin weights, pinned host
  transport, and asynchronous CUDA event coordination;
- DSpark speculative-decoding integration, tuning, and policy tests;
- planner and native-kernel parity tests; and
- one-expert-at-a-time helper and draft-primary weight preparation to bound
  startup memory.

The 26/10/0/0 target placement is now implemented: 26 experts per layer remain
on the RTX PRO 6000, 10 run on the RTX 4090, and the complement stays in native
packed host memory. The two RTX 3090s remain unused by the target.

The bundled DSpark MTP draft can now split every stage's 384 routed experts
192/192 across the two RTX 3090s. Its non-expert backbone and KV cache stay on
the first RTX 3090, while embeddings and the LM head reuse the copies already
resident on the RTX PRO 6000. This is an experimental eager-only path until it
passes the hardware gates below. Always validate the target-only launch first.

## Target configuration

The procedure assumes:

| Component | Target |
| --- | --- |
| OS | Ubuntu 24.04 LTS, Linux x86-64 |
| Host memory | 768 GiB installed, with at least 64 GiB reserved |
| Primary GPU | RTX PRO 6000 Blackwell Workstation Edition, 96 GiB |
| Helper GPUs | RTX 4090 24 GiB and 2 x RTX 3090 24 GiB |
| CPU | Intel AMX, AVX-512, and BF16 capable |
| CUDA toolkit | 13.3.x under `/usr/local/cuda-13.3` |
| Python | 3.11 in a dedicated Conda environment |
| Model storage | At least 1.1 TB free on a local high-throughput filesystem |

CUDA 13.3 is the target compiler toolkit for this server. The Python stack uses
the official PyTorch CUDA 13.0 wheel because that is the current CUDA 13 wheel
line published for PyTorch 2.11. CUDA 13.0 wheel binaries run on the newer
CUDA 13.3 driver through backward compatibility, while source and JIT
extensions compile with the local CUDA 13.3 toolkit. Consequently,
`torch.version.cuda` should report `13.0` and `nvcc --version` should report
`13.3`; that difference is intentional.

If the machine differs from this table, do not copy the placement numbers
blindly. Record its actual capacities and pass them to the planner.

## 1. Establish an administrative shell and working paths

Use Bash for the commands in this guide. Choose paths on the large local
filesystem, not on a small root volume or network home directory.

```bash
bash
set -o pipefail

export DSV4_ROOT=/mnt/home_extend/deepseek-v4-pro
export DSV4_SRC="$DSV4_ROOT/src"
export DSV4_MODEL="$DSV4_ROOT/models/DeepSeek-V4-Pro"
export DSV4_REPORT="$DSV4_ROOT/reports"
export DSV4_CONDA_ROOT=/mnt/home_extend/miniconda3
export SGLANG_SRC="$DSV4_SRC/sglang"
export KTRANSFORMERS_SRC="$DSV4_SRC/ktransformers"

mkdir -p "$DSV4_SRC" "$DSV4_MODEL" "$DSV4_REPORT"
```

All later commands assume these variables remain set in the current shell.

## 2. Perform the hardware and OS preflight

Run the inventory before installing anything and save it with the test
artifacts:

```bash
date -Is | tee "$DSV4_REPORT/preflight.txt"
uname -a | tee -a "$DSV4_REPORT/preflight.txt"
lscpu | tee -a "$DSV4_REPORT/preflight.txt"
free -b | tee -a "$DSV4_REPORT/preflight.txt"
df -h "$DSV4_ROOT" | tee -a "$DSV4_REPORT/preflight.txt"
nvidia-smi | tee -a "$DSV4_REPORT/preflight.txt"
nvidia-smi --query-gpu=index,pci.bus_id,name,memory.total,memory.free,compute_cap \
  --format=csv,noheader | tee "$DSV4_REPORT/gpus.csv"
nvidia-smi topo -m | tee "$DSV4_REPORT/gpu-topology.txt"
nvidia-smi topo -p2p r | tee "$DSV4_REPORT/gpu-p2p-read.txt"
nvidia-smi topo -p2p w | tee "$DSV4_REPORT/gpu-p2p-write.txt"
if command -v numactl >/dev/null 2>&1; then
  numactl --hardware | tee "$DSV4_REPORT/numa.txt"
else
  echo 'numactl is not installed yet; rerun this check after Section 3.' \
    | tee "$DSV4_REPORT/numa.txt"
fi
```

Confirm all of the following before continuing:

1. All four GPUs appear and have their expected memory capacities.
2. The RTX PRO 6000's compute capability is 12.0, the RTX 4090 is 8.9, and
   the RTX 3090s are 8.6.
3. The NVIDIA Linux driver is version 610.43.02 or newer and reports CUDA 13.3
   or newer capability.
4. `nvcc --version` reports release 13.3.
5. `/proc/cpuinfo` contains `amx_tile`, `amx_bf16`, `amx_int8`, `avx512f`,
   `avx512bw`, and `avx512_bf16`.
6. The model filesystem has at least 1.1 TB available before download.
7. The topology report confirms the expected lack of GPU peer access. This is
   a design constraint, not a reason to enable tensor parallelism.

Use this compact CPU flag check:

```bash
awk -F: '/^flags/{print $2; exit}' /proc/cpuinfo \
  | tr ' ' '\n' \
  | sort -u \
  | sed -n '/^amx_/p;/^avx512/p'
```

If AMX flags are absent, stop. The AVX2 fallback can validate correctness but
does not satisfy the intended Pro performance posture.

## 3. Install system build tools

Do not install or replace the NVIDIA driver from inside the Conda environment.
Have the administrator install NVIDIA Linux driver 610.43.02 or newer and the
CUDA 13.3 toolkit using the server's normal package-management policy first.
For an NVIDIA APT repository configured for Ubuntu 24.04, the versioned toolkit
metapackage is `cuda-toolkit-13-3`; using the versioned name prevents an
unplanned upgrade to a later toolkit family.

Install the CPU build and diagnostic dependencies:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  curl \
  git \
  git-lfs \
  jq \
  libhwloc-dev \
  libnuma-dev \
  ninja-build \
  numactl \
  pkg-config \
  ripgrep \
  tmux

# Run only when the NVIDIA CUDA repository is already configured and the
# toolkit is not already installed by the administrator.
sudo apt-get install -y cuda-toolkit-13-3

git lfs install
numactl --hardware | tee "$DSV4_REPORT/numa.txt"
```

Verify the CUDA compiler explicitly. Adjust `CUDA_HOME` if the toolkit is in a
different versioned directory:

```bash
export CUDA_HOME=/usr/local/cuda-13.3
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

nvcc --version
test -x "$CUDA_HOME/bin/nvcc"
nvcc --version | rg -q 'release 13\.3'
```

Do not continue with an older toolkit merely because `nvidia-smi` displays a
new CUDA version. The CUDA version shown by `nvidia-smi` is the driver's
maximum supported API level; `nvcc --version` identifies the installed build
toolkit.

## 4. Install Conda if necessary

If `conda --version` already succeeds, skip this section. Otherwise download
the current Linux x86-64 Miniconda installer from the official Anaconda
repository. In security-sensitive environments, compare its SHA-256 value
with the value published in the official Miniconda index before running it.

```bash
curl -fL \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  -o /tmp/Miniconda3-latest-Linux-x86_64.sh

bash /tmp/Miniconda3-latest-Linux-x86_64.sh \
  -b -p "$DSV4_CONDA_ROOT"

source "$DSV4_CONDA_ROOT/etc/profile.d/conda.sh"
conda --version
```

Starting a new shell later requires sourcing the same `conda.sh`, unless Conda
has already been initialized for that shell.

## 5. Create a fresh environment

Create a clean Python 3.11 environment. Do not reuse an environment containing
official `sglang`, another KT-Kernel wheel, or a different PyTorch CUDA build.

```bash
source "$DSV4_CONDA_ROOT/etc/profile.d/conda.sh"
conda create -n dsv4-pro python=3.11 pip -y
conda activate dsv4-pro

python -m pip install --upgrade pip setuptools wheel

conda env config vars set \
  DSV4_ROOT="$DSV4_ROOT" \
  DSV4_SRC="$DSV4_SRC" \
  DSV4_MODEL="$DSV4_MODEL" \
  DSV4_REPORT="$DSV4_REPORT" \
  SGLANG_SRC="$SGLANG_SRC" \
  KTRANSFORMERS_SRC="$KTRANSFORMERS_SRC" \
  CUDA_HOME="$CUDA_HOME"

conda deactivate
conda activate dsv4-pro

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

which python
python --version
python -m pip --version
```

Expected Python: `3.11.x`.

## 6. Clone the two pinned source branches

The SGLang and KT-Kernel work live in separate repositories. Both are required.

```bash
cd "$DSV4_SRC"

git clone --branch codex/dsv4 --single-branch \
  https://github.com/trilog-inc/sglang.git sglang

git clone --branch codex/nvfp4-kt-support --single-branch \
  https://github.com/trilog-inc/ktransformers.git ktransformers

```

Initialize only the KTransformers submodules required by KT-Kernel. The
KTransformers repository's own SGLang submodule is intentionally not used;
the `codex/dsv4` clone above is the source of truth.

```bash
git -C "$KTRANSFORMERS_SRC" submodule update --init --recursive \
  third_party/llama.cpp third_party/pybind11
```

Verify that both branches contain the implementation baselines:

```bash
test "$(git -C "$SGLANG_SRC" branch --show-current)" = codex/dsv4
rg -q 'kt_gpu_expert_devices' \
  "$SGLANG_SRC/python/sglang/srt/server_args.py"

git -C "$KTRANSFORMERS_SRC" merge-base --is-ancestor \
  26f008fbca8ba0f092f668554f40e4b091a054b7 HEAD

git -C "$SGLANG_SRC" status --short --branch
git -C "$KTRANSFORMERS_SRC" status --short --branch
```

Each validation command must exit with status 0. Fresh clones should have no
modified or untracked files.

## 7. Install the pinned Python and CUDA stack

Install PyTorch first so downstream packages cannot select a different CUDA
variant. PyTorch officially publishes 2.11 wheels for CUDA 13.0. Use that wheel
runtime with the CUDA 13.3 host compiler toolkit:

```bash
python -m pip install --force-reinstall --no-cache-dir \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu130

python -c 'import torch; assert torch.version.cuda == "13.0", torch.version.cuda; print("torch", torch.__version__, "cuda", torch.version.cuda, "from", torch.__file__)'
```

Do not continue until that check reports CUDA 13.0. `torch.version.cuda` is
compiled into the PyTorch wheel; it does not report the version selected by
`CUDA_HOME` or the host's `nvcc`.

Install the SGLang fork from the checked-out source:

```bash
python -m pip install -e "$SGLANG_SRC/python"
```

The merged SGLang baseline pins `sglang-kernel==0.4.5`. Force the CUDA 13.0
wheel from SGLang's wheel index after the editable install:

```bash
python -m pip install --force-reinstall --no-deps \
  sglang-kernel==0.4.5 \
  --index-url https://docs.sglang.ai/whl/cu130/
```

Keep the FlashInfer Python, cubin, and JIT-cache distributions aligned with the
`0.6.15.post1` version pinned by this branch. The Python package is available
from PyPI, the cubins use FlashInfer's common index, and the CUDA-specific JIT
cache uses its CUDA 13.0 index:

```bash
python -m pip install --force-reinstall --no-deps \
  flashinfer-python==0.6.15.post1

python -m pip install --force-reinstall --no-deps \
  flashinfer-cubin==0.6.15.post1 \
  --index-url https://flashinfer.ai/whl

python -m pip install --force-reinstall --no-deps \
  flashinfer-jit-cache==0.6.15.post1 \
  --index-url https://flashinfer.ai/whl/cu130
```

The non-Hopper sparse-MLA path also needs TileLang. Use the exact TileLang and
TVM FFI versions pinned by this branch:

```bash
python -m pip install \
  tilelang==0.1.11 \
  apache-tvm-ffi==0.1.11 \
  pytest
```

Do not override the source package's `transformers==5.12.1` pin.

## 8. Build KT-Kernel natively for this server

Build on the inference server itself. `NATIVE` uses the exact CPU instruction
set of the machine, while the CUDA architecture list covers both RTX 3090s,
the RTX 4090, and the RTX PRO 6000.

```bash
cd "$KTRANSFORMERS_SRC/kt-kernel"

export CPUINFER_FORCE_REBUILD=1
export CPUINFER_BUILD_TYPE=Release
export CPUINFER_PARALLEL=16
export CPUINFER_CPU_INSTRUCT=NATIVE
export CPUINFER_ENABLE_AMX=ON
export CPUINFER_ENABLE_AVX512=ON
export CPUINFER_USE_CUDA=1
export CPUINFER_CUDA_ARCHS='86;89;120'

bash ./install.sh build --manual 2>&1 \
  | tee "$DSV4_REPORT/kt-kernel-build.log"
```

The build must report AMX enabled, CUDA enabled, and architectures 86, 89, and
120. If CMake rejects architecture 120, the active `nvcc` is too old or
`CUDA_HOME` points at the wrong toolkit.

## 9. Verify imports, versions, and GPU visibility

Run all of these checks from the `dsv4-pro` environment:

```bash
python -c 'import torch; assert torch.version.cuda == "13.0", torch.version.cuda; print("torch", torch.__version__, "cuda", torch.version.cuda); print([(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())])'

python -m pip show \
  sglang-kernel \
  flashinfer-python \
  flashinfer-cubin \
  flashinfer-jit-cache

python -c 'import sglang; print("sglang import OK", sglang.__file__)'

python -c 'import flashinfer; print("flashinfer", flashinfer.__version__)'

python -c 'import tilelang; print("tilelang", tilelang.__version__)'

python -c 'import kt_kernel; print("kt-kernel", kt_kernel.__version__, "variant", getattr(kt_kernel, "__cpu_variant__", "source-build"))'

python -c 'from kt_kernel.utils import amx; assert amx.AMXMXFP4UE8M0_KGroup_MOE is not None; assert amx.AVX2MXFP4UE8M0_MOE is not None; print("native UE8M0 AMX and AVX2 bindings OK")'

python -m sglang.launch_server --help | rg -- \
  '--kt-weight-path|--kt-method|--kt-num-gpu-experts'

kt doctor
```

Check dependency consistency:

```bash
python -m pip check
```

`pip check` must exit cleanly. This branch directly pins the versions installed
in Section 7, so do not accept a dependency-conflict warning as expected.

Save the resolved environment:

```bash
conda list --explicit > "$DSV4_REPORT/conda-explicit.txt"
python -m pip freeze > "$DSV4_REPORT/pip-freeze.txt"
python -m torch.utils.collect_env > "$DSV4_REPORT/torch-environment.txt"
git -C "$SGLANG_SRC" rev-parse HEAD > "$DSV4_REPORT/sglang-commit.txt"
git -C "$KTRANSFORMERS_SRC" rev-parse HEAD \
  > "$DSV4_REPORT/ktransformers-commit.txt"
```

## 10. Run the repository tests before downloading the model

The planner tests use only small synthetic safetensor files:

```bash
cd "$SGLANG_SRC"
python -m pytest -q test/registered/unit/test_deepseek_v4_pro_memory_planner.py \
  | tee "$DSV4_REPORT/planner-tests.txt"

python -m pytest -q \
  test/registered/unit/tools/test_tune_dsv4_dspark_kt.py \
  test/registered/unit/layers/attention/test_dsv4_dspark_sm120_policy.py \
  | tee "$DSV4_REPORT/dspark-tests.txt"

python -m py_compile scripts/deepseek_v4_pro_memory_planner.py
```

Expected result: nine planner tests pass and the focused DSpark tests pass.

### Optional native parity executables

The following separate debug configuration exposes the two native UE8M0 test
executables. It does not replace the installed Release build:

```bash
cmake -S "$KTRANSFORMERS_SRC/kt-kernel" \
  -B "$KTRANSFORMERS_SRC/kt-kernel/build-parity" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE="$(command -v python)" \
  -DKTRANSFORMERS_CPU_DEBUG=ON \
  -DKTRANSFORMERS_CPU_USE_AMX=ON \
  -DKTRANSFORMERS_CPU_USE_AMX_AVX512=ON \
  -DKTRANSFORMERS_USE_CUDA=OFF \
  -DLLAMA_NATIVE=ON

cmake --build "$KTRANSFORMERS_SRC/kt-kernel/build-parity" \
  --target test_mxfp4_ue8m0 test_mxfp4_ue8m0_avx2 \
  --parallel 16

"$KTRANSFORMERS_SRC/kt-kernel/build-parity/test_mxfp4_ue8m0_avx2" \
  | tee "$DSV4_REPORT/ue8m0-avx2-parity.txt"

"$KTRANSFORMERS_SRC/kt-kernel/build-parity/test_mxfp4_ue8m0" \
  | tee "$DSV4_REPORT/ue8m0-amx-parity.txt"
```

Both executables should report parity across all 256 UE8M0 exponent encodings.
An illegal-instruction failure indicates that the process cannot execute the
CPU features used by the native build. Check bare-metal CPU flags, VM CPU
passthrough, BIOS settings, and kernel AMX support before rebuilding.

## 11. Authenticate and inspect the model download

The model may require accepting its Hugging Face terms in a browser before the
token can access it. Authenticate without placing the token in shell history:

```bash
hf auth login
hf auth whoami
```

Resolve and record the exact model revision before downloading:

```bash
python -c 'from huggingface_hub import HfApi; print(HfApi().model_info("deepseek-ai/DeepSeek-V4-Pro").sha)' \
  | tee "$DSV4_REPORT/model-revision.txt"

read -r DSV4_MODEL_REV < "$DSV4_REPORT/model-revision.txt"
export DSV4_MODEL_REV
```

Use the CLI dry run to verify access and estimate the transfer before spending
hours on it:

```bash
hf download deepseek-ai/DeepSeek-V4-Pro \
  --revision "$DSV4_MODEL_REV" \
  --local-dir "$DSV4_MODEL" \
  --dry-run \
  | tee "$DSV4_REPORT/model-download-dry-run.txt"
```

Recheck free storage. Keep at least 100 GB beyond the reported checkpoint size
for source builds, logs, Hugging Face metadata, and later test artifacts.

## 12. Download the checkpoint reproducibly

Download into its final directory. `hf download --local-dir` maintains local
metadata and resumes interrupted downloads, so rerun the same command after a
network interruption rather than deleting partial state.

```bash
hf download deepseek-ai/DeepSeek-V4-Pro \
  --revision "$DSV4_MODEL_REV" \
  --local-dir "$DSV4_MODEL" \
  --max-workers 4 \
  | tee "$DSV4_REPORT/model-download.txt"
```

Measure the actual payload and filesystem use:

```bash
find "$DSV4_MODEL" -maxdepth 1 -name '*.safetensors' -printf '%s\n' \
  | awk '{s += $1} END {printf "Safetensors: %.0f bytes, %.2f GiB\n", s, s / 1073741824}' \
  | tee "$DSV4_REPORT/checkpoint-size.txt"

du --apparent-size -sb "$DSV4_MODEL" \
  | tee "$DSV4_REPORT/checkpoint-apparent-size.txt"
du -sb "$DSV4_MODEL" | tee "$DSV4_REPORT/checkpoint-disk-use.txt"

jq '{model_type, architectures, num_hidden_layers, hidden_size, moe_intermediate_size, n_routed_experts, num_experts_per_tok, max_position_embeddings, quantization_config}' \
  "$DSV4_MODEL/config.json" \
  | tee "$DSV4_REPORT/model-config-summary.json"
```

Do not copy or convert the routed experts to AMXINT4. This implementation is
specifically intended to retain the checkpoint's native E2M1 and UE8M0 data.

## 13. Run the metadata-only capacity gate

Use the kernel's view of total host memory rather than assuming the DIMM label
is exactly 768 GiB:

```bash
awk '/MemTotal:/{printf "%.2f\n", $2 * 1024 / 1073741824}' /proc/meminfo \
  | tee "$DSV4_REPORT/host-ram-gib.txt"

read -r DSV4_HOST_RAM_GIB < "$DSV4_REPORT/host-ram-gib.txt"
export DSV4_HOST_RAM_GIB
```

Run the proposed 18/9/9/9 expert-per-layer placement with explicit reserves.
The extra runtime allowances make this slightly more conservative than a
weights-only estimate:

```bash
cd "$SGLANG_SRC"
set -o pipefail

python scripts/deepseek_v4_pro_memory_planner.py "$DSV4_MODEL" \
  --exclude-mtp \
  --host-ram-gib "$DSV4_HOST_RAM_GIB" \
  --host-reserve-gib 64 \
  --host-runtime-gib 8 \
  --gpu-names rtx-pro-6000,rtx-4090,rtx-3090-1,rtx-3090-2 \
  --gpu-capacities-gib 96,24,24,24 \
  --gpu-reserves-gib 12,3,3,3 \
  --gpu-runtime-gib 5,1,1,1 \
  --gpu-experts-per-layer 18,9,9,9 \
  --gpu-backends flashinfer_mxfp4,marlin_mxfp4,marlin_mxfp4,marlin_mxfp4 \
  --allocator-overhead-percent 0.5 \
  --verbose-layers \
  --json-output "$DSV4_REPORT/deepseek-v4-pro-memory-plan.json" \
  2>&1 | tee "$DSV4_REPORT/deepseek-v4-pro-memory-plan.txt"

DSV4_PLANNER_STATUS=${PIPESTATUS[0]}
printf 'planner exit status: %s\n' "$DSV4_PLANNER_STATUS" \
  | tee "$DSV4_REPORT/planner-status.txt"

jq '{go, placement_exclusions, warnings, diagnostics, placement}' \
  "$DSV4_REPORT/deepseek-v4-pro-memory-plan.json"
```

`--exclude-mtp` is required for this target-only gate. The normal DeepSeek-V4
target loader skips the checkpoint's MTP/NextN tensors when speculative
decoding is disabled. The planner still audits and reports those tensors, but
does not place them on the primary GPU in this mode. The resulting JSON records
both the complete `tensor_payload_bytes` and the smaller
`placement_payload_bytes`, plus the exact `placement_exclusions` map.

Also preserve a speculative-inclusive report by rerunning the same command
without `--exclude-mtp` and writing it to distinct `*-with-mtp` report files.
The measured checkpoint contains 39.10 GiB of MTP tensors, and the initial
18/9/9/9 placement does not fit those tensors while retaining the configured
reserves. That `NO-GO` applies to built-in MTP/DSpark enablement, not to the
target-only placement.

### Experimental two-GPU MTP capacity gate

The measured host and GPU headroom permits a deliberately tighter test plan.
Dedicate both RTX 3090s to equal halves of the MTP routed experts, keep the MTP
backbone on the first RTX 3090, retain the faster RTX 4090 as a target-expert
tier, and move the displaced 3090 target experts to the RTX PRO 6000 and host.
The modeled target per-layer expert split becomes 26/10/0/0. Use 48 GiB of host
reserve, 10 GiB on the primary, and 2 GiB on each helper only for this
experiment:

```bash
cd "$SGLANG_SRC"
set -o pipefail

python scripts/deepseek_v4_pro_memory_planner.py "$DSV4_MODEL" \
  --host-ram-gib "$DSV4_HOST_RAM_GIB" \
  --host-reserve-gib 48 \
  --host-runtime-gib 8 \
  --gpu-names rtx-pro-6000,rtx-4090,rtx-3090-1,rtx-3090-2 \
  --gpu-capacities-gib 96,24,24,24 \
  --gpu-reserves-gib 10,2,2,2 \
  --gpu-runtime-gib 5,1,1,1 \
  --gpu-experts-per-layer 26,10,0,0 \
  --gpu-mtp-fractions 0,0,0.5,0.5 \
  --gpu-backends flashinfer_mxfp4,marlin_mxfp4,marlin_mxfp4,marlin_mxfp4 \
  --allocator-overhead-percent 0.5 \
  --json-output "$DSV4_REPORT/deepseek-v4-pro-two-gpu-mtp-plan.json" \
  2>&1 | tee "$DSV4_REPORT/deepseek-v4-pro-two-gpu-mtp-plan.txt"

DSV4_MTP_PLANNER_STATUS=${PIPESTATUS[0]}
printf 'two-GPU MTP planner exit status: %s\n' "$DSV4_MTP_PLANNER_STATUS" \
  | tee "$DSV4_REPORT/two-gpu-mtp-planner-status.txt"

jq '{go, placement_exclusions, warnings, diagnostics, placement}' \
  "$DSV4_REPORT/deepseek-v4-pro-two-gpu-mtp-plan.json"
```

Based on the first server audit, the expected rounded result is:

| Tier | Weights | Runtime | Headroom | Test reserve |
| --- | ---: | ---: | ---: | ---: |
| Host | ~697.29 GiB | 8 GiB | ~50.12 GiB | 48 GiB |
| RTX PRO 6000 | ~78.95 GiB | 5 GiB | ~12.05 GiB | 10 GiB |
| RTX 4090 | ~20.03 GiB | 1 GiB | ~2.97 GiB | 2 GiB |
| RTX 3090 #1 | MTP expert half + MTP backbone (planner-derived) | 1 GiB | planner-derived | 2 GiB |
| RTX 3090 #2 | MTP expert half only (planner-derived) | 1 GiB | planner-derived | 2 GiB |

Treat these values as estimates until the command runs against the exact
checkpoint and capacities in `gpus.csv`. The updated planner assigns recognized
MTP expert tensors 50/50 but keeps MTP attention, norms, shared experts, and
other backbone tensors on RTX 3090 #1, matching the runtime. The first 3090 will
therefore report more weight than the second; an equal 19.65/19.65 report means
the checkout predates this executable-placement correction. The remaining
margins are narrow, especially on the RTX 4090 and first RTX 3090. Do not
proceed if the planner reports `NO-GO`, if the host has less measured headroom,
or if runtime profiling shows that the 1 GiB helper-runtime allowances are
insufficient.

This gate proves capacity only. The target router and two-device DSpark expert
split now implement the planned topology, but neither result substitutes for a
successful full-checkpoint load, deterministic greedy comparison, or sustained
memory observation on the actual server.

Interpretation:

- Exit status `0` and JSON `"go": true`: the metadata is understood and all
  configured steady-state reserves pass for the explicitly reported placement
  exclusions.
- Exit status `1`: valid checkpoint, but at least one capacity reserve,
  metadata warning, or placement diagnostic failed.
- Exit status `2`: invalid arguments, unreadable files, or malformed
  safetensor metadata.

Any warning makes the result a `NO-GO`. Do not suppress warnings merely to
obtain a green result. Inspect tensor names, dtypes, layer counts, and expert
counts first.

The GPU capacities above are the nominal target values. If `gpus.csv` reports
meaningfully less usable memory, round each capacity down and rerun the
planner. Never round up to the marketing capacity.

## 14. Target-helper and two-device MTP implementation gates

Confirm that the new per-device KT options are present:

```bash
python -m sglang.launch_server --help | rg -- \
  '--kt-gpu-expert-devices|--kt-num-gpu-experts-per-device|--kt-gpu-expert-backends|--speculative-draft-helper-device|--speculative-draft-num-gpu-experts-per-device'
```

Run the focused CPU-side topology and routing tests before loading the model:

```bash
cd "$SGLANG_SRC"
python -m pytest -q test/registered/unit/layers/test_kt_ep_wrapper.py \
  -k 'explicit_gpu_tier or remote_tier or remote_expert_topology or mtp_marlin_primary'

python -m pytest -q \
  test/registered/spec/dspark/test_dspark_draft_path_default.py \
  test/registered/spec/dspark/test_dspark_draft_device.py
```

For the first hardware smoke test, expose the GPUs in this exact logical order:

```text
cuda:0 = RTX PRO 6000
cuda:1 = RTX 3090 #1
cuda:2 = RTX 4090
cuda:3 = RTX 3090 #2
```

Adjust `CUDA_VISIBLE_DEVICES` if the physical indices in `gpus.csv` differ.
Keep the first launch eager, target-only, single-request, and limited to 4K
total tokens:

```bash
conda activate dsv4-pro
cd "$SGLANG_SRC"

export CUDA_HOME=/usr/local/cuda-13.3
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST='8.6;8.9;12.0+PTX'
export DSV4_CPUINFER_THREADS="${DSV4_CPUINFER_THREADS:-$(nproc)}"
# Correctness-first SM120 decode fallback. Remove only after the standalone
# FlashInfer sparse-MLA parity test passes on this exact driver/wheel stack.
export SGLANG_SM120_FLASHMLA_BACKEND=triton
# The HTTP listener starts before SGLang's prompt-plus-eight-token warmup. This
# CPU-offloaded 1.6T topology can exceed the generic 600-second watchdog on its
# first JIT-heavy forward.
export SGLANG_WARMUP_TIMEOUT=1800
# Two tokens cover prompt processing and one decode transition. The upstream
# default of eight is too costly for the initial CPU-offloaded smoke test.
export SGLANG_WARMUP_MAX_NEW_TOKENS=2

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m sglang.launch_server \
  --trust-remote-code \
  --model-path "$DSV4_MODEL" \
  --tp 1 \
  --moe-runner-backend flashinfer_mxfp4 \
  --kt-weight-path "$DSV4_MODEL" \
  --kt-method MXFP4 \
  --kt-mxfp4-backend amx \
  --kt-mxfp4-amx-min-tokens-per-expert 4 \
  --kt-gpu-expert-devices 0 2 \
  --kt-num-gpu-experts-per-device 26 10 \
  --kt-gpu-expert-backends flashinfer_mxfp4 marlin_mxfp4 \
  --kt-expert-placement-strategy uniform \
  --init-expert-location trivial \
  --kt-cpuinfer "$DSV4_CPUINFER_THREADS" \
  --kt-threadpool-count 2 \
  --kt-numa-nodes 0 1 \
  --kt-gpu-prefill-token-threshold 0 \
  --disable-shared-experts-fusion \
  --chunked-prefill-size 4096 \
  --max-total-tokens 4096 \
  --swa-full-tokens-ratio 0.20 \
  --max-running-requests 1 \
  --mem-fraction-static 0.89 \
  --disable-flashinfer-autotune \
  --disable-overlap-schedule \
  --disable-cuda-graph \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  --served-model-name dsv4pro \
  --host 0.0.0.0 \
  --port 60000 \
  2>&1 | tee "$DSV4_REPORT/target-helper-smoke-server.txt"
```

The startup log must report devices `(0, 2)`, counts `(26, 10)`, and a remote
Marlin tier with 10 experts for every routed layer. In a second shell, monitor
host RSS and all GPU allocations while weights load:

```bash
watch -n 2 'free -h; nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader'
```

Stop immediately if the RTX 4090 approaches its final 2 GiB reserve, host
available memory falls below 48 GiB, swap grows, or any layer reports a compact
mapping or Marlin preparation error. Do not increase context or concurrency
until a short greedy request completes deterministically.

After the target-only smoke test passes, stop that server and repeat the same
launch with the two-device DSpark options included:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m sglang.launch_server \
  --trust-remote-code \
  --model-path "$DSV4_MODEL" \
  --tp 1 \
  --moe-runner-backend flashinfer_mxfp4 \
  --kt-weight-path "$DSV4_MODEL" \
  --kt-method MXFP4 \
  --kt-mxfp4-backend amx \
  --kt-mxfp4-amx-min-tokens-per-expert 4 \
  --kt-gpu-expert-devices 0 2 \
  --kt-num-gpu-experts-per-device 26 10 \
  --kt-gpu-expert-backends flashinfer_mxfp4 marlin_mxfp4 \
  --kt-expert-placement-strategy uniform \
  --init-expert-location trivial \
  --kt-cpuinfer "$DSV4_CPUINFER_THREADS" \
  --kt-threadpool-count 2 \
  --kt-numa-nodes 0 1 \
  --kt-gpu-prefill-token-threshold 0 \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path "$DSV4_MODEL" \
  --speculative-draft-device 1 \
  --speculative-draft-helper-device 3 \
  --speculative-draft-num-gpu-experts-per-device 192 192 \
  --speculative-moe-runner-backend marlin \
  --disable-shared-experts-fusion \
  --chunked-prefill-size 4096 \
  --max-total-tokens 4096 \
  --swa-full-tokens-ratio 0.20 \
  --max-running-requests 1 \
  --mem-fraction-static 0.912 \
  --disable-flashinfer-autotune \
  --disable-overlap-schedule \
  --disable-cuda-graph \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  --served-model-name dsv4pro \
  --host 0.0.0.0 \
  --port 60000 \
  2>&1 | tee "$DSV4_REPORT/two-device-mtp-smoke-server.txt"
```

The MTP startup log must report primary draft `cuda:1`, helper `cuda:3`, counts
`(192, 192)`, an `mtp.N` weight prefix for every draft stage, and no CPU-resident
draft experts. Vocabulary operations intentionally execute on `cuda:0` so the
two 24 GiB devices do not duplicate the target embedding and LM-head weights.
This first version uses synchronous cross-device vocabulary transfers and must
remain eager.

The MTP smoke launch deliberately uses `--mem-fraction-static 0.912`, while the
target-only launch remains at `0.89`. On the measured first RTX 3090, draft
loading began with 23.26 GiB available and left 2.26 GiB free after 21.00 GiB
of weights. A fraction of `0.89` reserves 2.56 GiB, which is already larger
than the observed free memory and therefore gives the draft KV configurator a
negative budget. `0.912` reserves about 2.05 GiB and exposes about 0.21 GiB to
the capped 4K draft cache. Treat this as a machine-specific narrow setting:
recalculate it from the logged values if either number changes.

Both 4K smoke launches deliberately set `--swa-full-tokens-ratio 0.20`.
DeepSeek V4 uses a 128-token SWA window with 256-token allocator pages. At the
model's ordinary 0.1 ratio, a 4096-token full cache rounds the SWA tier down to
one 256-token page. That cannot cover the live window plus the current and next
paged allocations, so the scheduler repeatedly rejects even a one-token
request before model forward. A ratio of 0.20 produces a 768-token (three-page)
SWA tier. The server now rejects a smaller DSV4 tier during initialization with
the minimum ratio in the error message instead of starting an unusable server.

Each draft stage should then report both of these bounded preparation paths:

```text
Preparing KT remote expert tier one expert at a time: ... device=cuda:3 experts=192 ...
KT remote prepared expert tier ready: ... device=cuda:3 experts=192 ...
Preparing KT local expert tier one expert at a time: ... device=cuda:1 experts=192 ...
KT local prepared expert tier ready: ... device=cuda:1 experts=192 ...
```

The ordinary `Preparing MXFP4 experts for Marlin backend` message can still
appear once per stage for the internal one-slot dummy needed to initialize the
standard quantization method. It must not consume a 192-expert raw bank. If an
OOM traceback reaches `_repack_weight` with roughly 23 GiB already allocated,
the server is still running the older draft-primary loader; update the
`codex/dsv4` checkout before retrying. `PYTORCH_CUDA_ALLOC_CONF` and smaller
prefill/token limits do not fix that startup-weight duplication.

The smoke-test environment forces the Triton SM120 sparse-MLA decode fallback.
The native FlashInfer SM120 kernel can be evaluated separately after the server
is stable; do not combine that experiment with the first 800+ GiB model load.

Keep the same monitor running. Stop if either RTX 3090 falls below its 2 GiB
reserve, if the draft loads any `model.layers.N` expert in place of `mtp.N`, or
if target-only and MTP-enabled greedy output diverge before the speculative
accept/reject boundary can explain it.

Do not treat a successful `/v1/models` response as readiness. The HTTP listener
is live while the internal warmup is still running, so an external completion
can queue behind it. Wait for this log line before sending the first request:

```text
The server is fired up and ready to roll!
```

Do not substitute tensor parallelism across these heterogeneous GPUs. Do not
put target experts on either RTX 3090, and do not let Linux swap the expert bank
from disk during decode.

## 15. Preserve the audit bundle

Capture a final state snapshot for comparison with later runtime branches:

```bash
date -Is > "$DSV4_REPORT/audit-completed-at.txt"
free -b > "$DSV4_REPORT/free-after-audit.txt"
nvidia-smi -q > "$DSV4_REPORT/nvidia-smi-query.txt"
nvidia-smi topo -m > "$DSV4_REPORT/gpu-topology-after-audit.txt"
numactl --hardware > "$DSV4_REPORT/numa-after-audit.txt"
swapon --show > "$DSV4_REPORT/swap.txt"

find "$DSV4_REPORT" -maxdepth 1 -type f -printf '%f\n' | sort
```

Keep the report directory with the eventual benchmark results. At minimum it
should contain source commits, the exact model revision, package resolutions,
hardware inventory, topology, planner output, and parity-test output.

## 16. Updating these branches later

Never pull new source into a running server process. Stop the process, activate
the environment, fast-forward each clean worktree, then rebuild KT-Kernel.

```bash
conda activate dsv4-pro

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

git -C "$SGLANG_SRC" status --short
git -C "$KTRANSFORMERS_SRC" status --short

git -C "$SGLANG_SRC" pull --ff-only
git -C "$KTRANSFORMERS_SRC" pull --ff-only
git -C "$KTRANSFORMERS_SRC" submodule update --init --recursive \
  third_party/llama.cpp third_party/pybind11

python -m pip install -e "$SGLANG_SRC/python"

cd "$KTRANSFORMERS_SRC/kt-kernel"
export CPUINFER_FORCE_REBUILD=1
export CPUINFER_CPU_INSTRUCT=NATIVE
export CPUINFER_ENABLE_AMX=ON
export CPUINFER_ENABLE_AVX512=ON
export CPUINFER_USE_CUDA=1
export CPUINFER_CUDA_ARCHS='86;89;120'
bash ./install.sh build --manual
```

Verify `sglang-kernel==0.4.5` and all three FlashInfer distributions at
`0.6.15.post1`. If dependency resolution changed them, rerun the forced CUDA
13.0 wheel installs in Section 7. Then rerun Sections 9, 10, 13, and 14.

## Troubleshooting

### `torch.version.cuda` reports `12.8`

The environment contains a CUDA 12.8 PyTorch wheel. CUDA 13.3 in
`CUDA_HOME` cannot change a wheel that was compiled for CUDA 12.8. Confirm the
active interpreter, then replace the complete PyTorch package trio from the
CUDA 13.0 index:

```bash
conda activate dsv4-pro

which python
python -m pip --version
python -c 'import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda, torch.__file__)'

python -m pip install --force-reinstall --no-cache-dir \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu130

python -c 'import torch; assert torch.version.cuda == "13.0", torch.version.cuda; print("torch", torch.__version__, "cuda", torch.version.cuda, "from", torch.__file__)'
python -m pip check
```

If the assertion still reports 12.8, the displayed `torch.__file__` is being
imported from a different environment or user site. Do not build KT-Kernel
until that path belongs to the `dsv4-pro` Conda environment and the assertion
passes. If KT-Kernel was already built against the CUDA 12.8 wheel, set
`CPUINFER_FORCE_REBUILD=1` and repeat Section 8 after replacing PyTorch.

### `pip check` says KT-Kernel requires Torch 2.9.1

The KTransformers checkout predates the Torch 2.11 metadata alignment. Pull
`codex/nvfp4-kt-support`, confirm that it contains the minimum commit from
Section 6, and repeat the native build in Section 8. The rebuilt wheel must
declare `torch==2.11.0`; do not downgrade the SGLang environment to Torch
2.9.1 to satisfy a stale KT-Kernel wheel.

### `No kernel image is available` or unsupported SM120

- Confirm `CUDA_HOME=/usr/local/cuda-13.3` and `nvcc --version` reports release
  13.3.
- Confirm `CPUINFER_CUDA_ARCHS='86;89;120'` was present during the KT build.
- Set `CPUINFER_FORCE_REBUILD=1` and rebuild from a clean CMake cache.
- Confirm PyTorch sees capability `(12, 0)` for the RTX PRO 6000.

### `fp8e4nv not supported in this architecture` on the DSpark draft

The MTP checkpoint contains block-FP8 dense projection weights in addition to
its MXFP4 expert banks. RTX 3090 GPUs (SM86) can store those FP8 weights but
cannot execute native FP8 tensor-core GEMMs. The supported path is SGLang's
weight-only FP8 Marlin fallback; startup should print:

```text
Your GPU does not have native support for FP8 computation ... Marlin kernel
```

If the first request instead fails in `main_proj` /
`triton_w8a8_block_fp8_linear` with `fp8e4nv not supported`, the checkout is
older than the heterogeneous-device capability fix: its Marlin probe inspected
the SM120 target (`cuda:0`) while constructing the SM86 draft (`cuda:1`). Pull
the current `codex/dsv4` branch and restart. Do not use
`SGLANG_FORCE_FP8_MARLIN=1` as a blanket workaround: that process-wide setting
also repacks target linears on the Blackwell GPU.

If a one-token request succeeds but a longer request fails in
`dequantize_k_cache_paged` with the same `fp8e4nv` message, dense-layer Marlin
selection is already fixed and the failure is the next Ampere boundary: the
draft starts reading its packed SWA cache on the following decode step. Pull a
version containing the pre-SM89 byte-dequantization fallback and restart. That
path keeps the cache compact, decodes E4M3 through a 256-entry BF16 lookup table
inside Triton, and never exposes an unsupported FP8 pointer to the SM86
compiler. It does not change the native FP8 cache path used by the SM120 target.

### `sparse_mla_sm120_decode_dsv4` reports an unsupported shape or illegal memory access

The FlashInfer SM120 sparse-MLA kernel failed during eager-runner warmup. CUDA
graph disabling does not disable this autotuning pass. Terminate the failed
process because an illegal memory access leaves its CUDA context unusable, then
launch with both of these safeguards:

```bash
export SGLANG_SM120_FLASHMLA_BACKEND=triton
# Add to the launch arguments:
--disable-flashinfer-autotune
```

This selects SGLang's Triton sparse-MLA decode implementation and disables the
FlashInfer tuning forward. It is the conservative correctness path for the
initial load, not a statement about final attention performance.

### Draft weights leave no memory for the KV cache

If all local and remote MTP tiers finish but the draft KV configurator reports
that `--mem-fraction-static=0.89` is below a minimum near `0.903`, the bounded
expert loader succeeded. Use the two-device MTP value from Section 14:

```bash
--mem-fraction-static 0.912
```

For the measured 23.26/2.26 GiB pre-load/free values, this retains roughly
2.05 GiB of non-static slack while making roughly 0.21 GiB available for the
draft cache. Keep `--max-total-tokens 4096` and `--max-running-requests 1` for
the first retry. If SGLang next reports that the draft pool is smaller than the
target's 4096-token pool, reduce `--max-total-tokens` to `3072` for both workers
before raising the fraction above `0.912`. Stop if `nvidia-smi` shows less than
2 GiB free after pool initialization.

### Internal server warmup times out after 600 seconds

`/v1/models` can respond before SGLang finishes its internal `/generate`
warmup. A traceback from `_execute_server_warmup` with `read timeout=600` means
the warmup watchdog expired; it is not an OpenAI completions-client timeout.
Set the runbook's extended watchdog before launching:

```bash
export SGLANG_WARMUP_TIMEOUT=1800
export SGLANG_WARMUP_MAX_NEW_TOKENS=2
```

The updated server logs the warmup endpoint, token count, timeout, and elapsed
time. Do not submit an external completion until it prints `The server is fired
up and ready to roll!`. Keep the GPU/host monitor running during warmup. If the
two-token internal request still does not finish within 1,800 seconds, retry
once with `--skip-server-warmup --soft-watchdog-timeout 120` and issue a single
streaming completion with `max_tokens=1`; use that only as a diagnostic to
distinguish very slow first-token execution from a stuck scheduler. Preserve
the scheduler and py-spy output from the first `Prefill batch` or `Decode batch`
message through the failure.

```bash
curl -N --max-time 1800 http://127.0.0.1:60000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv4pro","prompt":"Hello","max_tokens":1,"temperature":0,"stream":true}'
```

### `/generate` returns HTTP 200 but never emits a token

If a scheduler stack repeatedly lands in `_compute_max_prefix_len`, the
scheduler is cycling through admission rather than running attention or MoE.
Query the live scheduler before stopping it:

```bash
curl -sS --max-time 10 \
  'http://127.0.0.1:60000/v1/loads?include=core,memory,queues,spec' \
  | python -m json.tool
```

For the 4K DSV4 smoke profile, `num_waiting_reqs=1`,
`num_running_reqs=0`, flat GPU utilization, and a stack in
`PrefillAdder`/`_compute_max_prefix_len` indicate an undersized SWA tier. Make
sure the launch includes `--swa-full-tokens-ratio 0.20`. Do not lengthen the
HTTP timeout: no forward has started. Updated code fails during pool sizing if
the ratio still rounds below the three-page admission floor.

### `CUDA_HOME environment variable is not set`

Set it to the versioned toolkit containing `bin/nvcc`, then update `PATH` and
`LD_LIBRARY_PATH` as shown in Section 3. Do not point it at the PyTorch wheel.

### FlashInfer import errors for MXFP4 symbols

Check that all three packages have base version `0.6.15.post1`. The JIT-cache
wheel may include a local `+cu130` suffix:

```bash
python -m pip show flashinfer-python flashinfer-cubin flashinfer-jit-cache
python -c 'from flashinfer import mxfp8_quantize; from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe; print("FlashInfer MXFP4 imports OK")'
```

### TileLang aborts with duplicate `__ffi_repr__` registration

Reinstall the constrained TVM FFI version:

```bash
python -m pip install --force-reinstall apache-tvm-ffi==0.1.11
```

### KT-Kernel cannot find HWLOC or NUMA

Verify `pkg-config --modversion hwloc`, then reinstall `libhwloc-dev`,
`libnuma-dev`, and `pkg-config`. Rebuild KT-Kernel afterward.

### KT-Kernel loads AVX2 instead of AMX

- Recheck the CPU flags from Section 2.
- Confirm this is bare metal or that the VM exposes AMX to the guest.
- Check BIOS and kernel support.
- Rebuild with `CPUINFER_CPU_INSTRUCT=NATIVE` and
  `CPUINFER_ENABLE_AMX=ON` on the inference server itself.

### Model download is interrupted

Rerun the exact `hf download` command with the same revision and local
directory. Do not remove the local Hugging Face metadata unless the client
reports that it is corrupt.

### Planner reports `NO-GO`

Treat the result as final until the cause is explained. Common legitimate
causes are less usable RAM than assumed, checkpoint layout changes, non-UE8M0
expert scales, insufficient GPU reserves, or unexpected expert counts. Do not
reduce reserves merely to force `GO`. The explicit two-GPU MTP experiment in
Section 13 is the only documented tighter profile; its smaller reserves must
still pass with the actual server capacities and runtime measurements.

### The host begins swapping or approaches OOM

Stop before attempting a model load. A paging expert bank is not a viable
runtime. Capture `free -b`, `swapon --show`, and the planner report; do not
disable swap on an already memory-pressured host without an administrator's
recovery plan.

## Reference links

- [Implementation strategy](DEEPSEEK_V4_PRO_SINGLE_SERVER_STRATEGY.md)
- [Official PyTorch 2.11 wheel commands](https://pytorch.org/get-started/previous-versions/)
- [Hugging Face download guide](https://huggingface.co/docs/huggingface_hub/guides/download)
- [NVIDIA CUDA 13.3 release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- [NVIDIA CUDA installation guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [DeepSeek-V4-Pro model page](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
