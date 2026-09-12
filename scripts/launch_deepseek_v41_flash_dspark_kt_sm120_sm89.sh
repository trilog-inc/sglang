#!/usr/bin/env bash
# DeepSeek-V4.1-Flash on RTX PRO 6000 (SM120), with its bundled DSPARK draft
# on RTX 4090 (SM89) and native MXFP4/AMX expert offload to system RAM.
set -euo pipefail

physical_cpu_count() {
  local count=""
  if command -v lscpu >/dev/null 2>&1; then
    count="$(lscpu -p=NODE,CORE,ONLINE 2>/dev/null \
      | awk -F, '$1 !~ /^#/ && $3 == "Y" { seen[$1 FS $2] = 1 } END { print length(seen) }')"
  fi
  if [[ "${count}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${count}"
  else
    getconf _NPROCESSORS_ONLN
  fi
}

ACTION="${1:-check}"
if [[ $# -gt 0 ]]; then
  shift
fi

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
TARGET_GPU="${TARGET_GPU:-0}"
DRAFT_GPU="${DRAFT_GPU:-2}"
MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-V4.1-Flash}"
KT_WEIGHT_PATH="${KT_WEIGHT_PATH:-${MODEL_PATH}}"
KT_KERNEL_ROOT="${KT_KERNEL_ROOT:-}"
KT_CPU_THREADS="${KT_CPU_THREADS:-$(physical_cpu_count)}"
KT_THREADPOOL_COUNT="${KT_THREADPOOL_COUNT:-2}"
KT_NUMA_NODES="${KT_NUMA_NODES:-}"
KT_NUM_GPU_EXPERTS="${KT_NUM_GPU_EXPERTS:-96}"
KT_AMX_MIN_TOKENS_PER_EXPERT="${KT_AMX_MIN_TOKENS_PER_EXPERT:-4}"
KT_GPU_PREFILL_TOKEN_THRESHOLD="${KT_GPU_PREFILL_TOKEN_THRESHOLD:-4096}"
KT_MXFP4_PREFILL_SLOTS="${KT_MXFP4_PREFILL_SLOTS:-auto}"
KT_MXFP4_PREFILL_HOST_STAGING_EXPERTS="${KT_MXFP4_PREFILL_HOST_STAGING_EXPERTS:-8}"
DSPARK_BLOCK_SIZE="${DSPARK_BLOCK_SIZE:-3}"
DISABLE_DSPARK="${DISABLE_DSPARK:-0}"
HOST_MEM_MIN_GIB="${HOST_MEM_MIN_GIB:-384}"
# The target weights plus two KT MXFP4 prefill slots consume about 88.8 GiB on
# the 96 GiB Blackwell card.  0.86 leaves no KV budget; 0.96 retains several
# GiB outside the static pool for activations while clearing the profiled 0.943
# minimum on this topology.
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.96}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-16}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v41-flash}"
SGLANG_BIND_HOST="${SGLANG_BIND_HOST:-0.0.0.0}"
SGLANG_BIND_PORT="${SGLANG_BIND_PORT:-30000}"

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export CPATH="${CUDA_HOME}/include${CPATH:+:${CPATH}}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${TARGET_GPU},${DRAFT_GPU}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-8.9 12.0f}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9;12.0+PTX}"
export KT_MXFP4_BACKEND=amx
# V4.1 has two roughly 94 GiB Engram tables.  Keep them in pinned host memory
# so the 96 GiB target GPU remains available for dense weights, GPU experts,
# KV cache, and CUDA graphs.  Private layout uses this host's anonymous THP.
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE="${SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE:-1}"
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT="${SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT:-private}"
export SGLANG_DSV41_ENGRAM_HOST_TABLE_PIN="${SGLANG_DSV41_ENGRAM_HOST_TABLE_PIN:-1}"

fail() {
  echo "error: $*" >&2
  exit 1
}

gpu_field() {
  local gpu="$1"
  local field="$2"
  nvidia-smi -i "${gpu}" --query-gpu="${field}" --format=csv,noheader 2>/dev/null \
    | head -1 \
    | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
}

check_kt_runtime() {
  python3 -c '
import kt_kernel
from kt_kernel import kt_kernel_ext
import sys
moe = getattr(kt_kernel_ext, "moe", None)
compiled = getattr(moe, "AMXFP4_KGroup_MOE", None) is not None
variant = getattr(kt_kernel, "__cpu_variant__", "unknown")
enabled = compiled and variant == "amx"
print(f"KT CPU variant: {variant}; AMX MXFP4 kernel: {compiled}")
sys.exit(0 if enabled else 1)
' || fail "KT-Kernel was not built with native AMX-BF16 support; run '$0 build-kt' first"
}

check_host() {
  [[ -x "${CUDA_HOME}/bin/nvcc" ]] \
    || fail "nvcc not found at ${CUDA_HOME}/bin/nvcc"

  local nvcc_release
  nvcc_release="$(${CUDA_HOME}/bin/nvcc --version \
    | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' \
    | tail -1)"
  [[ "${nvcc_release}" == "13.3" ]] \
    || fail "CUDA 13.3 is required; nvcc reports ${nvcc_release:-unknown}"

  command -v nvidia-smi >/dev/null || fail "nvidia-smi is required"
  [[ "${TARGET_GPU}" != "${DRAFT_GPU}" ]] \
    || fail "TARGET_GPU and DRAFT_GPU must name different physical GPUs"

  local target_cap draft_cap target_name draft_name
  target_cap="$(gpu_field "${TARGET_GPU}" compute_cap)"
  draft_cap="$(gpu_field "${DRAFT_GPU}" compute_cap)"
  target_name="$(gpu_field "${TARGET_GPU}" name)"
  draft_name="$(gpu_field "${DRAFT_GPU}" name)"
  [[ "${target_cap}" == "12.0" ]] \
    || fail "TARGET_GPU=${TARGET_GPU} must be SM120; found ${target_name:-unknown} (${target_cap:-unknown})"
  [[ "${draft_cap}" == "8.9" ]] \
    || fail "DRAFT_GPU=${DRAFT_GPU} must be SM89; found ${draft_name:-unknown} (${draft_cap:-unknown})"

  [[ -r /proc/cpuinfo ]] || fail "/proc/cpuinfo is unavailable"
  grep -qm1 -w amx_tile /proc/cpuinfo || fail "CPU does not advertise AMX-TILE"
  grep -qm1 -w amx_int8 /proc/cpuinfo || fail "CPU does not advertise AMX-INT8"
  grep -qm1 -w amx_bf16 /proc/cpuinfo || fail "CPU does not advertise AMX-BF16"
  grep -qm1 -w avx512f /proc/cpuinfo || fail "CPU does not advertise AVX512-F"
  grep -qm1 -w avx512bw /proc/cpuinfo || fail "CPU does not advertise AVX512-BW"
  grep -qm1 -w avx512_vnni /proc/cpuinfo || fail "CPU does not advertise AVX512-VNNI"
  grep -qm1 -w avx512_bf16 /proc/cpuinfo || fail "CPU does not advertise AVX512-BF16"

  local physical_cpus
  physical_cpus="$(physical_cpu_count)"
  [[ "${KT_CPU_THREADS}" =~ ^[1-9][0-9]*$ ]] \
    || fail "KT_CPU_THREADS must be a positive integer; got ${KT_CPU_THREADS}"
  (( KT_CPU_THREADS <= physical_cpus )) \
    || fail "KT_CPU_THREADS=${KT_CPU_THREADS} exceeds the ${physical_cpus} physical CPU cores available"

  local mem_available_kib mem_available_gib
  mem_available_kib="$(awk '/^MemAvailable:/ {print $2; exit}' /proc/meminfo)"
  mem_available_gib="$((mem_available_kib / 1024 / 1024))"
  (( mem_available_gib >= HOST_MEM_MIN_GIB )) \
    || fail "only ${mem_available_gib} GiB host RAM is available; ${HOST_MEM_MIN_GIB} GiB is required"

  echo "validated: CUDA ${nvcc_release}"
  echo "validated: target physical GPU ${TARGET_GPU}: ${target_name} (SM120)"
  echo "validated: draft physical GPU ${DRAFT_GPU}: ${draft_name} (SM89)"
  echo "validated: AMX-TILE/INT8/BF16, AVX512-F/BW/VNNI/BF16, ${physical_cpus} physical CPU cores, ${mem_available_gib} GiB host RAM available"
}

build_kt() {
  check_host
  [[ -n "${KT_KERNEL_ROOT}" ]] \
    || fail "set KT_KERNEL_ROOT to the ktransformers/kt-kernel directory"
  [[ -f "${KT_KERNEL_ROOT}/setup.py" ]] \
    || fail "${KT_KERNEL_ROOT}/setup.py does not exist"

  export CPUINFER_CPU_INSTRUCT=NATIVE
  export CPUINFER_ENABLE_AMX=ON
  export CPUINFER_ENABLE_AVX512=ON
  export CPUINFER_ENABLE_AVX512_BF16=ON
  export CPUINFER_USE_CUDA=1
  export CPUINFER_CUDA_ARCHS="89;120"
  export CPUINFER_BUILD_TYPE=Release
  python3 -m pip install --no-build-isolation --no-deps -v "${KT_KERNEL_ROOT}"
  check_kt_runtime
}

serve() {
  check_host
  check_kt_runtime

  local numa_args=()
  local speculative_args=()
  if [[ -n "${KT_NUMA_NODES}" ]]; then
    local numa_nodes=()
    read -r -a numa_nodes <<<"${KT_NUMA_NODES}"
    numa_args=(--kt-numa-nodes "${numa_nodes[@]}")
  fi
  if [[ "${DISABLE_DSPARK}" != "1" ]]; then
    speculative_args=(
      --speculative-algorithm DSPARK
      --speculative-dspark-block-size "${DSPARK_BLOCK_SIZE}"
      --speculative-draft-device cuda:1
      --speculative-moe-runner-backend marlin
    )
  fi

  # The checkpoint advertises gamma=5. DSPARK_BLOCK_SIZE defaults to 3 here
  # because the public SM120 depth-5 correctness report is still open. Raise it
  # only after comparing deterministic outputs against a non-speculative run.
  exec python3 -m sglang.launch_server \
    --trust-remote-code \
    --model-path "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tp 1 \
    --moe-runner-backend flashinfer_mxfp4 \
    --enable-deepseek-v4-fp4-indexer \
    "${speculative_args[@]}" \
    --enable-decoder-swa-bounded-replay \
    --kt-weight-path "${KT_WEIGHT_PATH}" \
    --kt-method MXFP4 \
    --kt-mxfp4-backend amx \
    --kt-mxfp4-amx-min-tokens-per-expert "${KT_AMX_MIN_TOKENS_PER_EXPERT}" \
    --kt-gpu-prefill-token-threshold "${KT_GPU_PREFILL_TOKEN_THRESHOLD}" \
    --kt-mxfp4-prefill-slots "${KT_MXFP4_PREFILL_SLOTS}" \
    --kt-mxfp4-prefill-host-staging-experts "${KT_MXFP4_PREFILL_HOST_STAGING_EXPERTS}" \
    --kt-num-gpu-experts "${KT_NUM_GPU_EXPERTS}" \
    --kt-expert-placement-strategy uniform \
    --init-expert-location trivial \
    --kt-cpuinfer "${KT_CPU_THREADS}" \
    --kt-threadpool-count "${KT_THREADPOOL_COUNT}" \
    "${numa_args[@]}" \
    --disable-shared-experts-fusion \
    --weight-loader-drop-cache-after-load \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
    --context-length "${CONTEXT_LENGTH}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS_DECODE}" \
    --cuda-graph-backend-decode breakable \
    --cuda-graph-backend-prefill disabled \
    --swa-full-tokens-ratio 0.1 \
    --reasoning-parser deepseek-v41 \
    --tool-call-parser deepseekv41 \
    --watchdog-timeout 18000 \
    --host "${SGLANG_BIND_HOST}" \
    --port "${SGLANG_BIND_PORT}" \
    "$@"
}

case "${ACTION}" in
  check) check_host ;;
  check-kt) check_host; check_kt_runtime ;;
  build-kt) build_kt ;;
  serve) serve "$@" ;;
  *) fail "usage: $0 {check|check-kt|build-kt|serve} [additional sglang arguments]" ;;
esac
