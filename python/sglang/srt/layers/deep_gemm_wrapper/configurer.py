import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import get_device_sm, is_blackwell_supported

logger = logging.getLogger(__name__)


# Capabilities where the configured DeepGEMM build has native kernels.
# SM120 support requires a recent nv_dev/source build; the Docker source override
# pins the validated commit used for RTX PRO 6000 / GLM-5.2.
DEEPGEMM_CAPS = {(9, 0), (10, 0), (10, 3), (12, 0)}


def _compute_enable_deep_gemm():
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability() not in DEEPGEMM_CAPS:
        return False

    try:
        import deep_gemm  # noqa: F401
    except ImportError:
        return False

    return envs.SGLANG_ENABLE_JIT_DEEPGEMM.get()


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()

DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and is_blackwell_supported()
DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL
