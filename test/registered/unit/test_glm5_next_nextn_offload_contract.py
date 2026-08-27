import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]


class _FakeCuda:
    def __init__(self):
        self.properties = [
            types.SimpleNamespace(uuid="GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
            types.SimpleNamespace(uuid="GPU-11111111-2222-3333-4444-555555555555"),
        ]

    def device_count(self):
        return len(self.properties)

    def get_device_properties(self, index):
        return self.properties[index]


def _load_draft_device_module():
    fake_torch = types.SimpleNamespace(cuda=_FakeCuda())
    module_path = ROOT / "python/sglang/srt/speculative/draft_device.py"
    spec = importlib.util.spec_from_file_location(
        "_glm5_next_test_draft_device", module_path
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"torch": fake_torch}):
        spec.loader.exec_module(module)
    return module


class TestGlm5NextDraftDevice(unittest.TestCase):
    def test_resolves_logical_indices_and_cuda_prefix(self):
        module = _load_draft_device_module()
        self.assertEqual(module.resolve_speculative_draft_device("1"), 1)
        self.assertEqual(module.resolve_speculative_draft_device("cuda:0"), 0)

    def test_resolves_case_and_dash_insensitive_cuda_uuid(self):
        module = _load_draft_device_module()
        self.assertEqual(
            module.resolve_speculative_draft_device("11111111222233334444555555555555"),
            1,
        )

    def test_rejects_invisible_index_and_uuid(self):
        module = _load_draft_device_module()
        with self.assertRaisesRegex(ValueError, "only 2 CUDA devices"):
            module.resolve_speculative_draft_device("cuda:2")
        with self.assertRaisesRegex(ValueError, "Could not resolve"):
            module.resolve_speculative_draft_device("GPU-deadbeef")


class TestGlm5NextNextNSourceBoundary(unittest.TestCase):
    def test_remote_worker_has_explicit_state_transport_boundary(self):
        source = (ROOT / "python/sglang/srt/speculative/eagle_worker.py").read_text()
        for required in (
            "_sync_remote_req_to_token",
            "_remote_owner_for_request",
            "_model_worker_batch_to_draft",
            "_copy_draft_capture_to_target",
            "_handoff_draft_to_target",
            "remote MTP request mapping is smaller than the target",
            "FLASHINFER_CUDA_ARCH_LIST",
            "Draft CUDA graphs are disabled for heterogeneous MTP",
            "prepare_kpool_request",
        ):
            self.assertIn(required, source)

    def test_draft_runner_does_not_allocate_target_kda_states(self):
        source = (ROOT / "python/sglang/srt/model_executor/model_runner.py").read_text()
        self.assertIn("and not self.is_draft_worker", source)

        pool_source = (
            ROOT / "python/sglang/srt/mem_cache/glm5_next_memory_pool.py"
        ).read_text()
        self.assertIn("if model_runner.is_draft_worker:", pool_source)
        self.assertIn("layer_num=1", pool_source)

    def test_nextn_model_uses_checkpoint_appended_decoder_contract(self):
        source = (ROOT / "python/sglang/srt/models/glm5_next_nextn.py").read_text()
        self.assertIn("is_nextn=True", source)
        self.assertIn("DeepseekV2WeightLoaderMixin.do_load_weights", source)
        self.assertIn("EntryClass = [Glm5NextForConditionalGenerationNextN]", source)

    def test_target_verification_commits_only_the_accepted_kpool_prefix(self):
        source = (ROOT / "python/sglang/srt/speculative/eagle_info.py").read_text()
        self.assertIn("commit_speculative_kpool", source)
        self.assertIn("accepted_length + 1", source)
        self.assertIn("supports topk=1 only", source)

    def test_target_verify_kpool_transaction_remains_eager(self):
        source = (ROOT / "python/sglang/srt/server_args.py").read_text()
        self.assertIn("KPool target verification currently", source)
        self.assertIn("self.disable_cuda_graph = True", source)


if __name__ == "__main__":
    unittest.main()
