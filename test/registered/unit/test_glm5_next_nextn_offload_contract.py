import ast
import copy
import dataclasses
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
NEXTN_MODEL_PATH = ROOT / "python/sglang/srt/models/glm5_next_nextn.py"
ATTENTION_REGISTRY_PATH = (
    ROOT / "python/sglang/srt/layers/attention/attention_registry.py"
)
EAGLE_WORKER_PATH = ROOT / "python/sglang/srt/speculative/eagle_worker.py"


def _compile_nextn_helper(name):
    tree = ast.parse(NEXTN_MODEL_PATH.read_text(encoding="utf-8"))
    function = copy.deepcopy(
        next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        )
    )
    function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {
        "__builtins__": __builtins__,
        "copy": copy,
        "Glm5NextTextConfig": object,
        "QuantizationConfig": object,
        "Optional": __import__("typing").Optional,
        "_NEXTN_SPECIFIC_WEIGHT_NAMES": (
            "shared_head.norm",
            "eh_proj",
            "enorm",
            "hnorm",
        ),
    }
    exec(compile(module, str(NEXTN_MODEL_PATH), "exec"), namespace)
    return namespace[name]


def _compile_eagle_worker_method(name, globals_):
    tree = ast.parse(EAGLE_WORKER_PATH.read_text(encoding="utf-8"))
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EAGLEWorker"
    )
    method = copy.deepcopy(
        next(
            node
            for node in worker.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
    )
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {"__builtins__": __builtins__, **globals_}
    exec(compile(module, str(EAGLE_WORKER_PATH), "exec"), namespace)
    return namespace[name]


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
    def test_remote_spec_copy_includes_dynamic_position_tensor(self):
        class FakeTensor:
            def __init__(self, name):
                self.name = name

            def to(self, *, device, non_blocking):
                self.last_copy = (device, non_blocking)
                return FakeTensor(f"{self.name}@{device}")

        @dataclasses.dataclass
        class DraftInput:
            hidden_states: FakeTensor
            seq_lens_cpu: FakeTensor

        fake_torch = types.SimpleNamespace(Tensor=FakeTensor)
        copy_spec = _compile_eagle_worker_method(
            "_copy_spec_to_device",
            {"copy": copy, "dataclasses": dataclasses, "torch": fake_torch},
        )
        cpu_metadata = FakeTensor("cpu-metadata")
        original = DraftInput(FakeTensor("hidden"), cpu_metadata)
        original.positions = FakeTensor("positions")

        copied = copy_spec(original, "cuda:1")

        self.assertIsNot(copied, original)
        self.assertEqual(copied.hidden_states.name, "hidden@cuda:1")
        self.assertEqual(copied.positions.name, "positions@cuda:1")
        self.assertIs(copied.seq_lens_cpu, cpu_metadata)

    def test_fp8_exclusions_follow_compact_nextn_namespace(self):
        remap = _compile_nextn_helper("_remap_nextn_quant_config")
        config = types.SimpleNamespace(num_hidden_layers=45)
        original = types.SimpleNamespace(
            ignored_layers=[
                "lm_head",
                "model.layers.45.eh_proj",
                "model.layers.45.self_attn.indexer.wq_b",
                "model.layers.45.self_attn.kv_b_proj",
                "model.layers.44.self_attn.o_proj",
            ]
        )

        draft = remap(config, original)

        self.assertIsNot(draft, original)
        self.assertEqual(len(original.ignored_layers), 5)
        self.assertIn("model.eh_proj", draft.ignored_layers)
        self.assertIn(
            "model.decoder.self_attn.indexer.wq_b", draft.ignored_layers
        )
        self.assertIn(
            "model.decoder.self_attn.kv_b_proj", draft.ignored_layers
        )
        self.assertNotIn(
            "model.decoder.self_attn.o_proj", draft.ignored_layers
        )

    def test_shared_draft_head_is_not_allocated_as_fp8(self):
        tree = ast.parse(NEXTN_MODEL_PATH.read_text(encoding="utf-8"))
        head_call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ParallelLMHead"
        )
        quant_keyword = next(
            keyword
            for keyword in head_call.keywords
            if keyword.arg == "quant_config"
        )
        self.assertIsInstance(quant_keyword.value, ast.Constant)
        self.assertIsNone(quant_keyword.value.value)

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

    def test_draft_runner_uses_native_dsa_backend_without_kda_sidecar(self):
        tree = ast.parse(ATTENTION_REGISTRY_PATH.read_text(encoding="utf-8"))
        wrapper_node = copy.deepcopy(
            next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "attn_backend_wrapper"
            )
        )
        wrapper_module = ast.fix_missing_locations(
            ast.Module(body=[wrapper_node], type_ignores=[])
        )
        namespace = {"__builtins__": __builtins__}
        exec(
            compile(wrapper_module, str(ATTENTION_REGISTRY_PATH), "exec"),
            namespace,
        )

        class NativeSparseAttnBackend:
            pass

        model_config_module = types.ModuleType(
            "sglang.srt.configs.model_config"
        )
        model_config_module.is_minimax_sparse = lambda _config: False
        nsa_backend_module = types.ModuleType(
            "sglang.srt.layers.attention.nsa_backend"
        )
        nsa_backend_module.NativeSparseAttnBackend = NativeSparseAttnBackend
        runner = types.SimpleNamespace(
            model_config=types.SimpleNamespace(
                hf_config=object(), is_glm5_next=True
            ),
            is_draft_worker=True,
            hybrid_gdn_config=None,
            use_mla_backend=True,
        )
        backend = NativeSparseAttnBackend()

        with patch.dict(
            sys.modules,
            {
                model_config_module.__name__: model_config_module,
                nsa_backend_module.__name__: nsa_backend_module,
            },
        ):
            result = namespace["attn_backend_wrapper"](runner, backend)

        self.assertIs(result, backend)

        glm_branch = next(
            node
            for node in wrapper_node.body
            if isinstance(node, ast.If)
            and "is_glm5_next" in ast.unparse(node.test)
        )
        draft_guard = next(
            node
            for node in glm_branch.body
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "runner.is_draft_worker"
        )
        self.assertEqual(len(draft_guard.body), 1)
        self.assertIsInstance(draft_guard.body[0], ast.Return)
        self.assertEqual(
            ast.unparse(draft_guard.body[0].value), "full_attn_backend"
        )

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

    def test_target_verification_commits_glm_linear_state(self):
        source = EAGLE_WORKER_PATH.read_text(encoding="utf-8")
        self.assertIn("model_runner.glm5_next_linear_config is not None", source)
        self.assertIn("self._mamba_verify_update(", source)

    def test_target_verify_kpool_transaction_remains_eager(self):
        source = (ROOT / "python/sglang/srt/server_args.py").read_text()
        self.assertIn("KPool target verification currently", source)
        self.assertIn("self.disable_cuda_graph = True", source)


if __name__ == "__main__":
    unittest.main()
