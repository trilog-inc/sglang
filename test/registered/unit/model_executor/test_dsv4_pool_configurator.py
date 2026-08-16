import unittest

from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSV4PoolConfigurator(unittest.TestCase):
    @staticmethod
    def _configurator(*, swa_ratio: float) -> DSV4PoolConfigurator:
        configurator = object.__new__(DSV4PoolConfigurator)
        configurator.swa_ratio = swa_ratio
        configurator.swa_page_size = 128
        configurator.c4_shrink_factor = 1
        configurator.c4_ring_size = 8
        return configurator

    def test_rejects_single_page_swa_pool_for_4k_smoke_profile(self):
        configurator = self._configurator(swa_ratio=0.1)

        with self.assertRaisesRegex(
            ValueError,
            r"DSV4 SWA pool \(256 tokens\).*minimum=768.*at least 0\.1875",
        ):
            configurator._compute_dsv4_sizes(full_token=4096, page_size=256)

    def test_20_percent_ratio_provides_three_page_admission_floor(self):
        configurator = self._configurator(swa_ratio=0.2)

        sizes = configurator._compute_dsv4_sizes(full_token=4096, page_size=256)

        self.assertEqual(sizes.full_max_total_num_tokens, 4096)
        self.assertEqual(sizes.swa_max_total_num_tokens, 768)


if __name__ == "__main__":
    unittest.main()
