import unittest

from inference_workbench.contracts import (
    AcceleratorModel,
    Component,
    ComponentKind,
    Diagram,
    MemoryModel,
    Position,
)
from inference_workbench.scenario import build_slice_zero_scenario, memory_variants
from inference_workbench.workloads import ProjectionPhaseModel
from inference_workbench.estimates import problem_02_estimate, projection_calculations


def component_ids(scenario: dict, graph_id: str) -> set[str]:
    return {
        component["component_id"]
        for component in scenario["diagrams"][graph_id]["components"]
    }


class SliceZeroTests(unittest.TestCase):
    def test_problem_02_estimates_match_answer_sheet(self) -> None:
        decode = problem_02_estimate(1)
        prefill = problem_02_estimate(512)

        self.assertEqual(decode.flops, 33_554_432)
        self.assertEqual(decode.total_hbm_bytes, 33_570_816)
        self.assertAlmostEqual(decode.arithmetic_intensity, 0.9995119571)
        self.assertAlmostEqual(decode.lower_bound_us, 55.95136)
        self.assertEqual(decode.bottleneck, "HBM bandwidth")

        self.assertEqual(prefill.flops, 17_179_869_184)
        self.assertEqual(prefill.total_hbm_bytes, 41_943_040)
        self.assertEqual(prefill.arithmetic_intensity, 409.6)
        self.assertAlmostEqual(prefill.lower_bound_us, 143.1655765)
        self.assertEqual(prefill.bottleneck, "FP16 compute")

        calculations = projection_calculations(prefill)
        expected = {
            "work", "weight_bytes", "input_bytes", "output_bytes", "total_bytes",
            "arithmetic_intensity", "compute_time", "memory_time", "lower_bound",
            "ridge_point", "bottleneck",
        }
        self.assertEqual(set(calculations), expected)
        for calculation in calculations.values():
            self.assertTrue(calculation.concept)
            self.assertTrue(calculation.formula)
            self.assertTrue(calculation.inputs)
            self.assertTrue(calculation.steps)
            self.assertTrue(calculation.unit_analysis)
            self.assertTrue(calculation.interpretation)

    def test_default_scenario_exposes_complete_drilldown(self) -> None:
        scenario = build_slice_zero_scenario().to_dict()

        self.assertEqual(scenario["initial_graph_id"], "system")
        self.assertEqual(set(scenario["diagrams"]), {
            "system", "gpu-0-detail",
            "prefill-detail", "prefill-hbm-boundary", "prefill-execution-path",
            "decode-detail", "decode-hbm-boundary", "decode-execution-path",
        })
        self.assertLessEqual(
            {"client", "server", "cpu", "host-memory", "host-link", "gpu-0"},
            component_ids(scenario, "system"),
        )
        self.assertLessEqual(
            {"hbm", "l2", "sm-array", "prefill", "decode"},
            component_ids(scenario, "gpu-0-detail"),
        )

    def test_memory_implementation_is_swappable_without_changing_consumers(self) -> None:
        hierarchical = build_slice_zero_scenario(memory_variant="hierarchical").to_dict()
        flat = build_slice_zero_scenario(memory_variant="flat").to_dict()

        self.assertIn("hbm", component_ids(hierarchical, "gpu-0-detail"))
        self.assertIn("device-memory", component_ids(flat, "gpu-0-detail"))
        self.assertIn("prefill", component_ids(flat, "gpu-0-detail"))
        self.assertIn("decode", component_ids(flat, "gpu-0-detail"))
        self.assertEqual(hierarchical["metadata"]["memory_variant"], "hierarchical")
        self.assertEqual(flat["metadata"]["memory_variant"], "flat")

    def test_registry_accepts_a_new_memory_model(self) -> None:
        class TestMemory(MemoryModel):
            def components(self):
                return ()

            def connections(self):
                return ()

        registry_name = "test-empty"
        if registry_name not in memory_variants.variants:
            memory_variants.register(registry_name, TestMemory)

        self.assertIsInstance(memory_variants.create(registry_name), TestMemory)

    def test_accelerator_can_be_replaced_at_the_composition_root(self) -> None:
        class TestAccelerator(AcceleratorModel):
            def system_component(self):
                return Component(
                    "test-accelerator", "Injected Accelerator", ComponentKind.ACCELERATOR,
                    "Test replacement.", Position(1010, 260),
                    drilldown_graph_id="test-accelerator-detail",
                )

            def diagram(self):
                return Diagram(
                    "test-accelerator-detail", "Injected Accelerator", "Replacement graph.",
                    "system", (), (),
                )

        scenario = build_slice_zero_scenario(
            accelerator_model=TestAccelerator()
        ).to_dict()

        self.assertIn("test-accelerator", component_ids(scenario, "system"))
        self.assertIn("test-accelerator-detail", scenario["diagrams"])
        system_edges = scenario["diagrams"]["system"]["connections"]
        self.assertTrue(any(edge["target_id"] == "test-accelerator" for edge in system_edges))

    def test_phase_models_are_injectable(self) -> None:
        custom_prefill = ProjectionPhaseModel(
            "prefill", "Custom Prefill", 128, "Injected workload phase.",
            problem_02_estimate(128),
        )
        scenario = build_slice_zero_scenario(
            prefill_model=custom_prefill
        ).to_dict()

        graph = scenario["diagrams"]["prefill-detail"]
        self.assertEqual(graph["title"], "Custom Prefill: Problem 02 projection")
        self.assertIn("X [128 × 4096]", {item["label"] for item in graph["components"]})

    def test_projection_uses_progressive_reusable_views_and_output_writeback(self) -> None:
        scenario = build_slice_zero_scenario().to_dict()
        operator = scenario["diagrams"]["prefill-detail"]
        boundary = scenario["diagrams"]["prefill-hbm-boundary"]
        execution = scenario["diagrams"]["prefill-execution-path"]
        components = {item["component_id"]: item for item in operator["components"]}
        connections = {item["connection_id"]: item for item in boundary["connections"]}

        self.assertEqual(components["prefill-input"]["kind"], "tensor")
        self.assertEqual(components["prefill-weight"]["kind"], "tensor")
        self.assertEqual(components["prefill-matmul"]["kind"], "operation")
        self.assertEqual(components["prefill-matmul"]["drilldown_graph_id"], "prefill-hbm-boundary")

        output_write = connections["prefill-y-boundary"]
        self.assertEqual(output_write["source_id"], "prefill-output-write")
        self.assertEqual(output_write["target_id"], "prefill-boundary-hbm")
        self.assertEqual(output_write["category"], "transfer")
        self.assertIn("GB/s", output_write["badge"])
        self.assertTrue(
            all(metric["calculation"] for metric in output_write["metrics"] if metric["name"] != "Assumed HBM rate")
        )

        self.assertEqual({edge["category"] for edge in operator["connections"]}, {"logical"})
        self.assertEqual({edge["category"] for edge in boundary["connections"]}, {"transfer", "mapping"})
        self.assertEqual({edge["category"] for edge in execution["connections"]}, {"physical"})
        execution_components = {item["component_id"]: item for item in execution["components"]}
        self.assertIn("prefill-physical-l2", execution_components)
        self.assertIn("Unknown—measure or calibrate", {metric["value"] for metric in execution_components["prefill-physical-l2"]["metrics"]})


if __name__ == "__main__":
    unittest.main()
