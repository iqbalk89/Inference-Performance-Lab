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


def component_ids(scenario: dict, graph_id: str) -> set[str]:
    return {
        component["component_id"]
        for component in scenario["diagrams"][graph_id]["components"]
    }


class SliceZeroTests(unittest.TestCase):
    def test_default_scenario_exposes_complete_drilldown(self) -> None:
        scenario = build_slice_zero_scenario().to_dict()

        self.assertEqual(scenario["initial_graph_id"], "system")
        self.assertEqual(
            set(scenario["diagrams"]),
            {"system", "gpu-0-detail", "prefill-detail", "decode-detail"},
        )
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
            "prefill", "Custom Prefill", 128, "Injected workload phase."
        )
        scenario = build_slice_zero_scenario(
            prefill_model=custom_prefill
        ).to_dict()

        graph = scenario["diagrams"]["prefill-detail"]
        self.assertEqual(graph["title"], "Custom Prefill: Problem 02 projection")
        self.assertIn("X [128 × 4096]", {item["label"] for item in graph["components"]})


if __name__ == "__main__":
    unittest.main()
