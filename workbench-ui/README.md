# Workbench UI — Slice 0

This React/TypeScript client renders the scenario exported by the composable
Python model. It contains no performance formulas or GPU-specific topology.
Changing the injected Python implementation and re-exporting the scenario is
enough to replace the visual hardware model.

From the repository root:

```bash
make workbench-install
make workbench-dev
```

Then open the local URL printed by Vite. Double-click components marked
`Push in` to navigate from the system into the GPU and phase views.

To prove memory-model substitution:

```bash
make workbench-export-flat
make workbench-dev
```

Return to the hierarchical default with:

```bash
make workbench-export
```
