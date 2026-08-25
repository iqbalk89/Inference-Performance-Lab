# Animated Visual Solutions

These Manim animations turn each modeling problem into a narrated visual
derivation. They are designed to make shapes, data movement, and arithmetic
mechanics visible—not merely display the final answer.

## Fastest Way to Run

Prerequisite: Docker Desktop must be installed and show **Engine running**.

From this directory:

```bash
./render.sh 01
```

The first run downloads the pinned Manim image and can take several minutes.
Later runs reuse that image. The script then:

1. renders Problem 01 at medium quality;
2. writes the MP4 under `media/videos/`;
3. opens the video automatically on macOS.

No local Manim, LaTeX, Cairo, or FFmpeg installation is required. The workflow
uses the official `manimcommunity/manim:v0.21.0` Docker image.

## Commands

```bash
./render.sh 01                  # medium quality, then open the video
./render.sh 01 --low            # fast draft render
./render.sh 01 --high           # high-quality render
./render.sh 01 --no-open        # render without opening the MP4
./render.sh 01 --show-command   # print the Docker command without running it
```

## Current Animations

| Problem | Scene | Concepts visualized |
| --- | --- | --- |
| [01 — Transformer Projection Cost](../problems/01-transformer-projection-cost/problem.md) | `TransformerProjectionCost` | Matrix shapes, inner dimension, dot products, parameter count, HBM bytes, FLOPs, assumptions |

## Directory Structure

```text
visual-solutions/
├── README.md
├── manim.cfg
├── render.sh
├── scenes/
│   └── problem_01.py
└── media/                 # generated and ignored by Git
```

Generated video files are reproducible outputs and are not committed. The
source animation and pinned renderer version are the durable artifacts.

## Troubleshooting

### Docker is not running

Open Docker Desktop and wait until it says **Engine running**, then retry.

### First render appears stuck

The image is roughly 500 MB compressed. Check Docker Desktop's image/download
activity and allow the initial pull to finish.

### Video does not open automatically

Locate the path printed by `render.sh`, or run:

```bash
find media/videos -name 'TransformerProjectionCost.mp4'
```

### Render only a final frame while editing

Run the printed Docker command manually and add Manim's `-s` flag. The wrapper
keeps its options deliberately small so the standard path remains easy.

## Adding a Visual Solution

1. Add `scenes/problem_NN.py` with one clearly named Manim `Scene`.
2. Add its problem-to-scene mapping in `render.sh`.
3. Use consistent colors: blue for tensors, green for compute, orange for HBM,
   purple for shapes, and yellow for conclusions.
4. Show the operation before showing the formula.
5. Keep units attached to values throughout the derivation.
6. End with assumptions and exclusions, not only the final number.

