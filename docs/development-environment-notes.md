# Development Environment Notes

## Hardware

- Machine: MacBook Pro (`MacBookPro14,3`)
- Processor: 2.8 GHz quad-core Intel Core i7
- CPU architecture: `x86_64`
- Memory: 16 GB
- Integrated graphics: Intel HD Graphics 630, up to 1536 MB dynamic VRAM
- Discrete graphics: AMD Radeon Pro 555, 2 GB VRAM
- Metal support: Yes
- NVIDIA GPU: No

## Operating System

- Operating system: macOS
- Version: 13.7.6
- Build: 22H625

## Installed Software

| Tool | Version | Status |
| --- | --- | --- |
| Python | 3.13.2 | Installed |
| Git | 2.45.2 | Installed |
| Docker Engine | 28.1.1 | Installed and verified |
| Docker Compose | 2.36.0 | Installed |
| VS Code | 1.121.0 (`x64`) | Installed |

## VS Code Extensions

- Python (`ms-python.python`)
- Pylance (`ms-python.vscode-pylance`)
- Docker (`ms-azuretools.vscode-docker`)
- Container Tools (`ms-azuretools.vscode-containers`)
- Jupyter (`ms-toolsai.jupyter`)
- Dev Containers (`ms-vscode-remote.remote-containers`)

GitHub Copilot is optional and is not currently installed.

## Verification

Docker Desktop was started and its engine was verified by successfully running
the official `hello-world` container.

Docker Desktop reported:

- Engine version: 28.1.1
- Architecture: `x86_64`
- Allocated CPUs: 8 logical CPUs
- Allocated memory: approximately 7.66 GiB

## Installation Issues

- Docker Desktop initially required its graphical application to be opened
  before the Docker engine became available.
- The VS Code command-line interface printed a macOS code-signing diagnostic,
  but the command completed normally and reported the installed version.

## Hardware Limitations

This machine does not have an NVIDIA GPU. CUDA, `nvidia-smi`, CUDA-enabled
PyTorch, and NVIDIA GPU inference cannot run locally.

Local work will include:

- Python development
- Git and GitHub workflows
- Docker and Docker Compose
- CPU-based tests
- Small CPU-based model inference

CUDA development, GPU profiling, vLLM, and GPU-intensive inference workloads
will run on remote Linux hardware with an NVIDIA GPU.

## Notes

- Both local GPUs support Apple's Metal graphics API.
- The discrete Radeon GPU has only 2 GB of VRAM and cannot be used as a CUDA
  device.
- Python 3.12 may be used for project environments when an inference library
  does not yet support Python 3.13.
