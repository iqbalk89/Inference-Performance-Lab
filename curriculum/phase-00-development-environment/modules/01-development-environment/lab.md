# Phase 0 Module 01 Lab — Environment Acceptance

## Objective

Verify that the local workstation and repository can support development while
GPU execution remains isolated to a remote NVIDIA environment.

## Procedure

1. Record operating system, architecture, Python, Git, Docker, Compose, and VS
   Code versions.
2. Verify the Python, Pylance, Docker, Jupyter, and Dev Containers extensions.
3. Create and activate a virtual environment, then confirm its interpreter is
   isolated from the system interpreter.
4. Run the Docker `hello-world` container.
5. Confirm the repository has a clean Git status and a working GitHub remote.
6. Verify `.gitignore` excludes virtual environments, caches, model artifacts,
   benchmark output, and raw profiler results.
7. Document why CUDA cannot run on this Intel Mac and how remote GPU work will
   be performed.

## Evidence

- `docs/development-environment-notes.md`
- `docs/learning-journal.md`
- Repository structure and Git history

## Pass Criteria

- All local commands complete successfully.
- No secrets or private SSH keys are tracked.
- Documentation reports actual rather than assumed versions.
- The repository can be cloned and understood by another engineer.
