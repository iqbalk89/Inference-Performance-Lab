# Lambda GPU Instance Runbook

This is the repeatable start-to-finish workflow for Phase 1 work on a Lambda
Cloud A10 instance. The bootstrap is deliberately manual: it runs after the
repository is cloned and does not execute automatically when the VM boots.

## What the Bootstrap Covers

`scripts/bootstrap_lambda_gpu.sh` provides the verified base GPU environment:

- Confirms Ubuntu 22.04, x86_64, NVIDIA visibility, and CUDA-enabled PyTorch.
- Installs Nsight Compute.
- Installs the working NVIDIA Nsight Systems CLI version, avoiding Lambda's
  previously observed report-importer mismatch.
- Optionally validates PyTorch, CPU sampling, scheduler events, CUDA tracing,
  GPU Metrics, and Nsight Compute counters.

It does not install models, inference servers, project-specific Python
dependencies, or create a virtual environment. Those will be added as explicit,
versioned layers when Phase 1 selects its first model and service. Lambda Stack's
system Python already supplies the CUDA-enabled PyTorch used by this check.

## 1. Before Launch

- Confirm the local repository is committed and pushed.
- Confirm the SSH private key exists at `~/.ssh/lambda_phase1`.
- Check Lambda Usage and set a spending alert if available.
- Never commit the private key or a temporary public IP.

## 2. Launch

In Lambda On-Demand Cloud, select:

- One NVIDIA A10 24 GB instance for Phase 1 foundation work.
- The Lambda Stack Ubuntu 22.04 image.
- The previously uploaded `lambda_phase1` public SSH key.

Wait until the instance status is **Running**, then copy its temporary IP. A VM
is billable while running, even if no SSH session is open.

## 3. Connect and Clone

Replace `<IP>` with the current instance address:

```bash
ssh -i ~/.ssh/lambda_phase1 ubuntu@<IP>
git clone https://github.com/iqbalk89/Inference-Performance-Lab.git
cd Inference-Performance-Lab
```

On a reused instance, update instead:

```bash
cd ~/Inference-Performance-Lab
git pull --ff-only
```

## 4. Bootstrap and Validate

Install and check the base tools:

```bash
./scripts/bootstrap_lambda_gpu.sh
```

On a new image, or after changing the bootstrap, run full acceptance:

```bash
./scripts/bootstrap_lambda_gpu.sh --validate
```

The validation takes several minutes and creates timestamped raw reports under
`profile-results/`. That directory is ignored by Git because profiler reports
are large and machine-specific. The command may request the `ubuntu` account's
sudo authorization. CPU sampling needs sudo because the image sets
`kernel.perf_event_paranoid=4`.

The bootstrap is idempotent: it is safe to rerun. Use `--help` to see its modes.

## 5. Work Normally

At the beginning of each session:

```bash
cd ~/Inference-Performance-Lab
git pull --ff-only
nvidia-smi
```

Keep source code and summarized findings in Git. Keep model weights, raw traces,
and other large generated artifacts out of Git unless the repository later
adopts explicit artifact storage.

For VS Code, install **Remote - SSH** locally, connect to
`ubuntu@<IP>` using `~/.ssh/lambda_phase1`, and open
`/home/ubuntu/Inference-Performance-Lab`.

## 6. Preserve Results

Commit and push reproducible code and Markdown summaries from the VM. Copy raw
artifacts to the Mac before terminating the instance when they are worth
keeping:

```bash
scp -i ~/.ssh/lambda_phase1 -r \
  ubuntu@<IP>:/home/ubuntu/Inference-Performance-Lab/profile-results/<run-directory> \
  /Users/Kaiser/Inference-Performance-Lab/profile-results/
```

Run that `scp` command in a local Mac terminal, not inside the VM.

## 7. End Every Session

1. Push all wanted code and documentation.
2. Copy any wanted ignored artifacts to the Mac.
3. In Lambda's dashboard, terminate the instance.
4. Confirm that no instance remains in Running or Booting state.
5. Check Usage for the final charge and delete paid persistent storage if any
   was created intentionally.

Closing SSH, closing the laptop, or stopping a program does **not** stop billing.
An ordinary terminated VM's local disk is ephemeral, so treat unpushed work as
disposable.

## Troubleshooting

- **SSH times out:** confirm the VM is Running, use its current IP, and check
  that the uploaded public key matches `~/.ssh/lambda_phase1.pub`.
- **Host-key warning after an IP is reused:** verify the IP in Lambda first,
  then remove only that IP's stale entry with `ssh-keygen -R <IP>`.
- **Nsight importer reports `LIBSSH_4_9_0 not found`:** rerun the bootstrap;
  the NVIDIA CLI package it installs is the verified fix.
- **CPU sampling permission failure:** collect the trace with `sudo nsys`; do
  not permanently weaken the kernel security setting.
- **CUDA unavailable in a new virtual environment:** the system PyTorch belongs
  to Lambda Stack. Do not install an arbitrary CPU-only PyTorch wheel; use the
  project's future pinned environment instructions.

The measured acceptance results for the current image are recorded in
[Lambda A10 Environment Acceptance](lambda-a10-acceptance.md).
