# Security Policy

Being straight about what this is: morphogenesis-ga is a **research sandbox** —
a from-scratch neuroevolution experiment that trains tiny models on a local
grid. There is no server, no network service and no user data. It runs on your
own machine against configs you control, so the realistic security surface is
small. This file is mainly about who to tell if something matters anyway.

## What could still go wrong

- **Dependencies.** The project pulls a normal PyTorch stack (`pyproject.toml`).
  A vulnerability in one of those is worth reporting even though it is not my
  code.
- **Untrusted config or checkpoint files.** Training is driven by YAML configs
  and loads `.pt` checkpoints. Loading a checkpoint from someone you do not
  trust is the usual PyTorch caveat — do not do it, and if you find a way the
  loader could be abused beyond the known `torch.load` behaviour, tell me.
- **Experiment tracking.** The optional MLflow and TensorBoard integrations
  write locally by default. If a code path sends run data somewhere off-machine
  without you asking, that is a bug.

## Reporting

Email **9637843238max@gmail.com** with what you found and how to reproduce it.
For anything that could harm a user before it is fixed, please email rather than
open a public issue.

I maintain this in my own time, so I cannot promise a schedule — but I will
acknowledge a real report within a few days and act on it.

## Not in scope

Bugs in the science — the fitness function, the GA, the reproducibility anchors
— are welcome, but they belong in a regular issue, not here. See
`CONTRIBUTING.md`. This file is only for security-relevant problems.
