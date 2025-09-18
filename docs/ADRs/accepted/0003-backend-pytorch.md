# ADR 0003 — Select PyTorch for Training and Inference

- Status: Accepted
- Date: 2025-09-18

## Context
The project initially experimented across frameworks. Maintaining multiple backends adds complexity and increases variance in results and dependencies.

## Decision
Standardize on PyTorch for the neural network backend.

## Rationale
- Simpler codebase and dependency set.
- Strong ecosystem and mature tooling.
- Works well with our small MLP and custom training loop.

## Consequences
- Single backend to maintain and test.
- All training/inference artifacts use the Torch state dict format (`.pt`).

## Alternatives Considered
- TensorFlow/Keras: familiar API but adds extra dep weight; not needed for current scope.

