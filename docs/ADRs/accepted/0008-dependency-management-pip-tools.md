# ADR 0008 — Dependency Management via pip‑tools

- Status: Accepted
- Date: 2025-09-18

## Context
Reproducible environments require pinning yet convenient updates.

## Decision
Use a two‑file setup with pip‑tools: human‑edited `requirements.in` and compiled `requirements.txt` (fully pinned), synced via Make targets.

## Rationale
- Clear separation of intent vs lock.
- Easy to update while keeping deterministic installs.

## Consequences
- Contributors edit `requirements.in` and run `make deps-compile` + `make deps-sync`.

## Alternatives Considered
- Hand‑editing a pinned requirements.txt: error‑prone and inconsistent.

