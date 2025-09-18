# ADR 0105 — Store Large Data and Run Artifacts in Git

- Status: Rejected
- Date: 2025-09-18

## Context
Large CSVs and per‑run artifacts bloat the repo and slow operations.

## Decision
Use Git LFS for original archives only; ignore unzipped full datasets and all local run folders.

## Rationale
- Keeps the repository lean and clone‑friendly; avoids accidental large commits.

## Alternatives Considered
- Commit everything: unacceptable repository growth and poor DX.

