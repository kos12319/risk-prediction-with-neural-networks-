# Platform Architecture Docs — Canon and Journal

This directory hosts a lightweight, Makefile-first system to keep one place up to date with all decisions and changes while the platform evolves.

- Platform Canon (generated): `docs/architecture/PLATFORM_SPEC.md`
  - Compiled from ADRs (accepted + proposed) and the Journal (reverse-chronological changes).
  - Regenerate via: `make docs-canon`.

- Journal (source of truth for changes): `docs/architecture/journal/`
  - Add entries with `make docs-journal-new TITLE="..." [TAGS="data,eval"] [ADRS="0001,0004"]`.
  - Each entry is a small, dated Markdown file that links to relevant ADRs, configs, and Make targets.

Why this setup
- ADRs capture the “why” (rationale, trade-offs, status).
- Journal captures the “what changed” over time with enough context to reconstruct a full narrative.
- The Canon stitches these into a single, readable document when needed.

Conventions
- Keep entries concise; link to code paths and configs rather than duplicating.
- Reference ADR IDs (e.g., 0001, 0004) so the Canon can surface both decision and change context together.
- Prefer Makefile targets in examples (Makefile-first policy).

Usage
- New change: `make docs-journal-new TITLE="Switch to time split for selection" TAGS="eval,cv" ADRS="0001,0002"`
- Rebuild Canon: `make docs-canon`

