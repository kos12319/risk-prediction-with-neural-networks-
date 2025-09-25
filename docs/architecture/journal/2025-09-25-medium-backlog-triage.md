# Medium backlog triage

- Date: 2025-09-25
- Status: done
- Tags: architecture, planning

## Summary
Reviewed the refreshed backend architecture and future extensions to surface the highest-value medium-term work. Promoted the config validation guardrails, temporal CV orchestrator, and run catalog manifest into the medium-priority queue and documented the follow-ups across pain points and the roadmap.

## Impact
- docs: docs/PAIN_POINTS.md, docs/architecture/FUTURE_EXTENSIONS.md, README.md, AGENTS.md, GEMINI.md

## Next
- Design schema-based config validation (Pydantic or dataclasses) that enforces backend-specific invariants before training runs start.
- Prototype a backend-agnostic temporal CV runner that reuses the shared pipeline and exposes Make targets for PyTorch and H2O.
- Draft the run catalog manifest format (JSON + Markdown summary) so future dashboards can rely on a stable contract.
