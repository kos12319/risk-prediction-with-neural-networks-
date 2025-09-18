# ADR 0009 — Optional Calibration Post‑Training

- Status: Proposed
- Date: 2025-09-18

## Context
Probability outputs from NNs can be miscalibrated, especially under class imbalance or with ROS.

## Proposal
Add an optional calibration step (Platt scaling or isotonic regression) fitted on validation scores/labels; report both calibrated and uncalibrated metrics.

## Rationale
- Improves decision quality when thresholds target specific operating points.
- Makes probability outputs more interpretable and actionable.

## Consequences
- Slight complexity and runtime increase.
- Must avoid leakage (fit on validation only).

## Alternatives Considered
- No calibration: acceptable baseline but may misstate risk probabilities.

