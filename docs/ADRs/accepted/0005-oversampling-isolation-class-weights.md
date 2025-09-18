# ADR 0005 — Oversampling Isolation and Class Weights

- Status: Accepted
- Date: 2025-09-18

## Context
Random over‑sampling (ROS) of the entire training set before creating validation can leak duplicated samples into validation, inflating signal and harming early stopping. For NNs, class weighting or focal loss is often preferable to heavy ROS.

## Decision
Split training into `(train_sub, val_sub)` deterministically first. Fit preprocessing on `train_sub` only. Apply any resampling strictly to `train_sub`. Support class weights as a first‑class option.

## Rationale
- Prevents leakage and improves early stopping reliability.
- Reduces prior distortion and calibration drift vs heavy ROS.

## Consequences
- Requires slight refactor in training orchestration (done).
- Config supports `training.class_weight` in addition to ROS.

## Alternatives Considered
- ROS before validation split: rejected due to leakage risk.

