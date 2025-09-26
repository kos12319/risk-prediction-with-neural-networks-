# Thesis Iteration 2

- Narrative: `THESIS_ITERATION_2.html`
- Proposal: `THESIS_PROPOSAL_ITERATION_2.md`
- Artifacts: `THESIS_ITERATION_2.pdf` (snapshot PDF)
- Focus: Neural-centric refinement with temporal CV, embeddings for categorical features, monotonic priors on known risk drivers, strong regularization, and calibration. Compares against Iteration 1 H2O AutoML baselines.
- Dataset scope: excludes the 1k subset due to small-sample overfitting and unstable thresholds; focuses on 10k, 100k, and full for robust, reproducible estimates.
- Additional exploration figures are linked relatively from `../../exploration/figures/`.

Upstream experiment reports remain in `docs/experiments/suites/thesis_iter1/`.

## Build Instructions (HTML/PDF with Bibliography)

Source of truth: `THESIS_ITERATION_2.md`. Do not edit the HTML/PDF directly; regenerate them from Markdown.

Requirements (local builds):
- `pandoc` (>= 3.x) and a LaTeX engine for PDF (e.g., `tectonic` or `xelatex`).
- CSL style and bibliographies are already in this repo.

Recommended setup (macOS/Homebrew):
- `brew install pandoc tectonic`

Paths:
- CSL: `../../csl/apa.csl`
- Bibliographies:
  - `../../bibliography/credit_risk_neural_networks_research_papers.bib`
  - `../../bibliography/feature_selection_transformer_credit_risk_papers.bib`
  - `../../bibliography/lendingclub_subtopics_white.bib`
  - `../../bibliography/lendingclub_subtopics_grey.bib`

Build HTML from Markdown (run from repo root):
```bash
pandoc docs/thesis/iteration-2/THESIS_ITERATION_2.md \
  --from markdown \
  --citeproc \
  --resource-path=.:docs:docs/thesis/iteration-2:docs/thesis/iteration-2/reports:docs/exploration \
  -o docs/thesis/iteration-2/THESIS_ITERATION_2.html
```

Build PDF from Markdown (run from repo root):
```bash
pandoc docs/thesis/iteration-2/THESIS_ITERATION_2.md \
  --from markdown \
  --citeproc \
  --pdf-engine=xelatex \
  --resource-path=.:docs:docs/thesis/iteration-2:docs/thesis/iteration-2/reports:docs/exploration \
  -V geometry:margin=1in -V fontsize=11pt \
  -o docs/thesis/iteration-2/THESIS_ITERATION_2.pdf
```

If only HTML is available, you can regenerate a PDF (without re-running citeproc) with:
```bash
pandoc THESIS_ITERATION_2.html -o THESIS_ITERATION_2.pdf
```
