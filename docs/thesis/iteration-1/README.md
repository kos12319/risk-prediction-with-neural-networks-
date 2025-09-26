# Thesis Iteration 1

- Narrative: `THESIS_ITERATION_1.md`
- Proposal: `THESIS_PROPOSAL_ITERATION_1.md`
- Artifacts: `THESIS_ITERATION_1.pdf` (snapshot PDF)
- Focus: Feature-set optimization with H2O AutoML across four regimes (agnostic, selected, aware, selected+providers) and four dataset sizes (1k, 10k, 100k, full).
- Key figures (copied snapshot from experiments suite):
  - `reports/aupr_roc_winners_by_size.svg`
  - `reports/1k/figures/aupr_by_feature_set.svg`
  - `reports/10k/figures/aupr_by_feature_set.svg`
  - `reports/100k/figures/aupr_by_feature_set.svg`
  - `reports/full/figures/aupr_by_feature_set.svg`
  - `reports/full/figures/pr_curve.png`
  - `reports/full/figures/roc_curve.png`
  - `reports/full/figures/h2o_varimp_heatmap_winners.png`
  - `reports/full/figures/h2o_leaderboard_pr.png`
  - `reports/full/figures/h2o_leaderboard_roc.png`

Upstream experiment configs and runs remain in `docs/experiments/suites/thesis_iter1/`.

## Build Instructions (HTML/PDF with Bibliography)

Source of truth: `THESIS_ITERATION_1.md`. Do not edit the HTML/PDF directly; regenerate them from Markdown.

Requirements (local builds):
- `pandoc` (>= 3.x) and a LaTeX engine for PDF (e.g., `tectonic` or `xelatex`).
- CSL style and bibliographies are already in this repo.

Recommended setup (macOS/Homebrew):
- `brew install pandoc tectonic`  # PDF via tectonic

Paths:
- CSL: `../../csl/apa.csl`
- Bibliographies:
  - `../../bibliography/credit_risk_neural_networks_research_papers.bib`
  - `../../bibliography/feature_selection_transformer_credit_risk_papers.bib`
  - `../../bibliography/lendingclub_subtopics_white.bib`
  - `../../bibliography/lendingclub_subtopics_grey.bib`

Build HTML (with citations):
```bash
pandoc THESIS_ITERATION_1.md \
  --from gfm \
  --citeproc \
  --csl ../../csl/apa.csl \
  --bibliography ../../bibliography/credit_risk_neural_networks_research_papers.bib \
  --bibliography ../../bibliography/feature_selection_transformer_credit_risk_papers.bib \
  --bibliography ../../bibliography/lendingclub_subtopics_white.bib \
  --bibliography ../../bibliography/lendingclub_subtopics_grey.bib \
  -o THESIS_ITERATION_1.html
```

Build PDF (via tectonic):
```bash
pandoc THESIS_ITERATION_1.md \
  --from gfm \
  --citeproc \
  --csl ../../csl/apa.csl \
  --bibliography ../../bibliography/credit_risk_neural_networks_research_papers.bib \
  --bibliography ../../bibliography/feature_selection_transformer_credit_risk_papers.bib \
  --bibliography ../../bibliography/lendingclub_subtopics_white.bib \
  --bibliography ../../bibliography/lendingclub_subtopics_grey.bib \
  -V geometry:margin=1in -V fontsize=11pt \
  --pdf-engine=tectonic \
  -o THESIS_ITERATION_1.pdf
```
