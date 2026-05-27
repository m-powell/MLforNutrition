# Tutorial Status Tracker

Use this file to track progress across all 16 tutorials (8 methods × 2 languages).

**Status codes:** ✅ Complete | 🔄 In Progress | ⚠️ Needs Revision | ❌ Not Started

## Status

| Method | R | Python | Notes |
|---|---|---|---|
| Obtaining NHANES Data | ❌ | ❌ | Priority — unblocks data loading for everything else |
| Clustering | ❌ | ❌ | |
| Deep Learning | ❌ | ❌ | |
| Ensemble Methods (XGBoost) | ❌ | ❌ | |
| Random Forests | ❌ | ❌ | |
| Linear Discriminant Analysis | ❌ | ❌ | |
| PCA / PCoA | ❌ | ❌ | |
| Support Vector Machines | ⚠️ Needs revision | ❌ | SVM-R exists; has hardcoded path + needs style pass |

## Revision Checklist (for existing tutorials)

When revising an existing tutorial to meet style guide standards, check off each item:

- [ ] Hardcoded file paths removed → URL + local fallback pattern added
- [ ] Section order matches style guide (Introduction → Setup → Data → Method → Results → Summary)
- [ ] Voice revised to first person plural ("We begin by...", "Here we...")
- [ ] Every code chunk preceded by a prose explanation
- [ ] All package imports commented with their purpose
- [ ] Random seed set, documented, and passed consistently
- [ ] `theme_nutr()` (R) or `set_plot_style()` (Python) applied to all figures
- [ ] Figure captions added in correct format ("Figure N: ...")
- [ ] Colorblind-friendly palette confirmed (`Set2` or equivalent)
- [ ] `sessionInfo()` / session info cell at end
- [ ] Cross-language link in Summary section

## Suggested Priority Order

1. **SVM-R revision** — the one real tutorial; revise it first to prove the workflow and produce a reference example for others
2. **Obtaining NHANES Data** (R + Python) — unblocks reproducible data loading for all other tutorials
3. Methods where a contributor draft exists in one language → apply style pass, then write companion
4. Methods with no drafts at all → start from `TEMPLATE.Rmd` or `TEMPLATE.ipynb`

## Who Is Working on What

| Method | Language | Owner | Status | Notes |
|---|---|---|---|---|
| SVM | R | [Name] | ⚠️ Needs revision | Existing draft at tutorials/svm/svm-r.Rmd |
| | | | | |
