# ML for Nutrition Research — Companion Code

This repository contains tutorial-style code implementations for the methods described in:

> *[Full citation — to be added upon publication]*

For each method, you will find implementations in both **R** (R Markdown → HTML) and
**Python** (Jupyter Notebook), each producing the analyses and figures shown in the paper.

## Table of Contents

| Method | R Tutorial | Python Tutorial |
|---|---|---|
| Obtaining NHANES Data | [R](tutorials/nhanes-data/nhanes-r.html) | [Python](tutorials/nhanes-data/nhanes-python.html) |
| Clustering | [R](tutorials/clustering/clustering-r.html) | [Python](tutorials/clustering/clustering-python.html) |
| Deep Learning | [R](tutorials/deep-learning/deep-learning-r.html) | [Python](tutorials/deep-learning/deep-learning-python.html) |
| Ensemble Methods (XGBoost) | [R](tutorials/ensemble-xgboost/xgboost-r.html) | [Python](tutorials/ensemble-xgboost/xgboost-python.html) |
| Random Forests | [R](tutorials/random-forest/random-forest-r.html) | [Python](tutorials/random-forest/random-forest-python.html) |
| Linear Discriminant Analysis | [R](tutorials/lda/lda-r.html) | [Python](tutorials/lda/lda-python.html) |
| PCA / PCoA | [R](tutorials/pca-pcoa/pca-r.html) | [Python](tutorials/pca-pcoa/pca-python.html) |
| Support Vector Machines | [R](tutorials/svm/svm-r.html) | [Python](tutorials/svm/svm-python.html) |

## Data

All tutorials use a processed NHANES dataset hosted in the [`data/`](data/) directory.
Tutorials load this file automatically from GitHub — no local download is needed to run the code.

To understand how the dataset was assembled and to reproduce the data collection yourself,
see the **[Obtaining NHANES Data](tutorials/nhanes-data/)** tutorial.

## Repository Structure

```
MLforNutrition/
├── README.md
├── data/
│   ├── nhanes_nutrition.csv        ← Bundled dataset (all tutorials load this)
│   └── README.md                   ← Variable dictionary and provenance notes
├── tutorials/
│   ├── nhanes-data/                ← One folder per method
│   │   ├── nhanes-r.Rmd
│   │   ├── nhanes-r.html
│   │   ├── nhanes-python.ipynb
│   │   └── nhanes-python.html
│   ├── svm/
│   │   ├── svm-r.Rmd
│   │   ├── svm-r.html
│   │   ├── svm-python.ipynb
│   │   └── svm-python.html
│   └── [other methods follow the same pattern]
└── docs/
    ├── STYLE_GUIDE.md              ← Voice, structure, and code conventions
    ├── TEMPLATE.Rmd                ← Starter R Markdown template
    ├── TEMPLATE.ipynb              ← Starter Jupyter Notebook template
    └── TUTORIAL_TRACKER.md        ← Progress tracker (16 tutorials total)
```

## Contributing

Before writing or revising a tutorial, read the [Style Guide](docs/STYLE_GUIDE.md).
It defines the expected voice, document structure, and code conventions so all tutorials feel consistent.

**Quick checklist:**
- [ ] Section order: Introduction → Setup → Data → Method → Results → Summary
- [ ] No hardcoded file paths — data loaded via URL with local fallback
- [ ] Random seed set and documented
- [ ] Every figure captioned ("Figure N: ...")
- [ ] Colorblind-friendly palette (Set2 or equivalent)
- [ ] sessionInfo() / session info cell at end

## Contact

[Your contact info or link to the paper]
