# MLforNutrition — Tutorial Style Guide

This guide defines the voice, structure, and code conventions for all tutorials in this repo. Every tutorial — regardless of method or language — should feel like it was written by the same person.

---

## 1. Audience

Write for a researcher in nutrition science or public health who:
- Is comfortable with basic statistics (regression, t-tests, confidence intervals)
- Has *some* R or Python experience, but is not a software developer
- Has little to no prior exposure to the ML method being covered
- Wants to understand what the method is doing, not just copy-paste working code

Do **not** assume the reader has seen the method before. Do **not** assume they know what a hyperparameter is, what cross-validation is, or why it matters — explain these the first time they appear.

---

## 2. Voice and Tone

- **Conversational, not textbook.** Write as if you're sitting next to the reader walking them through your analysis. First person plural ("We begin by...", "Here we split the data...") is preferred over passive voice ("The data were split...").
- **Explain, then code.** Before every code chunk, write a sentence or two explaining *what* you're about to do and *why*. The reader should never encounter a code block that surprises them.
- **Name things simply.** Prefer "training set" over "training partition". Prefer "outcome variable" over "response vector". Define any jargon the first time it appears, in plain language.
- **No unnecessary hedging.** Avoid phrases like "it is worth noting that" or "it should be mentioned that." Say the thing directly.
- **Keep it grounded in nutrition.** Every method should be introduced in the context of a specific nutrition/health research question using NHANES data. The reader should always know *why* this method is useful for their domain.

---

## 3. Document Structure

Every tutorial must follow this section order. Use these exact headings.

### Required Sections (in order)

1. **Introduction**
   - One paragraph: what is this method and when would a nutrition researcher use it?
   - One paragraph: what question are we answering in this tutorial, and what dataset are we using?
   - Keep it to ~200 words. No subheadings.

2. **Setup**
   - Load all packages/libraries here and nowhere else.
   - Comment every import with its purpose (one line).
   - For R: use `library()` calls. For Python: use `import` statements grouped logically (stdlib → third-party → local).

3. **Data**
   - Brief description of the NHANES dataset as used here: which variables, what time period, how many observations.
   - Load the data from the repo's bundled CSV (see data loading conventions below).
   - Always show the first few rows and print basic dimensions.
   - Describe any preprocessing steps (missing data handling, variable coding) *before* the code that performs them.

4. **Method**
   - This is the main body of the tutorial. Break it into logical sub-sections using level-3 headings (`###`).
   - Typical sub-sections: *Model setup and hyperparameters*, *Fitting the model*, *Evaluating performance*, *Interpreting results*.
   - Include at least one visualization. All figures must be labeled and captioned.
   - When introducing a hyperparameter, explain what it controls conceptually before showing how to set it.

5. **Results**
   - Summarize findings in plain language. What did the model tell us about the nutrition research question?
   - Include any key figures or tables produced in the Method section here (or reference them if already shown inline).
   - 1–2 short paragraphs.

6. **Summary**
   - Bullet list: 3–5 key takeaways from this tutorial.
   - One sentence on when *not* to use this method (limitations).
   - Optional: one sentence pointing the reader to the companion tutorial in the other language.

---

## 4. Code Conventions

### Both Languages

- **All hardcoded file paths are forbidden.** Data must be loaded from a URL pointing to the repo's `data/` folder on GitHub, with a fallback to a relative path. See the template for the exact pattern.
- **Set a random seed** at the start of any analysis involving randomness. Document the seed.
- **Name things descriptively.** `model_rf` not `m`. `train_data` not `df2`. `pred_bp` not `p`.
- **No silent warnings.** If a package produces known warnings, suppress them explicitly and add a comment explaining why.
- **Chunk/cell size:** Each code chunk/cell should do one logical thing. A chunk that loads data, cleans it, splits it, and fits a model is too long. Split it.

### R Conventions

- Tidyverse style throughout: `|>` native pipe, `tibble` not `data.frame`, `ggplot2` for all plots.
- Use `set.seed()` immediately before any call that uses randomness.
- Variable names: `snake_case`.
- Use `here::here()` for any relative file paths.
- Suppress package startup messages with `suppressPackageStartupMessages()` or `#| message: false` chunk options.
- All plots: use a consistent ggplot theme. The template defines `theme_nutr()` — use it for every plot.

### Python Conventions

- Use f-strings for string formatting, not `.format()` or `%`.
- Variable names: `snake_case`.
- Use `pandas` for all tabular data operations; `matplotlib` + `seaborn` for visualization.
- Set `random_state=SEED` (where `SEED` is the defined constant) on every object that accepts it.
- All plots: use the template's `set_plot_style()` function for consistent formatting.
- Prefer `scikit-learn` pipelines over manual preprocessing steps where possible.

---

## 5. Figures

- **Every figure must have a caption.** Captions go immediately below the figure.
- **Caption format:** "Figure N: [What is shown]. [One sentence interpretation relevant to the nutrition context.]"
- **Color palettes:** Use colorblind-friendly palettes. In R: `scale_color_brewer(palette = "Set2")` or `scale_fill_brewer(palette = "Set2")`. In Python: `sns.color_palette("Set2")`.
- **Figure size:** Standard width is 7 inches (R) / 7 inches at 100 dpi (Python). Specify explicitly.

---

## 6. Data Loading Convention

The NHANES dataset lives in two places:
1. Bundled in this repo at `data/MLforNutrition_NHANES.csv`
2. Assembled by the data generation notebook at `data/Redo of NHANES Data Pull.ipynb`

All tutorials load data using this pattern:

**R:**
```r
DATA_URL <- "https://raw.githubusercontent.com/m-powell/MLforNutrition/master/data/MLforNutrition_NHANES.csv"

nhanes <- tryCatch(
  read_csv(DATA_URL, show_col_types = FALSE),
  error = function(e) {
    message("Could not load from URL; falling back to local copy.")
    read_csv(here::here("data", "MLforNutrition_NHANES.csv"), show_col_types = FALSE)
  }
)
```

**Python:**
```python
DATA_URL = "https://raw.githubusercontent.com/m-powell/MLforNutrition/master/data/MLforNutrition_NHANES.csv"

try:
    nhanes = pd.read_csv(DATA_URL)
except Exception:
    print("Could not load from URL; falling back to local copy.")
    nhanes = pd.read_csv("../../data/MLforNutrition_NHANES.csv")
```

---

## 7. Cross-Language Parity

Every method has both an R tutorial and a Python tutorial. They are **not** line-for-line translations, but they must be comparable:

- Same dataset, same variables, same research question.
- Same model structure (same hyperparameter choices where possible).
- Same outcome metrics reported.
- Figures convey the same information, even if the aesthetics differ slightly.

When one language was written first, the second should follow the same section flow and introduce concepts in the same order. It's fine to acknowledge language-specific idioms ("In Python, scikit-learn uses pipelines to chain preprocessing and modeling steps, which differs from how we structured this in R...").

---

## 8. What to Avoid

- **Don't explain R to a Python reader, or vice versa.** Each tutorial is self-contained. Readers choose one language and follow it through.
- **Don't reproduce the paper.** Tutorials illustrate the methods with NHANES examples. They are not summaries of the paper's findings.
- **Don't over-explain the data.** The NHANES tutorial handles data provenance. Other tutorials load the data, describe the relevant columns, and move on.
- **Don't use `print()` statements as your only output.** Show results in formatted tables where possible (`knitr::kable()` in R, formatted DataFrames in Jupyter).
