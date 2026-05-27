# Data

## nhanes_nutrition.csv

This file contains the processed NHANES dataset used across all tutorials in this repository.

**Source:** National Health and Nutrition Examination Survey (NHANES), CDC/NCHS  
**Time period:** [Cycle(s) — e.g., 2017–2018]  
**Observations:** [N]  
**Processing:** See the [Obtaining NHANES Data tutorial](../tutorials/nhanes-data/) for a full walkthrough of how this file was assembled from the raw NHANES files.

### Variable Dictionary

| Variable | Type | Description |
|---|---|---|
| SEQN | integer | NHANES respondent sequence number |
| age | numeric | Age in years |
| gender | character | Gender ("Male" / "Female") |
| bp_sys | numeric | Systolic blood pressure (mmHg) |
| bp_di | numeric | Diastolic blood pressure (mmHg) |
| weight | numeric | Body weight (kg) |
| height | numeric | Height (cm) |
| bmi | numeric | Body mass index (kg/m²) |
| waist | numeric | Waist circumference (cm) |
| kcal | numeric | Total energy intake (kcal/day) |
| protein | numeric | Protein intake (g/day) |
| sugar | numeric | Total sugar intake (g/day) |
| med_hbp | logical | Currently taking blood pressure medication |
| med_chol | logical | Currently taking cholesterol medication |
| extreme_bp | logical | Extreme blood pressure (derived outcome variable) |
| extreme_waist | logical | Extreme waist circumference (derived outcome variable) |
| extreme_hdl | logical | Extreme HDL cholesterol (derived outcome variable) |

*Add or remove rows as needed to match the actual dataset.*

## Loading the Data

All tutorials load this file automatically from GitHub:

```
https://raw.githubusercontent.com/m-powell/m-powell.github.io/master/MLforNutrition/data/nhanes_nutrition.csv
```

A local copy is included here as a fallback for offline use. See `TEMPLATE.Rmd` or
`TEMPLATE.ipynb` for the exact loading pattern used in every tutorial.
