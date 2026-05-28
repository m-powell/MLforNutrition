# Data

## MLforNutrition_NHANES.csv

This file contains the processed NHANES dataset used across all tutorials in this repository.
It is the single data source for every method tutorial — tutorials load it automatically
from GitHub and do not require a local download to run.

**Source:** National Health and Nutrition Examination Survey (NHANES), CDC/NCHS  
**Cycles:** 1999–2000 through 2017–2018 (10 two-year cycles)  
**Observations:** ~21,800 adults with complete fasting laboratory and dietary recall data  
**Processing:** See `Redo of NHANES Data Pull.ipynb` in this folder for the full assembly pipeline.

---

## Loading the Data

All tutorials load this file using the following pattern:

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

## Variable Dictionary

### Identifiers

| Variable | Type | Description |
|---|---|---|
| `SEQN` | numeric | NHANES respondent sequence number (unique per person-cycle) |
| `cycle_begin_year` | integer | Survey cycle start year (1999, 2001, …, 2017) |

### Demographics

| Variable | Type | Description |
|---|---|---|
| `age` | numeric | Age in years (adults ≥ 18 only) |
| `gender` | character | `"male"` or `"female"` |

### Anthropometrics and Blood Pressure

Values may be imputed; see imputation flags below.

| Variable | Type | Description |
|---|---|---|
| `bp_sys` | numeric | Systolic blood pressure (mmHg) |
| `bp_di` | numeric | Diastolic blood pressure (mmHg); values ≤ 1 removed as implausible |
| `weight` | numeric | Body weight (kg) |
| `height` | numeric | Height (cm) |
| `bmi` | numeric | Body mass index (kg/m²) |
| `waist` | numeric | Waist circumference (cm) |

### Dietary Intake (Day 1 24-Hour Recall)

| Variable | Type | Description |
|---|---|---|
| `kcal` | numeric | Total energy intake (kcal/day) |
| `protein` | numeric | Protein intake (g/day) |
| `sugar` | numeric | Total sugar intake (g/day) |
| `carb` | numeric | Total carbohydrate intake (g/day) |
| `fat_total` | numeric | Total fat intake (g/day) |
| `fat_sat` | numeric | Saturated fat intake (g/day) |
| `fat_mon` | numeric | Monounsaturated fat intake (g/day) |
| `fat_poly` | numeric | Polyunsaturated fat intake (g/day) |

### Fasting Laboratory Measures

Only participants with all three fasting values present are included.

| Variable | Type | Description |
|---|---|---|
| `hdl` | numeric | HDL cholesterol (mg/dL) |
| `glucose` | numeric | Fasting plasma glucose (mg/dL) |
| `triglycerides` | numeric | Fasting triglycerides (mg/dL) |

### Medications

| Variable | Type | Description |
|---|---|---|
| `meds_hbp` | character | Taking blood pressure medication (`"TRUE"` / `"FALSE"` / `NA`) |
| `meds_chol` | character | Taking cholesterol medication (`"TRUE"` / `"FALSE"` / `NA`) |

### Imputation Flags

Each flag is `True` if the corresponding value was missing in the raw NHANES data and
was filled by multiple imputation (MICE via `IterativeImputer`); `False` otherwise.

| Variable | Type | Description |
|---|---|---|
| `imputed_weight` | boolean | Weight was imputed |
| `imputed_height` | boolean | Height was imputed |
| `imputed_bmi` | boolean | BMI was imputed |
| `imputed_waist` | boolean | Waist circumference was imputed |
| `imputed_bp_sys` | boolean | Systolic BP was imputed |
| `imputed_bp_di` | boolean | Diastolic BP was imputed |

### Derived Outcome Variables

These binary variables are computed from the continuous measurements above using
established clinical thresholds. They are pre-computed here so that all tutorials
use identical definitions.

| Variable | Type | Definition | Clinical Source |
|---|---|---|---|
| `extreme_bp` | boolean | `bp_sys >= 130` **or** `bp_di >= 80` | AHA/ACC 2017 Stage 1 hypertension |
| `extreme_waist` | boolean | `waist >= 102` cm (male) or `>= 88` cm (female) | ATP III abdominal obesity criterion |
| `extreme_hdl` | boolean | `hdl < 40` mg/dL (male) or `< 50` mg/dL (female) | ATP III low HDL criterion |
| `extreme_tri` | boolean | `triglycerides >= 150` mg/dL | ATP III elevated triglycerides criterion |
| `extreme_glu` | boolean | `glucose >= 100` mg/dL | ADA impaired fasting glucose threshold |
| `metsyn` | boolean | ≥ 3 of the above 5 criteria are `True` | ATP III metabolic syndrome definition |

**References:**
- Grundy et al. (2005). Diagnosis and management of the metabolic syndrome. *Circulation*, 112(17), 2735–2752.
- Whelton et al. (2018). 2017 ACC/AHA high blood pressure guideline. *Journal of the American College of Cardiology*, 71(19), e127–e248.
- American Diabetes Association (2004). Diagnosis and classification of diabetes mellitus. *Diabetes Care*, 27(Suppl 1), S5–S10.
