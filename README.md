# WMH Dementia Permutation Analysis

A permutation-based statistical investigation of white matter hyperintensity (WMH) volumes across cognitive status groups using ADNI data.

---

## Research Question

Is there a statistically significant difference in white matter hyperintensity (WMH) volume between:

- Cognitively Unimpaired (CU)
- Mild Cognitive Impairment (MCI)
- Dementia

Additionally, does gender moderate this relationship?

---

## Background

White matter hyperintensities (WMH) are commonly associated with aging and neurodegenerative processes. Increased WMH burden has been linked to cognitive decline and dementia progression. However, distributional assumptions may not always hold for volumetric neuroimaging measures.

To address this, a non-parametric permutation framework was implemented to assess group differences without relying on strict normality assumptions.

---

## Methods

- Dataset: ADNI (Alzheimer’s Disease Neuroimaging Initiative)
- Outcome variable: Log-transformed WMH volume
- Group comparison: CU vs MCI vs Dementia
- Statistical method: Non-parametric permutation testing
- Additional analysis: Gender-stratified permutation comparisons

The permutation approach provides robust inference by constructing empirical null distributions through label shuffling.

---

## Key Findings

- WMH burden demonstrates an increasing trend across cognitive impairment severity.
- Overall permutation testing indicated a trend-level group effect (p ≈ 0.06).
- Gender-stratified analysis suggested a stronger trend among males, while no statistically significant group effect was observed among females.

These findings suggest a potential association between WMH burden and cognitive decline, with possible gender-related differences warranting further investigation.

---

## Project Structure
notebooks/
wmh_permutation_analysis.ipynb

src/
(optional modular statistical code)

results/
figures/

data/
(data not included)


---

## Data Availability

The dataset used in this analysis is derived from ADNI and is not distributed in this repository.

To reproduce the analysis:

1. Obtain ADNI data access approval.
2. Download the required WMH dataset.
3. Place the dataset in the `data/` directory.
4. Run the analysis notebook.

---

## Contribution

This was a collaborative group project. I was primarily responsible for the analytical and statistical components of the study.

My contributions included:

- Implementation of the non-parametric permutation testing framework
- Data preprocessing and log-transformation of WMH measures
- Group-level statistical comparisons across cognitive status categories
- Gender-stratified permutation analyses
- Development of visualizations and interpretation of results

All statistical modeling and permutation analysis code were implemented by me.

---

## Tools & Libraries

- Python
- NumPy
- Pandas
- SciPy
- Matplotlib

---

## Reproducibility

All analyses are reproducible given access to the appropriate ADNI dataset. The notebook provides a full computational workflow from preprocessing to statistical inference.
