# WMH Dementia Permutation Analysis

Permutation-based statistical analysis of white matter hyperintensity (WMH) volumes across cognitive status groups using ADNI-derived data.

---

## Research Question

Is white matter hyperintensity (WMH) burden significantly different across:

- Cognitively Unimpaired (CU)
- Mild Cognitive Impairment (MCI)
- Dementia

Additionally, does gender moderate these group differences?

---

## Rationale

WMH volumes are typically right-skewed and may violate parametric assumptions.  
To obtain robust inference without relying on normality assumptions, a non-parametric permutation framework was implemented.

---

## Methods

- Dataset: ADNI-derived participant summary dataset  
- Outcome: Log-transformed WMH volume  
- Groups: CU, MCI, Dementia (derived from CDR scores)  
- Statistical approach: Permutation test using variance of group means  
- Additional analysis: Gender-stratified permutation testing  

The permutation framework constructs an empirical null distribution by randomly shuffling group labels.

---

## Key Findings

- WMH burden shows an increasing trend across cognitive impairment severity.
- Overall permutation testing indicates a trend-level group effect.
- Gender-stratified analysis suggests stronger group separation in males relative to females.

---

## Project Structure
src/
wmh_permutation_analysis.py

data/
(not included – see Data section)

results/
figures and summary outputs

requirements.txt

---

## Data Availability

The dataset used in this project is derived from ADNI and is not included in this repository due to data access restrictions.

To reproduce the analysis:

1. Obtain ADNI access approval.
2. Place the dataset inside the `data/` directory.
3. Run:
python src/wmh_permutation_analysis.py --data data/adni_participants_summary_modified.csv

---

## Contribution

This was a collaborative group project.  
I was responsible for:

- Designing and implementing the permutation testing framework
- Data preprocessing and log-transformation
- Group-level statistical comparisons
- Gender-stratified analysis
- Visualization and interpretation of results

All statistical modeling code was implemented by me.

---

## Tools

- Python
- NumPy
- Pandas
- Matplotlib
