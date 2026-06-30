# Clinical data

This folder is for NHS England clinical activity datasets used alongside wastewater data.

The initial regression workflow uses:

- Integrated Urgent Care Aggregate Data Collection, including NHS111 activity
- A&E Attendances and Emergency Admissions monthly CSV data

Download source CSV files with:

```bash
python scripts/download_clinical_data.py
```

Downloaded files are written under:

```text
data/clinical/raw/
```

A download manifest is written to:

```text
data/clinical/clinical_download_manifest.json
```

The regression notebook starts at:

```text
notebooks/02_nhs111_gp_regression.ipynb
```

Terminology note: NHS England's A&E collection contains emergency admissions. If you mean a more specific GP outcome, such as GP in-hours consultations or out-of-hours primary-care dispositions, adapt the outcome-selection step in the notebook.
