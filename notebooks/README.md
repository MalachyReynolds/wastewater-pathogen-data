# Analysis notebooks

Start with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
jupyter lab
```

Then open:

- `01_wastewater_analysis.ipynb`

The first notebook is intentionally exploratory. It inventories the raw files, inspects schemas, defines a canonical long-format target, and provides placeholders for country-specific cleaning adapters.
