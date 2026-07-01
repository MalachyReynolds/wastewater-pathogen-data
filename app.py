from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Respiratory Incidence Dashboard", page_icon="\U0001fac1", layout="wide")

st.title("Respiratory incidence predictive dashboard")
st.write(
    "Use the pages in the sidebar to import and explore datasets, build predictive "
    "models, forecast future incidence, and download or export data."
)

st.markdown(
    """
- **Data** -- load the canonical wastewater/clinical/UKHSA/Google Trends series panel, or preview a single raw file.
- **Explore** -- interactive time series and correlation charts.
- **Model** -- fit regression models or the spike early-warning neural network.
- **Forecast** -- project a target series past its last observed date.
- **Downloads** -- refresh source data and export datasets, predictions, and forecasts.
"""
)

if "series" in st.session_state:
    series = st.session_state["series"]
    st.success(f"A canonical series panel is loaded: {len(series):,} observations.")
else:
    st.info("No data is loaded yet. Start on the Data page.")
