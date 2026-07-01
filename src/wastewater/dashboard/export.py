from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


def dataframe_download_buttons(df: pd.DataFrame, label: str, filename_stem: str, key: str) -> None:
    """Render CSV and Parquet download buttons for a dataframe."""
    col1, col2 = st.columns(2)
    col1.download_button(
        f"{label} (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{filename_stem}.csv",
        mime="text/csv",
        key=f"{key}_csv",
    )
    col2.download_button(
        f"{label} (Parquet)",
        data=df.to_parquet(index=False),
        file_name=f"{filename_stem}.parquet",
        mime="application/octet-stream",
        key=f"{key}_parquet",
    )


def file_download_button(path: Path, label: str, key: str) -> None:
    """Render a download button for an existing file on disk, as raw bytes."""
    path = Path(path)
    if not path.exists():
        st.caption(f"{path.name} has not been generated yet.")
        return
    st.download_button(label, data=path.read_bytes(), file_name=path.name, key=key)
