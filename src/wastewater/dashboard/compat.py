from __future__ import annotations

import pandas as pd


def make_streamlit_safe(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce object columns to a Streamlit-safe format before display."""
    if frame is None:
        return frame

    safe = frame.copy()
    for column in safe.columns:
        if safe[column].dtype == object:
            values = safe[column]
            if values.empty:
                safe[column] = values.astype(str)
                continue
            try:
                safe[column] = values.astype("string")
            except Exception:
                safe[column] = values.apply(lambda value: str(value) if pd.notna(value) else "")
    return safe
