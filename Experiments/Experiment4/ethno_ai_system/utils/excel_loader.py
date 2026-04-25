"""Helpers for loading persona rows from Excel."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from config import MAX_ROWS_FOR_SAMPLE_DATA, PERSONAS_FILE, ensure_directories


REQUIRED_COLUMNS = [
    "Name",
    "Problem",
    "Person Details",
    "Ethnographic Solution",
]


def create_sample_dataset() -> None:
    """Create a starter Excel file so the project runs immediately."""
    ensure_directories()

    sample_rows = [
        {
            "Name": "Maria Santos",
            "Problem": "Maria feels invisible in her own home after her adult children moved back during the pandemic and never left, eroding her sense of autonomy and personal space.",
            "Person Details": "Maria is a 58-year-old retired school aide in Sacramento, CA. Her two adult children ages 29 and 32 returned home in 2020 and remain. She no longer cooks what she wants, watches TV in her room to avoid conflict, and has stopped inviting friends over. She describes herself as a guest in her own house. Her husband sides with the children.",
            "Ethnographic Solution": "Conduct a spatial ethnography of the home using diaries, room-use mapping, conflict triggers, and household negotiation routines to identify how autonomy can be restored.",
        },
        {
            "Name": "Devon Price",
            "Problem": "Devon is exhausted by rotating warehouse shifts and feels his sleep, eating habits, and mood are becoming unstable.",
            "Person Details": "Devon is a 34-year-old warehouse worker in Columbus, OH. His schedule changes every week, he often eats at odd hours, and he has started snapping at his partner because he feels constantly tired and disoriented.",
            "Ethnographic Solution": "Study Devon's shift transitions, food timing, commute strain, and recovery rituals through a one-week temporal ethnography to redesign rest and routine supports.",
        },
        {
            "Name": "Aisha Khan",
            "Problem": "Aisha feels isolated as a freelance designer because her workday has blurred into evenings and weekends, weakening friendships and making her anxious about money.",
            "Person Details": "Aisha is a 27-year-old freelance designer in Austin, TX. She works from a studio apartment, checks client messages late at night, cancels social plans, and worries that saying no will dry up future work.",
            "Ethnographic Solution": "Map the overlap between workspace, finances, client access expectations, and social withdrawal to identify boundary-setting interventions grounded in daily practice.",
        },
    ][:MAX_ROWS_FOR_SAMPLE_DATA]

    pd.DataFrame(sample_rows).to_excel(PERSONAS_FILE, index=False)


def validate_dataset_exists() -> None:
    """Create sample data if the Excel file is missing."""
    if not PERSONAS_FILE.exists():
        create_sample_dataset()


def load_personas() -> pd.DataFrame:
    """Load the Excel file and validate required columns."""
    validate_dataset_exists()
    dataframe = pd.read_excel(PERSONAS_FILE)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required Excel columns: {missing_columns}")
    return dataframe


def get_persona_by_row(row_number: int) -> Dict[str, str]:
    """Return one persona row using 1-based indexing."""
    dataframe = load_personas()
    if row_number < 1 or row_number > len(dataframe):
        raise IndexError(f"Row {row_number} is outside the dataset range 1..{len(dataframe)}")
    row = dataframe.iloc[row_number - 1]
    return {column: str(row[column]) for column in REQUIRED_COLUMNS}


def get_all_personas() -> List[Dict[str, str]]:
    """Return the whole dataset as a list of dictionaries."""
    dataframe = load_personas()
    return dataframe[REQUIRED_COLUMNS].astype(str).to_dict(orient="records")
