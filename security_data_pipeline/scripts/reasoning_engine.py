# security_data_pipeline/scripts/reasoning_engine.py
import pandas as pd
import numpy as np

def reason_about_data(intent: str, query: str, df: pd.DataFrame):
    """Generate natural-language insights from data patterns."""
    msg = None

    if intent == "crime_stats":
        if "total_crime_count" in df.columns:
            crimes = df.groupby("location_name")["total_crime_count"].sum(numeric_only=True)
            top = crimes.sort_values(ascending=False).head(3)
            msg = "Based on recorded data, the top crime areas are " + \
                  ", ".join([f"{loc} ({val:.0f} incidents)" for loc, val in top.items()]) + "."
    elif intent == "demographics":
        if {"population", "median_income"}.issubset(df.columns):
            avg_income = df["median_income"].mean()
            msg = f"The average median income across Toronto regions is roughly ${avg_income:,.0f}."
    elif intent == "urban_safety":
        msg = "Urban safety scores vary by district — FYND AI is integrating new response-time datasets."
    elif intent == "alerts":
        msg = "You can check real-time emergencies using Environment Canada or Alert Ready Canada feeds."
    elif intent == "news":
        msg = "Fetching current Canadian safety headlines…"
    elif intent == "map":
        msg = "Would you like a map of crimes or patrol zones?"
    else:
        msg = "FYND AI is working tirelessly to improve our datasets."

    return msg