import streamlit as st
import pandas as pd
import os

EVENTS_DIR = "events"

st.set_page_config(page_title="📅 View Events", layout="wide")
st.title("📅 View Saved Events")

if not os.path.exists(EVENTS_DIR):
    st.info("No events have been saved yet.")
else:
    event_files = [f for f in os.listdir(EVENTS_DIR) if f.endswith(".csv")]
    if not event_files:
        st.info("No events have been saved yet.")
    else:
        selected_event = st.selectbox("Select an event to view:", sorted(event_files))
        if selected_event:
            df = pd.read_csv(os.path.join(EVENTS_DIR, selected_event))
            st.subheader(f"Event: {selected_event.replace('.csv','')}")
            st.dataframe(df, use_container_width=True)
