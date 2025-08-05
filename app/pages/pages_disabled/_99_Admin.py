import os, sys, subprocess, time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import streamlit as st

st.set_page_config(layout="wide")
st.title("Admin / Maintenance")

if st.button("Fetch latest articles now"):
    with st.spinner("Running ingest..."):
        # call the ingest script in a subprocess
        result = subprocess.run([sys.executable, "scripts/ingest.py"], capture_output=True, text=True)
        st.code(result.stdout + "\n" + result.stderr)
    st.success("Done. Reload the other pages to see updates.")

