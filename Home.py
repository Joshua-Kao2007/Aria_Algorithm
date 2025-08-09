#min-max scalar, binn machine learning
import streamlit as st

st.set_page_config(page_title="Aria", page_icon="🎭", layout="centered")

st.title("🎭 Aria")
st.markdown("Welcome to the Aria Prediction System. Choose an option below:")

# Navigation buttons
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Admin")
    if st.button("Go to Admin Page"):
        st.switch_page("pages/1_Admin.py")

with col2:
    st.markdown("### Predict")
    if st.button("Go to Predict Page"):
        st.switch_page("pages/2_Predict.py")

# Footer
st.markdown("---")
st.markdown("Created by Joshua Kao © 2025")
