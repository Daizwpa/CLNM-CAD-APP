import __init__
import os
import streamlit as st




os.environ["LOKY_MAX_CPU_COUNT"] = "1"

st.set_page_config(initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* Hide entire sidebar container */
    section[data-testid="stSidebar"] {
        display: none !important;
    }

    /* Remove the gap left by sidebar */
    div[data-testid="stAppViewContainer"] {
        margin-left: 0px !important;
    }

    /* Optional: remove hamburger menu */
    button[kind="header"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.set_page_config(initial_sidebar_state="collapsed")
    st.title("🔐 Welcome to SRP-DZ Platform")
    st.write("### For Diangosis postoperative recurrent lymph node metastasis of Thyroid cancer")
    st.write("Please sign in with your corporate or personal account.")

    if st.button("Log in with Google", type="primary"):
        st.login("google")


if __name__ == "__main__":  
    


    if st.user.is_logged_in:
        st.switch_page("pages/Inference.py")
    main()
