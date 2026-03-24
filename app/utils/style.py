"""CSS partage minimal et sans selecteurs risqués."""
import streamlit as st

_CSS = """
<style>
.main .block-container {
    max-width: 1350px !important;
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
}

[data-testid="stSidebarNav"],
[data-testid="collapsedControl"] {
    display: none !important;
}

[data-testid="stRadio"] label,
[data-testid="stSelectbox"] label,
[data-testid="stButton"] button {
    font-size: 0.95rem !important;
}

[data-testid="stButton"] button {
    min-height: 48px !important;
}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
