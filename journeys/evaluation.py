import streamlit as st


def evaluation_mode_show() -> None:
    st.info(
        "Evaluation mode is not yet available for ClickZetta. "
        "You can validate locally by inspecting query outputs, but automated evaluation is disabled."
    )
