import streamlit as st


@st.experimental_dialog("Partner Semantic Support", width="large")
def show_placeholder() -> None:
    st.info(
        "Looker partner integration is not yet available for ClickZetta. "
        "Please reach out to the project maintainers if you require this feature."
    )
    if st.button("Close", use_container_width=True):
        st.experimental_rerun()


def show() -> None:
    show_placeholder()
