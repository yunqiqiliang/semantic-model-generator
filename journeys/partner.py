import streamlit as st


@st.experimental_dialog("Partner Semantic Support", width="large")
def partner_semantic_setup() -> None:
    """
    Renders the partner semantic setup dialog with instructions.
    """
    from partner.partner_utils import configure_partner_semantic

    st.write(
        """
        Have an existing semantic layer from a partner tool already mapped into ClickZetta?
        See the steps below to merge those partner semantic specs into the generated ClickZetta semantic YAML.
        """
    )
    configure_partner_semantic()


def show() -> None:
    """
    Runs partner setup dialog.
    """
    partner_semantic_setup()
