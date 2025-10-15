import streamlit as st

# set_page_config must be run as the first Streamlit command on the page, before any other streamlit imports.
st.set_page_config(layout="wide", page_icon="💬", page_title="Semantic Model Generator")

from app_utils.shared_utils import (  # noqa: E402
    GeneratorAppScreen,
    render_sidebar_title,
    get_clickzetta_connection,
    set_account_name,
    set_host_name,
    set_sit_query_tag,
    set_clickzetta_session,
    set_streamlit_location,
    set_user_name,
)
from semantic_model_generator.clickzetta_utils.env_vars import (  # noqa: E402
    CLICKZETTA_INSTANCE,
    CLICKZETTA_SERVICE,
    CLICKZETTA_USERNAME,
)


@st.experimental_dialog(title="Connection Error")
def failed_connection_popup() -> None:
    """
    Renders a dialog box detailing that the credentials provided could not be used to connect to ClickZetta.
    """
    st.markdown(
        """It looks like the credentials provided could not be used to connect to the account."""
    )
    st.stop()


def verify_environment_setup():
    """
    Ensures that the correct environment variables are set before proceeding.
    """

    # Instantiate the ClickZetta connection that gets reused throughout the app.
    try:
        with st.spinner(
            "Validating your connection to ClickZetta."
        ):
            return get_clickzetta_connection()
    except Exception:
        failed_connection_popup()


if __name__ == "__main__":
    from journeys import builder, iteration, partner

    st.session_state["sis"] = set_streamlit_location()

    def onboarding_dialog() -> None:
        """
        Renders the initial screen where users can choose to create a new semantic model or edit an existing one.
        """

        sidebar = st.sidebar
        render_sidebar_title(sidebar, margin_bottom=28)
        sidebar.subheader("Get Started")
        if sidebar.button(
            "Create a new semantic model",
            use_container_width=True,
        ):
            builder.show()
        if sidebar.button(
            "Edit an existing semantic model",
            use_container_width=True,
        ):
            iteration.show()
        if sidebar.button(
            "Start with partner semantic model",
            use_container_width=True,
        ):
            set_sit_query_tag(
                get_clickzetta_connection(),
                vendor="",
                action="start",
            )
            partner.show()

        sidebar.divider()
        iteration.render_metadata_sections(sidebar)

        st.markdown(
            """
<div style="margin-top: 56px; text-align: left; width: 100%;">
  <h1 style="margin-bottom: 1.5rem;">Welcome to the ClickZetta Semantic Model Generator</h1>
  <p style="margin-bottom: 1.25rem;">
    ⚠️ Heads up! Our original Streamlit experience is now a companion to ClickZetta’s native semantic modeling tooling. Think of this app as your local workbench for rapid iteration before promoting changes back into the platform.
  </p>
  <p style="margin-bottom: 1.25rem;">
    👉 Start in the ClickZetta workspace to create or update official semantic models. When you need to inspect YAML, prototype adjustments, or run partner workflows, hop back here—the generator reads and writes directly from the same ClickZetta volumes.
  </p>
  <p style="margin-bottom: 1.25rem;">
    ✅ Within this app you can edit semantic YAML side by side with chat-based validation, run the ClickZetta validators, upload/download files from volumes, and integrate specs exported from partner tools.
  </p>
  <h3 style="margin-bottom: 0.75rem;">Why Semantic Models Matter</h3>
  <p style="margin-bottom: 1.25rem;">
    <strong>ClickZetta Semantic Model Generator provides a standard, open semantic layer</strong> that standardizes data annotation and unifies data definitions across the enterprise. Through structured dimensions, measures, and business logic definitions, semantic models enable LLMs to better understand data meaning and relationships, thereby <strong>avoiding LLM hallucinations and significantly improving data analysis accuracy</strong>. Both data teams and business users can collaborate based on the same semantic definitions, ensuring consistency and reliability of analytical results.
  </p>
  <h3 style="margin-bottom: 0.75rem;">Typical workflows supported by this app</h3>
  <p style="margin-bottom: 0.75rem;">
    • <strong>Author and refine semantic models.</strong> Start from ClickZetta metadata, stitch dimensions and measures, and save the resulting semantic YAML straight into a Lakehouse volume.<br>
    • <strong>Iterate safely on existing files.</strong> Pull a YAML from your production volume, try changes in the editor, validate them with the built-in ClickZetta rules, and push the update back when it is ready.<br>
    • <strong>Use semantic models as prompts for analytics.</strong> Feed the YAML into the integrated chat assistant—the app uses it as context to generate SQL and run it against Lakehouse tables so you can test questions before sharing with stakeholders.<br>
    • <strong>Enrich and align documentation.</strong> With DashScope enabled, automatically expand descriptions, relationships, and starter questions to keep the business-facing layer up to date.<br>
  </p>
  <p>
    Use this generator as a safe sandbox: pull a semantic model from a volume, iterate quickly with the integrated chat assistant, then push the refined YAML back when you are ready.
  </p>
</div>
            """,
            unsafe_allow_html=True,
        )

        st.image("images/semantic-model-overview.svg", use_column_width=True)

    conn = verify_environment_setup()
    set_clickzetta_session(conn)

    # Populating common state between builder and iteration apps.
    set_account_name(conn, CLICKZETTA_INSTANCE)
    set_host_name(conn, CLICKZETTA_SERVICE)
    set_user_name(conn, CLICKZETTA_USERNAME)

    # When the app first loads, show the onboarding screen.
    if "page" not in st.session_state:
        st.session_state["page"] = GeneratorAppScreen.ONBOARDING

    # Depending on the page state, we either show the onboarding menu or the chat app flow.
    # The builder flow is simply an intermediate dialog before the iteration flow.
    if st.session_state["page"] == GeneratorAppScreen.ITERATION:
        iteration.show()
    else:
        onboarding_dialog()
