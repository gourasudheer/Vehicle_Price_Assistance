"""
Streamlit frontend for the Bike & Car Price/Mileage Chatbot.
Clean, user-facing UI — no internal/technical details shown.
Uses Streamlit's own theme variables (not hardcoded colors) so it looks
correct in both light and dark mode.

Run:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import config  # noqa: E402
from rag_pipeline import RAGPipeline  # noqa: E402

st.set_page_config(page_title="Vehicle Price Assistant", layout="centered")

# ---------------------------------------------------------------------------
# Minimal styling — uses Streamlit's theme CSS variables so colors adapt
# automatically to light/dark mode instead of being hardcoded.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .block-container { padding-top: 2.5rem; max-width: 780px; }
    .subtitle {
        color: var(--text-color);
        opacity: 0.65;
        font-size: 1.02rem;
        margin-top: -0.6rem;
        margin-bottom: 1.6rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Vehicle Price Assistant")
st.markdown(
    '<div class="subtitle">Ask about on-road price, ex-showroom price, or mileage '
    'for any bike or car in the catalog.</div>',
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_pipeline():
    return RAGPipeline()


# ---------------------------------------------------------------------------
# Sidebar — kept minimal, no technical/internal details
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    category_filter = st.selectbox("Vehicle type", ["All", "Bike", "Car"])
    st.divider()
    if st.button("Clear Chat", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Readiness checks (friendly, non-technical messaging)
# ---------------------------------------------------------------------------
if not config.OPENROUTER_API_KEY:
    st.error("The assistant isn't set up yet. Please add your API key and restart the app.")
    st.stop()

if not config.CHROMA_PERSIST_DIR.exists() or not any(config.CHROMA_PERSIST_DIR.iterdir()):
    st.error("No vehicle data has been loaded yet. Please run the data setup step first.")
    st.stop()

try:
    with st.spinner("Getting things ready..."):
        rag = load_pipeline()
except Exception as e:
    st.error(f"Something went wrong while starting the assistant.\n\nDetails: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🚘"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Example prompts (only shown before the conversation starts)
# Plain st.button, no custom CSS override -> always readable in any theme
# ---------------------------------------------------------------------------
clicked_example = None
if not st.session_state.messages:
    st.write("Try asking:")
    cols = st.columns(3)
    examples = [
        "Honda Activa 6G price & mileage?",
        "Compare Tata Nexon vs Hyundai Creta",
        "Cheapest electric bike?",
    ]
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            clicked_example = ex

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
user_query = st.chat_input("Ask about a bike or car...")
query = clicked_example or user_query

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🚘"):
        with st.spinner("Looking that up..."):
            search_query = query
            if category_filter != "All":
                search_query = f"{query} (category: {category_filter.lower()})"
            result = rag.ask(search_query)
            st.markdown(result["answer"])

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})