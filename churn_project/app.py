# =============================================================================
# app.py
# Streamlit chatbot UI for the Cell2Cell Churn Analytics RAG system.
# Run: streamlit run app.py
# =============================================================================

import streamlit as st
import sys, os
sys.path.append(os.path.dirname(__file__))

from src.rag.chatbot import ChurnChatbot

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Analytics Assistant",
    page_icon="📉",
    layout="wide",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8FAFC; }
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }
    .stTextInput > div > div > input { border-radius: 10px; }
    h1 { color: #1B3A6B; }
    .metric-card {
        background: white; border-radius: 10px;
        padding: 16px; text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📉 Churn Analytics")
    st.markdown("**Dataset:** Cell2Cell Telecom")
    st.markdown("**Records:** 51,047 customers")
    st.markdown("**Churn Rate:** 29%")
    st.divider()

    st.markdown("### 💡 Example Questions")
    example_qs = [
        "Why are new customers churning more?",
        "What is the churn rate for low credit rating customers?",
        "Which features drive churn the most?",
        "What retention strategies work best?",
        "What does DropRate mean?",
        "Compare churn between rural and suburban customers",
    ]
    for q in example_qs:
        if st.button(q, use_container_width=True):
            st.session_state["prefill_question"] = q

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state["messages"] = []
        if "chatbot" in st.session_state:
            st.session_state["chatbot"].reset_history()

# ─── Initialize Chatbot ───────────────────────────────────────────────────────
@st.cache_resource
def load_chatbot():
    bot = ChurnChatbot()
    bot.connect()
    return bot

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("📉 Cell2Cell Churn Analytics Assistant")
st.markdown("Ask questions about customer churn patterns, model predictions, and retention strategies.")



st.divider()

# ─── Chat Interface ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle example question prefill
prefill = st.session_state.pop("prefill_question", None)
if prefill:
    user_input = prefill
else:
    user_input = st.chat_input("Ask a question about customer churn...")

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Get answer from RAG chatbot
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            try:
                bot = load_chatbot()
                result = bot.ask(user_input)
                answer = result["answer"]

                st.markdown(answer)

                # Show retrieved context in expander
                with st.expander("🔍 Retrieved Context", expanded=False):
                    tabs = st.tabs(["Segments", "Customer Examples", "Feature Definitions"])
                    with tabs[0]:
                        for doc in result["retrieved_docs"].get("segment_docs", []):
                            st.markdown(f"- {doc}")
                    with tabs[1]:
                        for doc in result["retrieved_docs"].get("customer_docs", [])[:3]:
                            st.markdown(f"- {doc[:300]}...")
                    with tabs[2]:
                        for doc in result["retrieved_docs"].get("feature_docs", []):
                            st.markdown(f"- {doc}")

            except Exception as e:
                answer = f"⚠️ Error: {e}. Ensure ChromaDB is indexed and OPENAI_API_KEY is set."
                st.error(answer)

    st.session_state["messages"].append({"role": "assistant", "content": answer})
