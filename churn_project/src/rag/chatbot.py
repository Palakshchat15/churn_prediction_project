# =============================================================================
# src/rag/chatbot.py
# LLM query handler using Ollama (mistral 7B).
# Handles exact CustomerID lookups directly from the engineered CSV
# so every customer can be queried regardless of FAISS sample size.
# =============================================================================

import sys
import os
import re
import requests
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import (
    LLM_PROVIDER, LLM_MODEL, LLM_MAX_TOKENS,
    OLLAMA_BASE_URL, OPENAI_API_KEY, ANTHROPIC_API_KEY,
    DATA_ENGINEERED, TARGET_COL,
)
from .retriever import FAISSRetriever

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a precise Customer Churn Analytics Assistant for
the Cell2Cell telecom dataset.

DATASET FACTS:
- 51,047 customers | 29% churn rate | 76 features after engineering
- Primary model: XGBoost | Secondary: LightGBM
- Key churn drivers: CreditRating, MonthsInService, RetentionCalls,
  MonthlyRevenue, DropRate, MadeCallToRetentionTeam, OverageRatio

YOUR INSTRUCTIONS:
1. Answer using ONLY the context provided — do not use general knowledge.
2. Always quote specific numbers and values directly from the context.
3. If a specific customer's data is provided in the context, answer the
   question about that customer directly and completely.
4. If the exact customer is not found, clearly state that and then describe
   the most similar or highest-risk customer available in the context.
5. Structure answers clearly — use bullet points for multi-part answers.
6. End every answer with one concrete, actionable retention recommendation.

STRICT RULE: Do not invent statistics. Only use the provided context.
""".strip()


class ChurnChatbot:
    """
    RAG-powered chatbot with direct CSV lookup for CustomerID queries.

    For specific CustomerID queries:
        1. First tries exact lookup from the full engineered CSV
        2. Falls back to FAISS semantic search if CSV lookup fails
        3. If customer not found anywhere, returns the highest churn
           risk customer as an alternative

    For general queries:
        Standard FAISS semantic search across all collections.
    """

    def __init__(self):
        self.retriever    = FAISSRetriever()
        self.chat_history = []
        self._llm_ready   = False
        self._df          = None   # full engineered dataset for direct lookup

    # --------------------------------------------------------------------------
    def connect(self):
        self.retriever.connect()
        self._load_dataframe()
        self._verify_llm()
        return self

    # --------------------------------------------------------------------------
    def _load_dataframe(self):
        """Load the full engineered CSV for direct CustomerID lookups."""
        try:
            self._df = pd.read_csv(DATA_ENGINEERED)
            print(f"[BOT] Loaded dataset: {len(self._df):,} customers "
                  f"for direct lookup")
        except Exception as e:
            print(f"[BOT] WARNING: Could not load dataset for lookup: {e}")

    # --------------------------------------------------------------------------
    def _verify_llm(self):
        if LLM_PROVIDER == "ollama":
            try:
                r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
                if r.status_code == 200:
                    models  = [m["name"] for m in r.json().get("models", [])]
                    matched = [m for m in models
                               if m.split(":")[0] == LLM_MODEL.split(":")[0]]
                    if matched:
                        print(f"[LLM] Ollama ready  —  model: {matched[0]}")
                        self._llm_ready = True
                    else:
                        print(f"[LLM] WARNING: '{LLM_MODEL}' not found.")
                        print(f"      Run: ollama pull {LLM_MODEL}")
            except requests.exceptions.ConnectionError:
                print("[LLM] WARNING: Ollama not running. Run: ollama serve")
            except Exception as e:
                print(f"[LLM] WARNING: {e}")

        elif LLM_PROVIDER == "openai" and OPENAI_API_KEY:
            print(f"[LLM] OpenAI ready  —  model: {LLM_MODEL}")
            self._llm_ready = True

        elif LLM_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
            print(f"[LLM] Anthropic ready  —  model: {LLM_MODEL}")
            self._llm_ready = True

    # --------------------------------------------------------------------------
    def _extract_customer_id(self, query: str):
        """Extract a CustomerID integer from a query string if present."""
        patterns = [
            r'customer\s*id\s*[:#]?\s*(\d+)',
            r'customerid\s*[:#]?\s*(\d+)',
            r'\bid\s*[:#]?\s*(\d+)',
            r'#\s*(\d+)',
            r'\b(\d{5,})\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return int(match.group(1))
        return None

    # --------------------------------------------------------------------------
    def _customer_row_to_text(self, row: pd.Series) -> str:
        """Convert a single customer dataframe row into a rich text document."""
        cid         = int(row.get("CustomerID", 0))
        churn_label = "CHURNED" if row.get(TARGET_COL, 0) == 1 else "RETAINED"

        return (
            f"Customer ID {cid}: {churn_label}.\n"
            f"  Monthly Revenue         : ${row.get('MonthlyRevenue', 0):.2f}\n"
            f"  Monthly Minutes         : {row.get('MonthlyMinutes', 0):.0f}\n"
            f"  Months in Service       : {row.get('MonthsInService', 0):.0f}\n"
            f"  Credit Rating           : {row.get('CreditRating', 'N/A')}\n"
            f"  Retention Calls         : {row.get('RetentionCalls', 0):.0f}\n"
            f"  Retention Offers Accept : {row.get('RetentionOffersAccepted', 0):.0f}\n"
            f"  Called Retention Team   : "
            f"{'Yes' if row.get('MadeCallToRetentionTeam', 0) == 1 else 'No'}\n"
            f"  Drop Rate               : {row.get('DropRate', 0):.4f}\n"
            f"  Overage Ratio           : {row.get('OverageRatio', 0):.4f}\n"
            f"  Revenue Per Minute      : ${row.get('RevenuePerMinute', 0):.4f}\n"
            f"  Equipment Age (years)   : {row.get('EquipmentAgeYears', 0):.1f}\n"
            f"  Total Recurring Charge  : ${row.get('TotalRecurringCharge', 0):.2f}\n"
            f"  Income Group            : {row.get('IncomeGroup', 'N/A')}\n"
            f"  Overage Minutes         : {row.get('OverageMinutes', 0):.0f}\n"
            f"  Customer Care Calls     : {row.get('CustomerCareCalls', 0):.0f}\n"
            f"  Owns Computer           : "
            f"{'Yes' if row.get('OwnsComputer', 0) == 1 else 'No'}\n"
            f"  Has Credit Card         : "
            f"{'Yes' if row.get('HasCreditCard', 0) == 1 else 'No'}"
        )

    # --------------------------------------------------------------------------
    def _direct_customer_lookup(self, customer_id: int) -> str:
        """
        Look up a customer directly from the full engineered CSV.
        Returns formatted text or empty string if not found.
        """
        if self._df is None:
            return ""

        match = self._df[self._df["CustomerID"] == customer_id]
        if match.empty:
            return ""

        return self._customer_row_to_text(match.iloc[0])

    # --------------------------------------------------------------------------
    def _highest_churn_risk_customer(self) -> str:
        """
        Return the customer with the highest churn probability from the CSV.
        Used as fallback when a queried CustomerID doesn't exist.
        Proxy: churned customers sorted by lowest revenue (highest risk signal).
        """
        if self._df is None:
            return ""

        churned = self._df[self._df[TARGET_COL] == 1].copy()
        if churned.empty:
            return ""

        # Sort by most retention calls + lowest revenue as high-risk proxy
        churned["risk_score"] = (
            churned.get("RetentionCalls", 0) * 2 +
            (1 / (churned.get("MonthlyRevenue", 1) + 1))
        )
        top = churned.sort_values("risk_score", ascending=False).iloc[0]
        return self._customer_row_to_text(top)

    # --------------------------------------------------------------------------
    def _build_context(self, query: str) -> tuple:
        """
        Build context for the LLM.

        For CustomerID queries:
            1. Direct CSV lookup (covers ALL 51k customers)
            2. If not found — highest risk customer as alternative
            3. Plus FAISS segment/feature docs

        For general queries:
            Standard FAISS retrieval across all collections.

        Returns:
            (context_string, retrieval_dict)
        """
        customer_id = self._extract_customer_id(query)
        retrieval   = self.retriever.retrieve(query)

        if customer_id is not None:
            # Try direct CSV lookup first
            customer_text = self._direct_customer_lookup(customer_id)

            if customer_text:
                print(f"[BOT] Direct CSV lookup: CustomerID {customer_id} found")
                customer_section = (
                    f"=== CUSTOMER ID {customer_id} — DIRECT LOOKUP ===\n"
                    f"{customer_text}"
                )
            else:
                print(f"[BOT] CustomerID {customer_id} not found in dataset")
                highest_risk  = self._highest_churn_risk_customer()
                customer_section = (
                    f"=== CUSTOMER LOOKUP RESULT ===\n"
                    f"Customer ID {customer_id} does not exist in the dataset.\n\n"
                    f"=== HIGHEST CHURN RISK CUSTOMER (shown as alternative) ===\n"
                    f"{highest_risk}"
                )

            # Combine with segment and feature context from FAISS
            parts = [customer_section]
            if retrieval["segment_docs"]:
                parts.append(
                    "=== SEGMENT INSIGHTS ===\n" +
                    "\n".join(retrieval["segment_docs"])
                )
            if retrieval["feature_docs"]:
                parts.append(
                    "=== FEATURE DEFINITIONS ===\n" +
                    "\n".join(retrieval["feature_docs"])
                )
            context = "\n\n".join(parts)

        else:
            context = retrieval["context_string"]

        return context, retrieval

    # --------------------------------------------------------------------------
    def _build_messages(self, question: str, context: str,
                        use_history: bool) -> list:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if use_history and self.chat_history:
            messages.extend(self.chat_history[-6:])
        user_content = (
            f"RETRIEVED CONTEXT:\n"
            f"{'-' * 50}\n"
            f"{context}\n"
            f"{'-' * 50}\n\n"
            f"QUESTION: {question}\n\n"
            f"Answer based strictly on the context above."
        )
        messages.append({"role": "user", "content": user_content})
        return messages

    # --------------------------------------------------------------------------
    def _call_ollama(self, messages: list) -> str:
        payload = {
            "model":    LLM_MODEL,
            "messages": messages,
            "stream":   False,
            "options":  {
                "temperature":    0.1,
                "num_predict":    LLM_MAX_TOKENS,
                "top_p":          0.9,
                "repeat_penalty": 1.1,
            },
        }
        try:
            r = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload, timeout=180,
            )
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except requests.exceptions.Timeout:
            return "Response timed out — try a shorter question or retry."
        except requests.exceptions.ConnectionError:
            return "Cannot reach Ollama. Run: ollama serve"
        except Exception as e:
            return f"Ollama error: {e}"

    # --------------------------------------------------------------------------
    def _call_openai(self, messages: list) -> str:
        try:
            from openai import OpenAI
            client   = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=LLM_MODEL, messages=messages,
                max_tokens=LLM_MAX_TOKENS, temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"OpenAI error: {e}"

    # --------------------------------------------------------------------------
    def _call_anthropic(self, messages: list) -> str:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            system = next(
                (m["content"] for m in messages if m["role"] == "system"), ""
            )
            user_msgs = [m for m in messages if m["role"] != "system"]
            response  = client.messages.create(
                model=LLM_MODEL, max_tokens=LLM_MAX_TOKENS,
                system=system, messages=user_msgs,
            )
            return response.content[0].text.strip()
        except Exception as e:
            return f"Anthropic error: {e}"

    # --------------------------------------------------------------------------
    def _call_llm(self, messages: list) -> str:
        if LLM_PROVIDER == "ollama":
            return self._call_ollama(messages)
        elif LLM_PROVIDER == "openai":
            return self._call_openai(messages)
        elif LLM_PROVIDER == "anthropic":
            return self._call_anthropic(messages)
        return f"Unknown LLM_PROVIDER '{LLM_PROVIDER}' in config.py"

    # --------------------------------------------------------------------------
    def ask(self, question: str, use_history: bool = True) -> dict:
        """Submit a question and get a grounded answer."""
        context, retrieval = self._build_context(question)

        if not context.strip():
            context = "No relevant documents found. Run the RAG indexing step."

        messages = self._build_messages(question, context, use_history)

        if not self._llm_ready:
            answer = (
                f"LLM not ready. Ensure Ollama is running and "
                f"'{LLM_MODEL}' is pulled."
            )
        else:
            answer = self._call_llm(messages)

        self.chat_history.append({"role": "user",      "content": question})
        self.chat_history.append({"role": "assistant",  "content": answer})

        return {
            "question":       question,
            "context":        context,
            "answer":         answer,
            "retrieved_docs": retrieval,
        }

    # --------------------------------------------------------------------------
    def reset_history(self):
        self.chat_history = []
        print("[BOT] Conversation history cleared.\n")

    # --------------------------------------------------------------------------
    def interactive(self):
        print("\n" + "=" * 60)
        print("  CELL2CELL CHURN ANALYTICS CHATBOT")
        print(f"  Model    : {LLM_MODEL}  via  {LLM_PROVIDER}")
        print("  Commands : quit | reset | context")
        print("=" * 60)

        print("\nExample questions:")
        examples = [
            "What is the monthly revenue for customer ID 3000002?",
            "Is customer 3000050 at risk of churning?",
            "Which customers have the highest churn risk?",
            "What is the churn rate for low credit rating customers?",
            "Which features drive churn the most?",
            "What retention strategies work for high overage customers?",
        ]
        for i, q in enumerate(examples, 1):
            print(f"  {i}. {q}")
        print()

        last_context = ""

        while True:
            try:
                question = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye.")
                break

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye.")
                break
            if question.lower() == "reset":
                self.reset_history()
                continue
            if question.lower() == "context":
                print("\n--- Last Retrieved Context ---")
                print(last_context if last_context else "(none yet)")
                print("------------------------------\n")
                continue

            print("\nAssistant: thinking...\n")
            result       = self.ask(question)
            last_context = result["context"]
            print(f"Assistant:\n{result['answer']}\n")
            print("-" * 60 + "\n")


if __name__ == "__main__":
    bot = ChurnChatbot()
    bot.connect()
    bot.interactive()