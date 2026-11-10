"""Merrick Bank Customer Retention Assistant (RAG + LangChain + Gradio).

Standalone version of Merrick_Retention_Agent.ipynb. Run with:  python app.py
Set REBUILD_INDEX=1 to re-embed docs/ instead of loading the saved FAISS index.
"""
import json
import os
import re
import warnings
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # picks up OPENAI_API_KEY, LLM_PROVIDER, etc. from a local .env file

# Project folders (Task 1.4). Works both as a notebook and as app.py.
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = BASE_DIR / "docs"
INDEX_DIR = BASE_DIR / "index"

CHURN_CSV = DATA_DIR / "Churn_Modeling1.csv"
SENTIMENT_CSV = DATA_DIR / "Churn_With_Sentiment.csv"

# Retrieval settings
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
TOP_K = 4
MIN_SIMILARITY = 0.25  # cosine similarity floor; chunks below this are treated as "not relevant"

# LLM settings. LLM_PROVIDER can be "openai" or "ollama".
# If it is not set, OpenAI is used when a key is present, otherwise Ollama.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
MAX_HISTORY_TURNS = 6  # how many past question/answer pairs are sent back to the LLM


def prepare_folders():
    """Task 1.4: make sure data/, docs/, and index/ exist."""
    for folder in (DATA_DIR, DOCS_DIR, INDEX_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    return {f.name: f.exists() for f in (DATA_DIR, DOCS_DIR, INDEX_DIR)}


def list_doc_files():
    """All .txt and .md files under docs/, skipping Jupyter checkpoint copies."""
    files = [p for ext in ("*.txt", "*.md") for p in DOCS_DIR.rglob(ext)]
    return sorted(p for p in files if ".ipynb_checkpoints" not in p.parts)


def verify_documents():
    """Task 1.3: confirm each document decodes as UTF-8 and is not empty."""
    rows = []
    for path in list_doc_files():
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
            readable, encoding = True, "utf-8"
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
            readable, encoding = False, "NOT utf-8 (fell back to latin-1)"
        rows.append({
            "file": path.name,
            "bytes": len(raw),
            "characters": len(text),
            "lines": text.count("\n") + 1,
            "encoding": encoding,
            "windows_line_endings": "\r\n" in text,
            "readable": readable and len(text.strip()) > 0,
        })
    return pd.DataFrame(rows)


def load_documents():
    """Task 2.1: load every .txt and .md file in docs/ with DirectoryLoader + TextLoader."""
    loader = DirectoryLoader(
        str(DOCS_DIR),
        glob=["**/*.txt", "**/*.md"],
        exclude=["**/.ipynb_checkpoints/**"],
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    for doc in docs:
        doc.page_content = doc.page_content.replace("\r\n", "\n").strip()
        doc.metadata["source"] = Path(doc.metadata["source"]).name  # keep just the file name for citations
    return sorted(docs, key=lambda d: d.metadata["source"])


def split_documents(docs):
    """Task 2.2: split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def get_embeddings():
    """Task 2.3: Hugging Face sentence-transformer embeddings (normalized, so distances map to cosine similarity)."""
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_or_load_index(embeddings, rebuild=False):
    """Task 2.4: build the FAISS index from docs/ and save it to index/, or load the saved copy."""
    index_file = INDEX_DIR / "index.faiss"
    if index_file.exists() and not rebuild:
        # The index was written by this project, so loading its pickle metadata is safe.
        return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
    chunks = split_documents(load_documents())
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(INDEX_DIR))
    return vectorstore


def retrieve(vectorstore, query, k=TOP_K, min_similarity=MIN_SIMILARITY):
    """Top-k chunks for a query, filtered by cosine similarity.

    FAISS returns squared L2 distance. With normalized embeddings,
    cosine similarity = 1 - distance / 2.
    """
    results = []
    for doc, distance in vectorstore.similarity_search_with_score(query, k=k):
        similarity = 1 - float(distance) / 2
        if similarity >= min_similarity:
            # Return a copy so the score from this query is not overwritten by a later query that hits the same chunk.
            results.append(Document(page_content=doc.page_content,
                                    metadata={**doc.metadata, "similarity": round(similarity, 3)}))
    return results


def format_docs(docs):
    """Number the retrieved chunks so the LLM can cite them as [1], [2], ..."""
    if not docs:
        return "NO RELEVANT DOCUMENTS WERE FOUND."
    return "\n\n".join(
        f"[{i}] (source: {d.metadata['source']})\n{d.page_content}" for i, d in enumerate(docs, start=1)
    )


class CustomerDB:
    """Structured customer data: churn profile plus optional sentiment label."""

    def __init__(self, churn_path=CHURN_CSV, sentiment_path=SENTIMENT_CSV):
        df = pd.read_csv(churn_path)
        self.has_sentiment = False
        if sentiment_path and Path(sentiment_path).exists():
            sentiment = pd.read_csv(sentiment_path)[["CustomerId", "Sentiment_Label"]]
            df = df.merge(sentiment, on="CustomerId", how="left")
            self.has_sentiment = True
        self.df = df.set_index("CustomerId")

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _segment(tenure, credit_score):
        """Segmentation rules from retention_policy.txt, highest matching tier wins."""
        if tenure > 3 or credit_score > 750:
            return "High-value"
        if 1 <= tenure <= 3 or 650 <= credit_score <= 750:
            return "Moderate-value"
        return "Low-value"

    @staticmethod
    def _fee_waiver_signals(tenure, credit_score, is_active):
        """Eligibility criteria from fee_waiver_policy.txt, limited to what the data can answer."""
        return {
            "active_12_months_or_more": bool(is_active and tenure >= 1),
            "credit_score_above_700": bool(credit_score > 700),
            "auto_approval_tenure_over_24_months": bool(tenure > 2),
            "late_payment_history": "unknown (not in the dataset)",
            "financial_hardship": "unknown (not in the dataset)",
        }

    def get_context(self, customer_id):
        """Task 3.2: profile, churn details, sentiment, and derived retention fields for one customer.

        Returns None when the ID is not a valid number or is not in the data.
        """
        try:
            customer_id = int(str(customer_id).strip())
        except (TypeError, ValueError):
            return None
        if customer_id not in self.df.index:
            return None

        row = self.df.loc[customer_id]
        tenure, score = int(row["Tenure"]), int(row["CreditScore"])
        exited, active = int(row["Exited"]), int(row["IsActiveMember"])
        sentiment = row["Sentiment_Label"] if self.has_sentiment and pd.notna(row["Sentiment_Label"]) else None

        risk_signals = []
        if exited:
            risk_signals.append("flagged as churned in the churn data (Exited = 1)")
        if not active:
            risk_signals.append("not an active member")
        if sentiment == "Negative":
            risk_signals.append("negative sentiment in recent feedback")

        return {
            "customer_id": customer_id,
            "surname": str(row["Surname"]),
            "geography": str(row["Geography"]),
            "gender": str(row["Gender"]),
            "age": int(row["Age"]),
            "tenure_years": tenure,
            "credit_score": score,
            "balance": float(row["Balance"]),
            "num_products": int(row["NumOfProducts"]),
            "has_credit_card": bool(row["HasCrCard"]),
            "is_active_member": bool(active),
            "estimated_salary": float(row["EstimatedSalary"]),
            "exited": bool(exited),
            "sentiment": sentiment,
            "retention_segment": self._segment(tenure, score),
            "fee_waiver_signals": self._fee_waiver_signals(tenure, score, active),
            "risk_signals": risk_signals,
        }


def resolve_customer_id(customer_id_field, question):
    """Use the ID typed in the UI field, or fall back to an 8-digit ID mentioned in the question."""
    field = str(customer_id_field or "").strip()
    if field:
        return field
    match = re.search(r"\b(\d{8})\b", question or "")
    return match.group(1) if match else None


def format_customer(lookup):
    """Turn a customer lookup result into the text block that goes into the system prompt."""
    if lookup["status"] == "none":
        return "No customer ID was provided. Answer in general terms and do not assume any customer details."
    if lookup["status"] == "not_found":
        return (f"A customer ID was provided ({lookup['id']}) but NO MATCHING RECORD exists in the customer database. "
                "Say that the customer could not be found, ask the user to check the ID, and answer only in general terms.")
    c = lookup["profile"]
    yes_no = lambda flag: "Yes" if flag else "No"
    waiver = c["fee_waiver_signals"]
    return "\n".join([
        f"Customer ID: {c['customer_id']}",
        f"Name on file: {c['surname']} (surname only; no first name is stored)",
        f"Geography: {c['geography']} | Gender: {c['gender']} | Age: {c['age']}",
        f"Tenure: {c['tenure_years']} years | Products held: {c['num_products']} | "
        f"Has credit card: {yes_no(c['has_credit_card'])} | Active member: {yes_no(c['is_active_member'])}",
        f"Credit score: {c['credit_score']} | Balance: {c['balance']:,.2f} | Estimated salary: {c['estimated_salary']:,.2f}",
        f"Churn flag (Exited): {yes_no(c['exited'])}",
        f"Feedback sentiment: {c['sentiment'] or 'not available'}",
        f"Retention segment (derived from the retention policy rules): {c['retention_segment']}",
        "Fee waiver signals (derived from the fee waiver policy criteria): "
        f"active 12+ months = {yes_no(waiver['active_12_months_or_more'])}; "
        f"credit score above 700 = {yes_no(waiver['credit_score_above_700'])}; "
        f"auto-approval (tenure over 24 months) = {yes_no(waiver['auto_approval_tenure_over_24_months'])}; "
        f"late payment history = {waiver['late_payment_history']}; "
        f"financial hardship = {waiver['financial_hardship']}",
        "Risk signals: " + ("; ".join(c["risk_signals"]) if c["risk_signals"] else "none in the data"),
    ])


SYSTEM_PROMPT = """You are the Merrick Bank Customer Retention Assistant. You help retention agents and \
customers with questions about accounts, cards, fees, rewards, complaints, and retention offers.

COMPLIANCE GUIDELINES (always follow these)
1. Grounding: every statement about a policy, product, fee, rate, timeline, or offer must come from the \
RETRIEVED CONTEXT below. Whenever you use the context, you MUST cite the chunk right after the statement, \
using its number in square brackets, like [1]. Put each number in its own brackets, for example [1][3]. \
An answer that uses the context but has no [n] citation is not acceptable. \
Never invent fees, rates, offers, or eligibility rules.
2. Missing information: if the context does not contain the answer, say plainly that you do not have that \
information in the bank's documents and suggest contacting customer support. Do not guess.
3. Offers: only suggest retention offers that the retention policy lists for this customer's segment. Never \
promise anything beyond policy limits. Any exception must be approved by the Retention Desk Manager.
4. No guarantees: describe eligibility conditions instead of promising approval, unless the policy says the \
request is auto-approved for this customer. If a criterion is marked unknown in the profile, say it still \
has to be verified.
5. Privacy: only discuss the customer in the CUSTOMER PROFILE below. Never reveal or speculate about any \
other customer. Use the profile to personalize, but do not quote the customer's salary or exact balance.
6. Tone: be empathetic, respectful, and transparent. Acknowledge the concern first, use plain language \
without jargon, offer one relevant option where a retention offer applies, and end with a clear next step.
7. Keep amounts and currency symbols exactly as they appear in the context.

CUSTOMER PROFILE
{customer_block}

RETRIEVED CONTEXT
{context}

Keep the answer concise (under about 180 words) unless the user asks for more detail."""

FALLBACK_ANSWER = (
    "I'm sorry, I couldn't find anything in Merrick Bank's policy and product documents that answers that, "
    "so I don't want to guess. I can help with annual fees and fee waivers, card and account products, "
    "rewards, complaints, and retention offers. If you include a customer ID, I can tailor the answer to "
    "that customer. For anything else, please contact customer support."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


def get_llm():
    """Task 4.3: ChatOpenAI or ChatOllama, chosen by LLM_PROVIDER or by whether an OpenAI key is present."""
    provider = os.getenv("LLM_PROVIDER") or ("openai" if os.getenv("OPENAI_API_KEY") else "ollama")
    if provider.lower() == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)
    from langchain_ollama import ChatOllama
    return ChatOllama(model=OLLAMA_MODEL, temperature=0.2)


def build_chain(vectorstore, customer_db, llm):
    """Task 4.1: retriever + customer lookup + prompt + LLM, composed from Runnables."""

    def lookup_customer(x):
        cid = resolve_customer_id(x.get("customer_id"), x["question"])
        if cid is None:
            return {"status": "none", "id": None, "profile": None}
        profile = customer_db.get_context(cid)
        return {"status": "found" if profile else "not_found", "id": cid, "profile": profile}

    gather = RunnablePassthrough.assign(
        # The customer ID is stripped before searching: digits add noise to the embedding and match nothing in the docs.
        docs=RunnableLambda(lambda x: retrieve(vectorstore, re.sub(r"\b\d{8}\b", "", x["question"]).strip())),
        customer=RunnableLambda(lookup_customer),
    )
    shape = RunnablePassthrough.assign(
        context=lambda x: format_docs(x["docs"]),
        customer_block=lambda x: format_customer(x["customer"]),
        history=lambda x: x.get("history", []),
    )
    generate = prompt | llm | StrOutputParser()

    nothing_to_go_on = lambda x: not x["docs"] and x["customer"]["status"] == "none" and not x["history"]
    answer = RunnableBranch(
        (nothing_to_go_on, RunnableLambda(lambda x: FALLBACK_ANSWER)),
        generate,
    )
    return gather | shape | RunnablePassthrough.assign(answer=answer)


def to_messages(turns):
    """Convert [(question, answer), ...] into LangChain messages, keeping only the most recent turns."""
    messages = []
    for question, answer in turns[-MAX_HISTORY_TURNS:]:
        messages += [HumanMessage(content=question), AIMessage(content=answer)]
    return messages


def ask(chain, question, customer_id="", turns=None):
    """Run the chain once and return the full result (answer, docs, customer lookup)."""
    return chain.invoke({
        "question": question.strip(),
        "customer_id": customer_id,
        "history": to_messages(turns or []),
    })


def format_citations(docs):
    """Markdown list of the sources behind an answer, numbered to match the [n] citations."""
    if not docs:
        return "_No policy documents were retrieved for this answer._"
    lines = []
    for i, d in enumerate(docs, start=1):
        snippet = " ".join(d.page_content.split())[:160]
        lines.append(f"**[{i}] {d.metadata['source']}** (similarity {d.metadata.get('similarity', 0):.2f}): {snippet}...")
    return "\n\n".join(lines)


EXAMPLES = [
    ["How can I waive my annual fee?", ""],
    ["What should we offer to retain customer 15634602?", ""],
    ["Explain the difference between Gold and Platinum cards.", ""],
    ["I'm thinking about closing my account. The fees are too high.", "15737452"],
]


def customer_panel(lookup):
    """Markdown summary of the customer for the side panel."""
    if lookup["status"] == "none":
        return "_No customer selected. Enter an 8-digit customer ID for a personalized answer._"
    if lookup["status"] == "not_found":
        return f"**Customer {lookup['id']} was not found.** Check the ID. Answers will be general."
    c = lookup["profile"]
    return (
        f"**{c['surname']}** (ID {c['customer_id']})\n\n"
        f"- Segment: **{c['retention_segment']}**\n"
        f"- Tenure: {c['tenure_years']} years, credit score {c['credit_score']}\n"
        f"- Products: {c['num_products']}, active member: {'yes' if c['is_active_member'] else 'no'}\n"
        f"- Churn flag: {'yes' if c['exited'] else 'no'}, sentiment: {c['sentiment'] or 'n/a'}\n"
        f"- Risk signals: {'; '.join(c['risk_signals']) or 'none'}"
    )


def build_ui(chain):
    """Task 5: Gradio chat UI with customer ID input, citations, and chat history."""
    import gradio as gr

    def respond(message, customer_id, chat, turns):
        message = (message or "").strip()
        if not message:
            return "", chat, turns, gr.update(), gr.update()
        try:
            result = ask(chain, message, customer_id, turns)
            answer = result["answer"].strip()
            sources = sorted({d.metadata["source"] for d in result["docs"]})
            shown = answer + ("\n\n**Sources:** " + ", ".join(sources) if sources else "")
            panel, cites = customer_panel(result["customer"]), format_citations(result["docs"])
            turns = turns + [(message, answer)]
        except Exception as exc:  # most often: no API key, or Ollama is not running
            shown = ("Sorry, I couldn't reach the language model, so I can't answer right now. "
                     f"Please check the LLM setup and try again. (Details: {type(exc).__name__}: {exc})")
            panel, cites = gr.update(), gr.update()
        chat = chat + [{"role": "user", "content": message}, {"role": "assistant", "content": shown}]
        return "", chat, turns, panel, cites

    def clear():
        return [], [], customer_panel({"status": "none"}), "_Sources for the latest answer appear here._"

    with gr.Blocks(title="Merrick Bank Retention Assistant") as demo:
        gr.Markdown("# Merrick Bank Customer Retention Assistant\n"
                    "Answers are grounded in the bank's policy documents and the customer's churn profile.")
        turns = gr.State([])
        with gr.Row():
            with gr.Column(scale=3):
                try:
                    chatbot = gr.Chatbot(label="Conversation", height=460, type="messages")
                except TypeError:  # Gradio 6 uses the messages format by default and has no "type" argument
                    chatbot = gr.Chatbot(label="Conversation", height=460)
                with gr.Row():
                    customer_id = gr.Textbox(label="Customer ID (optional)", placeholder="e.g. 15634602", scale=1)
                    question = gr.Textbox(label="Question", placeholder="Ask about fees, cards, rewards, retention offers...", scale=3)
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear chat")
                gr.Examples(examples=EXAMPLES, inputs=[question, customer_id], label="Sample queries")
            with gr.Column(scale=2):
                gr.Markdown("### Customer")
                profile_md = gr.Markdown(customer_panel({"status": "none"}))
                gr.Markdown("### Sources for the latest answer")
                sources_md = gr.Markdown("_Sources for the latest answer appear here._")

        io = dict(inputs=[question, customer_id, chatbot, turns],
                  outputs=[question, chatbot, turns, profile_md, sources_md])
        send.click(respond, **io)
        question.submit(respond, **io)
        clear_btn.click(clear, outputs=[chatbot, turns, profile_md, sources_md])
    return demo


TEST_CASES = [
    {"name": "Fee waiver (general)", "question": "How can I waive my annual fee?", "customer_id": "",
     "expected_source": "fee_waiver_policy.txt", "expect_personalized": False, "expect_missing": False},
    {"name": "Retention offer (ID in question)", "question": "What should we offer to retain customer 15634602?", "customer_id": "",
     "expected_source": "retention_policy.txt", "expect_personalized": True, "expect_missing": False},
    {"name": "Gold vs Platinum", "question": "Explain the difference between Gold and Platinum cards.", "customer_id": "",
     "expected_source": "product_summary.txt", "expect_personalized": False, "expect_missing": False},
    {"name": "At-risk customer wants to leave", "question": "I'm thinking about closing my account. The fees are too high.", "customer_id": "15737452",
     "expected_source": None, "expect_personalized": True, "expect_missing": False},
    {"name": "Unknown customer ID", "question": "What retention offer does this customer qualify for?", "customer_id": "99999999",
     "expected_source": "retention_policy.txt", "expect_personalized": False, "expect_missing": True},
    {"name": "Off-topic question", "question": "What is the weather in Columbus today?", "customer_id": "",
     "expected_source": None, "expect_personalized": False, "expect_missing": True},
]

MISSING_PHRASES = ("couldn't find", "could not find", "could not be found", "not found", "don't have", "do not have",
                   "no matching", "unable to find", "check the id", "verify the id", "not able to")

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are grading a bank customer-retention assistant. Score the ANSWER from 1 (poor) to 5 (excellent) on:\n"
     "- grounding: every factual claim is supported by the CONTEXT or CUSTOMER PROFILE, nothing is invented. "
     "If the context is empty and the answer correctly says it has no information, that is a 5.\n"
     "- personalization: uses the customer profile appropriately. If no customer applies, score how well it avoids assuming one.\n"
     "- tone: empathetic, respectful, plain language, compliant (no promises beyond policy).\n"
     "- coherence: clear, well organized, directly answers the question.\n"
     'Reply with JSON only, like {{"grounding": 5, "personalization": 4, "tone": 5, "coherence": 5, "comment": "one short sentence"}}'),
    ("human", "QUESTION:\n{question}\n\nCUSTOMER PROFILE:\n{customer_block}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"),
])


def extract_citations(answer):
    """Chunk numbers cited in an answer. Accepts [1], [1][2], [1, 2], and [1-3]."""
    cited = []
    for group in re.findall(r"\[([\d\s,\-]+)\]", answer):
        for part in group.split(","):
            part = part.strip()
            if re.fullmatch(r"\d+\s*-\s*\d+", part):
                low, high = (int(n) for n in part.split("-"))
                cited += list(range(low, high + 1))
            elif part.isdigit():
                cited.append(int(part))
    return cited


def automated_checks(case, result):
    """Objective checks on one result."""
    answer, docs, lookup = result["answer"], result["docs"], result["customer"]
    sources = [d.metadata["source"] for d in docs]
    cited = extract_citations(answer)

    checks = {
        "expected_source_retrieved": (case["expected_source"] in sources) if case["expected_source"] else None,
        "citations_valid": (bool(cited) and all(1 <= n <= len(docs) for n in cited)) if docs and not case["expect_missing"] else None,
        "personalized": None,
        "handled_missing": None,
    }
    if case["expect_personalized"] and lookup["status"] == "found":
        profile = lookup["profile"]
        segment_word = profile["retention_segment"].split("-")[0].lower()  # high / moderate / low
        checks["personalized"] = profile["surname"].lower() in answer.lower() or segment_word in answer.lower()
    if case["expect_missing"]:
        checks["handled_missing"] = any(p in answer.lower() for p in MISSING_PHRASES)
    return checks


def judge_answer(llm, result):
    """LLM-as-judge scores for one result. Returns {} if the judge reply cannot be parsed."""
    reply = (JUDGE_PROMPT | llm | StrOutputParser()).invoke({
        "question": result["question"],
        "customer_block": result["customer_block"],
        "context": result["context"],
        "answer": result["answer"],
    })
    match = re.search(r"\{.*\}", reply, flags=re.DOTALL)
    try:
        return json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        return {}


def run_evaluation(chain, llm, cases=TEST_CASES):
    """Run every test case and return (results table, raw results)."""
    rows, raw = [], []
    for case in cases:
        result = ask(chain, case["question"], case["customer_id"])
        scores = judge_answer(llm, result)
        rows.append({
            "test": case["name"],
            "sources": ", ".join(sorted({d.metadata["source"] for d in result["docs"]})) or "(none)",
            **automated_checks(case, result),
            **{k: scores.get(k) for k in ("grounding", "personalization", "tone", "coherence", "comment")},
        })
        raw.append({"case": case, "result": result})
    return pd.DataFrame(rows), raw


def main():
    """Build everything and launch the Gradio app."""
    prepare_folders()
    report = verify_documents()
    if report.empty or not report.readable.all():
        raise SystemExit("docs/ is empty or contains an unreadable file. See verify_documents().")
    vectorstore = build_or_load_index(get_embeddings(), rebuild=os.getenv("REBUILD_INDEX") == "1")
    chain = build_chain(vectorstore, CustomerDB(), get_llm())
    build_ui(chain).launch()


if __name__ == "__main__":
    main()
