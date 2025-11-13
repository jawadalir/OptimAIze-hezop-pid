# app_streamlit_rag_agent.py
import os
import json
import uuid
import re
import hashlib
from datetime import datetime
from difflib import SequenceMatcher

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import httpx

from embedding import (  # type: ignore[import]
    PINECONE_IMPORT_ERROR,
    get_pinecone_index,
    load_metas,
    normalize_vector,
    pinecone_index_ready,
)

# ==========================
# Configuration
# ==========================
load_dotenv()

# Streamlit page config must be called before any other Streamlit command
st.set_page_config(page_title="PID + HAZOP RAG Agent (auto-input)", layout="wide")

# OpenAI client
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY")
if not api_key:
    st.error("❌ No OpenAI API key found. Set OPENAI_API_KEY or OPEN_AI_KEY in your .env.")
    st.stop()
http_client = httpx.Client(trust_env=False)
client = OpenAI(api_key=api_key, http_client=http_client)

# Embedding model config
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072  # same as model

# Paths
INDEX_DIR = "./rag_store"
META_PATH = os.path.join(INDEX_DIR, "metadata.json")
CLASSIFIED_JSON = "classified_pipeline_tags2.json"
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs("temp_pdfs", exist_ok=True)

# session state initialization
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []
if "display_history" not in st.session_state:
    st.session_state.display_history = []

# ==========================
# Plant metadata loading
# ==========================
DATA = {}
PIPELINES = {}
PROCESS_DATA = {}
TAG_TO_PIPELINES = {}


def normalize_tag(tag: str) -> str:
    if not isinstance(tag, str):
        return ""
    return re.sub(r"[^a-zA-Z0-9]", "", tag).lower()


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def build_tag_index(pipelines):
    index = {}
    for pipe_tag, pipe_info in pipelines.items():
        for step in pipe_info.get("complete_flow", []):
            step_tag = step.get("tag")
            details = step.get("details") or {}
            detail_tag = details.get("Tag")
            for t in {step_tag, detail_tag}:
                if not t:
                    continue
                norm = normalize_tag(t)
                index.setdefault(norm, set()).add(pipe_tag)
    return index


def load_plant_data():
    global DATA, PIPELINES, PROCESS_DATA, TAG_TO_PIPELINES
    if not os.path.exists(CLASSIFIED_JSON):
        DATA = {}
        PIPELINES = {}
        PROCESS_DATA = {}
        TAG_TO_PIPELINES = {}
        return {}
    try:
        with open(CLASSIFIED_JSON, "r", encoding="utf-8") as f:
            DATA = json.load(f)
    except json.JSONDecodeError:
        st.error("❌ 'classified_pipeline_tags2.json' is not valid JSON.")
        st.stop()
    PIPELINES = DATA.get("complete_pipeline_flows", {}) or {}
    PROCESS_DATA = DATA.get("process_data", {}) or {}
    TAG_TO_PIPELINES = build_tag_index(PIPELINES)
    return DATA


load_plant_data()
# ==========================
# Retrieval & agentic tool
# ==========================
def embed_query(q):
    emb = client.embeddings.create(input=q, model=EMBED_MODEL).data[0].embedding
    v = np.array(emb, dtype="float32")
    return normalize_vector(v)


def retrieve(query, topk=6):
    if topk <= 0:
        return []
    query_vec = embed_query(query)
    if query_vec.size == 0:
        return []
    pinecone_index = get_pinecone_index(EMBED_DIM)
    try:
        response = pinecone_index.query(
            vector=query_vec.tolist(),
            top_k=topk,
            include_metadata=True,
        )
    except Exception as e:
        st.warning(f"Failed to query Pinecone index: {e}")
        return []

    matches = response.get("matches") or []
    results = []
    for match in matches:
        meta = match.get("metadata") or {}
        score = match.get("score")
        try:
            score = float(score) if score is not None else None
        except Exception:
            score = None
        results.append((meta, score))
    return results
# Minimal local context builders (kept from original code)
def find_best_tag_matches(query, data_list, threshold=0.6):
    results = []
    if not data_list:
        return results
    q_raw = query.lower()
    q_norm = normalize_tag(query)
    for item in data_list:
        tag = item.get("Tag", "")
        tag_lower = tag.lower()
        tag_norm = normalize_tag(tag)
        if tag_lower and tag_lower in q_raw:
            results.append(item)
            continue
        if tag_norm and similarity(q_norm, tag_norm) >= threshold:
            results.append(item)
    return results

def find_pipeline_matches(query, threshold=0.6):
    q_raw = query.lower()
    q_norm = normalize_tag(query)
    matches = {}
    for pipe_tag, pipe_info in PIPELINES.items():
        tag_lower = pipe_tag.lower()
        tag_norm = normalize_tag(pipe_tag)
        if tag_lower in q_raw:
            matches[pipe_tag] = pipe_info
            continue
        if similarity(q_norm, tag_norm) >= threshold:
            matches[pipe_tag] = pipe_info
    return matches

def find_pipelines_for_tag(tag_query: str, threshold: float = 0.7):
    results = {}
    if not tag_query:
        return results
    q_norm = normalize_tag(tag_query)
    if not q_norm:
        return results
    direct = TAG_TO_PIPELINES.get(q_norm, set())
    for p_tag in direct:
        if p_tag in PIPELINES:
            results[p_tag] = PIPELINES[p_tag]
    for t_norm, pipe_ids in TAG_TO_PIPELINES.items():
        if t_norm in q_norm or q_norm in t_norm or similarity(q_norm, t_norm) >= threshold:
            for p_tag in pipe_ids:
                if p_tag in PIPELINES:
                    results[p_tag] = PIPELINES[p_tag]
    return results

def build_local_context(query):
    context = {"equipment": [], "instrumentation": [], "handvalves": [], "pipelines": {}}
    q = query.lower()
    if any(word in q for word in ["pipeline", "line", "flow path", "pipe"]):
        context["pipelines"] = PIPELINES
    elif any(word in q for word in ["equipment", "pump", "tank", "vessel", "reactor"]):
        context["equipment"] = PROCESS_DATA.get("Equipment", [])
    elif any(word in q for word in ["instrument", "valve", "controller", "sensor"]):
        context["instrumentation"] = PROCESS_DATA.get("Instrumentation", [])
        context["handvalves"] = PROCESS_DATA.get("HandValves", [])
    else:
        context["equipment"] = find_best_tag_matches(query, PROCESS_DATA.get("Equipment", []))
        context["instrumentation"] = find_best_tag_matches(query, PROCESS_DATA.get("Instrumentation", []))
        context["handvalves"] = find_best_tag_matches(query, PROCESS_DATA.get("HandValves", []))
        context["pipelines"] = find_pipeline_matches(query)

    extra_pipes = {}
    extra_pipes.update(find_pipelines_for_tag(query))
    for section in ["equipment", "instrumentation", "handvalves"]:
        for item in context[section]:
            t = item.get("Tag", "")
            extra_pipes.update(find_pipelines_for_tag(t))

    if context["pipelines"]:
        context["pipelines"].update(extra_pipes)
    else:
        context["pipelines"] = extra_pipes

    if context["pipelines"]:
        existing_tags = {e.get("Tag") for e in context["equipment"]}
        for pipe_info in context["pipelines"].values():
            for end_key in ["start", "end"]:
                node = pipe_info.get(end_key, {})
                if node.get("category") == "equipment":
                    det = node.get("details") or {}
                    tag = det.get("Tag")
                    if det and tag and tag not in existing_tags:
                        context["equipment"].append(det)
                        existing_tags.add(tag)

    return context

def summarize_context(context):
    lines = []
    if context["equipment"]:
        lines.append("Equipment:")
        for e in context["equipment"]:
            tag = e.get("Tag", "")
            typ = e.get("Type", "")
            spec = e.get("EquipmentSpec", "")
            lines.append(f"- {tag} (type {typ}): spec = {spec}")
    if context["instrumentation"]:
        lines.append("Instrumentation:")
        for i in context["instrumentation"]:
            tag = i.get("Tag", "")
            typ = i.get("Type", "")
            details = i.get("Details", "")
            lines.append(f"- {tag} (type {typ}): details = {details}")
    if context["handvalves"]:
        lines.append("Hand valves:")
        for h in context["handvalves"]:
            tag = h.get("Tag", "")
            code = h.get("Code", h.get("ValveCode", ""))
            normally = h.get("Normally", "")
            lines.append(f"- {tag} (code {code}, normally {normally})")
    if context["pipelines"]:
        lines.append("Pipelines:")
        for tag, info in context["pipelines"].items():
            start = info.get("start", {})
            end = info.get("end", {})
            s_tag = (start.get("details") or {}).get("Tag") or start.get("tag", "unknown")
            e_tag = (end.get("details") or {}).get("Tag") or end.get("tag", "unknown")
            lines.append(f"- {tag}: from {s_tag} to {e_tag}")
            flow = info.get("complete_flow", [])
            for step in flow:
                if step.get("category") == "node":
                    node_tag = step.get("tag", "")
                    if node_tag:
                        lines.append(f"  • node '{node_tag}' is present in pipeline {tag}")
                elif step.get("category") == "instrumentation":
                    inst_details = step.get("details") or {}
                    inst_tag = inst_details.get("Tag") or step.get("tag", "")
                    if inst_tag:
                        lines.append(f"  • instrumentation '{inst_tag}' is on pipeline {tag}")
                elif step.get("category") == "handvalve":
                    hv_details = step.get("details") or {}
                    hv_tag = hv_details.get("Tag") or step.get("tag", "")
                    if hv_tag:
                        lines.append(f"  • handvalve '{hv_tag}' is on pipeline {tag}")
    if not lines:
        return "No matching data found in plant model."
    return "\n".join(lines)

# ==========================
# Agentic tool: get_react_context
# ==========================
def get_react_context(question: str, top_k: int = 6) -> str:
    local_context = build_local_context(question)
    summary = summarize_context(local_context)

    # retrieve from vector store
    retrieved = retrieve(question, topk=top_k)
    if retrieved:
        retrieve_lines = ["Retrieved references:"]
        for meta, score in retrieved:
            snippet = meta["text"].strip().replace("\n", " ")
            snippet = snippet[:400] + ("..." if len(snippet) > 400 else "")
            source = meta.get("source", "unknown")
            page = meta.get("page")
            score_pct = f"{score:.2f}"
            page_info = f" p.{page}" if page else ""
            retrieve_lines.append(f"- [{source}{page_info}] (score {score_pct}): {snippet}")
        retrieved_text = "\n".join(retrieve_lines)
    else:
        retrieved_text = "Retrieved references: none found."

    combined = f"Local plant context:\n{summary}\n\n{retrieved_text}"
    return combined.strip()

# ==========================
# Chat/Agent orchestration
# ==========================
react_system = {
    "role": "system",
    "content": (
        "You are a process engineer expert in P&ID and HAZOP interpretation. "
        "You follow a ReAct pattern: reason step-by-step, decide when to call tools, "
        "and use only the provided plant and RAG context. "
        "Always cite sources (file name + page when available) and quote tag names exactly."
    ),
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_react_context",
            "description": (
                "Retrieve and summarize plant context relevant to the question, "
                "including structured tag matches and retrieved document excerpts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "top_k": {"type": "integer", "default": 6},
                },
                "required": ["question"],
            },
        }
    }
]

def ensure_agent_initialized():
    if not st.session_state.agent_messages:
        st.session_state.agent_messages.append(react_system)

def append_display_message(role, content):
    st.session_state.display_history.append(
        {"role": role, "content": content, "time": datetime.utcnow().isoformat()}
    )

def call_react_agent():
    """
    Agent loop:
    - Sends messages to LLM (gpt-4o).
    - Model may request tool calls (get_react_context).
    - When tool call is present, we run it synchronously and append result as a tool message.
    - Continue until model returns final response (no tool_calls).
    """
    ensure_agent_initialized()
    messages = st.session_state.agent_messages
    while True:
        # call the model
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.2,
            tools=TOOLS,
            tool_choice="auto",
        )
        message = response.choices[0].message
        assistant_entry = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        messages.append(assistant_entry)

        # if model did not call tools, we have final answer
        if not message.tool_calls:
            return message.content or "I couldn't generate a response."

        # otherwise handle each tool call
        for tool_call in message.tool_calls:
            if tool_call.type != "function":
                continue
            fname = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except:
                args = {}
            if fname == "get_react_context":
                q = args.get("question", "")
                top_k = int(args.get("top_k", 6))
                tool_result = get_react_context(q, top_k=top_k)
            else:
                tool_result = f"Unsupported tool requested: {fname}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": fname,
                "content": tool_result,
            })

# ==========================
# Streamlit UI
# ==========================
st.title("🧠 PID + HAZOP RAG Agent (auto-input)")

st.sidebar.header("Vector Index Status")
metas = load_metas(META_PATH)
if metas:
    st.sidebar.success(f"{len(metas)} chunks cached from Pinecone.")
else:
    if pinecone_index_ready(EMBED_DIM):
        st.sidebar.info("Vectors found in Pinecone (metadata cache empty).")
    else:
        st.sidebar.warning("No vectors detected in Pinecone for this index.")

# chat interface
if pinecone_index_ready(EMBED_DIM):
    user_q = st.chat_input("Ask about pipelines, equipment, instruments, or HAZOP findings...")
    if user_q:
        append_display_message("user", user_q)
        st.session_state.agent_messages.append({"role": "user", "content": user_q})
        try:
            reply = call_react_agent()
        except Exception as e:
            reply = f"⚠️ Error calling GPT: {e}"
        append_display_message("assistant", reply)
else:
    st.warning("⚠️ Pinecone index is empty. Run `python embedding.py` to ingest PDFs.")

st.markdown("---")
st.header("Session conversation")
for msg in st.session_state.display_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
