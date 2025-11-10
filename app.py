import os
import json
import uuid
import re
from difflib import SequenceMatcher
from datetime import datetime

import faiss
import numpy as np
import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from dutch_to_eng import translate_pdf

# ==========================
# 2) Streamlit App Config
# ==========================
st.set_page_config(page_title="PID + HAZOP RAG Chatbot", layout="wide")
st.title("🧠 PID + HAZOP RAG Chatbot")

# ==========================
# 3) Env + OpenAI
# ==========================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY")
if not api_key:
    st.error("❌ No OpenAI API key found. Set OPENAI_API_KEY or OPEN_AI_KEY in your .env.")
    st.stop()
client = OpenAI(api_key=api_key)

# ==========================
# 4) Data Sources and Paths
# ==========================
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072
INDEX_PATH = "./rag_store/faiss.index"
META_PATH = "./rag_store/metadata.json"
CLASSIFIED_JSON = "classified_pipeline_tags2.json"
os.makedirs("rag_store", exist_ok=True)
os.makedirs("temp_pdfs", exist_ok=True)

# Initialize session state for uploaded PDFs and chat history
if "uploaded_pdfs" not in st.session_state:
    st.session_state.uploaded_pdfs = []  # List of (original_name, translated_path) tuples
if "display_history" not in st.session_state:
    st.session_state.display_history = []  # Only current session chats
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []
if "index_built" not in st.session_state:
    st.session_state.index_built = False

# ==========================
# 5) Helpers: PDF + JSON → blocks
# ==========================
def extract_pdf_text(path: str):
    blocks = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            for ptxt in txt.split("\n\n"):
                ptxt = ptxt.strip()
                if ptxt:
                    blocks.append({"text": ptxt, "source": os.path.basename(path), "page": i})
    return blocks

def json_to_blocks(data):
    source_name = os.path.basename(CLASSIFIED_JSON)
    blocks = []
    # Pipelines
    for tag, flow in data.get("complete_pipeline_flows", {}).items():
        desc = [f"Pipeline {tag}:"]
        start_tag = flow.get("start", {}).get("tag")
        end_tag = flow.get("end", {}).get("tag")
        desc.append(f"Starts at {start_tag}, ends at {end_tag}.")
        for step in flow.get("complete_flow", []):
            desc.append(f"- {step.get('tag')} ({step.get('category')})")
        blocks.append({"text": "\n".join(desc), "source": source_name})
    # Process data
    for cat in ["Equipment", "Instrumentation", "HandValves"]:
        for item in data.get("process_data", {}).get(cat, []):
            spec = item.get("EquipmentSpec") or item.get("Details") or ""
            line = f"{cat} {item.get('Tag')} ({item.get('Type')}): {spec}"
            blocks.append({"text": line, "source": source_name})
    return blocks

def chunk_texts(blocks, max_chars=2500):
    chunks = []
    for b in blocks:
        txt = b["text"]
        for i in range(0, len(txt), max_chars):
            part = txt[i:i+max_chars]
            if len(part.strip()) > 50:
                chunks.append({
                    "text": part,
                    "source": b.get("source", "pdf"),
                    "page": b.get("page"),
                })
    return chunks

def embed_text(text: str):
    emb = client.embeddings.create(input=text, model=EMBED_MODEL).data[0].embedding
    return np.array(emb, dtype="float32")


# ==========================
# 5b) Plant JSON helpers (from main.py)
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
# 6) Build / Load FAISS Index
# ==========================
def build_or_load_index(data, uploaded_pdfs_list):
    # Check if we have uploaded PDFs to process
    if not uploaded_pdfs_list and not st.session_state.index_built:
        # If no PDFs uploaded and index doesn't exist, return None
        return None, None
    
    # If index already built and exists, try to load it first
    if st.session_state.index_built and os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        try:
            # Verify that all current PDFs are in the index
            existing_metas = json.load(open(META_PATH, "r", encoding="utf-8"))
            existing_sources = {meta.get("source", "") for meta in existing_metas}
            current_sources = {os.path.basename(translated) for _, translated in uploaded_pdfs_list}
            
            # If all current PDFs are already in the index, just load it
            if current_sources.issubset(existing_sources) and len(current_sources) > 0:
                # Index is up to date, just load it
                index = faiss.read_index(INDEX_PATH)
                metas = json.load(open(META_PATH, "r", encoding="utf-8"))
                return index, metas
        except Exception as e:
            st.warning(f"⚠️ Error loading existing index: {e}. Rebuilding...")

    # If we get here, we need to build/rebuild the index
    all_blocks = []
    
    # From uploaded and translated PDFs
    for original_name, translated_path in uploaded_pdfs_list:
        if os.path.exists(translated_path):
            all_blocks += extract_pdf_text(translated_path)
        else:
            st.warning(f"⚠️ Missing translated file: {translated_path}")

    # From classified JSON (if exists)
    if os.path.exists(CLASSIFIED_JSON):
        all_blocks += json_to_blocks(data)

    if not all_blocks:
        st.warning("⚠️ No content available to build the vector index. Please upload PDFs first.")
        return None, None

    # Chunk and embed
    chunks = chunk_texts(all_blocks)
    vecs = []
    for c in chunks:
        vecs.append(embed_text(c["text"]))
    vecs = np.vstack(vecs) if vecs else np.zeros((0, EMBED_DIM), dtype="float32")
    if vecs.size == 0:
        st.error("No content available to build the vector index.")
        return None, None
    faiss.normalize_L2(vecs)

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vecs)

    metas = [
        {
            "chunk_id": str(uuid.uuid4()),
            "text": c["text"],
            "source": c["source"],
            "page": c.get("page"),
            "created": datetime.utcnow().isoformat(),
        }
        for c in chunks
    ]

    faiss.write_index(index, INDEX_PATH)
    json.dump(metas, open(META_PATH, "w", encoding="utf-8"), indent=2)
    st.session_state.index_built = True
    return index, metas

def load_plant_data():
    global DATA, PIPELINES, PROCESS_DATA, TAG_TO_PIPELINES

    if not os.path.exists(CLASSIFIED_JSON):
        st.warning(f"⚠️ Missing plant JSON: {CLASSIFIED_JSON}")
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
    except OSError as exc:
        st.error(f"❌ Unable to read {CLASSIFIED_JSON}: {exc}")
        st.stop()

    PIPELINES = DATA.get("complete_pipeline_flows", {}) or {}
    PROCESS_DATA = DATA.get("process_data", {}) or {}
    TAG_TO_PIPELINES = build_tag_index(PIPELINES)

    return DATA


DATA = load_plant_data()

# ==========================
# PDF Upload and Processing Section
# ==========================
st.sidebar.header("📄 Document Upload")
st.sidebar.markdown("Upload Dutch PDF files to process and add to the knowledge base.")

uploaded_file = st.sidebar.file_uploader(
    "Choose a Dutch PDF file",
    type=["pdf"],
    help="Upload a PDF file in Dutch. It will be translated and processed."
)

if uploaded_file is not None:
    # Show uploaded file info
    st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")
    
    # Check if file is already in the list
    file_names = [name for name, _, _ in st.session_state.uploaded_pdfs]
    if uploaded_file.name not in file_names:
        # Save uploaded file temporarily
        temp_input_path = os.path.join("temp_pdfs", f"input_{uuid.uuid4().hex}_{uploaded_file.name}")
        with open(temp_input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Generate translated file path
        translated_name = f"{os.path.splitext(uploaded_file.name)[0]}_translated.pdf"
        temp_translated_path = os.path.join("temp_pdfs", f"translated_{uuid.uuid4().hex}_{translated_name}")
        
        # Add to list (not processed yet)
        st.session_state.uploaded_pdfs.append((uploaded_file.name, temp_input_path, temp_translated_path))
        st.sidebar.info(f"📋 Added to queue: {uploaded_file.name}")
        st.rerun()

# Show list of uploaded files
if st.session_state.uploaded_pdfs:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Uploaded Files:")
    for i, (original_name, input_path, translated_path) in enumerate(st.session_state.uploaded_pdfs):
        status = "✅ Translated" if os.path.exists(translated_path) else "⏳ Pending"
        st.sidebar.text(f"{i+1}. {original_name} - {status}")
    
    # Check if index needs rebuilding
    processed_pdfs = [(name, translated) for name, _, translated in st.session_state.uploaded_pdfs if os.path.exists(translated)]
    needs_rebuild = False
    if processed_pdfs:
        # Check if index exists and contains all current PDFs
        if os.path.exists(META_PATH):
            try:
                existing_metas = json.load(open(META_PATH, "r", encoding="utf-8"))
                existing_sources = {meta.get("source", "") for meta in existing_metas}
                current_sources = {os.path.basename(translated) for _, translated in processed_pdfs}
                # Check if all current PDFs are in the index
                if not current_sources.issubset(existing_sources):
                    needs_rebuild = True
            except:
                needs_rebuild = True
        else:
            needs_rebuild = True
    else:
        needs_rebuild = False
    
    # Process Document button
    if st.sidebar.button("🔄 Process Documents", type="primary"):
        with st.sidebar:
            with st.spinner("Processing documents..."):
                translation_done = False
                # Translate all queued PDFs (only if not already translated)
                for original_name, input_path, translated_path in st.session_state.uploaded_pdfs:
                    if os.path.exists(translated_path):
                        st.info(f"⏭️ Skipping {original_name} - already translated")
                        continue
                    
                    st.info(f"Translating {original_name}...")
                    try:
                        # Use smaller chunks (1000) for faster translation and less timeout risk
                        translate_pdf(input_path, translated_path, chunk_size=1000, max_retries=3, timeout=30)
                        st.success(f"✅ Translated: {original_name}")
                        translation_done = True
                    except Exception as e:
                        error_msg = str(e)
                        st.error(f"❌ Error translating {original_name}")
                        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                            st.warning("⏱️ Translation timed out. The function will retry automatically. If it fails completely, try processing again - it will skip already translated chunks.")
                        else:
                            st.warning(f"💡 Error: {error_msg[:150]}. Try processing again.")
                
                # Rebuild index only if needed
                if translation_done or needs_rebuild:
                    st.info("Rebuilding vector index...")
                    st.session_state.index_built = False  # Force rebuild
                else:
                    st.info("⏭️ Skipping index rebuild - already up to date")
                
                st.rerun()

# Build or load index
processed_pdfs = [(name, translated) for name, _, translated in st.session_state.uploaded_pdfs if os.path.exists(translated)]
index, metas = build_or_load_index(DATA, processed_pdfs)

# ==========================
# 7) Retrieval utilities
# ==========================
def embed_query(q):
    v = embed_text(q).reshape(1, -1)
    faiss.normalize_L2(v)
    return v

def retrieve(q, topk=6):
    if index is None or metas is None:
        return []

    v = embed_query(q)
    D, I = index.search(v, topk)
    results = []
    for n, i in enumerate(I[0]):
        if i >= 0 and i < len(metas):
            results.append((metas[i], float(D[0][n])))
    return results

# Chat history is now only in session state (memory), not loaded from file

# ==========================
# 9) Chat UI + ReAct-style system
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
                    "question": {
                        "type": "string",
                        "description": "The user question needing plant context.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of retrieval chunks to include.",
                        "default": 6,
                    },
                },
                "required": ["question"],
            },
        },
    }
]


def get_react_context(question: str, top_k: int = 6) -> str:
    local_context = build_local_context(question)
    summary = summarize_context(local_context)

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


def ensure_agent_initialized():
    if not st.session_state.agent_messages:
        st.session_state.agent_messages.append(react_system)


def append_display_message(role, content):
    st.session_state.display_history.append(
        {"role": role, "content": content, "time": datetime.utcnow().isoformat()}
    )


def call_react_agent():
    ensure_agent_initialized()
    messages = st.session_state.agent_messages

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.2,
            tools=TOOLS,
            tool_choice="auto",
        )
        message = response.choices[0].message

        assistant_entry = {
            "role": "assistant",
            "content": message.content or "",
        }
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

        if not message.tool_calls:
            return message.content or "I couldn't generate a response."

        for tool_call in message.tool_calls:
            if tool_call.type != "function":
                continue

            function_name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if function_name == "get_react_context":
                question = args.get("question", "")
                top_k = args.get("top_k", 6)
                tool_result = get_react_context(question, top_k=top_k)
            else:
                tool_result = "Unsupported tool requested."

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result,
                }
            )


# Display only current session conversation (from memory)
for msg in st.session_state.display_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Check if index is available before allowing chat
if index is None or metas is None:
    st.warning("⚠️ Please upload and process at least one PDF document before starting a chat.")
    st.info("Use the sidebar to upload Dutch PDF files and click 'Process Documents' to build the knowledge base.")
else:
    user_q = st.chat_input("Ask about pipelines, equipment, instruments, or HAZOP findings...")

    if user_q:
        append_display_message("user", user_q)
        st.session_state.agent_messages.append({"role": "user", "content": user_q})

        try:
            reply = call_react_agent()
        except Exception as e:
            reply = f"⚠️ Error calling GPT: {e}"

        append_display_message("assistant", reply)
        st.rerun()