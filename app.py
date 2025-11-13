# app_streamlit_rag_agent.py
import os
import json
import uuid
import re
import hashlib
from datetime import datetime
from difflib import SequenceMatcher

import numpy as np
import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import httpx
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_IMPORT_ERROR = None
except ImportError as err:
    Pinecone = None  # type: ignore[assignment]
    ServerlessSpec = None  # type: ignore[assignment]
    PINECONE_IMPORT_ERROR = err

# Try to import translation helper (depends on PyMuPDF / fitz native libs)
try:
    from dutch_to_eng import translate_pdf
    TRANSLATION_AVAILABLE = True
    TRANSLATION_IMPORT_ERROR = None
except (ImportError, OSError) as err:
    TRANSLATION_AVAILABLE = False
    TRANSLATION_IMPORT_ERROR = err

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

# INPUT_PDFS: you can set either via env var "INPUT_PDFS" (comma-separated full paths)
# or edit this list in-code (absolute or relative paths)
INPUT_PDFS_ENV = os.getenv("INPUT_PDFS")
if INPUT_PDFS_ENV:
    INPUT_PDFS = [p.strip() for p in INPUT_PDFS_ENV.split(",") if p.strip()]
else:
    # default - update these to point to your local files (example names)
    INPUT_PDFS = [
        "hezop1.pdf",
        "hezop2.pdf",
        "hezop3.pdf",
        # add more files or use env var
    ]

# session state initialization
if "index_built" not in st.session_state:
    st.session_state.index_built = False
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []
if "display_history" not in st.session_state:
    st.session_state.display_history = []
if "uploaded_files_list" not in st.session_state:
    # store tuples: (original_path, saved_input_path, translated_path, file_hash)
    st.session_state.uploaded_files_list = []

# ==========================
# Utility helpers
# ==========================
def file_hash(path: str) -> str:
    """Return sha256 hash for file contents (fast, deterministic)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def extract_pdf_text(path: str):
    blocks = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                txt = page.extract_text() or ""
                # split by double newline blocks to keep paragraphs
                for ptxt in txt.split("\n\n"):
                    ptxt = ptxt.strip()
                    if ptxt:
                        blocks.append({"text": ptxt, "source": os.path.basename(path), "page": i})
    except Exception as e:
        st.warning(f"Could not extract text from {path}: {e}")
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

def embed_texts(texts):
    """Batch embed a list of texts and return a float32 numpy array."""
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype="float32")
    resp = client.embeddings.create(input=texts, model=EMBED_MODEL)
    vecs = [r.embedding for r in resp.data]
    arr = np.array(vecs, dtype="float32")
    return arr


@st.cache_resource(show_spinner=False)
def get_pinecone_index():
    if PINECONE_IMPORT_ERROR:
        st.error(
            f"❌ Pinecone SDK not available: {PINECONE_IMPORT_ERROR}. "
            "Install it with 'pip install pinecone'."
        )
        st.stop()

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        st.error("❌ No Pinecone API key found. Set PINECONE_API_KEY in your .env.")
        st.stop()

    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_ENVIRONMENT") or os.getenv("PINECONE_REGION") or "us-east-1"
    index_name = os.getenv("PINECONE_INDEX_NAME", "hezop-rag")

    client = Pinecone(api_key=api_key)
    try:
        listed = client.list_indexes()
        if hasattr(listed, "names"):
            existing_indexes = set(listed.names())
        elif hasattr(listed, "__iter__"):
            existing_indexes = {getattr(idx, "name", str(idx)) for idx in listed}
        else:
            existing_indexes = set()
    except Exception as err:
        st.error(f"❌ Failed to list Pinecone indexes: {err}")
        st.stop()

    if index_name not in existing_indexes:
        try:
            client.create_index(
                name=index_name,
                dimension=EMBED_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
            st.info(f"Created Pinecone index '{index_name}'.")
        except Exception as err:
            st.error(f"❌ Failed to create Pinecone index '{index_name}': {err}")
            st.stop()

    return client.Index(index_name)


def normalize_vectors(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

# ==========================
# Plant JSON loader (same as yours)
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
    PIPELINES = DATA.get("complete_pipeline_flows", {}) or {}
    PROCESS_DATA = DATA.get("process_data", {}) or {}
    TAG_TO_PIPELINES = build_tag_index(PIPELINES)
    return DATA

DATA = load_plant_data()

# ==========================
# Index & metadata helpers (dedupe-aware)
# ==========================
def load_metas():
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Failed to load metadata cache: {e}")
    return []


def save_metas(metas):
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metas, f, indent=2)

def get_existing_file_hashes(metas):
    """Return set of file_hash values present in metadata."""
    hashes = set()
    for m in metas:
        fh = m.get("file_hash")
        if fh:
            hashes.add(fh)
    return hashes

# ==========================
# Input files processing (auto ingest)
# ==========================
def prepare_input_files(input_list):
    """
    Ensure input files exist, copy to temp folder (to avoid locking originals),
    compute file hashes, and create translated paths.
    Store into session_state.uploaded_files_list as tuples:
    (original_path, saved_input_path, translated_path, file_hash)
    """
    prepared = []
    for path in input_list:
        if not os.path.exists(path):
            st.warning(f"Input file not found: {path}")
            continue
        # save a local copy to temp_pdfs to normalize path names
        base = os.path.basename(path)
        saved_input_path = os.path.join("temp_pdfs", f"input_{uuid.uuid4().hex}_{base}")
        # only copy if not already present (compare by file size or name)
        if not os.path.exists(saved_input_path):
            with open(path, "rb") as src, open(saved_input_path, "wb") as dst:
                dst.write(src.read())
        fh = file_hash(saved_input_path)
        translated_name = f"{os.path.splitext(base)[0]}_translated.pdf"
        translated_path = os.path.join("temp_pdfs", f"translated_{fh}_{translated_name}")
        prepared.append((path, saved_input_path, translated_path, fh))
    # update session state
    st.session_state.uploaded_files_list = prepared
    return prepared

# ==========================
# Build / update Pinecone index with dedupe
# ==========================
def build_or_update_index(force_rebuild=False, translate_if_needed=True, chunk_size=2500):
    """
    - Loads cached metadata (local JSON) for dedupe.
    - Optionally clears the Pinecone index on force rebuild.
    - For new files: extract text, chunk, embed, and upsert into Pinecone.
    - Also optionally includes the CLASSIFIED_JSON as a source (only once).
    """
    if not st.session_state.uploaded_files_list and not os.path.exists(CLASSIFIED_JSON):
        st.warning("No input files provided and no classified JSON present. Nothing to index.")
        return []

    pinecone_index = get_pinecone_index()
    metas = []
    if force_rebuild:
        try:
            pinecone_index.delete(delete_all=True)
            st.info("Cleared Pinecone index before rebuilding.")
        except Exception as e:
            st.warning(f"Failed to clear Pinecone index: {e}")
        save_metas([])
    else:
        metas = load_metas()

    existing_hashes = get_existing_file_hashes(metas)

    if translate_if_needed and not TRANSLATION_AVAILABLE:
        st.warning(
            "PDF translation disabled because the translation module failed to import.\n"
            "Please reinstall PyMuPDF (pip install --force-reinstall PyMuPDF==1.24.10) "
            "or install the Microsoft Visual C++ Redistributable for Visual Studio 2015-2022."
        )

    new_chunks = []
    for original_path, saved_input_path, translated_path, fh in st.session_state.uploaded_files_list:
        if fh in existing_hashes and not force_rebuild:
            st.info(f"Skipping embedding for {os.path.basename(original_path)} — already indexed.")
            continue

        if translate_if_needed and TRANSLATION_AVAILABLE:
            try:
                if not os.path.exists(translated_path):
                    st.info(f"Translating {original_path} ...")
                    translate_pdf(saved_input_path, translated_path, chunk_size=1000, max_retries=3, timeout=30)
                    st.success(f"Translated: {os.path.basename(original_path)}")
            except Exception as e:
                st.warning(f"Translation error for {original_path}: {e}. Proceeding with original file.")
                translated_path = saved_input_path
        elif translate_if_needed and not TRANSLATION_AVAILABLE:
            translated_path = saved_input_path

        src_for_text = translated_path if os.path.exists(translated_path) else saved_input_path
        blocks = extract_pdf_text(src_for_text)
        chunks = chunk_texts(blocks, max_chars=chunk_size)
        for c in chunks:
            c["file_hash"] = fh
            c["original_path"] = os.path.basename(original_path)
            c["chunk_id"] = str(uuid.uuid4())
        new_chunks.extend(chunks)

    if os.path.exists(CLASSIFIED_JSON):
        with open(CLASSIFIED_JSON, "r", encoding="utf-8") as f:
            plant_data = json.load(f)
        classified_blocks = json_to_blocks(plant_data)
        try:
            mtime = os.path.getmtime(CLASSIFIED_JSON)
            cj_hash = hashlib.sha256(f"{CLASSIFIED_JSON}-{mtime}".encode()).hexdigest()
        except Exception:
            cj_hash = hashlib.sha256(CLASSIFIED_JSON.encode()).hexdigest()
        if cj_hash not in existing_hashes or force_rebuild:
            for b in classified_blocks:
                b["file_hash"] = cj_hash
                b["original_path"] = os.path.basename(CLASSIFIED_JSON)
                b["chunk_id"] = str(uuid.uuid4())
            new_chunks.extend(classified_blocks)
        else:
            st.info("Classified JSON already indexed — skipping.")

    if not new_chunks:
        st.success("Index is up-to-date. Nothing new to embed.")
        if metas:
            st.session_state.index_built = True
        return metas

    BATCH = 16
    text_batches = [new_chunks[i:i + BATCH] for i in range(0, len(new_chunks), BATCH)]
    new_metas = []
    upserted = 0

    for batch in text_batches:
        texts = [c["text"] for c in batch]
        arr = embed_texts(texts)
        arr = normalize_vectors(arr)
        if arr.size == 0:
            continue

        items = []
        batch_metas = []
        for vec, c in zip(arr, batch):
            meta = {
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "source": c.get("source", c.get("original_path", "unknown")),
                "page": c.get("page"),
                "created": datetime.utcnow().isoformat(),
                "file_hash": c.get("file_hash"),
                "original_path": c.get("original_path"),
            }
            items.append(
                {
                    "id": c["chunk_id"],
                    "values": vec.tolist(),
                    "metadata": meta,
                }
            )
            batch_metas.append(meta)

        try:
            pinecone_index.upsert(items=items)
            upserted += len(items)
            new_metas.extend(batch_metas)
        except Exception as e:
            st.error(f"Failed to upsert batch into Pinecone: {e}")

    if upserted > 0:
        metas.extend(new_metas)
        save_metas(metas)
        st.success(f"Added {upserted} chunks to Pinecone index.")
        st.session_state.index_built = True
    else:
        st.warning("No chunks were added to Pinecone.")

    return metas

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
    pinecone_index = get_pinecone_index()
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


def pinecone_index_ready():
    try:
        stats = get_pinecone_index().describe_index_stats()
        total_vectors = stats.get("total_vector_count", 0)
        return total_vectors and total_vectors > 0
    except Exception:
        return False

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

# show configured input files
st.sidebar.header("Input files (auto)")
st.sidebar.markdown("Files specified inside the app or via env var INPUT_PDFS (comma-separated).")
if INPUT_PDFS:
    st.sidebar.markdown("**Configured input files:**")
    for p in INPUT_PDFS:
        st.sidebar.text(p)
else:
    st.sidebar.warning("No input files configured. Set env var INPUT_PDFS or edit the script.")

# prepare input files (copy to temp and compute hash)
prepared = prepare_input_files(INPUT_PDFS)

# show list of prepared files
if prepared:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Prepared files")
    for orig, saved, translated, fh in prepared:
        status = "translated" if os.path.exists(translated) else "pending-translation"
        st.sidebar.text(f"- {os.path.basename(orig)} (hash: {fh[:8]}...) - {status}")

# Index building controls
col1, col2, col3 = st.sidebar.columns([1,1,1])
with col1:
    if st.button("▶️ Build / Update Index"):
        with st.spinner("Building or updating index (dedupe-aware)..."):
            metas = build_or_update_index(force_rebuild=False, translate_if_needed=True)
            if metas:
                st.success("Index ready.")
            else:
                st.warning("Index build/update did not add any new chunks.")
with col2:
    if st.button("🔁 Force Rebuild Index"):
        with st.spinner("Force rebuilding full index..."):
            metas = build_or_update_index(force_rebuild=True, translate_if_needed=True)
            if metas:
                st.success("Index force-rebuilt.")
with col3:
    if st.button("🧾 Show metadata summary"):
        metas = load_metas()
        if metas:
            st.write(f"Total chunks: {len(metas)}")
            # show top 5 metas
            st.write(metas[:5])
        else:
            st.info("No metadata found yet.")

# Load metadata cache to decide whether index is populated
metas = load_metas()

if not metas and not pinecone_index_ready():
    st.warning("⚠️ No vector index available. Build/update index first (sidebar).")
    st.info("Use 'Build / Update Index' to embed configured files automatically.")
else:
    if not metas:
        st.info("Metadata cache empty — relying on Pinecone index metadata only.")
    # chat interface
    user_q = st.chat_input("Ask about pipelines, equipment, instruments, or HAZOP findings...")
    if user_q:
        append_display_message("user", user_q)
        st.session_state.agent_messages.append({"role": "user", "content": user_q})
        try:
            reply = call_react_agent()
        except Exception as e:
            reply = f"⚠️ Error calling GPT: {e}"
        append_display_message("assistant", reply)
        # st.experimental_rerun()

# show conversation
st.markdown("---")
st.header("Session conversation")
for msg in st.session_state.display_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
