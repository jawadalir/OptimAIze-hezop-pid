import os
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import streamlit as st

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_IMPORT_ERROR: Optional[Exception] = None
except ImportError as err:  # pragma: no cover - handled gracefully in app
    Pinecone = None  # type: ignore[assignment]
    ServerlessSpec = None  # type: ignore[assignment]
    PINECONE_IMPORT_ERROR = err


def _require_pinecone_client() -> "Pinecone":
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

    return Pinecone(api_key=api_key)  # type: ignore[arg-type]


@st.cache_resource(show_spinner=False)
def get_pinecone_index(embed_dim: int):
    """
    Return a cached Pinecone index handle, creating the index if it does not exist.
    """
    client = _require_pinecone_client()

    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_ENVIRONMENT") or os.getenv("PINECONE_REGION") or "us-east-1"
    index_name = os.getenv("PINECONE_INDEX_NAME", "hezop-rag")

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
                dimension=embed_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
            st.info(f"Created Pinecone index '{index_name}'.")
        except Exception as err:
            st.error(f"❌ Failed to create Pinecone index '{index_name}': {err}")
            st.stop()

    return client.Index(index_name)


def load_metas(meta_path: str) -> List[Dict[str, Any]]:
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as err:
            st.warning(f"Failed to load metadata cache: {err}")
    return []


def save_metas(meta_path: str, metas: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(list(metas), f, indent=2)


def get_existing_file_hashes(metas: Iterable[Dict[str, Any]]) -> set[str]:
    hashes: set[str] = set()
    for meta in metas:
        file_hash = meta.get("file_hash")
        if file_hash:
            hashes.add(file_hash)
    return hashes


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


def pinecone_index_ready(embed_dim: int) -> bool:
    try:
        stats = get_pinecone_index(embed_dim).describe_index_stats()
        total_vectors = stats.get("total_vector_count", 0)
        return bool(total_vectors and total_vectors > 0)
    except Exception:
        return False


__all__ = [
    "PINECONE_IMPORT_ERROR",
    "get_pinecone_index",
    "load_metas",
    "save_metas",
    "get_existing_file_hashes",
    "normalize_vectors",
    "normalize_vector",
    "pinecone_index_ready",
]

