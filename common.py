"""Shared config, SQLite image store, Qdrant, and llama.cpp (llama-server) clients. Minimal dependencies."""
import base64, json, math, os, re, sqlite3
from functools import lru_cache

import cv2
import numpy as np
import requests

# ---------- config (override with env vars) ----------
LLM_URL = os.getenv("LLM_URL", "http://127.0.0.1:8080")      # llama-server: vision + chat model
EMBED_URL = os.getenv("EMBED_URL", "http://127.0.0.1:8081")  # llama-server: embedding model (--embedding)
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))               # all-MiniLM-L6-v2 = 384
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
SQLITE_PATH = os.getenv("SQLITE_PATH", "frames.db")
QDRANT_URL = os.getenv("QDRANT_URL")                         # unset -> embedded local mode (no server)
QDRANT_PATH = os.getenv("QDRANT_PATH", "qdrant_data")
COLLECTION = "video_frames"


# ---------- SQLite: images ----------
def db():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS frames (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            image BLOB NOT NULL)"""
    )
    return conn


def save_frame(video_id: str, ts: float, jpeg: bytes) -> int:
    with db() as c:
        return c.execute("INSERT INTO frames (video_id, timestamp, image) VALUES (?,?,?)", (video_id, ts, jpeg)).lastrowid


def get_images(frame_ids):
    """Returns [(id, timestamp, RGB numpy array)] ordered by time."""
    if not frame_ids:
        return []
    q = ",".join("?" * len(frame_ids))
    with db() as c:
        rows = c.execute(f"SELECT id, timestamp, image FROM frames WHERE id IN ({q}) ORDER BY timestamp", list(frame_ids)).fetchall()
    return [(i, t, cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)) for i, t, b in rows]


def list_videos():
    with db() as c:
        return [r[0] for r in c.execute("SELECT DISTINCT video_id FROM frames")]


# ---------- Qdrant ----------
@lru_cache(maxsize=1)
def qdrant():
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    client = QdrantClient(url=QDRANT_URL) if QDRANT_URL else QdrantClient(path=QDRANT_PATH)
    if not client.collection_exists(COLLECTION):
        client.create_collection(COLLECTION, vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE))
    return client


# ---------- llama.cpp clients (OpenAI-compatible endpoints) ----------
def _post(url: str, payload: dict, timeout: int = 600) -> dict:
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        if not r.ok:
            raise SystemExit(f"HTTP {r.status_code} from {url}: {r.text[:1500]}")
        return r.json()
    except requests.ConnectionError:
        raise SystemExit(f"Cannot reach llama-server at {url}. Did you run ./start_servers.sh ?")


def embed(text: str) -> list[float]:
    v = _post(f"{EMBED_URL}/v1/embeddings", {"input": text})["data"][0]["embedding"]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def llm(prompt: str, images: list[bytes] | None = None, system: str | None = None,
        max_tokens: int = 800, json_mode: bool = False) -> str:
    content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(i).decode()}}
               for i in (images or [])] + [{"type": "text", "text": prompt}]
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    payload = {"model": "local", "messages": messages, "max_tokens": max_tokens, "temperature": 0.2}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}  # grammar-constrained valid JSON
    return _post(f"{LLM_URL}/v1/chat/completions", payload)["choices"][0]["message"]["content"]


def parse_json(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def fmt_ts(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"
