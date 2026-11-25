"""Part 1: video -> distinct frames -> batched VLM captions -> embeddings -> Qdrant (+ images in SQLite).

Usage: python ingest.py path/to/video.mp4 [--fps 1] [--hash-thresh 20] [--batch 4]
"""
import argparse, os

import cv2
import numpy as np
from qdrant_client.models import PointStruct

from common import BATCH_SIZE, COLLECTION, embed, fmt_ts, llm, parse_json, qdrant, save_frame

BATCH_PROMPT = """You are given {n} frames from a video, in order. They are Frame 1 to Frame {n}, in the same order as the images.
Describe EACH frame independently and in detail for later search: people (count, gender, age, clothing, actions), objects,
animals, vehicles, setting, lighting, readable text, overall activity. Each caption must stand completely alone: never use words like 'same', 'still', 'again', 'now', 'continues' or refer to other frames. Always describe people by their appearance, as if seeing them for the first time.
Return ONLY JSON:
{{"frames": [{{"frame": 1, "caption": "<detailed 2-4 sentences>", "objects": ["..."], "people_count": 0, "setting": "<short>", "visible_text": ""}}, ...]}}
The list must contain exactly {n} entries."""


def dhash(bgr: np.ndarray) -> np.ndarray:
    """256-bit difference hash using only OpenCV/numpy (no scipy/imagehash)."""
    small = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (17, 16), interpolation=cv2.INTER_AREA)
    return small[:, 1:] > small[:, :-1]


def distinct_frames(path: str, sample_fps: float, hash_thresh: int):
    """Yield (timestamp_sec, BGR frame) for frames that differ from the last kept frame."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(fps / sample_fps))
    i, last = 0, None
    while cap.grab():
        if i % step == 0:
            ok, bgr = cap.retrieve()
            if ok:
                h = dhash(bgr)
                if last is None or np.count_nonzero(h != last) >= hash_thresh:
                    last = h
                    yield i / fps, bgr
        i += 1
    cap.release()


def to_jpeg(bgr: np.ndarray, max_side: int) -> bytes:
    scale = max_side / max(bgr.shape[:2])
    if scale < 1:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def caption_batch(vlm_jpegs: list[bytes]) -> list[dict]:
    """Caption N frames in one VLM call; fall back to one-by-one if the output is malformed."""
    n = len(vlm_jpegs)
    raw = llm(BATCH_PROMPT.format(n=n), images=vlm_jpegs, max_tokens=450 * n, json_mode=True)
    frames = (parse_json(raw) or {}).get("frames")
    if isinstance(frames, list) and len(frames) == n and all(isinstance(f, dict) and f.get("caption") for f in frames):
        return frames
    if n == 1:
        return [(frames[0] if frames else None) or {"caption": raw.strip()}]
    print(f"  ! batch output malformed, retrying {n} frames individually")
    return [caption_batch([j])[0] for j in vlm_jpegs]


def flush(batch, video_id, client) -> int:
    for b, info in zip(batch, caption_batch([b["vlm"] for b in batch])):
        caption = info.get("caption", "")
        extra = {k: v for k, v in info.items() if k not in ("caption", "frame")}
        frame_id = save_frame(video_id, b["ts"], b["store"])  # SQLite id doubles as Qdrant point id
        text = f"{caption} Objects: {', '.join(map(str, extra.get('objects', [])))}. Setting: {extra.get('setting', '')}."
        client.upsert(COLLECTION, [PointStruct(
            id=frame_id, vector=embed(text),
            payload={"video_id": video_id, "frame_id": frame_id, "timestamp": b["ts"], "caption": caption, "extra": extra})])
        print(f"[{fmt_ts(b['ts'])}] frame {frame_id}: {caption[:90]}...")
    return len(batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--fps", type=float, default=1.0, help="candidate frames sampled per second")
    ap.add_argument("--hash-thresh", type=int, default=20, help="min dHash distance (of 256 bits) to be 'distinct'")
    ap.add_argument("--batch", type=int, default=BATCH_SIZE, help="frames per VLM call")
    ap.add_argument("--vlm-side", type=int, default=512, help="longest image side sent to the VLM (smaller = faster)")
    a = ap.parse_args()

    video_id, client, batch, n = os.path.basename(a.video), qdrant(), [], 0
    for ts, bgr in distinct_frames(a.video, a.fps, a.hash_thresh):
        batch.append({"ts": ts, "vlm": to_jpeg(bgr, a.vlm_side), "store": to_jpeg(bgr, 768)})
        if len(batch) == a.batch:
            n += flush(batch, video_id, client)
            batch = []
    if batch:
        n += flush(batch, video_id, client)
    client.close()
    print(f"Done. Indexed {n} distinct frames from {video_id}")


if __name__ == "__main__":
    main()
