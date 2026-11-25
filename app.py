"""Part 2: Gradio Q&A over the indexed video. Run: python app.py"""
import gradio as gr
from qdrant_client.models import FieldCondition, Filter, MatchValue

from common import COLLECTION, embed, fmt_ts, get_images, list_videos, llm, parse_json, qdrant

SYSTEM = """You answer questions about a video using ONLY the timestamped frame descriptions provided.
Be concise and cite timestamps. If the descriptions don't contain the answer, say so plainly.
Return ONLY JSON: {"answer": "<text>", "frame_ids": [<ids of up to 4 frames that best visually support the answer, or [] if images aren't useful>]}"""


def ask(question: str, video: str, top_k: int):
    if not question.strip():
        return "Please ask a question.", []
    flt = Filter(must=[FieldCondition(key="video_id", match=MatchValue(value=video))]) if video and video != "All" else None
    hits = qdrant().query_points(COLLECTION, query=embed(question), limit=int(top_k), query_filter=flt).points
    if not hits:
        return "Nothing has been indexed yet. Run ingest.py first.", []

    hits.sort(key=lambda h: h.payload["timestamp"])
    context = "\n".join(
        f"[frame_id={h.payload['frame_id']} | t={fmt_ts(h.payload['timestamp'])} | video={h.payload['video_id']}] {h.payload['caption']}"
        for h in hits
    )
    raw = llm(f"Frame descriptions:\n{context}\n\nQuestion: {question}", system=SYSTEM, max_tokens=700, json_mode=True)
    out = parse_json(raw) or {"answer": raw, "frame_ids": []}

    valid = {h.payload["frame_id"] for h in hits}
    ids = [i for i in out.get("frame_ids", []) if i in valid]
    gallery = [(img, f"t={fmt_ts(ts)} (frame {fid})") for fid, ts, img in get_images(ids)]
    return out.get("answer", ""), gallery


with gr.Blocks(title="Video Q&A") as demo:
    gr.Markdown("## Ask questions about your video")
    with gr.Row():
        q = gr.Textbox(label="Question", placeholder="Was there a man in the video?", scale=4)
        video = gr.Dropdown(["All"] + list_videos(), value="All", label="Video", scale=1)
        k = gr.Slider(3, 20, value=8, step=1, label="Frames to retrieve", scale=1)
    btn = gr.Button("Ask", variant="primary")
    ans = gr.Markdown()
    gal = gr.Gallery(label="Relevant frames", columns=4, height="auto")
    btn.click(ask, [q, video, k], [ans, gal])
    q.submit(ask, [q, video, k], [ans, gal])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860,share=True)  # reachable from other devices on your network
