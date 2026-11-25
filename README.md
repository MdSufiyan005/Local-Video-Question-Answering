# VidSeek

**VidSeek** is a local video question-answering system that lets you ask natural-language questions about a video and receive an answer supported by relevant video frames and timestamps.

It runs entirely on an NVIDIA Jetson Orin using llama.cpp, a vision-language model, vector search, SQLite, and Gradio. No cloud API is required after the models are downloaded.

## About the project

VidSeek processes a video in two stages:

1. **Ingestion** — The video is split into frames. Visually similar frames are filtered out, and distinct frames are sent to a vision-language model (VLM), which generates a detailed caption for each frame. The captions are converted into embeddings and stored in Qdrant. The actual frame images are stored in SQLite.

2. **Question answering** — A user asks a question in the Gradio interface. The question is embedded and compared with the stored caption embeddings. The most relevant frames are retrieved, and an LLM generates an answer using that context. Relevant frame images and timestamps are displayed alongside the answer.

Example:

```text
Question: Was there a man in the video?

Answer: Yes. A man wearing an orange shirt and black pants appears walking along a paved path.

Supporting frames:
- Frame 1 at 00:00:00
- Frame 2 at 00:00:02
- Frame 3 at 00:00:04
```

## Features

- Extracts frames from a video
- Removes visually similar or duplicate frames
- Generates captions using a local VLM
- Stores captions and metadata in Qdrant
- Stores frame images in SQLite
- Answers questions using retrieved frame context
- Displays relevant frames and timestamps
- Runs locally with llama.cpp
- Provides a simple Gradio web interface
- Designed for NVIDIA Jetson Orin

## Project structure

```text
ECHS-END-TERM-PROJ/
├── app.py               # Gradio question-answering interface
├── ingest.py            # Video ingestion and indexing pipeline
├── common.py            # Shared configuration and helper functions
├── start_servers.sh     # Starts the VLM and embedding servers
├── requirements.txt     # Python dependencies
├── llama.cpp/           # llama.cpp source and build files
├── models/
│   ├── vlm/             # Vision-language model files
│   └── embed/           # Embedding model files
├── frames.db            # SQLite database containing frame images
├── qdrant_data/         # Qdrant vector database storage
├── vlm.log              # VLM server logs
└── embed.log            # Embedding server logs
```

## Requirements

- NVIDIA Jetson Orin
- JetPack with CUDA support
- Python 3.10 or later
- llama.cpp built with CUDA
- Internet access for the initial model download
- At least 8 GB of unified memory

The project uses:

- Qwen2.5-VL-3B-Instruct for frame captioning and answering
- nomic-embed-text-v1.5 for text embeddings
- Qdrant for vector storage
- SQLite for frame image storage
- Gradio for the web interface

## Setup

### 1. Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Build llama.cpp

Clone and build llama.cpp with CUDA support:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DLLAMA_OPENSSL=ON

cmake --build build -j4 --config Release
cd ..
```

For Jetson Orin, use CUDA architecture `87`.

Add the built binaries to your path:

```bash
export PATH=$HOME/ECHS-END-TERM-PROJ/llama.cpp/build/bin:$PATH
```

Verify that `llama-server` is available:

```bash
which llama-server
```

### 3. Download the models

Create model folders:

```bash
mkdir -p models/vlm models/embed
```

Download the vision-language model:

```bash
hf download ggml-org/Qwen2.5-VL-3B-Instruct-GGUF \
  Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf \
  mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf \
  --local-dir models/vlm
```

Download the embedding model:

```bash
hf download nomic-ai/nomic-embed-text-v1.5-GGUF \
  nomic-embed-text-v1.5.f16.gguf \
  --local-dir models/embed
```

Check that the model files were downloaded:

```bash
find models -name "*.gguf" -exec ls -lh {} \;
```

You should see:

- One Qwen2.5-VL main model file
- One Qwen2.5-VL multimodal projector file, usually named `mmproj-*`
- One embedding model file

## Running the project

### 1. Start the servers

Make the startup script executable:

```bash
chmod +x start_servers.sh
```

Start the VLM and embedding servers:

```bash
./start_servers.sh
```

The script starts:

- VLM server on port `8080`
- Embedding server on port `8081`

Check that both servers are running:

```bash
pgrep -a llama-server
```

Check server health:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8081/health
```

Both should return:

```json
{"status":"ok"}
```

You can view live server logs with:

```bash
tail -f vlm.log embed.log
```

### 2. Ingest a video

Run:

```bash
python ingest.py your_video.mp4
```

Example:

```bash
python ingest.py 76621-559757958.mp4
```

This will:

- Extract frames from the video
- Filter out similar frames
- Generate captions using the VLM
- Create embeddings from the captions
- Store embeddings and metadata in Qdrant
- Store frame images in SQLite

### 3. Open the app

Keep both llama.cpp servers running, then start the Gradio app:

```bash
python app.py
```

Open:

```text
http://localhost:7860
```

If you are accessing the Jetson from another device:

```text
http://<jetson-ip>:7860
```

Enter a question such as:

```text
Was there a man in the video?
What color shirt was the person wearing?
Was anyone walking?
```

The app will return an answer, relevant frame captions, and supporting frames with timestamps.

## Useful options

Use a smaller batch size if captions become inaccurate:

```bash
python ingest.py video.mp4 --batch 2
```

Use a lower perceptual-hash threshold to keep more frames:

```bash
python ingest.py video.mp4 --hash-thresh 12
```

Increase the frame sampling rate:

```bash
python ingest.py video.mp4 --fps 2
```

## Jetson performance

Enable maximum performance mode before ingesting or using the app:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

Check GPU usage while the VLM is processing frames:

```bash
tegrastats | grep -o --line-buffered "GR3D_FREQ [0-9]*%"
```

During captioning, `GR3D_FREQ` should increase if the GPU is being used.

## Important notes

- Run `ingest.py` before `app.py`.
- Do not run ingestion and the app at the same time if using embedded Qdrant.
- Running ingestion again on the same video may create duplicate entries.
- To reset the database, remove the existing data:

```bash
rm -rf frames.db qdrant_data
```

Then ingest the video again:

```bash
python ingest.py your_video.mp4
```

- On an 8 GB Jetson Orin, use the 3B VLM. A 7B model may not fit in available memory.
- If captions become mixed between frames, reduce the batch size to 2 or 1.
- If you run out of memory, reduce the VLM image resolution or batch size.
