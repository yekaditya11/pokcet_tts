"""
Pocket TTS FastAPI Server
Run with: python server.py
"""
import io
import os
import threading
import logging
from pathlib import Path
from queue import Queue
from contextlib import asynccontextmanager

import scipy.io.wavfile
import uvicorn
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pocket_tts import TTSModel
from pocket_tts.data.audio import stream_audio_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────
DEVICE = os.getenv("DEVICE", "cpu")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
LSD_DECODE_STEPS = int(os.getenv("LSD_DECODE_STEPS", "1"))
VOICE_PATH = Path(__file__).parent / "homesoul.wav"

# ── Global State ───────────────────────────────────────────
tts_model: TTSModel | None = None
default_voice_state: dict | None = None
model_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and default voice on startup."""
    global tts_model, default_voice_state
    logger.info(f"Loading Pocket TTS model on device={DEVICE}...")
    tts_model = TTSModel.load_model(temp=TEMPERATURE, lsd_decode_steps=LSD_DECODE_STEPS)

    if DEVICE != "cpu":
        os.environ["NO_CUDA_GRAPH"] = "1"
        tts_model.to(DEVICE)

    logger.info(f"Loading default voice from {VOICE_PATH}...")
    default_voice_state = tts_model.get_state_for_audio_prompt(VOICE_PATH)
    logger.info("Model and voice loaded. Server ready.")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Pocket TTS API",
    description="Text-to-Speech API powered by Kyutai Pocket TTS",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────
def stream_tts(text: str):
    """Generate audio chunks and yield as streaming WAV."""
    queue = Queue()

    class QueueWriter(io.IOBase):
        def __init__(self, q):
            self.queue = q
        def write(self, data):
            self.queue.put(data)
        def flush(self):
            pass
        def close(self):
            self.queue.put(None)

    def _produce():
        chunks = tts_model.generate_audio_stream(
            model_state=default_voice_state, text_to_generate=text
        )
        stream_audio_chunks(QueueWriter(queue), chunks, tts_model.config.mimi.sample_rate)

    thread = threading.Thread(target=_produce)
    thread.start()

    while True:
        data = queue.get()
        if data is None:
            break
        yield data

    thread.join()


# ── Endpoints ──────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "healthy", "device": DEVICE}


@app.get("/system")
async def system_info():
    """Return current system load and memory info."""
    import psutil
    load_1, load_5, load_15 = os.getloadavg()
    mem = psutil.virtual_memory()
    return {
        "load": [round(load_1, 2), round(load_5, 2), round(load_15, 2)],
        "memory_percent": mem.percent,
        "memory_used_gb": round(mem.used / (1024**3), 2),
        "cpu_count": psutil.cpu_count(),
    }


@app.post("/tts")
async def text_to_speech(text: str = Form(...)):
    """Generate streaming speech from text."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    return StreamingResponse(
        stream_tts(text),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=generated_speech.wav",
            "Transfer-Encoding": "chunked",
        },
    )


@app.post("/tts/sync")
async def text_to_speech_sync(text: str = Form(...)):
    """Generate complete audio from text (non-streaming)."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    with model_lock:
        audio = tts_model.generate_audio(default_voice_state, text)

    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, tts_model.sample_rate, audio.numpy())
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=generated_speech.wav"},
    )


WORKERS = int(os.getenv("WORKERS", "4"))

if __name__ == "__main__":
    uvicorn.run("server:app", host=HOST, port=PORT, workers=WORKERS, reload=False)
