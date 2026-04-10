from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, field_validator, ValidationError
import json
import base64
import socketio

from prompt import build_prompt
from service import stream_extract
from utils import AudioProcessor
from contextlib import asynccontextmanager

# NEW: Backend Session Memory (In-memory storage for clinical context)
SESSIONS = {}
processor = AudioProcessor(use_vad=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lazy load VAD model on startup
    processor._load_vad()
    yield
    # Cleanup if needed

class ExtractionSchema(BaseModel):
    fields: dict | list[str]
    instructions: str | None = None
    knowledgebase: str | None = None

    @field_validator("fields")
    @classmethod
    def validate_fields(cls, v):
        if not v:
            raise ValueError("fields cannot be empty")
        if isinstance(v, list):
            if len(v) > 50:
                raise ValueError("Too many fields — max 50 allowed")
            cleaned = [f.strip() for f in v if f.strip()]
            if not cleaned:
                raise ValueError("fields cannot be blank strings")
            return cleaned
        elif isinstance(v, dict):
            if len(v) > 50:
                raise ValueError("Too many fields — max 50 allowed")
            cleaned = {k.strip(): val.strip() if isinstance(val, str) else val for k, val in v.items() if k.strip()}
            if not cleaned:
                raise ValueError("fields cannot be blank strings")
            return cleaned
        return v


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Socket.IO Setup ---
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio, app)

# --- Extraction Events ---

@sio.on("reset_extraction_session", namespace="/extraction")
async def handle_reset(sid, data):
    session_id = str(data.get("session_id", "default"))
    SESSIONS[session_id] = {}
    await sio.emit("extraction_status", {
        "status": "reset",
        "message": f"Session {session_id} reset."
    }, to=sid, namespace="/extraction")
    print(f"Extraction session reset: {session_id}")

@sio.on("audio_chunk", namespace="/extraction")
async def handle_audio_chunk(sid, data):
    session_id = str(data.get("session_id", "default"))
    audio_data_base64 = data.get("audio_data")
    mime_type = data.get("mime_type", "audio/webm")
    schema = data.get("schema")
    instructions = data.get("instructions")
    knowledgebase = data.get("knowledgebase")

    if not audio_data_base64:
        await sio.emit("extraction_error", {"message": "No audio data provided"}, to=sid, namespace="/extraction")
        return

    try:
        audio_bytes = base64.b64decode(audio_data_base64)
    except Exception as e:
        await sio.emit("extraction_error", {"message": f"Invalid base64: {str(e)}"}, to=sid, namespace="/extraction")
        return

    if not processor.is_valid_audio(audio_bytes):
        await sio.emit("extraction_error", {"message": "No speech detected or audio too quiet."}, to=sid, namespace="/extraction")
        return

    previous_data = SESSIONS.get(session_id, {})
    
    if not schema:
        schema = {"fields": ["Chief complaint", "duration", "Cause", "severity"]}

    prompt = build_prompt(
        schema,
        context=previous_data,
        instructions=instructions,
        knowledgebase=knowledgebase
    )

    await sio.emit("extraction_stream_start", {"stream_id": session_id}, to=sid, namespace="/extraction")

    full_response_text = ""
    try:
        async for chunk in stream_extract(audio_bytes, prompt, mime_type=mime_type):
            cleaned_chunk = processor.clean_text(chunk)
            if not cleaned_chunk: continue
            
            full_response_text += cleaned_chunk
            await sio.emit("extraction_stream_chunk", {"text": cleaned_chunk}, to=sid, namespace="/extraction")
        
        if full_response_text.strip().startswith("{"):
            try:
                extracted_data = json.loads(full_response_text)
                SESSIONS[session_id] = extracted_data
                print(f"Saved extraction state for {session_id}")
            except Exception as e:
                print(f"Failed to save extraction session state: {e}")

    except Exception as e:
        await sio.emit("extraction_error", {"message": str(e)}, to=sid, namespace="/extraction")
    
    # Emit final event with full data and status tag
    await sio.emit("extraction_stream_end", {
        "stream_id": session_id,
        "data": SESSIONS.get(session_id),
        "text": full_response_text,
        "status": "completed"
    }, to=sid, namespace="/extraction")

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB
ALLOWED_MIMES = {
    "audio/wav",
    "audio/mpeg",
    "audio/mp4",
    "audio/webm",
    "audio/ogg",
    "audio/x-wav",
    "audio/x-m4a",
}


@app.post("/extract")
async def extract_stream(
        audio: UploadFile = File(...),
        schema: str = Form(...),
        instructions: str = Form(None),
        knowledgebase: str = Form(None)
):
    if audio.content_type not in ALLOWED_MIMES:
        return JSONResponse(
            status_code=415,
            content={
                "detail": f"Unsupported file type '{audio.content_type}'. "
                          f"Allowed: {sorted(ALLOWED_MIMES)}"
            }
        )

    audio_bytes = await audio.read()

    if len(audio_bytes) == 0:
        return JSONResponse(
            status_code=400,
            content={"detail": "Uploaded audio file is empty."}
        )

    if len(audio_bytes) > MAX_FILE_SIZE:
        return JSONResponse(
            status_code=413,
            content={
                "detail": f"File too large ({len(audio_bytes) // (1024 * 1024)} MB). "
                          f"Maximum allowed is 25 MB."
            }
        )

    try:
        raw = json.loads(schema)
    except json.JSONDecodeError:
        schema = schema.replace(".", ",")
        if "," in schema and not schema.strip().startswith(("{", "[")):
            raw = [s.strip() for s in schema.split(",")]
        else:
            return JSONResponse(
                status_code=422,
                content={
                    "detail": 'Invalid schema. Send valid JSON e.g. ["field1", "field2"]'
                }
            )

    if isinstance(raw, list):
        raw = {"fields": raw}
    elif isinstance(raw, dict) and "fields" not in raw:
        # NEW: Allow direct JSON dictionary without "fields" wrapper
        raw = {"fields": raw}

    try:
        validated_schema = ExtractionSchema.model_validate(raw)
    except ValidationError as e:
        return JSONResponse(
            status_code=422,
            content={"detail": e.errors()}
        )

    prompt = build_prompt(
        validated_schema.fields, 
        instructions=validated_schema.instructions or instructions,
        knowledgebase=validated_schema.knowledgebase or knowledgebase
    )

    async def generator():
        try:
            async for chunk in stream_extract(
                audio_bytes,
                prompt,
                mime_type=audio.content_type
            ):
                yield chunk
        except Exception as e:
            yield json.dumps({"error": str(e), "status": "failed"})

    return StreamingResponse(
        generator(),
        media_type="text/plain",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        }
    )