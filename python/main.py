from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, field_validator, ValidationError
import json

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
    fields: list[str]

    @field_validator("fields")
    @classmethod
    def validate_fields(cls, v):
        if not v:
            raise ValueError("fields list cannot be empty")
        if len(v) > 50:
            raise ValueError("Too many fields — max 50 allowed")
        cleaned = [f.strip() for f in v if f.strip()]
        if not cleaned:
            raise ValueError("fields cannot be blank strings")
        return cleaned


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        schema: str = Form(...)
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

    try:
        validated_schema = ExtractionSchema.model_validate(raw)
    except ValidationError as e:
        return JSONResponse(
            status_code=422,
            content={"detail": e.errors()}
        )

    prompt = build_prompt(validated_schema.model_dump())

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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")
            session_id = str(message.get("session_id", "default"))

            # NEW: Allow clearing the session context
            if msg_type == "reset":
                SESSIONS[session_id] = {}
                await websocket.send_json({
                    "type": "status",
                    "status": "reset",
                    "message": f"Session {session_id} reset."
                })
                print(f"Session reset: {session_id}")
                continue

            if msg_type == "audio":
                audio_data_base64 = message.get("audio_data")
                mime_type = message.get("mime_type", "audio/webm")
                
                # NEW: Retrieve data from BACKEND memory instead of frontend
                previous_data = SESSIONS.get(session_id, {})
                
                print(f"Received audio chunk for session: {session_id}")
                if previous_data:
                    print(f"Backend context found for {session_id}: {list(previous_data.keys())}")
                else:
                    print(f"New or empty session for {session_id}")
                
                import base64
                audio_bytes = base64.b64decode(audio_data_base64)
                
                if not processor.is_valid_audio(audio_bytes):
                    await websocket.send_json({
                        "type": "error",
                        "message": "No speech detected or audio too quiet."
                    })
                    continue

                # Static schema for medical extraction
                schema = {
                    "fields": [
                        "Chief complaint",
                        "duration",
                        "Cause",
                        "severity"
                    ]
                }
                prompt = build_prompt(schema, context=previous_data)

                await websocket.send_json({
                    "type": "stream_start",
                    "stream_id": session_id
                })

                full_response_text = ""
                try:
                    async for chunk in stream_extract(
                        audio_bytes,
                        prompt,
                        mime_type=mime_type
                    ):
                        cleaned_chunk = processor.clean_text(chunk)
                        if not cleaned_chunk: continue
                        
                        full_response_text += cleaned_chunk
                        await websocket.send_json({
                            "type": "stream_chunk",
                            "text": cleaned_chunk
                        })
                    
                    # Store final extraction for the next chunk
                    # Robust check: clean the full text one last time
                    if full_response_text.strip().startswith("{"):
                        try:
                            extracted_data = json.loads(full_response_text)
                            SESSIONS[session_id] = extracted_data
                            print(f"Saved state for {session_id}")
                        except Exception as e:
                            print(f"Failed to save session state: {e}")

                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
                
                await websocket.send_json({
                    "type": "stream_end"
                })


    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass