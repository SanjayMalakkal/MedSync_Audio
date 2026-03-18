import google.generativeai as genai
from config import GEMINI_API_KEY, MODEL_NAME
import asyncio
import logging

logger = logging.getLogger(__name__)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

GENERATION_CONFIG = {
    "temperature": 0,
    "max_output_tokens": 8192,
    "top_p": 0.9,
    "top_k": 40,
    "response_mime_type": "application/json"
}


async def stream_extract(audio_bytes: bytes, prompt: str, mime_type: str = "audio/wav"):

    logger.info(f"Gemini request — audio size: {len(audio_bytes)} bytes, mime: {mime_type}")

    response = await model.generate_content_async(
        [
            prompt,
            {
                "mime_type": mime_type,
                "data": audio_bytes
            }
        ],
        stream=True,
        generation_config=GENERATION_CONFIG,
    )

    chunk_count = 0

    async for chunk in response:

        if not chunk.candidates:
            continue

        for candidate in chunk.candidates:

            if not candidate.content:
                continue

            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    chunk_count += 1
                    yield part.text

    logger.info(f"Stream complete — {chunk_count} chunks yielded.")