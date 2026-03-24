import os
import wave
import json
import numpy as np
import re
from typing import Optional, List, Tuple

class AudioProcessor:
    """
    Audio processing utilities including VAD and filtering, 
    adapted from m.py for use with Gemini.
    """
    
    # Common noise/hallucination patterns to filter out
    PATTERNS = {
        "social_media": [
            r"thanks?\s*(for)?\s*(watching|listening|viewing)",
            r"(please\s*)?(like\s*(and|&))?\s*subscribe",
            r"(see you|catch you)\s*(in\s*the)?\s*next\s*(video|episode|time)",
            r"don't\s*forget\s*to\s*subscribe",
            r"hit\s*(that|the)\s*(like|subscribe|bell)",
        ],
        "attribution": [
            r"subtitles?\s*(by|created|provided)",
            r"captions?\s*(by|created|provided)",
            r"transcribed?\s*by",
            r"translated?\s*by",
            r"copyright\s*\d{4}",
        ],
        "noise": [
            r"^\.{2,}$",
            r"^[ \.\,\!\?]+$",
            r"^(uh|um|hmm|hm|ah|oh|eh)\s*$",
        ]
    }

    # Minimum thresholds
    MIN_AUDIO_DURATION = 0.3
    MIN_RMS_ENERGY = 0.002
    MIN_SPEECH_RATIO = 0.02

    def __init__(self, use_vad: bool = True):
        self.use_vad = use_vad
        self._vad_model = None
        self._vad_utils = None
        self._compiled_patterns = {
            cat: [re.compile(p, re.IGNORECASE) for p in pats]
            for cat, pats in self.PATTERNS.items()
        }

    @property
    def vad_model(self):
        """Lazy load VAD model."""
        if self._vad_model is None and self.use_vad:
            try:
                import torch
                torch.set_num_threads(1)
                self._vad_model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    trust_repo=True
                )
                self._vad_utils = utils
                print("[AudioProcessor] Silero VAD loaded.")
            except Exception as e:
                print(f"[AudioProcessor] VAD load failed: {e}. Using RMS only.")
                self.use_vad = False
        return self._vad_model

    def is_valid_audio(self, audio_bytes: bytes) -> bool:
        """Check if audio contains sufficient speech content."""
        if len(audio_bytes) < 1000: # Very basic size check
            return False
            
        try:
            # Basic RMS check
            import io
            arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            rms = np.sqrt(np.mean(arr**2))
            
            if rms < self.MIN_RMS_ENERGY:
                print(f"[AudioProcessor] Rejected: Low energy ({rms:.5f})")
                return False

            if self.use_vad and self.vad_model:
                import torch
                get_speech_timestamps, _, _, *_ = self._vad_utils
                wav = torch.from_numpy(arr)
                
                # We assume 16kHz for Gemini usually, but check if we need to adjust
                # Silero VAD expects 16k or 8k
                speech_timestamps = get_speech_timestamps(
                    wav, self.vad_model, threshold=0.4, sampling_rate=16000, return_seconds=True
                )
                
                if not speech_timestamps:
                    print("[AudioProcessor] Rejected: No speech detected by VAD")
                    return False
                
                total_speech = sum(ts['end'] - ts['start'] for ts in speech_timestamps)
                duration = len(arr) / 16000
                if (total_speech / duration) < self.MIN_SPEECH_RATIO:
                    print(f"[AudioProcessor] Rejected: Low speech ratio ({total_speech/duration:.2f})")
                    return False

            return True
        except Exception as e:
            print(f"[AudioProcessor] Error in validation: {e}")
            return True # Fail-safe: process anyway

    def clean_text(self, text: str) -> str:
        """Filter out hallucinated patterns from text output."""
        if not text:
            return ""
            
        cleaned = text
        for cat, patterns in self._compiled_patterns.items():
            for p in patterns:
                cleaned = p.sub("", cleaned)
        
        return re.sub(r'\s+', ' ', cleaned).strip()
