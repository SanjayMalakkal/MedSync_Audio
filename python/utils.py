import os
import wave
import numpy as np
import re
from typing import Optional, List, Tuple

class AudioProcessor:
    """
    Handles VAD (Voice Activity Detection) and Hallucination Filtering.
    """
    
    # regex patterns from m.py
    HALLUCINATION_PATTERNS = [
        r"^thanks?\s*(for)?\s*(watching|listening|viewing)",
        r"^(please\s*)?(like\s*(and|&))?\s*subscribe",
        r"^(see you|catch you)\s*(in\s*the)?\s*next\s*(video|episode|time)",
        r"^subtitles?\s*(by|created|provided)",
        r"^captions?\s*(by|created|provided)",
        r"^transcribed?\s*by",
        r"^translated?\s*by",
        r"^\[?(music|applause|laughter|silence)\]?$",
        r"^♪+$",
        r"^\.{2,}$",
        r"^(uh|um|hmm|hm|ah|oh|eh)\s*$",
        r"^you\s*$",
        r"^indoctrinate\.?$",
        r"medical\s*consultation\.?$"
    ]

    MIN_RMS_ENERGY = 0.002
    MIN_AUDIO_DURATION = 0.3

    def __init__(self, use_vad: bool = True):
        self.use_vad = use_vad
        self._vad_model = None
        self._vad_utils = None
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.HALLUCINATION_PATTERNS]

    def _load_vad(self):
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
                print("[AudioProcessor] Silero VAD loaded")
            except Exception as e:
                print(f"[AudioProcessor] VAD Load Error: {e}")
                self.use_vad = False

    def is_valid_audio(self, audio_bytes: bytes) -> bool:
        """Check if audio has sufficient energy and speech."""
        import tempfile
        import os
        
        if not audio_bytes: return False
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name
            
            try:
                import torch
                import torchaudio
                import torchaudio.transforms as T
                
                # Load via torchaudio (handles webm/opus if ffmpeg is present)
                waveform, sample_rate = torchaudio.load(temp_path)
                
                # Resample to 16kHz for Silero VAD
                if sample_rate != 16000:
                    resampler = T.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)
                
                # Convert to mono
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                audio_data = waveform.squeeze().numpy()
                
                if len(audio_data) == 0: return False
                
                duration = len(audio_data) / 16000
                if duration < self.MIN_AUDIO_DURATION: return False
                
                rms = np.sqrt(np.mean(audio_data**2))
                if rms < self.MIN_RMS_ENERGY:
                    print(f"[AudioProcessor] Rejected: Silent (RMS: {rms:.4f})")
                    return False

                if self.use_vad:
                    self._load_vad()
                    if self._vad_model:
                        get_speech_timestamps, _, _, *_ = self._vad_utils
                        wav = torch.from_numpy(audio_data)
                        ts = get_speech_timestamps(wav, self._vad_model, sampling_rate=16000)
                        if not ts:
                            print(f"[AudioProcessor] Rejected: No speech detected by VAD")
                            return False
                
                return True
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        except Exception as e:
            print(f"[AudioProcessor] Validation Error: {e}")
            return True # Fail open


    def clean_text(self, text: str) -> str:
        """Filter out hallucination patterns."""
        if not text: return ""
        t = text.strip()
        for p in self._compiled_patterns:
            if p.search(t):
                print(f"[AudioProcessor] Filtered hallucination: '{t}'")
                return ""
        return t
