from utils.llm import get_llm_client
from utils.asr import get_asr_model

class DataExtractor:
    def __init__(self):
        self.llm = get_llm_client()
        self.asr = get_asr_model()

    def _transcribe_audio(self, audio_path):
        with open(audio_path, "rb") as f:
            result = self.asr.transcribe(audio_path)
        return result["text"]