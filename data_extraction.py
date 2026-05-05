from utils.llm import OpenAIModel
from utils.asr import ASRModel
from .video_processing import VideoProcessor
from typing import TypedDict
from itertools import batched


class ExtractedData(TypedDict):
    transcript: str
    caption: str


class DataExtractor:
    def __init__(self, video_path: str, llm: OpenAIModel, asr: ASRModel):
        self.llm = llm
        self.asr = asr
        self.processor = VideoProcessor(video_path)

    def _transcribe_audio(self) -> str:
        audio_path = self.processor.get_audio()
        with open(audio_path, "rb") as f:
            return self.asr.transcribe(audio_path)

    def _caption_video(self) -> str:
        frames = self.processor.sample_video()

        system_prompt = """
        You are a video analysis assistant. Given frames from a video segment and an optional transcript, write a concise 2–4 sentence description of 
        what is visually happening. Include key objects, on-screen text, people, and actions. Be specific and factual.
        """

        user_prompt = "Describe the frames."

        caption = ""

        sampling_size = 8

        for sample in batched(frames, sampling_size):
            try:
                caption += self.llm.call_llm(messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt}] + [
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{frame}"}}
                        for frame in sample
                    ]},
                ])
            finally:
                continue

        return caption

    def extract_data(self) -> ExtractedData:
        transcript = self._transcribe_audio()
        caption = self._caption_video()

        return {transcript, caption}
