import os
from itertools import batched
import streamlit as st

from models import get_llm, get_asr_model
from .video_processing import VideoProcessor


class DataExtractor:
    def __init__(self, video_path: str):
        self.llm = get_llm()
        self.asr = get_asr_model()
        self.processor = VideoProcessor(video_path)

    def _transcribe_audio(self) -> str:
        audio_path = self.processor.get_audio()
        try:
            with open(audio_path, "rb") as f:
                return self.asr.transcribe(audio_path)
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def _caption_video(self, frames: list[str]) -> str:
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

    def extract_data(self) -> tuple[str, str]:
        progress = st.progress(0, text="Extracting audio...")
        transcript = self._transcribe_audio()

        progress.progress(
            40, text="Audio transcribed. Extracting dynamic frames...")
        frames, thumbnail = self.processor.sample_video()

        progress.progress(60, text="Analyzing visuals with VLM...")
        caption = self._caption_video(frames)

        progress.progress(100, text="Processing complete! Ready for review.")
        return transcript, caption, f"data:image/jpeg;base64,{thumbnail}"
