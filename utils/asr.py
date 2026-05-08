import whisper


class ASRModel:
    def __init__(self):
        self._model = whisper.load_model("base")

    def transcribe(self, audio):
        result = self._model.transcribe(audio)
        return result["text"]
