from utils.vector_store import VectorStore
from utils.asr import ASRModel
from utils.llm import OpenAIModel
import os

from dotenv import load_dotenv
load_dotenv()


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.environ.get("OPENAI_ENDPOINT")

QDRANT_URL = os.environ.get("QDRANT_URL")
COLLECTION_NAME = "video_contents"

llm = OpenAIModel(api_key=OPENAI_API_KEY, endpoint=OPENAI_ENDPOINT)
asr = ASRModel()
vector_store = VectorStore(qdrant_url=QDRANT_URL,
                           collection_name=COLLECTION_NAME, llm=llm)
