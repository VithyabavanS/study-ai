import os

from dotenv import load_dotenv
from google import genai

from config import LLM_MODEL

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.models.generate_content(
    model=LLM_MODEL,
    contents="Say hello in Tamil and in English.",
)
print(response.text)