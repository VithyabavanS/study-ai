import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.models.generate_content(
    # model="gemini-2.5-flash-lite",
    model="gemini-3.5-flash-lite",
    contents="Say hello in Tamil and in English.",
)
print(response.text)