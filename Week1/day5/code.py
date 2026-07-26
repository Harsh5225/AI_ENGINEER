import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

response = llm.invoke("Explain LangChain in simple words.")

# print(response.content[0]["text"])
# streaming output
for chunk in llm.stream(
    "Explain LangChain in simple words. I need a detailed response with examples."
):
    print(chunk.content, end="", flush=True)