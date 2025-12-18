import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# 2. Initialize the Model
# We use "gemini-1.5-flash" as it is fast and free-tier friendly
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# 3. Ask a simple question
try:
    response = llm.invoke("Hello, explain what RAG is in one sentence.")
    print("SUCCESS! Gemini says:")
    print(response.content)
except Exception as e:
    print("Error:", e)
