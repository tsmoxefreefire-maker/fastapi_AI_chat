import os
from fastapi import FastAPI
from google import genai

app = FastAPI()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


@app.get("/User_name&Age")
def get_user_name_and_age(name: str, age: int):
    return {"your Name is:": name, "your Age is:": age}


@app.get("/ask_ai")
def ask_ai(user_question: str):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_question,
    )

    return {"question": user_question, "ai_answer": response.text}
