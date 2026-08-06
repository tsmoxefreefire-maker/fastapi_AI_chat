import os
from fastapi import FastAPI, HTTPException
from google import genai

app = FastAPI()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


@app.get("/User_name&Age")
def get_user_name_and_age(name: str, age: int):
    return {"your Name is:": name, "your Age is:": age}


@app.get("/ask_ai")
def ask_ai(user_question: str):
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY environment variable is not set.",
        )

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",  # النموذج الجديد المعتمد في المكتبة
            contents=user_question,
        )
        return {"question": user_question, "ai_answer": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
