import os
from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.get("/User_name&Age")
def get_user_name_and_age(name: str, age: int):
    return {"your Name is:": name, "your Age is:": age}


@app.get("/ask_ai")
def ask_ai(user_question: str):
    messages_list: list = [
        {"role": "system", "content": "you are a helpful coding tutor"},
        {"role": "user", "content": user_question},
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages_list,
    )

    answer = response.choices[0].message.content
    return {"question": user_question, "ai_answer": answer}
