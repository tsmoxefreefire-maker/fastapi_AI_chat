import os
from fastapi import FastAPI, HTTPException
from google import genai

app = FastAPI()


@app.get("/ask_ai")
def ask_ai(user_question: str):
    # 1. جلب المفتاح من متغيرات البيئة
    api_key = os.getenv("GEMINI_API_KEY")

    # 2. التحقق من وجود المفتاح
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set in Render environment variables.",
        )

    try:
        # 3. إنشاء العميل بيمرّر المفتاح مباشرة
        client = genai.Client(api_key=api_key)

        # 4. إرسال الطلب
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=user_question,
        )
        return {"question": user_question, "ai_answer": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
