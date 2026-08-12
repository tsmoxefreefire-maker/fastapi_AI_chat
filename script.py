import io
import os
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from google import genai
from google.genai import types

app = FastAPI()


@app.post("/generate_questions")
async def generate_questions_file(
    num_questions: int = Form(5, description="إجمالي عدد الأسئلة"),
    mcq_percent: int = Form(0, description="نسبة أسئلة الاختيار من متعدد (0-100)"),
    tf_percent: int = Form(100, description="نسبة أسئلة الصح والخطأ (0-100)"),
    essay_percent: int = Form(0, description="نسبة الأسئلة المقالية / الإجابة القصيرة (0-100)"),
    file: UploadFile = File(...),
):
    # 1. التحقق من عدد الأسئلة
    if num_questions < 1 or num_questions > 20:
        raise HTTPException(
            status_code=400, detail="Please enter a question count between 1 and 20."
        )

    # 2. التحقق من مجموع النسب المئوية
    total_percentage = mcq_percent + tf_percent + essay_percent
    if total_percentage != 100:
        raise HTTPException(
            status_code=400,
            detail=f"مجموع النسب المئوية يجب أن يكون 100%. المجموع الحالي: {total_percentage}%"
        )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500, detail="GEMINI_API_KEY is not set in Render Environment"
        )

    # 3. حساب عدد الأسئلة لكل نوع
    count_mcq = round((mcq_percent / 100) * num_questions)
    count_tf = round((tf_percent / 100) * num_questions)
    count_essay = num_questions - (count_mcq + count_tf)

    try:
        file_bytes = await file.read()
        mime_type = file.content_type or "text/plain"

        client = genai.Client(api_key=api_key)

        # تحويل الملف المرفوع (مهما كان نوعه: PDF, PNG, TXT...) إلى Part يفهمه Gemini
        uploaded_part = types.Part.from_bytes(
            data=file_bytes,
            mime_type=mime_type,
        )

        prompt = f"""
        Analyze the provided document/file and generate exactly {num_questions} comprehension questions based on it.

        Required Distribution of Question Types:
        - Multiple Choice Questions (MCQ): {count_mcq}
        - True/False Questions: {count_tf}
        - Short Answer Questions: {count_essay}

        Formatting Requirements for the Output (.txt):
        1. Format the entire response as clean, readable plain text (TXT format).
        2. Group the questions clearly by section or list them sequentially with clear question type tags.
        3. For MCQs, list choices as A), B), C), D).
        4. For True/False, list choices as [ True / False ].
        5. At the very bottom of the document, provide a separate "ANSWER KEY / دليل الإجابات" section with all correct answers numbered clearly.
        6. Use the same primary language as the uploaded file (Arabic or English).
        7. Do NOT wrap the output inside markdown code fences like ```text. Output plain text directly.
        """

        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=[uploaded_part, prompt],
        )

        txt_data = response.text

        if not txt_data or not txt_data.strip():
            raise HTTPException(
                status_code=500, detail="Generated content was empty."
            )

        file_stream = io.BytesIO(txt_data.encode("utf-8"))  # type: ignore

        # استخراج اسم الملف واستبدال امتداده بـ .txt
        original_name = file.filename or "file"
        base_name = os.path.splitext(original_name)[0]
        new_filename = f"questions_{base_name}.txt"

        return StreamingResponse(
            file_stream,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename={new_filename}"},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
