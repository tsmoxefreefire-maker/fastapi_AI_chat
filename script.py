import io
import os
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from google import genai
from google.genai import types

app = FastAPI()


@app.get("/")
def home():
    return {"message": "API is running. Go to /docs to test it."}


@app.post("/generate_questions")
async def generate_questions_file(
    num_questions: int = Form(5, description="Total number of questions"),
    mcq_percent: int = Form(10, description="Multiple choice percentage {0-100}"),
    tf_percent: int = Form(10, description="True/False percentage {0-100}"),
    essay_percent: int = Form(30, description="Short answer percentage {0-100}"),
    file: UploadFile = File(...),
):
    # Validate number of questions
    if num_questions < 1 or num_questions > 20:
        raise HTTPException(
            status_code=400, detail="Please enter a question count between 1 and 20."
        )

    # Validate percentage total
    total_percentage = mcq_percent + tf_percent + essay_percent
    if total_percentage != 100:
        raise HTTPException(
            status_code=400,
            detail=f"Percentages must sum to 100%. Current sum: {total_percentage}%",
        )

    # Check API Key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500, detail="GEMINI_API_KEY environment variable is not set."
        )

    # Calculate question distribution
    count_mcq = round((mcq_percent / 100) * num_questions)
    count_tf = round((tf_percent / 100) * num_questions)
    count_essay = num_questions - (count_mcq + count_tf)

    try:
        file_bytes = await file.read()
        mime_type = file.content_type or "text/plain"

        client = genai.Client(api_key=api_key)

        # Convert uploaded file bytes into Gemini Part format
        uploaded_part = types.Part.from_bytes(
            data=file_bytes,
            mime_type=mime_type,
        )

        prompt = f"""
        Read the attached file and generate exactly {num_questions} comprehension questions based on its content.

        Question Type Distribution:
        - Multiple Choice Questions (MCQ): {count_mcq}
        - True/False Questions: {count_tf}
        - Short Answer Questions: {count_essay}

        Formatting Rules:
        1. Output plain text directly. Do NOT use markdown code blocks like ```text.
        2. List each question clearly with its number and type.
        3. For MCQs, list options as A), B), C), D).
        4. For True/False, list options as [ True / False ].
        5. Add an "ANSWER KEY" section at the end of the file with all correct answers.
        6. Match the language of the uploaded file.
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

        # Set output file name
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
