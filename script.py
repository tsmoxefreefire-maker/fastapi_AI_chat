import io
import os
from enum import Enum
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from google import genai
from google.genai import types

app = FastAPI(
    title="AI Developer & Learning Toolkit API",
    description="A multi-purpose AI service using Gemini 3.5 Flash",
    version="1.0.0",
)


# -------------------------------------------------------------------
# Helper: Get Gemini Client
# -------------------------------------------------------------------
def get_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500, detail="GEMINI_API_KEY environment variable is not set."
        )
    return genai.Client(api_key=api_key)


@app.get("/", tags=["General"])
def home():
    return {"message": "API is online. Go to /docs to use the interactive UI."}


# -------------------------------------------------------------------
# Part 1: Programming-Only AI Assistant (Form Field Input)
# -------------------------------------------------------------------
@app.post("/chat/programming", tags=["Part 1: Programming Chat"])
async def programming_chat(
    question: str = Form(..., description="Type your programming question here")
):
    """
    Ask Gemini anything related strictly to programming and computer science.
    """
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    client = get_gemini_client()

    system_instruction = """
    You are a strict programming assistant.
    Answer ONLY questions related to programming, software engineering, databases, algorithms, web development, and computer science.
    If the user asks about ANY topic outside programming, politely decline.

    Formatting Rules:
    - Write in clean, simple plain text.
    - Do NOT use markdown symbols like ###, **, or LaTeX math symbols like $.
    - Use simple numbers (1, 2, 3) and simple dashes (-) for lists.
    """

    prompt = f"{system_instruction}\n\nUser Question: {question}"

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
        )
        return {"answer": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------------
# Part 2: Document Summarizer (.txt File Export)
# -------------------------------------------------------------------
@app.post("/summarize", tags=["Part 2: Document Summarizer"])
async def summarize_document(file: UploadFile = File(...)):
    """
    Upload any file (PDF, TXT, Image, Document) to extract key points and download as .txt file.
    """
    client = get_gemini_client()

    try:
        file_bytes = await file.read()
        mime_type = file.content_type or "text/plain"

        uploaded_part = types.Part.from_bytes(
            data=file_bytes,
            mime_type=mime_type,
        )

        prompt = """
        Analyze the attached file carefully and extract the most important information.
        
        Formatting Rules:
        1. Output clean plain text without any markdown code blocks (no ```text).
        2. Do NOT use markdown headers like ### or bold symbols like **.
        3. Use clear, simple headings and simple dashes (-) for bullet points.
        4. Keep the output language matching the original file.
        """

        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=[uploaded_part, prompt],
        )

        txt_data = response.text
        if not txt_data or not txt_data.strip():
            raise HTTPException(
                status_code=500, detail="Generated summary was empty."
            )

        file_stream = io.BytesIO(txt_data.encode("utf-8"))  # type: ignore

        original_name = file.filename or "file"
        base_name = os.path.splitext(original_name)[0]
        new_filename = f"summary_{base_name}.txt"

        return StreamingResponse(
            file_stream,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename={new_filename}"},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------------
# Part 3: Custom Question Generator (.txt File Export)
# -------------------------------------------------------------------
@app.post("/generate_questions", tags=["Part 3: Question Generator"])
async def generate_questions_file(
    num_questions: int = Form(5, description="Total number of questions"),
    mcq_percent: int = Form(0, description="Multiple choice percentage (0-100)"),
    tf_percent: int = Form(100, description="True/False percentage (0-100)"),
    essay_percent: int = Form(0, description="Short answer percentage (0-100)"),
    file: UploadFile = File(...),
):
    """
    Generate customized exam questions based on an uploaded file and download as .txt file.
    """
    if num_questions < 1 or num_questions > 20:
        raise HTTPException(
            status_code=400, detail="Please enter a question count between 1 and 20."
        )

    total_percentage = mcq_percent + tf_percent + essay_percent
    if total_percentage != 100:
        raise HTTPException(
            status_code=400,
            detail=f"Percentages must sum to 100%. Current sum: {total_percentage}%",
        )

    count_mcq = round((mcq_percent / 100) * num_questions)
    count_tf = round((tf_percent / 100) * num_questions)
    count_essay = num_questions - (count_mcq + count_tf)

    client = get_gemini_client()

    try:
        file_bytes = await file.read()
        mime_type = file.content_type or "text/plain"

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
        1. Output plain text directly without markdown formatting (no ###, no **, no ```text).
        2. List each question clearly with simple numbers.
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


# -------------------------------------------------------------------
# Part 4: Core School Subjects Tutor (Dropdown Selection)
# -------------------------------------------------------------------
class SubjectName(str, Enum):
    MATHEMATICS = "Mathematics"
    SCIENCE = "Science"
    ARABIC = "Arabic"
    ENGLISH = "English"


@app.post("/chat/subjects", tags=["Part 4: Core Subjects Tutor"])
async def subjects_chat(
    subject: SubjectName = Form(..., description="Select the subject"),
    question: str = Form(..., description="Type your question or request here"),
):
    """
    Ask specialized questions in Mathematics, Science, Arabic, or English.
    Returns clean, plain text responses without complex formatting symbols.
    """
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    client = get_gemini_client()

    system_instruction = f"""
    You are an expert tutor dedicated EXCLUSIVELY to teaching: {subject.value}.

    STRICT BOUNDARIES:
    - Answer ONLY questions related to {subject.value}. If the student asks about other topics, politely decline.

    CRITICAL FORMATTING RULES (VERY IMPORTANT):
    1. Write in plain, highly readable text ONLY.
    2. Do NOT use Markdown symbols like ###, **, __, or markdown tables.
    3. Do NOT use LaTeX math code or dollar signs like $, $$, \\frac, etc. Write math expressions in simple plain text (e.g. write "x + 2 = 5" or "1/2" directly).
    4. Use simple numbered lists (1, 2, 3) and simple dashes (-) for points.

    TEACHING CONTENT:
    1. Explain step-by-step in very simple, clean language.
    2. If the user asks for basics or learning from scratch, provide a clear "Step-by-Step Basics Guide" in simple numbered steps.
    3. Match the student's input language (Arabic or English).
    """

    prompt = f"{system_instruction}\n\nStudent Question: {question}"

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
        )
        return {
            "selected_subject": subject.value,
            "answer": response.text,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
