import io
import os
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from google import genai

app = FastAPI()


@app.post("/generate_questions")
async def generate_questions_file(
    num_questions: int = Form(5, description="Number of questions to generate"),
    file: UploadFile = File(...),
):
    if not file.filename or not file.filename.endswith(".txt"):
        raise HTTPException(
            status_code=400, detail="Please upload a .txt file only."
        )

    if num_questions < 1 or num_questions > 20:
        raise HTTPException(
            status_code=400, detail="Please enter a question count between 1 and 20."
        )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500, detail="GEMINI_API_KEY is not set in Render Environment"
        )

    try:
        file_bytes = await file.read()
        story_text = file_bytes.decode("utf-8")

        client = genai.Client(api_key=api_key)

        prompt = f"""
        Read the following story and generate exactly {num_questions} comprehension questions based on it.
        Return the result EXCLUSIVELY as a valid JSON object with this structure:
        {{
            "story_title": "Title or summary of story",
            "questions_count": {num_questions},
            "questions": [
                {{
                    "id": 1,
                    "question": "Question text",
                    "answer": "Correct answer"
                }}
            ]
        }}

        Story text:
        {story_text}
        """

        interaction = client.interactions.create(
            model="gemini-2.0-flash",
            input=prompt,
            response_format={"type": "object"},
        )

        # استخراج النص بطريقة آمنة تضمن الحصول على JSON غير فارغ
        json_data = ""
        if hasattr(interaction, "outputs") and interaction.outputs:
            json_data = interaction.outputs[0].text
        elif hasattr(interaction, "result") and interaction.result:
            json_data = str(interaction.result)
        else:
            json_data = str(interaction)

        if not json_data or json_data.strip() == "":
            raise HTTPException(
                status_code=500, detail="Generated JSON response was empty."
            )

        file_stream = io.BytesIO(json_data.encode("utf-8"))  # type: ignore

        new_filename = f"questions_{file.filename.replace('.txt', '.json')}"

        return StreamingResponse(
            file_stream,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={new_filename}"},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
