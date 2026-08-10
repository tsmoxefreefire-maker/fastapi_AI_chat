import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from google import genai

app = FastAPI()


@app.post("/summarize")
async def summarize_story(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".txt"):
        raise HTTPException(
            status_code=400, detail="Please upload a .txt file only."
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

        prompt = f"Summarize this story simply and highlight the main events:\n\n{story_text}"

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )

        return {"file_name": file.filename, "summary": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
