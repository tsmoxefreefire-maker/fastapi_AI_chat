import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.post("/summarize")
async def summarize_story(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".txt"):
        raise HTTPException(
            status_code=400, detail="Please upload a .txt file only."
        )

    file_bytes = await file.read()
    story_text = file_bytes.decode("utf-8")

    messages_list = [
        {
            "role": "system",
            "content": "You are a helpful assistant that summarizes stories simply.",
        },
        {
            "role": "user",
            "content": f"Summarize this story:\n\n{story_text}",
        },
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages_list,  # type: ignore
    )

    ai_summary = response.choices[0].message.content  # type: ignore

    return {"file_name": file.filename, "summary": ai_summary}
