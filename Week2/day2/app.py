import os
import csv
import time

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from src.helper import llm_pipeline

app = FastAPI(title="Interview Question Creator")

# Serve everything inside /static at the URL path /static/...
# This is how the browser will later be able to load the uploaded PDF
# and download the generated CSV.
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2Templates lets FastAPI render .html files from the templates/ folder,
# and inject Python values into them using {{ variable }} syntax.
templates = Jinja2Templates(directory="templates")

os.makedirs("static/docs", exist_ok=True)
os.makedirs("static/output", exist_ok=True)


@app.get("/")
async def index(request: Request):
    """
    Renders the upload page.
    FastAPI requires the raw `Request` object to be passed into
    TemplateResponse - Jinja2 needs it internally to render correctly.
    """
    return templates.TemplateResponse(request, "index.html", {})


@app.post("/upload")
async def upload(pdf_file: UploadFile = File(...)):
    """
    Step 1: Save the uploaded PDF to static/docs/ so the browser
    can preview it (via an <iframe src="/static/docs/filename.pdf">).
    Returns the path so the frontend JS knows where to point the iframe.
    """
    file_path = f"static/docs/{pdf_file.filename}"

    with open(file_path, "wb") as f:
        content = await pdf_file.read()
        f.write(content)

    return JSONResponse({
        "pdf_filename": file_path,
        "original_name": pdf_file.filename
    })


@app.post("/analyze")
async def analyze(pdf_filename: str):
    """
    Step 2: Run the actual LangChain pipeline on the saved PDF -
    generates ~10 questions, builds a retriever, and answers each
    question. Saves everything to a CSV for download, and also
    returns the Q&A pairs directly so the page can display them
    immediately without needing to download the file.
    """
    answer_generation_chain, questions_list = llm_pipeline(pdf_filename)

    timestamp = str(int(time.time()))
    output_file = f"static/output/QA_{timestamp}.csv"

    qa_pairs = []

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Question", "Answer"])

        for question in questions_list:
            answer = answer_generation_chain.run(question)
            writer.writerow([question, answer])
            qa_pairs.append({"question": question, "answer": answer})

    return JSONResponse({
        "qa_pairs": qa_pairs,
        "csv_path": output_file
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
