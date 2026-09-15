from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from analyzer import analyze_resume, calculate_match_score
from extractor import extract_resume_text
import os
import tempfile


app = FastAPI(title = "Career Gap Analyzer")

@app.get('/health')
def health_check():
    return {"status":"ok"}

@app.post('/analyze')
async def analyse(resume: UploadFile = File(), job_description : str = Form()):

    if not resume.filename.endswith((".pdf", ".docx")):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX files are supported."
        )

    if not job_description.strip():
        raise HTTPException(
            status_code=400,
            detail="Job description cannot be empty."
        )
    
    suffix = os.path.splitext(resume.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:

        temp_file.write(await resume.read())
        temp_file_path = temp_file.name

    try:
        #Extract
        resume_text = extract_resume_text(temp_file_path)

        #Analyze
        result = analyze_resume(
            resume_text,
            job_description
        )

        #Calculate score
        score = calculate_match_score(result)

        return {
            "match_score": score,
            "analysis": result.model_dump()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

    finally:
        os.remove(temp_file_path)