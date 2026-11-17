
from typing import List

import ollama
from pydantic import BaseModel, Field
from pypdf import PdfReader


# =====================================
# MODEL CONFIGURATION
# =====================================

MODEL_NAME = "llama3.2:1b"


# =====================================
# STRUCTURED RESPONSE SCHEMA
# =====================================

class ResumeReview(BaseModel):

    match_score: int = Field(
        ge=0,
        le=100,
        description="Resume match score between 0 and 100"
    )

    strengths: List[str] = Field(
        description="Relevant skills and qualifications in the resume"
    )

    missing_skills: List[str] = Field(
        description="Required job skills not demonstrated in the resume"
    )

    improvements: List[str] = Field(
        description="Suggestions to improve the resume"
    )

    summary: str = Field(
        description="Overall resume assessment"
    )


# =====================================
# RESUME ANALYSIS
# =====================================

def analyze_resume(resume_text, job_description):

    if not resume_text.strip():
        raise ValueError("Resume text is empty.")

    if not job_description.strip():
        raise ValueError("Job description is empty.")

    prompt = f"""
You are an experienced technical recruiter.

Compare the candidate's resume with the job description.

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

INSTRUCTIONS:

1. Identify relevant skills, projects and qualifications.
2. Include every relevant strength supported by the resume.
3. Do not return an empty strengths list if relevant
   skills or qualifications are present.
4. Identify job requirements missing from the resume.
5. Suggest practical resume improvements.
6. Never invent skills, experience or qualifications.
7. Give a reasonable match score from 0 to 100.
8. Explain your assessment clearly.

Return the result using the requested JSON structure.
"""

    # Send prompt to local Ollama API
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        format=ResumeReview.model_json_schema(),
        options={
            "temperature": 0
        }
    )

    # Debug model response
    print("\n========== RAW MODEL RESPONSE ==========")
    print(response.message.content)
    print("========================================\n")

    # Validate JSON using Pydantic
    result = ResumeReview.model_validate_json(
        response.message.content
    )

    if not result.strengths:
        print("Warning: No strengths identified by the model.")

    return result


# =====================================
# PDF TEXT EXTRACTION
# =====================================

def extract_pdf_text(pdf_file):

    reader = PdfReader(pdf_file)

    extracted_text = []

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            extracted_text.append(page_text)

    return "\n".join(extracted_text)
