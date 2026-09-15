from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from schema import AnalysisResult
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()  

llm_model = ChatGroq(model = "openai/gpt-oss-120b")

llm = llm_model.with_structured_output(AnalysisResult, method="json_schema", strict=True)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4)
)
def analyze_resume(resume_text: str, job_description: str):
    """compare resume against Job Description."""
    system_prompt = """
        You are an experienced technical recruiter and career analyst.
    
    Compare the candidate's resume against the job description.
    
    Your analysis must be based ONLY on the information provided in resume and job description. 
    Do not make assumptions or add information that is not present in the provided text.
    
    Identify:
        - skills mentioned in the job description
        - Skills that match the requirements
        - Required skills that are missing or weakly demonstrated
        - Specific improvements to the resume
        - Topics the candidate should prepare for an interview
    
    For every requirement in jd_skills, determine which specific components
    of that requirement are clearly demonstrated in the resume and which
    components are not demonstrated.

    For each JD requirement:

    - matched_components should contain only components that are clearly
    demonstrated in the resume.
    - missing_components should contain components that are required by the
    JD but are not demonstrated in the resume.

    Do not infer skills from related technologies, job titles, education, or
    general background.

    For compound requirements, evaluate each important component separately.

        """
    messages = [
        ("system", system_prompt),
        ("user", f"Resume: {resume_text}\n\nJob Description: {job_description}")
    ]
    response = llm.invoke(messages)
    return response


def classify_match(match):

    if not match.matched_components:
        return "MISSING"

    if not match.missing_components:
        return "MET"

    return "PARTIAL"


def calculate_match_score(result):
    """Calculate a deterministic match score from requirement statuses."""

    status = {
        "MET": 1,
        "PARTIAL": 0.5,
        "MISSING": 0
    }

    statuses = [
        classify_match(match)
        for match in result.skill_matches
    ]

    if not statuses:
        return 0

    score = sum(status[s] for s in statuses)

    return round((score / len(statuses)) * 100, 2)