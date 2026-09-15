from langchain_groq import ChatGroq
from dotenv import load_dotenv
from schema import Roadmap


load_dotenv()

llm_model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

llm = llm_model.with_structured_output(Roadmap)


def generate_roadmap(
    skill: str,
    resume_text: str,
    job_description: str
):
    """Generate a personalized learning roadmap for a missing skill."""

    system_prompt = """
    You are a technical career tutor.

    Create a personalized learning roadmap for the candidate to learn
    the selected missing skill.

    Use ONLY the information provided in the resume and job description
    to understand the candidate's current background and the requirements
    of the target role.

    The roadmap should:

    - Start from the candidate's likely current level.
    - Avoid teaching concepts the candidate already clearly knows.
    - Focus on the selected skill.
    - Prioritize knowledge that is relevant to the target job.
    - Progress from fundamentals to practical application.
    - Include hands-on exercises or projects.
    - End with a practical project that can demonstrate the skill.
    - Be realistic and achievable.
    - Do not assume experience that is not present in the resume.
    """

    messages = [
        ("system", system_prompt),
        (
            "user",
            f"""
            Selected skill to learn:
            {skill}

            Candidate resume:
            {resume_text}

            Target job description:
            {job_description}
            """
        )
    ]

    response = llm.invoke(messages)

    return response