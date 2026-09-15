from pydantic import BaseModel, Field

class SkillMatch(BaseModel):
    jd_skill: str
    matched_components: list[str]
    missing_components: list[str]

class AnalysisResult(BaseModel):

    jd_skills: list[str] = Field(
        description="Skills explicitly required or strongly implied by the job description."
    )

    resume_skills: list[str] = Field(
        description="Skills clearly demonstrated in the candidate's resume."
    )

    matched_skills: list[str] = Field(
        description="Skills that are semantically matched between the job description and resume."
    )

    missing_skills: list[str] = Field(
        description="Important skills required by the job description that are not demonstrated in the resume."
    )

    skill_matches: list[SkillMatch]
    
    experience_gaps: list[str] = Field(
        description="Experience or qualification gaps relevant to the job description."
    )

    resume_improvements: list[str] = Field(
        description="Specific, actionable improvements to better align the resume with the job."
    )


class Phase(BaseModel):
    title: str
    description: str
    topics: list[str]
    exercise: list[str]


class Roadmap(BaseModel):
    skill: str
    current_level: str
    goal: str
    phases: list[Phase]
