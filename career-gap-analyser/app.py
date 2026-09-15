import streamlit as st
import requests

from extractor import extract_resume_text
from tutor import generate_roadmap


API_URL = "http://127.0.0.1:8000/analyze"


st.title("Career Gap Analyzer")

st.write(
    "Upload your resume and paste a job description to see how well "
    "your profile matches the role."
)

# Resume upload
resume_file = st.file_uploader(
    "Upload your resume",
    type=["pdf", "docx"]
)


# Job description
job_description = st.text_area(
    "Paste the job description",
    height=300
)


# Analyze
if st.button("Analyze"):

    if resume_file is None:
        st.warning("Please upload your resume.")

    elif not job_description.strip():
        st.warning("Please paste the job description.")

    else:

        with st.spinner("Analyzing your resume..."):

            files = {
                "resume": (
                    resume_file.name,
                    resume_file.getvalue(),
                    resume_file.type
                )
            }

            data = {
                "job_description": job_description
            }

            try:

                response = requests.post(
                    API_URL,
                    files=files,
                    data=data
                )

                if response.status_code == 200:

                    response_data = response.json()

                    # Store API response
                    st.session_state.result = response_data["analysis"]
                    st.session_state.match_score = response_data["match_score"]

                    # Keep resume text for Tutor
                    with open(resume_file.name, "wb") as f:
                        f.write(resume_file.getbuffer())

                    resume_text = extract_resume_text(resume_file.name)

                    st.session_state.resume_text = resume_text
                    st.session_state.job_description = job_description

                else:

                    st.error(
                        f"API Error ({response.status_code}): "
                        f"{response.text}"
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to FastAPI. "
                    "Make sure the FastAPI server is running."
                )


# Display results
if "result" in st.session_state:

    result = st.session_state.result
    match_score = st.session_state.match_score


    # Match Score
    st.subheader("Match Score")

    st.metric(
        "Skill Match",
        f"{match_score}/100"
    )


    # Job Description Skills
    st.subheader("Job Description Skills")

    for skill in result["jd_skills"]:
        st.write(f"✓ {skill}")


    # Matched Skills
    st.subheader("Matched Skills")

    for skill in result["matched_skills"]:
        st.write(f"✓ {skill}")


    # Resume Skills
    st.subheader("Resume Skills")

    for skill in result["resume_skills"]:
        st.write(f"✓ {skill}")


    # Missing / Weak Skills
    st.subheader("Missing / Weak Skills")

    for skill in result["missing_skills"]:
        st.write(f"• {skill}")


    # Skill Status
    st.subheader("Skill Status")

    for match in result["skill_matches"]:

        matched_components = match["matched_components"]
        missing_components = match["missing_components"]

        if not matched_components:
            status = "MISSING"

        elif not missing_components:
            status = "MET"

        else:
            status = "PARTIAL"


        if status == "MET":

            st.write(
                f" **{match['jd_skill']}** — MET"
            )

        elif status == "PARTIAL":

            st.write(
                f" **{match['jd_skill']}** — PARTIAL"
            )

        else:

            st.write(
                f" **{match['jd_skill']}** — MISSING"
            )


    # Resume Improvements
    st.subheader("Resume Improvements")

    for improvement in result["resume_improvements"]:
        st.write(f"• {improvement}")


    # Experience Gaps
    st.subheader("Experience Gaps")

    for gap in result["experience_gaps"]:
        st.write(f"• {gap}")


    # Tutor
    if result["missing_skills"]:

        st.subheader("Learn a Missing Skill")

        selected_skill = st.selectbox(
            "Select a skill to generate a learning roadmap",
            result["missing_skills"]
        )


        if st.button("Generate Roadmap"):

            with st.spinner("Generating learning roadmap..."):

                roadmap = generate_roadmap(
                    selected_skill,
                    st.session_state.resume_text,
                    st.session_state.job_description
                )


            st.subheader(
                f"Learning Roadmap for {selected_skill}"
            )

            st.markdown(
                f"**Current Level:** {roadmap.current_level}"
            )


            for phase in roadmap.phases:

                with st.expander(f"📍 {phase.title}"):

                    st.markdown(
                        phase.description
                    )

                    st.markdown("**📚 Learn**")

                    for topic in phase.topics:
                        st.write(f"• {topic}")

                    st.markdown("**🛠 Practice**")

                    for exercise in phase.exercise:
                        st.write(f"• {exercise}")