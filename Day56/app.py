
import streamlit as st

from utils.resume_analyzer import (
    analyze_resume,
    extract_pdf_text
)


# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="AI Resume Reviewer",
    page_icon="📄",
    layout="wide"
)


# =====================================
# HEADER
# =====================================

st.title("📄 AI Resume Reviewer")

st.write(
    "Analyze your resume against a job description "
    "using a locally running Large Language Model."
)

st.info("Powered by Ollama and Llama 3.2 (1B)")


# =====================================
# USER INPUT
# =====================================

uploaded_file = st.file_uploader(
    "Upload your resume (PDF)",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste the job description",
    height=200,
    placeholder="Enter the job requirements here..."
)


# =====================================
# ANALYZE RESUME
# =====================================

if st.button("Analyze Resume", type="primary"):

    if uploaded_file is None:

        st.warning("Please upload your resume.")

    elif not job_description.strip():

        st.warning("Please enter a job description.")

    else:

        try:

            with st.spinner("Analyzing your resume..."):

                # Extract resume text
                resume_text = extract_pdf_text(uploaded_file)

                if not resume_text.strip():

                    st.error(
                        "No readable text found in this PDF. "
                        "Please upload a text-based PDF."
                    )

                    st.stop()

                # Call LLM backend
                result = analyze_resume(
                    resume_text,
                    job_description
                )

            # =====================================
            # DISPLAY RESULTS
            # =====================================

            st.success("Resume analysis completed!")

            st.divider()

            # Match score
            st.subheader("📊 Resume Match Score")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Match Score",
                    f"{result.match_score}/100"
                )

            with col2:

                st.metric(
                    "Number of Strengths",
                    len(result.strengths)
                )

            st.progress(result.match_score / 100)

            st.divider()

            # Strengths
            st.subheader("✅ Strengths")

            if result.strengths:

                for strength in result.strengths:
                    st.write("✅", strength)

            else:

                st.warning(
                    "No strengths were identified. "
                    "Check the extracted resume text and raw model response."
                )

            # Missing skills
            st.subheader("⚠️ Missing Skills")

            if result.missing_skills:

                for skill in result.missing_skills:
                    st.write("•", skill)

            else:

                st.write("No missing skills identified.")

            # Improvements
            st.subheader("💡 Suggested Improvements")

            if result.improvements:

                for suggestion in result.improvements:
                    st.write("•", suggestion)

            else:

                st.write("No improvement suggestions returned.")

            # Summary
            st.subheader("📝 Overall Summary")

            st.write(result.summary)

            # JSON response
            with st.expander("View Raw JSON Response"):

                st.json(result.model_dump())

            # Extracted text for debugging
            with st.expander("View Extracted Resume Text"):

                st.text(resume_text)

        except Exception as error:

            st.error(f"Analysis failed: {error}")

            st.exception(error)


# =====================================
# FOOTER
# =====================================

st.divider()

st.caption(
    "AI Resume Reviewer | Python | Streamlit | "
    "Ollama | Pydantic | Llama 3.2"
)
