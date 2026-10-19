
import streamlit as st
import joblib
import re
from pathlib import Path
from bs4 import BeautifulSoup

# Load saved machine learning model and supporting files
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

model = joblib.load(MODEL_DIR / "jobshield_model.pkl")
tfidf = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
threshold = float(joblib.load(MODEL_DIR / "threshold.pkl"))

# Clean the job posting text
def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return text.lower().strip()

# Page configuration
st.set_page_config(
    page_title="JobShield AI",
    page_icon="🛡️",
    layout="wide"
)

# Application heading
st.title("🛡️ JobShield AI")
st.subheader("AI-Powered Fake Job Detection")
st.write(
    "Screen job postings for potential fraud using machine learning."
)

st.info(
    "Enter the available information about the job posting. "
    "You can leave optional fields blank."
)

# Job information form
with st.form("job_analysis_form"):

    st.markdown("### 1. Job Information")

    title = st.text_input("Job Title")

    description = st.text_area(
        "Job Description *",
        height=200,
        placeholder="Paste the complete job description here..."
    )

    st.markdown("### 2. Company and Requirements")

    company_profile = st.text_area(
        "Company Profile",
        placeholder="Company background, website information, etc."
    )

    requirements = st.text_area(
        "Candidate Requirements",
        placeholder="Skills, qualifications, and experience required..."
    )

    benefits = st.text_area(
        "Salary, Benefits or Perks",
        placeholder="Salary, incentives, benefits, or other details..."
    )

    col1, col2 = st.columns(2)

    with col1:
        employment_type = st.text_input("Employment Type")
        experience = st.text_input("Required Experience")

    with col2:
        education = st.text_input("Required Education")
        industry = st.text_input("Industry")

    function = st.text_input("Job Function")

    submitted = st.form_submit_button(
        "Analyze Job Posting",
        type="primary",
        use_container_width=True
    )

# Analyze the submitted job
if submitted:

    if not description.strip():
        st.warning("Please enter the job description.")

    else:
        fields = [
            title,
            company_profile,
            description,
            requirements,
            benefits,
            employment_type,
            experience,
            education,
            industry,
            function
        ]

        combined_text = " ".join(
            value.strip()
            for value in fields
            if value.strip()
        )

        cleaned_text = clean_text(combined_text)

        job_vector = tfidf.transform([cleaned_text])

        fraud_probability = float(
            model.predict_proba(job_vector)[0][1]
        )

        prediction = int(fraud_probability >= threshold)

        st.divider()
        st.markdown("## Analysis Results")

        result_col, score_col = st.columns(2)

        with result_col:

            if prediction == 1:
                st.error("⚠️ Potentially Fraudulent")
                st.write(
                    "The model's estimated fraud score meets or "
                    "exceeds its decision threshold."
                )

            else:
                st.success("✅ Likely Legitimate")
                st.write(
                    "The estimated fraud score is below the decision "
                    "threshold. This does not guarantee the job is safe."
                )

        with score_col:

            st.metric(
                "Estimated Fraud Risk Score",
                f"{fraud_probability * 100:.2f}%"
            )

            st.progress(
                max(0.0, min(1.0, fraud_probability))
            )

            st.caption(
                f"Decision threshold: {threshold:.2f}"
            )
                # Detect common warning signs
        warning_patterns = {
            "Upfront registration or recruitment fee":
                r"\b(registration fee|upfront fee|processing fee|pay.*fee)\b",

            "Unrealistic or guaranteed income claims":
                r"\b(guaranteed income|earn \$?[\d,]+\s*(per week|weekly|per day|daily))\b",

            "Requests for sensitive banking information":
                r"\b(bank details|bank information|otp|bank password)\b",

            "Hiring without an interview":
                r"\b(without an interview|no interview required|hired immediately)\b"
        }

        detected_warnings = [
            warning
            for warning, pattern in warning_patterns.items()
            if re.search(pattern, combined_text, re.IGNORECASE)
        ]

        st.markdown("### 🔎 Potential Warning Signs")

        if detected_warnings:
            for warning in detected_warnings:
                st.warning("⚠️ " + warning)
        else:
            st.info(
                "No warning phrases from the basic checklist were detected. "
                "This does not guarantee that the job is legitimate."
            )

        st.markdown("### Before You Apply")

        st.markdown("""
        - Verify the employer through its official website.
        - Be cautious of upfront registration or recruitment fees.
        - Never share banking passwords or OTPs with recruiters.
        - Confirm that the recruiter and vacancy are genuine.
        """)

        st.caption(
            "This score is an uncalibrated model estimate, not a "
            "verified probability or proof of fraud. Independently "
            "verify the job before making a decision."
        )
