# ============================================================
# RESUME SCREENING AI
# Machine Learning Capstone Project
# ============================================================

import re
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from pypdf import PdfReader
from scipy.special import expit
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Resume Screening AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */
    .main {
        padding-top: 1rem;
    }

    /* Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }

    /* Result cards */
    .result-card {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-bottom: 1rem;
    }

    /* Fit result */
    .fit-result {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #28a745;
        margin-bottom: 15px;
    }

    /* Not fit result */
    .not-fit-result {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dc3545;
        margin-bottom: 15px;
    }

    /* Skill badges */
    .skill-badge {
        display: inline-block;
        padding: 5px 10px;
        margin: 3px;
        border-radius: 15px;
        background-color: #e8f4fd;
        font-size: 0.85rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #777;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid #ddd;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. LOAD MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("Models/resume_screening_model.pkl")
        vectorizer = joblib.load("Models/tfidf_vectorizer.pkl")
        skills = joblib.load("Models/skills.pkl")
        return model, vectorizer, skills
    except FileNotFoundError as e:
        st.error(
            f"Required model file was not found: {e}")
        
        st.info(
            """
            Make sure the following files are in
            the Models folder relative to app.py:

            - Models/resume_screening_model.pkl
            - Models/tfidf_vectorizer.pkl
            - Models/skills.pkl
            """
        )
        st.stop()


    except Exception as e:

        st.error(
            f"Error loading model artifacts: {e}"
        )

        st.stop()


model, tfidf, skills = load_artifacts()


# ============================================================
# 4. TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    """
    Basic text preprocessing.

    Converts text to lowercase, removes unnecessary
    characters and normalizes whitespace.
    """

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 5. STOPWORD REMOVAL
# ============================================================

def remove_stopwords(text):

    """
    Remove English stopwords.
    """

    words = text.split()

    filtered_words = [
        word
        for word in words
        if word not in ENGLISH_STOP_WORDS
    ]

    return " ".join(
        filtered_words
    )


# ============================================================
# 6. PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file):

    """
    Extract text from an uploaded PDF resume.
    """

    try:

        reader = PdfReader(
            uploaded_file
        )

        pages_text = []

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                pages_text.append(
                    page_text
                )

        return "\n".join(
            pages_text
        )

    except Exception as e:

        st.error(
            f"Could not read {uploaded_file.name}: {e}"
        )

        return ""


# ============================================================
# 7. SKILL EXTRACTION
# ============================================================

def extract_skills(
    text,
    skill_list
):

    """
    Extract known skills from a document.
    """

    text = str(
        text
    ).lower()

    matched_skills = []

    for skill in skill_list:

        skill_lower = skill.lower()

        # Use word boundaries for short skills
        # and normal substring matching for phrases.
        if " " in skill_lower:

            if skill_lower in text:

                matched_skills.append(
                    skill
                )

        else:

            pattern = (
                r"\b"
                + re.escape(skill_lower)
                + r"\b"
            )

            if re.search(
                pattern,
                text
            ):

                matched_skills.append(
                    skill
                )

    return sorted(
        set(matched_skills)
    )


# ============================================================
# 8. FIT SCORE
# ============================================================

def calculate_fit_score(
    resume_vector
):

    """
    Calculate a 0–100 Fit Score.

    Uses predict_proba when available.
    Otherwise uses decision_function.
    """

    try:

        # Models such as Logistic Regression
        if hasattr(
            model,
            "predict_proba"
        ):

            probability = (
                model.predict_proba(
                    resume_vector
                )[0, 1]
            )

            score = probability * 100

        # Models such as Linear SVM
        elif hasattr(
            model,
            "decision_function"
        ):

            decision_score = (
                model.decision_function(
                    resume_vector
                )[0]
            )

            score = (
                expit(
                    decision_score
                ) * 100
            )

        # Fallback
        else:

            prediction = model.predict(
                resume_vector
            )[0]

            score = (
                100
                if prediction == 1
                else 0
            )

        return round(
            float(score),
            2
        )

    except Exception:

        prediction = model.predict(
            resume_vector
        )[0]

        return (
            100.0
            if prediction == 1
            else 0.0
        )


# ============================================================
# 9. SCREEN ONE RESUME
# ============================================================

def screen_resume(
    resume_text,
    job_description
):

    """
    Screen a single resume against a job description.
    """

    # --------------------------------------------------------
    # Preprocess resume
    # --------------------------------------------------------

    processed_resume = preprocess_text(
        resume_text
    )

    processed_resume = remove_stopwords(
        processed_resume
    )

    # --------------------------------------------------------
    # TF-IDF transformation
    # --------------------------------------------------------

    resume_vector = tfidf.transform(
        [processed_resume]
    )

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    prediction = model.predict(
        resume_vector
    )[0]

    classification = (
        "Fit"
        if prediction == 1
        else "Not Fit"
    )

    # --------------------------------------------------------
    # Fit Score
    # --------------------------------------------------------

    fit_score = calculate_fit_score(
        resume_vector
    )

    # --------------------------------------------------------
    # Extract resume skills
    # --------------------------------------------------------

    resume_skills = extract_skills(
        resume_text,
        skills
    )

    # --------------------------------------------------------
    # Extract required job skills
    # --------------------------------------------------------

    required_skills = extract_skills(
        job_description,
        skills
    )

    # --------------------------------------------------------
    # Matched skills
    # --------------------------------------------------------

    matched_skills = sorted(
        set(resume_skills)
        &
        set(required_skills)
    )

    # --------------------------------------------------------
    # Missing skills
    # --------------------------------------------------------

    missing_skills = sorted(
        set(required_skills)
        -
        set(resume_skills)
    )

    # --------------------------------------------------------
    # Skill match percentage
    # --------------------------------------------------------

    if len(required_skills) > 0:

        skill_match = (
            len(matched_skills)
            /
            len(required_skills)
        ) * 100

    else:

        skill_match = 0

    return {

        "Classification": classification,

        "Fit Score": round(
            fit_score,
            2
        ),

        "Skill Match": round(
            skill_match,
            2
        ),

        "Resume Skills": resume_skills,

        "Required Skills": required_skills,

        "Matched Skills": matched_skills,

        "Missing Skills": missing_skills
    }


# ============================================================
# 10. RECOMMENDATION LEVEL
# ============================================================

def recommendation_level(
    score
):

    """
    Convert Fit Score into an easy-to-understand
    recommendation category.
    """

    if score >= 75:

        return "Strong Match"

    elif score >= 50:

        return "Moderate Match"

    else:

        return "Low Match"


# ============================================================
# 11. DISPLAY SKILLS
# ============================================================

def display_skill_badges(
    skill_list
):

    """
    Display skills as visual badges.
    """

    if not skill_list:

        st.write(
            "None detected."
        )

        return

    badges = ""

    for skill in skill_list:

        badges += (
            f'<span class="skill-badge">'
            f'{skill}'
            f'</span>'
        )

    st.markdown(
        badges,
        unsafe_allow_html=True
    )


# ============================================================
# 12. APPLICATION HEADER
# ============================================================

st.markdown(
    '<div class="main-header">'
    '📄 Resume Screening AI'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-header">
    Intelligent resume screening and candidate ranking
    powered by Machine Learning.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 13. SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ About the System"
    )

    st.markdown(
        """
        This application uses a trained
        Machine Learning model to assist
        recruiters with initial resume screening.

        ### Features

        ✅ Resume classification

        ✅ Fit Score

        ✅ Skill matching

        ✅ Missing skill detection

        ✅ Candidate ranking

        ✅ Multiple resume screening

        ✅ CSV results download

        ### Technology

        - Python
        - Scikit-learn
        - TF-IDF
        - Machine Learning
        - Streamlit
        - PDF text extraction
        """
    )

    st.divider()

    st.warning(
        """
        **Important**

        This system is an AI-assisted
        screening tool.

        It should not be used as the
        sole basis for employment
        decisions.
        """
    )


# ============================================================
# 14. JOB DESCRIPTION SECTION
# ============================================================

st.header(
    "📋 Job Description"
)

job_description = st.text_area(
    "Paste the job description below",
    height=250,
    placeholder=(
        "Example:\n\n"
        "We are looking for a Machine Learning "
        "Engineer with experience in Python, "
        "SQL, Scikit-learn, Pandas and Machine "
        "Learning..."
    )
)


# ============================================================
# 15. JOB DESCRIPTION SKILLS PREVIEW
# ============================================================

if job_description.strip():

    required_skills_preview = extract_skills(
        job_description,
        skills
    )

    if required_skills_preview:

        st.markdown(
            "### 🔎 Detected Job Skills"
        )

        display_skill_badges(
            required_skills_preview
        )

    else:

        st.info(
            "No skills from the configured "
            "skill vocabulary were detected."
        )


# ============================================================
# 16. RESUME UPLOAD SECTION
# ============================================================

st.header(
    "📄 Upload Candidate Resumes"
)

uploaded_files = st.file_uploader(
    "Upload one or more PDF resumes",
    type=["pdf"],
    accept_multiple_files=True,
    help=(
        "You can upload multiple PDF resumes "
        "for batch screening."
    )
)


# ============================================================
# 17. UPLOAD PREVIEW
# ============================================================

if uploaded_files:

    st.success(
        f"{len(uploaded_files)} resume(s) uploaded."
    )

    uploaded_names = [
        file.name
        for file in uploaded_files
    ]

    with st.expander(
        "View uploaded resumes"
    ):

        for name in uploaded_names:

            st.write(
                f"📄 {name}"
            )


# ============================================================
# 18. SCREEN BUTTON
# ============================================================

st.divider()

screen_button = st.button(
    "🔍 Screen Resumes",
    type="primary",
    use_container_width=True
)


# ============================================================
# 19. SCREENING PROCESS
# ============================================================

if screen_button:

    # --------------------------------------------------------
    # Validate Job Description
    # --------------------------------------------------------

    if not job_description.strip():

        st.error(
            "Please enter a job description before screening resumes."
        )

        st.stop()

    # --------------------------------------------------------
    # Validate Resumes
    # --------------------------------------------------------

    if not uploaded_files:

        st.error(
            "Please upload at least one PDF resume."
        )

        st.stop()

    # --------------------------------------------------------
    # Screening starts
    # --------------------------------------------------------

    st.header(
        "🔍 Screening Results"
    )

    results = []

    progress_bar = st.progress(
        0
    )

    status_text = st.empty()

    total_files = len(
        uploaded_files
    )

    # --------------------------------------------------------
    # Process every resume
    # --------------------------------------------------------

    for index, uploaded_file in enumerate(
        uploaded_files
    ):

        status_text.write(
            f"Processing {uploaded_file.name}..."
        )

        # Extract PDF text
        resume_text = extract_pdf_text(
            uploaded_file
        )

        # Check extraction
        if not resume_text.strip():

            st.warning(
                f"⚠️ No readable text was found "
                f"in {uploaded_file.name}. "
                f"It may be an image-based/scanned PDF."
            )

            progress_bar.progress(
                (index + 1)
                / total_files
            )

            continue

        try:

            # Screen resume
            screening = screen_resume(
                resume_text,
                job_description
            )

            # Add candidate information
            screening["Candidate"] = (
                uploaded_file.name
            )

            screening["Resume Text"] = (
                resume_text
            )

            results.append(
                screening
            )

        except Exception as e:

            st.error(
                f"Error processing "
                f"{uploaded_file.name}: {e}"
            )

        # Update progress
        progress_bar.progress(
            (index + 1)
            / total_files
        )

    status_text.empty()

    # --------------------------------------------------------
    # Check results
    # --------------------------------------------------------

    if not results:

        st.error(
            "No resumes could be successfully processed."
        )

        st.stop()

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Add Recommendation
    # --------------------------------------------------------

    results_df[
        "Recommendation"
    ] = results_df[
        "Fit Score"
    ].apply(
        recommendation_level
    )

    # --------------------------------------------------------
    # Rank candidates
    # --------------------------------------------------------

    results_df = (
        results_df
        .sort_values(
            by="Fit Score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Add rank
    # --------------------------------------------------------

    results_df.insert(
        0,
        "Rank",
        range(
            1,
            len(results_df) + 1
        )
    )


    # ========================================================
    # 20. SUMMARY METRICS
    # ========================================================

    st.subheader(
        "📊 Screening Summary"
    )

    total_candidates = len(
        results_df
    )

    fit_candidates = (
        results_df[
            results_df[
                "Classification"
            ] == "Fit"
        ].shape[0]
    )

    not_fit_candidates = (
        results_df[
            results_df[
                "Classification"
            ] == "Not Fit"
        ].shape[0]
    )

    highest_score = (
        results_df[
            "Fit Score"
        ].max()
    )

    average_score = (
        results_df[
            "Fit Score"
        ].mean()
    )

    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )

    col1.metric(
        "Total Resumes",
        total_candidates
    )

    col2.metric(
        "Fit Candidates",
        fit_candidates
    )

    col3.metric(
        "Not Fit",
        not_fit_candidates
    )

    col4.metric(
        "Highest Score",
        f"{highest_score:.1f}%"
    )

    col5.metric(
        "Average Score",
        f"{average_score:.1f}%"
    )


    # ========================================================
    # 21. CANDIDATE RANKING
    # ========================================================

    st.subheader(
        "🏆 Candidate Ranking"
    )

    ranking_columns = [
        "Rank",
        "Candidate",
        "Classification",
        "Fit Score",
        "Skill Match",
        "Recommendation"
    ]

    ranking_display = results_df[
        ranking_columns
    ].copy()

    ranking_display[
        "Fit Score"
    ] = ranking_display[
        "Fit Score"
    ].map(
        lambda x: f"{x:.2f}%"
    )

    ranking_display[
        "Skill Match"
    ] = ranking_display[
        "Skill Match"
    ].map(
        lambda x: f"{x:.2f}%"
    )

    st.dataframe(
        ranking_display,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # 22. TOP CANDIDATE
    # ========================================================

    top_candidate = (
        results_df.iloc[0]
    )

    st.subheader(
        "🥇 Top Candidate"
    )

    top_col1, top_col2, top_col3 = (
        st.columns(3)
    )

    top_col1.metric(
        "Candidate",
        top_candidate[
            "Candidate"
        ]
    )

    top_col2.metric(
        "Fit Score",
        f"{top_candidate['Fit Score']:.2f}%"
    )

    top_col3.metric(
        "Skill Match",
        f"{top_candidate['Skill Match']:.2f}%"
    )


    # ========================================================
    # 23. CANDIDATE DETAILS
    # ========================================================

    st.subheader(
        "👤 Candidate Details"
    )

    for _, row in results_df.iterrows():

        candidate_name = row[
            "Candidate"
        ]

        rank = row[
            "Rank"
        ]

        score = row[
            "Fit Score"
        ]

        classification = row[
            "Classification"
        ]

        recommendation = row[
            "Recommendation"
        ]

        # ----------------------------------------------------
        # Candidate expander
        # ----------------------------------------------------

        with st.expander(
            f"#{rank} — {candidate_name} "
            f"| {classification} "
            f"| {score:.1f}%"
        ):

            # -----------------------------------------------
            # Candidate metrics
            # -----------------------------------------------

            col1, col2, col3, col4 = (
                st.columns(4)
            )

            col1.metric(
                "Classification",
                classification
            )

            col2.metric(
                "Fit Score",
                f"{score:.2f}%"
            )

            col3.metric(
                "Skill Match",
                f"{row['Skill Match']:.2f}%"
            )

            col4.metric(
                "Recommendation",
                recommendation
            )

            st.divider()

            # -----------------------------------------------
            # Matched Skills
            # -----------------------------------------------

            st.markdown(
                "### ✅ Matched Skills"
            )

            display_skill_badges(
                row[
                    "Matched Skills"
                ]
            )

            # -----------------------------------------------
            # Missing Skills
            # -----------------------------------------------

            st.markdown(
                "### ⚠️ Missing Skills"
            )

            if row[
                "Missing Skills"
            ]:

                display_skill_badges(
                    row[
                        "Missing Skills"
                    ]
                )

            else:

                st.success(
                    "No missing required skills detected."
                )

            # -----------------------------------------------
            # All Resume Skills
            # -----------------------------------------------

            st.markdown(
                "### 🧠 Detected Resume Skills"
            )

            display_skill_badges(
                row[
                    "Resume Skills"
                ]
            )


    # ========================================================
    # 24. SCORE DISTRIBUTION
    # ========================================================

    st.subheader(
        "📈 Fit Score Distribution"
    )

    chart_data = (
        results_df[
            [
                "Candidate",
                "Fit Score"
            ]
        ]
        .set_index(
            "Candidate"
        )
    )

    st.bar_chart(
        chart_data
    )


    # ========================================================
    # 25. DOWNLOAD RESULTS
    # ========================================================

    st.subheader(
        "📥 Export Results"
    )

    # -----------------------------------------------
    # Create export dataframe
    # -----------------------------------------------

    export_df = results_df[
        [
            "Rank",
            "Candidate",
            "Classification",
            "Fit Score",
            "Skill Match",
            "Recommendation",
            "Matched Skills",
            "Missing Skills"
        ]
    ].copy()

    # Convert lists to readable text
    export_df[
        "Matched Skills"
    ] = export_df[
        "Matched Skills"
    ].apply(
        lambda x: ", ".join(x)
    )

    export_df[
        "Missing Skills"
    ] = export_df[
        "Missing Skills"
    ].apply(
        lambda x: ", ".join(x)
    )

    csv_data = export_df.to_csv(
        index=False
    )

    st.download_button(
        label="⬇️ Download Screening Results",
        data=csv_data,
        file_name=(
            "resume_screening_results.csv"
        ),
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# 26. APPLICATION FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    <strong>
    Resume Screening AI
    </strong>

    <br>

    Machine Learning Capstone Project

    <br>

    Built with Python, Scikit-learn,
    TF-IDF and Streamlit.

    <br><br>

    <em>
    This system is designed to assist recruiters
    during initial screening and should not replace
    human judgment in employment decisions.
    </em>

    </div>
    """,
    unsafe_allow_html=True
)