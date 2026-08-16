# Resume Screening Classifier

## Recruiters Sift Many CVs for Graduate Roles

A Machine Learning–based resume screening system that automatically classifies candidates as **Fit** or **Not Fit** for a specified job role using resume text and job requirements.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Problem Statement](#problem-statement)
* [Aim](#aim)
* [Objectives](#objectives)
* [Expected Outcome](#expected-outcome)
* [Machine Learning Workflow](#machine-learning-workflow)
* [Project Features](#project-features)
* [Technology Stack](#technology-stack)
* [Dataset](#dataset)
* [Dataset Structure](#dataset-structure)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Environment Setup](#environment-setup)
* [Data Preprocessing](#data-preprocessing)
* [Exploratory Data Analysis](#exploratory-data-analysis)
* [Feature Engineering](#feature-engineering)
* [Model Training](#model-training)
* [Model Evaluation](#model-evaluation)
* [Resume Screening](#resume-screening)
* [Fit Score](#fit-score)
* [Matched Skills](#matched-skills)
* [Model Saving](#model-saving)
* [Deployment](#deployment)
* [Example Prediction](#example-prediction)
* [Ethical Considerations](#ethical-considerations)
* [Limitations](#limitations)
* [Future Improvements](#future-improvements)
* [Conclusion](#conclusion)
* [Author](#author)

---

## Project Overview

The recruitment process has become increasingly competitive as organizations receive hundreds or thousands of applications for a single job opening. Manually reviewing every resume is time-consuming, expensive, and prone to human error, inconsistency, and oversight.

Recruiters may spend only a few seconds initially reviewing each resume. As a result, qualified candidates may be overlooked while unsuitable candidates may progress through the recruitment process.

Machine Learning provides an opportunity to automate part of the initial resume-screening process by analyzing resume text and comparing candidate information with the requirements of a target role.

This project develops a supervised Machine Learning system that analyzes resume content and classifies candidates into two categories:

* **Fit (1)** — the resume appears to satisfy the requirements of the target role.
* **Not Fit (0)** — the resume appears not to satisfy the requirements of the target role.

The system also provides a **fit score** and identifies **matched skills** to make the prediction more interpretable.

> **Important:** This system is intended as a decision-support and research project. It should not be used as the sole basis for employment decisions.

---

## Problem Statement

Organizations receive a large number of applications for graduate and entry-level positions. Manually screening these applications can be inefficient and inconsistent.

The primary challenge is to automatically analyze unstructured resume text and determine whether a candidate appears suitable for a particular job role.

The project addresses the following problem:

> **How can supervised Machine Learning and Natural Language Processing be used to automatically classify resumes as Fit or Not Fit for a specified job role?**

The system must process textual resume information, extract meaningful features, train classification models, evaluate their performance, and use the selected model to screen previously unseen resumes.

---

## Aim

To develop an intelligent resume screening system capable of automatically classifying resumes as **Fit** or **Not Fit** for a specific job role using supervised Machine Learning and Natural Language Processing techniques.

---

## Objectives

The objectives of this project are to:

1. Understand the structure and characteristics of the resume dataset.
2. Load, inspect, and validate resume and job-related data.
3. Clean and preprocess resume text.
4. Perform Exploratory Data Analysis (EDA).
5. Identify important skills, keywords, and textual patterns.
6. Convert resume text into numerical representations using TF-IDF Vectorization.
7. Engineer relevant text-based features where necessary.
8. Split the dataset into training and testing sets.
9. Train multiple Machine Learning classification algorithms.
10. Compare model performance using appropriate evaluation metrics.
11. Select the best-performing model.
12. Develop a resume screening prediction pipeline.
13. Generate a candidate fit score.
14. Identify skills from the job requirements that appear in each resume.
15. Save the trained model and preprocessing pipeline.
16. Prepare the system for future deployment using Streamlit or another web application framework.

---

## Expected Outcome

At the end of the project, a Machine Learning resume screening system will be developed that can classify new resumes into:

| Class | Meaning |
| ----- | ------- |
| `0`   | Not Fit |
| `1`   | Fit     |

For each candidate, the system can provide:

* Classification result
* Fit score
* Matched skills
* Missing or unmatched skills where applicable
* Model confidence/probability where supported

The trained model and text-processing pipeline will be saved for future deployment.

---

## Machine Learning Workflow

```text
Business Problem Understanding
            ↓
Dataset Loading
            ↓
Data Understanding
            ↓
Data Cleaning
            ↓
Text Preprocessing
            ↓
Exploratory Data Analysis
            ↓
Feature Extraction
            ↓
TF-IDF Vectorization
            ↓
Feature Engineering
            ↓
Train-Test Split
            ↓
Model Training
            ↓
Model Evaluation
            ↓
Model Comparison
            ↓
Best Model Selection
            ↓
Resume Screening
            ↓
Fit Score & Matched Skills
            ↓
Model Saving
            ↓
Deployment Preparation
```

---

## Project Features

### 1. Resume Text Processing

The system cleans and prepares resume text for Machine Learning by:

* Converting text to lowercase
* Removing unnecessary characters
* Handling whitespace
* Removing irrelevant text where appropriate
* Tokenizing text where required
* Removing stop words where appropriate
* Preparing text for vectorization

### 2. TF-IDF Feature Extraction

Term Frequency–Inverse Document Frequency (TF-IDF) is used to transform resume text into numerical features.

TF-IDF gives greater importance to terms that are relevant to particular documents while reducing the importance of terms that occur frequently across the entire dataset.

### 3. Machine Learning Classification

Multiple classification algorithms can be trained and compared, including:

* Logistic Regression
* Linear Support Vector Machine
* Naive Bayes
* Random Forest
* Other suitable classifiers

### 4. Model Evaluation

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC-AUC where applicable

For resume screening, **precision and recall should be considered alongside accuracy**, because incorrectly rejecting a suitable candidate can be more important than achieving a high overall accuracy.

### 5. Fit Score

The system produces a score indicating how strongly the resume matches the target classification criteria.

### 6. Matched Skills

The system compares relevant job skills with the candidate's resume and identifies skills that appear to match.

Example:

```text
Required Skills:
Python, SQL, Machine Learning, Git, Pandas

Matched Skills:
Python
SQL
Machine Learning
Pandas
```

### 7. Model Persistence

The trained model and TF-IDF vectorizer can be saved using `joblib` so that they can be reused without retraining.

---

## Technology Stack

### Programming Language

* Python 3.10+

### Data Analysis

* Pandas
* NumPy

### Natural Language Processing

* NLTK
* Scikit-learn

### Machine Learning

* Scikit-learn

### Visualization

* Matplotlib
* Seaborn

### Model Persistence

* Joblib

### Optional Resume Extraction

For PDF-based resumes:

* PyMuPDF

For Word documents:

* python-docx

### Deployment

* Streamlit

### Development Environment

The project can be developed using:

* Jupyter Notebook
* VS Code
* PyCharm
* Google Colab

---

## Dataset

The model requires a labeled resume dataset containing resume text and a target classification.

A typical dataset should contain:

```text
Resume Text
Job Role / Category
Label
```

Where the label represents whether the candidate is considered suitable for the target role.

Example:

| Resume                                         | Category     | Label |
| ---------------------------------------------- | ------------ | ----: |
| Python developer with SQL and ML experience... | Data Science |     1 |
| Graphic designer with Adobe experience...      | Data Science |     0 |
| Data analyst with Python and SQL...            | Data Science |     1 |

### Dataset Requirements

The dataset should ideally contain:

* A sufficiently large number of resumes
* Resume text
* Relevant job categories or roles
* A clearly defined target label
* Sufficient examples of both Fit and Not Fit candidates

The dataset should be checked for:

* Duplicate resumes
* Missing values
* Incorrect labels
* Class imbalance
* Personally identifiable information
* Data leakage

---

## Dataset Structure

A CSV dataset may follow this structure:

```csv
resume_text,label
"Python developer with experience in pandas, SQL and machine learning",1
"Graphic designer with experience in Photoshop and Illustrator",0
"Data analyst experienced in Python, SQL and Power BI",1
```

If a job description is included:

```csv
resume_text,job_description,label
"Python developer with SQL experience","Python developer with Python SQL and Git requirements",1
"Marketing specialist with SEO experience","Python developer with Python SQL and Git requirements",0
```

---

## Project Structure

A recommended project structure is:

```text
resume-screening-classifier/
│
├── data/
│   ├── raw/
│   │   └── resumes.csv
│   │
│   └── processed/
│       └── cleaned_resumes.csv
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   └── 06_model_evaluation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   ├── resume_classifier.joblib
│   └── tfidf_vectorizer.joblib
│
├── app/
│   └── app.py
│
├── reports/
│   ├── figures/
│   └── model_results.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone or download the project repository and navigate into the project directory.

Create a virtual environment:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Setup

Recommended Python version:

```text
Python 3.10+
```

Verify the Python installation:

```bash
python --version
```

Verify pip:

```bash
pip --version
```

---

## Data Preprocessing

Resume text is unstructured and can contain unnecessary characters, formatting, and inconsistent representations.

The preprocessing pipeline may include:

1. Missing-value handling
2. Duplicate removal
3. Lowercasing
4. Special-character removal
5. Whitespace normalization
6. Stop-word removal
7. Tokenization
8. Optional stemming or lemmatization

Example:

### Original

```text
JOHN DOE

Experienced Python Developer with 3+ years of experience in Python,
Pandas, SQL, Machine Learning and Git.
```

### Processed

```text
experienced python developer years experience python pandas sql machine learning git
```

Care should be taken not to remove meaningful technical terms during preprocessing.

---

## Exploratory Data Analysis

EDA is performed to understand the dataset before model training.

The analysis may include:

### Class Distribution

Determine how many resumes belong to each class.

```text
Fit
Not Fit
```

This is important for identifying class imbalance.

### Resume Length

Analyze:

* Number of words
* Number of characters
* Average resume length

### Common Terms

Identify frequently occurring:

* Skills
* Job titles
* Technologies
* Certifications
* Qualifications

### Category Distribution

If multiple job roles are represented, analyze the number of resumes per category.

---

## Feature Engineering

The main text representation used in this project is **TF-IDF**.

Additional features may include:

* Resume length
* Number of matched skills
* Number of required skills
* Skill-match ratio
* Years of experience where reliably extractable
* Education indicators
* Certification indicators

Feature engineering should be performed carefully to avoid introducing information that would not be available when screening a new candidate.

---

## TF-IDF Vectorization

TF-IDF converts textual information into numerical features that Machine Learning algorithms can process.

The vectorizer may be configured with parameters such as:

```python
TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2)
)
```

The use of unigrams and bigrams allows the system to capture both individual terms and phrases.

For example:

```text
python
machine learning
data science
sql database
```

---

## Train-Test Split

The dataset should be divided into training and testing sets.

A typical split is:

```text
80% Training
20% Testing
```

Stratification should be considered when the target classes are imbalanced.

Example:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The test set should remain unseen during model training.

---

## Machine Learning Models

Several classification algorithms can be compared.

### Logistic Regression

Logistic Regression provides a strong baseline for text classification and can produce class probabilities.

### Linear Support Vector Machine

Linear SVM is often effective for high-dimensional sparse text data.

### Multinomial Naive Bayes

Naive Bayes is a lightweight model that can perform well on text classification problems.

### Random Forest

Random Forest can be evaluated as an additional model, although tree-based methods may not always be the strongest choice for sparse TF-IDF text features.

---

## Model Evaluation

Models should be evaluated using several metrics.

### Accuracy

Measures the proportion of correctly classified resumes.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Measures how many candidates predicted as Fit were actually Fit.

High precision helps reduce unsuitable candidates being incorrectly classified as suitable.

### Recall

Measures how many truly Fit candidates were successfully identified.

High recall is important when the goal is to minimize the number of suitable candidates that are missed.

### F1-Score

The F1-score provides a balance between precision and recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### Confusion Matrix

The confusion matrix provides:

* True Positives
* True Negatives
* False Positives
* False Negatives

For this project:

```text
                    Predicted
                  Not Fit    Fit

Actual Not Fit      TN        FP
Actual Fit          FN        TP
```

---

## Model Comparison

A model comparison table can be generated:

| Model               | Accuracy | Precision | Recall | F1 |
| ------------------- | -------: | --------: | -----: | -: |
| Logistic Regression |        — |         — |      — |  — |
| Linear SVM          |        — |         — |      — |  — |
| Naive Bayes         |        — |         — |      — |  — |
| Random Forest       |        — |         — |      — |  — |

The final values should be populated from the actual experimental results rather than being manually entered.

The best model should be selected based on the project's evaluation criteria rather than accuracy alone.

---

## Resume Screening

After training, a new resume can be passed to the prediction pipeline.

Example:

```python
prediction = model.predict(vectorized_resume)
```

The system returns:

```text
Classification: Fit
```

or:

```text
Classification: Not Fit
```

---

## Fit Score

Where the selected classifier supports probabilities, the system can use the predicted probability of the Fit class as a model-based score.

Example:

```python
probability = model.predict_proba(vectorized_resume)[0][1]
fit_score = probability * 100
```

Example output:

```text
Fit Score: 87.4%
Classification: Fit
```

### Important

A fit score should be interpreted as a **model score**, not as an objective measure of a candidate's quality.

It should not be presented as a guarantee that a candidate is qualified.

---

## Matched Skills

The system can maintain a list of skills associated with the target role.

Example:

```python
required_skills = [
    "python",
    "sql",
    "machine learning",
    "pandas",
    "numpy",
    "git"
]
```

The candidate resume is then checked for these skills.

Example output:

```text
Matched Skills:
- Python
- SQL
- Machine Learning
- Pandas

Missing Skills:
- Git
- NumPy
```

A skill matching system should account for variations and synonyms where possible.

For example:

```text
Machine Learning
machine-learning
ML
```

may refer to the same underlying skill.

---

## Batch Resume Screening

The system can be extended to process multiple resumes at once.

Example workflow:

```text
resumes/
│
├── candidate_001.pdf
├── candidate_002.pdf
├── candidate_003.pdf
├── candidate_004.pdf
└── candidate_005.pdf
```

The application can:

1. Read each resume.
2. Extract the text.
3. Clean the text.
4. Transform the text using the trained TF-IDF vectorizer.
5. Generate a prediction.
6. Calculate the fit score.
7. Identify matched skills.
8. Store the results.
9. Rank candidates by score.

Example output:

| Candidate     | Classification | Fit Score | Matched Skills  |
| ------------- | -------------- | --------: | --------------- |
| Candidate 001 | Fit            |     91.2% | Python, SQL, ML |
| Candidate 002 | Fit            |     84.6% | Python, Pandas  |
| Candidate 003 | Not Fit        |     32.1% | SQL             |
| Candidate 004 | Fit            |     78.4% | Python, Git     |

---

## PDF Resume Processing

For a deployment version that accepts PDF resumes, PyMuPDF can be used to extract text.

Example:

```python
import fitz

def extract_text_from_pdf(file_path):
    document = fitz.open(file_path)
    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text
```

Scanned/image-only PDFs may require OCR because ordinary PDF text extraction may not be able to read text embedded in images.

---

## Model Saving

The trained model should be saved so that it can be reused without retraining.

Example:

```python
import joblib

joblib.dump(model, "models/resume_classifier.joblib")
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
```

The files can later be loaded:

```python
model = joblib.load("models/resume_classifier.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
```

For production-style deployment, saving the entire preprocessing-and-model pipeline as a single artifact can reduce the risk of applying inconsistent preprocessing during inference.

---

## Streamlit Deployment

The trained model can be integrated into a Streamlit application.

A possible interface may contain:

```text
------------------------------------------
       RESUME SCREENING CLASSIFIER
------------------------------------------

Upload Resume:
[ Choose PDF ]

Job Role:
[ Data Scientist ]

Job Requirements:
[ Python, SQL, Machine Learning, ... ]

              [ SCREEN RESUME ]

------------------------------------------

Classification: FIT

Fit Score: 87.4%

Matched Skills:
✓ Python
✓ SQL
✓ Machine Learning
✓ Pandas

Missing Skills:
• Git
------------------------------------------
```

Run the application using:

```bash
streamlit run app/app.py
```

---

## Example Prediction

Example input:

```text
Candidate Resume:

Data Analyst with 2 years of experience in Python,
SQL, Pandas, NumPy and Power BI. Experienced in
data cleaning, visualization and machine learning.
```

Target role:

```text
Junior Data Scientist

Required Skills:
Python
SQL
Pandas
NumPy
Machine Learning
Git
```

Possible system output:

```text
Classification: Fit

Fit Score: 88.6%

Matched Skills:
Python
SQL
Pandas
NumPy
Machine Learning

Missing Skills:
Git
```

The exact score will depend on the trained model and dataset.

---

## Reproducibility

To make experiments reproducible:

* Use a fixed `random_state`.
* Record the dataset version.
* Record preprocessing parameters.
* Record model parameters.
* Save the trained vectorizer.
* Save the final model.
* Record evaluation metrics.
* Keep training and testing data separate.
* Avoid fitting preprocessing steps on the test set.

Example:

```python
RANDOM_STATE = 42
```

---

## Ethical Considerations

Automated recruitment systems can have significant consequences for candidates.

This project should therefore be treated as a **decision-support tool rather than an autonomous hiring system**.

Potential concerns include:

### Bias

Historical recruitment data may contain biases. A Machine Learning model trained on biased data can reproduce or amplify those patterns.

### Protected Characteristics

The system should not make decisions based on protected characteristics such as:

* Race
* Ethnicity
* Religion
* Gender
* Disability
* Age
* Other legally protected characteristics

Where possible, sensitive information should be removed or excluded from model features.

### Transparency

Candidates and recruiters should understand that an automated system is being used as part of the screening process.

### Human Oversight

A qualified human recruiter should review important decisions, particularly rejection decisions.

### Privacy

Resume data may contain personally identifiable information such as:

* Names
* Email addresses
* Telephone numbers
* Addresses
* Employment history

Data should be handled securely and according to applicable privacy and data-protection requirements.

---

## Limitations

The project has several limitations.

### 1. Dataset Quality

Model performance depends heavily on the quality, size, representativeness, and labeling quality of the dataset.

### 2. Keyword Dependence

TF-IDF-based systems may rely heavily on words and phrases rather than fully understanding the context of a candidate's experience.

### 3. Synonyms

Different candidates may describe the same skill differently.

For example:

```text
Machine Learning
ML
machine-learning
```

A simple keyword matcher may treat these as different terms unless normalization is implemented.

### 4. Resume Formatting

Complex PDF layouts, tables, columns, images, and scanned documents can make text extraction difficult.

### 5. Generalization

A model trained on one job category or dataset may not perform equally well on different roles, industries, or populations.

### 6. Bias

The model may learn biases present in the training data.

### 7. Fit Score Interpretation

A model probability should not be interpreted as a definitive measure of candidate suitability.

---

## Future Improvements

Future versions of the project could include:

* BERT-based text classification
* Transformer-based resume embeddings
* Sentence Transformers
* Semantic similarity between resumes and job descriptions
* Named Entity Recognition
* Automatic skill extraction
* Experience extraction
* Education extraction
* Certification extraction
* Job-specific scoring
* Multi-class job classification
* Explainable AI techniques
* Fairness and bias monitoring
* OCR for scanned resumes
* DOCX support
* Batch PDF upload
* Candidate ranking
* Recruiter dashboard
* Database integration
* REST API
* Cloud deployment
* Authentication and access control

A particularly useful improvement would be replacing simple keyword matching with **semantic similarity between the job description and resume**, allowing the system to recognize related terms and concepts rather than relying only on exact keyword matches.

---

## Conclusion

The Resume Screening Classifier demonstrates how Natural Language Processing and supervised Machine Learning can be applied to automate an initial stage of the recruitment process.

The project transforms unstructured resume text into numerical features using TF-IDF, trains and compares multiple classification algorithms, evaluates their performance, and uses the selected model to classify new resumes.

The resulting system provides:

* Resume classification
* Fit score
* Matched skills
* Model-based candidate ranking
* Saved Machine Learning artifacts
* A foundation for Streamlit deployment

Although the system can improve the efficiency of initial resume screening, it should not replace human judgment. Recruitment decisions have significant consequences, and automated screening systems should be evaluated for accuracy, fairness, transparency, privacy, and bias before being used in real-world hiring.

---

## Author
Sunday Shaibu Onetokole

**Capstone Project — Resume Screening Classifier**

Machine Learning / Data Science Project

---

## License

This project is intended for educational and research purposes.

Before using any third-party resume dataset, verify its license, terms of use, privacy requirements, and permitted use.
