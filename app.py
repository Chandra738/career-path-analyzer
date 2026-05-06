import streamlit as st
import pandas as pd
import time
from sklearn.tree import DecisionTreeClassifier
import plotly.express as px
from fpdf import FPDF

st.set_page_config(page_title="AI Career System", layout="centered")

# ---------------- HEADER ----------------
st.title("🚀 AI Career Intelligence System")
st.caption("ML-powered career prediction + roadmap + AI assistant")

st.divider()

# ---------------- PROFILE ----------------
st.subheader("👤 User Profile")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Name")

with col2:
    target = st.selectbox(
        "Target Career",
        ["Software Developer", "AI Engineer", "Data Scientist", "Cybersecurity Analyst", "HR / Manager"]
    )

study_hours = st.slider("Study hours per week", 1, 40, 10)

st.divider()

# ---------------- DATASET ----------------
data = pd.DataFrame({
    "python": [8, 5, 9, 4, 7],
    "sql": [7, 6, 9, 5, 8],
    "ml": [8, 5, 9, 4, 7],
    "stats": [7, 6, 9, 5, 8],
    "problem_solving": [8, 6, 9, 5, 8],
    "communication": [5, 8, 6, 7, 6],
    "leadership": [6, 7, 5, 8, 7],
    "cloud": [6, 7, 8, 5, 7],
    "business": [7, 8, 6, 7, 8],
    "career": [
        "Software Developer",
        "HR / Manager",
        "AI Engineer",
        "IT Support",
        "Software Developer"
    ]
})

features = [
    "python","sql","ml","stats","problem_solving",
    "communication","leadership","cloud","business"
]

X = data[features]
y = data["career"]

model = DecisionTreeClassifier()
model.fit(X, y)

# ---------------- INPUT ----------------
st.subheader("📥 Enter Skills")

values = [st.slider(f, 0, 10, 0) for f in features]

st.divider()

# ---------------- PDF ----------------
def create_pdf(name, career, skills, missing):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="AI Career Report", ln=True, align='C')
    pdf.ln(5)

    pdf.cell(200, 10, txt=f"Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"Career: {career}", ln=True)

    pdf.ln(5)
    pdf.cell(200, 10, txt="Skills:", ln=True)

    for k, v in skills.items():
        pdf.cell(200, 8, txt=f"{k}: {v}", ln=True)

    pdf.ln(5)
    pdf.cell(200, 10, txt="Missing Skills:", ln=True)

    for m in missing:
        pdf.cell(200, 8, txt=f"- {m}", ln=True)

    file = "career_report.pdf"
    pdf.output(file)
    return file

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Career"):

    with st.spinner("AI analyzing your profile..."):
        time.sleep(1.5)

    prediction = model.predict([values])[0]

    st.success(f"🎯 Best Career: {prediction}")

    st.write("---")

    # ---------------- SKILL GAP ----------------
    st.subheader("📚 Skill Gap Analysis")

    career_skills = {
        "Software Developer": ["python","problem_solving","sql"],
        "AI Engineer": ["python","ml","stats"],
        "Data Scientist": ["python","stats","sql"],
        "Cybersecurity Analyst": ["problem_solving","cloud"],
        "HR / Manager": ["communication","leadership"]
    }

    user = dict(zip(features, values))
    required = career_skills.get(prediction, [])

    missing = [s for s in required if user.get(s,0) < 6]

    if missing:
        for m in missing:
            st.warning(f"📌 Improve: {m}")
    else:
        st.success("🎉 Strong match!")

    st.divider()

    # ---------------- CHART ----------------
    st.subheader("📊 Skill Profile")

    fig = px.bar(x=features, y=values)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------------- ROADMAP ----------------
    st.subheader("📅 Learning Roadmap")

    if missing:
        weeks = max(1, int(3 * (10 / study_hours)))
        for i, s in enumerate(missing):
            st.info(f"{s} → {weeks} weeks")

    st.divider()

    # ---------------- PDF ----------------
    pdf_file = create_pdf(name, prediction, user, missing)

    st.download_button(
        "📥 Download Report",
        open(pdf_file, "rb"),
        file_name="career_report.pdf"
    )

# ---------------- CHAT ----------------
st.write("---")
st.subheader("💬 AI Career Assistant")

query = st.text_input("Ask anything about careers:")

if query:
    q = query.lower()

    if "ai" in q:
        st.info("AI Engineer is best for machine learning and data-driven roles.")
    elif "salary" in q:
        st.info("AI Engineer & Software Developer have highest salary growth.")
    elif "skill" in q:
        st.info("Focus on Python, ML, SQL, Problem Solving.")
    elif "roadmap" in q:
        st.info("Start with basics → projects → internships.")
    else:
        st.info("Ask about skills, careers, salary, or roadmap.")

st.caption("🚀 AI Career System | ML + Dashboard + Chat + PDF")