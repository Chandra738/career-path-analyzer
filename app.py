import streamlit as st
import pandas as pd
import numpy as np
import time
import random
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

# ══════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Career Path Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════
#  CUSTOM CSS
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
  .block-container { padding: 1.2rem 2rem 2rem; }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: #12122a;
    border: 1px solid #3a3a7a;
    border-radius: 12px;
    padding: 14px 18px;
  }

  /* Skill pills */
  .pill-green {
    display:inline-block; background:#0f2d0f; color:#4ade80;
    border:1px solid #166534; border-radius:16px;
    padding:5px 13px; font-size:12px; margin:3px;
  }
  .pill-amber {
    display:inline-block; background:#2d1f00; color:#fbbf24;
    border:1px solid #92400e; border-radius:16px;
    padding:5px 13px; font-size:12px; margin:3px;
  }

  /* Institution cards */
  .inst-card {
    background:#0f0f2a; border:1px solid #2a2a5a;
    border-radius:12px; padding:16px 18px; height:100%;
    border-left:3px solid #7F77DD;
  }

  /* Tip box */
  .tip-box {
    background:#0f2040; border:1px solid #1a4080;
    border-radius:8px; padding:10px 14px;
    font-size:13px; color:#93c5fd; margin-top:10px;
  }

  /* Welcome banner */
  .banner {
    background: linear-gradient(135deg, #1a1a3e, #0f0f2a);
    border: 1px solid #3a3a7a;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 20px;
    text-align: center;
  }
  .banner h1 { font-size: 28px; margin: 0 0 8px; }
  .banner p  { font-size: 14px; color: #aaa; margin: 0 0 16px; }
  .steps-row {
    display: flex; justify-content: center;
    gap: 16px; flex-wrap: wrap; margin-top: 12px;
  }
  .step-box {
    background: #1e1e4a; border-radius: 10px;
    padding: 10px 18px; font-size: 13px;
    border: 1px solid #3a3a7a; min-width: 130px;
    text-align: center;
  }

  /* Skill level badge */
  .lvl-badge {
    font-size: 11px; font-weight: 600;
    padding: 2px 8px; border-radius: 8px;
    margin-left: 6px;
  }
  .lvl-beg  { background:#3a0000; color:#f87171; }
  .lvl-int  { background:#3a2d00; color:#fbbf24; }
  .lvl-adv  { background:#0f2d0f; color:#4ade80; }
  .lvl-exp  { background:#1e1258; color:#a78bfa; }

  /* Compare table */
  .cmp-table { width:100%; border-collapse:collapse; font-size:13px; }
  .cmp-table th { background:#1e1e4a; padding:10px 14px; text-align:left; }
  .cmp-table td { padding:10px 14px; border-bottom:1px solid #1e1e3a; }
  .cmp-table tr:hover td { background:#12122a; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  DATA & MODEL
# ══════════════════════════════════════════════════════
data = pd.DataFrame({
    "python":          [9,2,8,3,7,5,9,4,6,8,3,7,9,2,5,8,6,4,7,9,3,6,8,5,7,2,9,4,6,8,7,3,8,5,9,2,6,4,7,8],
    "sql":             [6,5,9,4,7,6,8,4,7,7,5,8,7,3,6,9,6,5,7,8,4,6,9,5,7,3,8,5,7,8,6,4,9,5,8,3,6,4,7,8],
    "ml":              [9,2,9,3,7,4,9,3,5,8,2,7,9,2,4,8,5,3,7,9,2,5,9,3,7,2,9,3,5,8,6,2,8,4,9,2,5,3,6,8],
    "stats":           [8,4,9,4,7,5,8,4,6,8,4,8,8,3,6,9,6,4,7,8,3,6,8,5,7,3,8,4,6,8,7,3,8,5,8,3,6,4,7,8],
    "problem_solving": [9,5,8,6,7,6,9,5,7,8,5,8,9,4,7,8,7,5,8,9,4,7,8,6,8,4,9,5,7,8,7,5,8,6,9,4,7,5,7,8],
    "communication":   [4,9,5,9,6,9,4,8,9,5,9,6,4,9,8,5,9,9,6,4,8,7,5,9,6,9,4,8,8,5,7,9,5,8,4,9,7,8,6,5],
    "leadership":      [3,9,4,9,6,8,3,8,9,4,9,5,3,9,8,4,9,9,6,3,9,8,4,9,5,9,3,8,8,4,6,9,4,8,3,9,7,8,6,4],
    "cloud":           [8,3,8,3,6,4,9,3,5,8,2,7,8,2,4,9,5,3,7,9,2,5,8,3,7,2,9,3,5,8,6,2,8,4,9,2,5,3,6,8],
    "business":        [4,9,5,8,6,8,4,7,8,5,8,6,4,8,7,5,8,8,7,4,8,7,5,8,6,8,4,7,7,5,7,8,5,7,4,8,7,7,6,5],
    "career": [
        "AI Engineer","HR Manager","Data Scientist","IT Support",
        "Software Developer","Business Analyst","Data Scientist","IT Support",
        "Product Manager","Software Developer","HR Manager","Data Scientist",
        "AI Engineer","Teacher/Trainer","Business Analyst","Data Scientist",
        "Product Manager","HR Manager","Software Developer","AI Engineer",
        "Teacher/Trainer","Product Manager","Data Scientist","Business Analyst",
        "Software Developer","Teacher/Trainer","AI Engineer","IT Support",
        "Product Manager","Data Scientist","Software Developer","HR Manager",
        "Data Scientist","Business Analyst","AI Engineer","Teacher/Trainer",
        "Product Manager","IT Support","Software Developer","Data Scientist"
    ]
})

FEATURES = ["python","sql","ml","stats","problem_solving",
            "communication","leadership","cloud","business"]

ROLE_NEEDS = {
    "AI Engineer":        {"python":9,"ml":9,"stats":8,"cloud":7,"sql":6,"problem_solving":8,"communication":5,"leadership":3,"business":4},
    "Data Scientist":     {"python":8,"ml":8,"stats":9,"sql":8,"cloud":5,"problem_solving":8,"communication":6,"leadership":4,"business":5},
    "Software Developer": {"python":8,"sql":6,"ml":4,"stats":5,"cloud":7,"problem_solving":8,"communication":6,"leadership":4,"business":4},
    "Product Manager":    {"python":3,"sql":5,"ml":3,"stats":6,"cloud":4,"problem_solving":8,"communication":9,"leadership":9,"business":9},
    "Business Analyst":   {"python":4,"sql":8,"ml":4,"stats":7,"cloud":3,"problem_solving":8,"communication":8,"leadership":6,"business":9},
    "HR Manager":         {"python":2,"sql":4,"ml":2,"stats":4,"cloud":2,"problem_solving":6,"communication":9,"leadership":9,"business":8},
    "IT Support":         {"python":5,"sql":5,"ml":3,"stats":3,"cloud":7,"problem_solving":8,"communication":7,"leadership":4,"business":4},
    "Teacher/Trainer":    {"python":4,"sql":3,"ml":3,"stats":6,"cloud":2,"problem_solving":7,"communication":9,"leadership":7,"business":5},
}

ROLE_INFO = {
    "AI Engineer":        {"salary":"₹10–25 LPA","growth":"+38%","remote":"🟢 High",   "emoji":"🤖","desc":"Build AI systems, train ML models, deploy intelligent solutions."},
    "Data Scientist":     {"salary":"₹8–20 LPA", "growth":"+34%","remote":"🟢 High",   "emoji":"📊","desc":"Analyse large data, find patterns, build predictive models."},
    "Software Developer": {"salary":"₹6–18 LPA", "growth":"+25%","remote":"🟢 High",   "emoji":"💻","desc":"Build apps and systems using Python, Java, or JavaScript."},
    "Product Manager":    {"salary":"₹10–22 LPA","growth":"+22%","remote":"🟡 Medium", "emoji":"🎯","desc":"Lead product strategy, work with teams, deliver user value."},
    "Business Analyst":   {"salary":"₹5–14 LPA", "growth":"+18%","remote":"🟡 Medium", "emoji":"📈","desc":"Bridge business and tech — analyse data, define requirements."},
    "HR Manager":         {"salary":"₹4–10 LPA", "growth":"+10%","remote":"🔴 Low",    "emoji":"🤝","desc":"Hire talent, manage teams, build workplace culture."},
    "IT Support":         {"salary":"₹3–7 LPA",  "growth":"+12%","remote":"🟡 Medium", "emoji":"🛠️","desc":"Maintain systems, troubleshoot issues, support users."},
    "Teacher/Trainer":    {"salary":"₹3–8 LPA",  "growth":"+15%","remote":"🟡 Medium", "emoji":"📚","desc":"Educate and train others — in schools, colleges, or edtech."},
}

COURSE_LINKS = {
    "python":          ("Python – freeCodeCamp",       "https://www.freecodecamp.org/learn/scientific-computing-with-python/"),
    "sql":             ("SQL – SQLZoo (free)",          "https://sqlzoo.net/"),
    "ml":              ("ML – Andrew Ng Coursera",     "https://www.coursera.org/specializations/machine-learning-introduction"),
    "stats":           ("Stats – Khan Academy",        "https://www.khanacademy.org/math/statistics-probability"),
    "problem_solving": ("Problem Solving – HackerRank","https://www.hackerrank.com/domains/algorithms"),
    "communication":   ("Communication – Coursera",    "https://www.coursera.org/learn/wharton-communication-skills"),
    "leadership":      ("Leadership – LinkedIn Learn", "https://www.linkedin.com/learning/topics/leadership"),
    "cloud":           ("Cloud – AWS Free Tier",       "https://aws.amazon.com/free/"),
    "business":        ("Business – Google Certs",     "https://grow.google/certificates/"),
}

INSTITUTIONS = [
    {"name":"NPTEL (IIT Online)",    "skills":["python","ml","stats","sql","cloud"],             "duration":"3–6 mo","cost":"Free",         "placement":"85%","link":"https://nptel.ac.in"},
    {"name":"Coursera (Google/IBM)", "skills":["python","sql","cloud","leadership","business"],   "duration":"3–4 mo","cost":"₹3,000–8,000", "placement":"88%","link":"https://www.coursera.org"},
    {"name":"upGrad",                "skills":["python","ml","stats","sql","communication"],      "duration":"6–12 mo","cost":"₹50,000+",    "placement":"92%","link":"https://www.upgrad.com"},
    {"name":"Udemy Bootcamp",        "skills":["python","cloud","problem_solving","sql"],         "duration":"1–3 mo","cost":"₹500–2,000",   "placement":"75%","link":"https://www.udemy.com"},
    {"name":"Great Learning",        "skills":["python","ml","cloud","sql","leadership"],         "duration":"3–6 mo","cost":"₹10,000–40,000","placement":"89%","link":"https://www.greatlearning.in"},
    {"name":"NASSCOM FutureSkills",  "skills":["cloud","python","business","communication"],      "duration":"2–4 mo","cost":"Free–₹5,000",  "placement":"80%","link":"https://futureskills.nasscom.in"},
]

CAREER_TIPS = [
    "💡 Python is required in 91% of Data Science jobs in India!",
    "💡 SQL is the #1 most in-demand skill across all tech roles.",
    "💡 Cloud certifications can increase your salary by 25–40%.",
    "💡 Communication skills matter in every single career path.",
    "💡 AI Engineer is the fastest growing role in India in 2025.",
    "💡 Consistent 10 hrs/week of learning = job-ready in 4 months!",
    "💡 NPTEL certificates from IIT are free and highly respected.",
    "💡 A portfolio of 3 projects beats a degree in many tech companies.",
]

# ── Train model ──
X = data[FEATURES]
y = data["career"]
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# ══════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════
def match_score(user: dict, role: str) -> int:
    req = ROLE_NEEDS[role]
    return int(np.mean([min(user.get(s, 0) / max(v, 1), 1) * 100 for s, v in req.items()]))

def skill_gap(user: dict, role: str):
    req = ROLE_NEEDS[role]
    have, need = [], []
    for s, v in req.items():
        if user.get(s, 0) >= v - 1:
            have.append(s)
        else:
            need.append((s, v, user.get(s, 0)))
    return have, sorted(need, key=lambda x: x[1] - x[2], reverse=True)

def level_badge(val: int) -> str:
    if val <= 3:   return '<span class="lvl-badge lvl-beg">Beginner</span>'
    elif val <= 6: return '<span class="lvl-badge lvl-int">Intermediate</span>'
    elif val <= 9: return '<span class="lvl-badge lvl-adv">Advanced</span>'
    else:          return '<span class="lvl-badge lvl-exp">⭐ Expert</span>'

def voice_speak(text: str):
    safe = text.replace('"', "'").replace('\n', ' ')
    components.html(f"""
    <script>
      window.speechSynthesis.cancel();
      var u = new SpeechSynthesisUtterance("{safe}");
      u.rate = 0.93; u.pitch = 1.05; u.volume = 1;
      window.speechSynthesis.speak(u);
    </script>
    """, height=0)

def ai_response(q: str, top_role: str = "", name: str = "") -> str:
    q   = q.lower()
    nm  = name.strip() or "there"
    rol = top_role or "AI Engineer"
    inf = ROLE_INFO.get(rol, {})

    if any(w in q for w in ["your name","who are you","what are you","hello","hi","hey"]):
        return f"Hello {nm}! I am your AI Career Assistant. Rate your skills above and click Predict to get your personalised career roadmap. I can also answer questions about careers, salaries, and learning paths!"
    elif any(w in q for w in ["career","job","role","predict","suggest","best","suit"]):
        return f"Hi {nm}! Based on your skills, your top career match is {rol}. {inf.get('desc','')} Expected salary: {inf.get('salary','N/A')} with {inf.get('growth','N/A')} job growth in 2025!"
    elif any(w in q for w in ["skill","learn","improve","missing","study","what should"]):
        return f"For {rol}, focus on Python, SQL, Machine Learning, and Cloud skills. These are the top in-demand skills in India for 2025 tech jobs. Start with Python on freeCodeCamp — it is completely free!"
    elif any(w in q for w in ["salary","pay","money","lpa","earn","income"]):
        return f"The average salary for {rol} in India is {inf.get('salary','₹6–15 LPA')} per year. This grows significantly with experience and certifications. Cloud and AI skills give the highest salary boost!"
    elif any(w in q for w in ["time","long","month","week","fast","how soon","ready","when"]):
        return f"With 10 hours of study per week, most people become job-ready in 3 to 6 months. The key is consistency every single day. Your personalised timeline is shown below after you click Predict!"
    elif any(w in q for w in ["course","certificate","platform","where","nptel","coursera","udemy"]):
        return f"Top free platforms: NPTEL from IIT is excellent and free. Coursera has Google and IBM certificates. Udemy has affordable practical bootcamps. For {rol}, I recommend starting with Python and then SQL!"
    elif any(w in q for w in ["remote","wfh","work from home","online work"]):
        return f"Remote work for {rol} is rated {inf.get('remote','Medium')}. AI Engineer, Data Scientist, and Software Developer roles have the best remote opportunities in India right now!"
    elif any(w in q for w in ["institution","college","university","which platform"]):
        return "Top platforms for you: NPTEL is free and from IIT professors. upGrad has the highest placement rate at 92 percent. Coursera has globally recognised certificates from Google and IBM!"
    elif any(w in q for w in ["compare","vs","versus","difference","better"]):
        return f"Use the Career Comparison tab in the results to compare {rol} with any other career side by side — salary, growth, remote work, and effort all compared!"
    elif any(w in q for w in ["download","save","roadmap","pdf","report"]):
        return "Scroll down to the Download section after clicking Predict! You can download your full career roadmap as a text file with all your matches, skill gaps, timeline, and recommended platforms."
    else:
        return f"I can help you with careers, skills, salaries, learning timelines, and institutions, {nm}! Your current top match is {rol}. Try asking: What salary can I earn? How long to learn ML? Which platform is best?"

# ══════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🚀 AI Career Analyzer")
    st.caption("Your personal AI career guide")
    st.divider()

    st.markdown("#### 🔥 Trending Skills 2025")
    trending = {
        "Python / AI":      96,
        "Cloud (AWS/GCP)":  91,
        "Machine Learning": 88,
        "SQL / Data":       82,
        "Prompt Engineering":71,
        "Cybersecurity":    73,
        "DevOps / CI-CD":   77,
        "Communication":    65,
    }
    for skill, pct in trending.items():
        color = "#e24b4a" if pct >= 85 else "#fbbf24" if pct >= 75 else "#4ade80"
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px">
          <span>{skill}</span><span style="color:{color};font-weight:600">{pct}%</span>
        </div>
        <div style="height:5px;background:#1e1e4a;border-radius:3px;margin-bottom:8px">
          <div style="width:{pct}%;height:100%;background:{color};border-radius:3px"></div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 📖 How To Use")
    st.markdown("""
    1. 👤 Fill your profile  
    2. 🎙️ Ask voice assistant  
    3. 📥 Rate all 9 skills  
    4. 🔮 Click **Predict**  
    5. 📊 Explore your results  
    6. 📥 Download roadmap  
    """)

    st.divider()
    st.markdown("#### 💡 Did You Know?")
    st.info(random.choice(CAREER_TIPS))

# ══════════════════════════════════════════════════════
#  WELCOME BANNER
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="banner">
  <h1>🚀 AI Career Path Analyzer</h1>
  <p>Rate your skills → AI finds your best career + skill gaps + learning roadmap + voice guidance</p>
  <div class="steps-row">
    <div class="step-box">👤 1. Fill Profile</div>
    <div class="step-box">🎙️ 2. Ask AI</div>
    <div class="step-box">📥 3. Rate Skills</div>
    <div class="step-box">🔮 4. Predict</div>
    <div class="step-box">📊 5. Get Roadmap</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════
if "last_role"   not in st.session_state: st.session_state.last_role   = ""
if "last_name"   not in st.session_state: st.session_state.last_name   = ""
if "did_predict" not in st.session_state: st.session_state.did_predict = False

# ══════════════════════════════════════════════════════
#  SECTION 1 — USER PROFILE
# ══════════════════════════════════════════════════════
with st.expander("👤 Your Profile — fill for personalised results", expanded=True):
    p1, p2, p3 = st.columns(3)
    with p1: user_name    = st.text_input("Your Name",               placeholder="e.g. Ravi Kumar")
    with p2: current_role = st.text_input("Current Role/Background", placeholder="e.g. Student, IT Support")
    with p3: location     = st.text_input("Location",                placeholder="e.g. Hyderabad")
    p4, p5 = st.columns(2)
    with p4:
        target = st.selectbox("Career you're most interested in",
                    ["(Let AI decide)","AI Engineer","Data Scientist","Software Developer",
                     "Product Manager","Business Analyst","HR Manager","IT Support","Teacher/Trainer"])
    with p5:
        hrs_week = st.slider("Study hours per week", 1, 40, 10)

    # Profile completion bar
    filled = sum([
        bool(user_name.strip()),
        bool(current_role.strip()),
        bool(location.strip()),
        target != "(Let AI decide)",
        hrs_week != 10
    ])
    pct = int(filled / 5 * 100)
    bar_color = "#4ade80" if pct >= 80 else "#fbbf24" if pct >= 40 else "#f87171"
    st.markdown(f"""
    <div style="margin-top:10px">
      <div style="display:flex;justify-content:space-between;font-size:12px;color:#aaa;margin-bottom:4px">
        <span>Profile Completion</span><span style="color:{bar_color};font-weight:600">{pct}%</span>
      </div>
      <div style="height:6px;background:#1e1e4a;border-radius:3px">
        <div style="width:{pct}%;height:100%;background:{bar_color};border-radius:3px;transition:width .4s"></div>
      </div>
      <div style="font-size:11px;color:#888;margin-top:4px">
        {'✅ Profile complete — results will be fully personalised!' if pct == 100 else 'Fill more fields for better personalised results'}
      </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════
#  SECTION 2 — VOICE ASSISTANT
# ══════════════════════════════════════════════════════
st.subheader("🎙️ Voice Assistant — Ask me anything")

# Language toggle
lang_col, _ = st.columns([1, 3])
with lang_col:
    lang_mode = st.radio("Voice Language", ["🇬🇧 English", "🇮🇳 Hindi"], horizontal=True)
lang_code = "hi-IN" if "Hindi" in lang_mode else "en-IN"

voice_html = f"""
<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;padding:4px 0">
  <button id="micBtn" onclick="startVoice()" style="
    padding:11px 26px; border-radius:24px; border:none;
    background:#7F77DD; color:white; cursor:pointer;
    font-size:15px; font-weight:600; letter-spacing:.3px;
    box-shadow:0 4px 15px rgba(127,119,221,0.35);
    transition:all .2s;">
    🎙️ Click &amp; Speak
  </button>
  <span id="micStatus" style="font-size:13px;color:#aaa">
    Ask: "What career suits me?" · "What salary can I earn?" · "What should I learn?"
  </span>
</div>
<div id="voiceOut" style="
  margin-top:10px; padding:12px 16px; border-radius:10px;
  background:#0f0f2a; border:1px solid #2a2a5a;
  font-size:13px; color:#ccc; min-height:40px;
  transition: all .3s;"></div>
<script>
function startVoice() {{
  var btn    = document.getElementById('micBtn');
  var status = document.getElementById('micStatus');
  var out    = document.getElementById('voiceOut');
  var SR     = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {{ status.innerText = '❌ Use Google Chrome for voice support.'; return; }}
  var rec = new SR();
  rec.lang = '{lang_code}';
  rec.continuous = false; rec.interimResults = false;
  btn.innerText = '🔴 Listening...';
  btn.style.background = '#e24b4a';
  btn.style.boxShadow = '0 4px 15px rgba(226,75,74,0.4)';
  status.innerText = 'Speak now...';
  rec.onresult = function(e) {{
    var text = e.results[0][0].transcript;
    out.innerText = '🗣️ You said: "' + text + '"';
    btn.innerText = '🎙️ Click & Speak';
    btn.style.background = '#7F77DD';
    btn.style.boxShadow = '0 4px 15px rgba(127,119,221,0.35)';
    status.innerText = '✅ Got it! Type the same below to see AI response ↓';
  }};
  rec.onerror = function() {{
    btn.innerText = '🎙️ Click & Speak';
    btn.style.background = '#7F77DD';
    status.innerText = '⚠️ Could not hear. Try again in Chrome.';
  }};
  rec.onend = function() {{
    btn.innerText = '🎙️ Click & Speak';
    btn.style.background = '#7F77DD';
  }};
  rec.start();
}}
</script>
"""
components.html(voice_html, height=130)

voice_input = st.text_input(
    "Type your question here for AI voice response 👇",
    placeholder="e.g. What career suits me?  |  What salary can I earn?  |  How long to learn ML?"
)

# Voice chat response (works at any time)
if voice_input.strip():
    last_role = st.session_state.get("last_role", "")
    response  = ai_response(voice_input, last_role, user_name)
    st.info(f"**You:** {voice_input}\n\n**🤖 AI:** {response}")
    voice_speak(response)
    st.markdown(
        '<div class="tip-box">💡 Try asking: "What career suits me?" · "What should I learn?" · '
        '"How long will it take?" · "What salary?" · "Which platform is best?"</div>',
        unsafe_allow_html=True
    )

st.divider()

# ══════════════════════════════════════════════════════
#  SECTION 3 — SKILL SLIDERS (with live level badges)
# ══════════════════════════════════════════════════════
st.subheader("📥 Rate Your Skills  (0 = none  →  10 = expert)")

col_a, col_b = st.columns(2)
with col_a:
    with st.expander("💻 Technical Skills", expanded=True):
        python = st.slider("Python",            0, 10, 0, key="py")
        st.markdown(f"Level: {level_badge(python)}", unsafe_allow_html=True)
        sql    = st.slider("SQL / Databases",   0, 10, 0, key="sq")
        st.markdown(f"Level: {level_badge(sql)}", unsafe_allow_html=True)
        ml     = st.slider("Machine Learning",  0, 10, 0, key="ml")
        st.markdown(f"Level: {level_badge(ml)}", unsafe_allow_html=True)
        stats  = st.slider("Statistics / Math", 0, 10, 0, key="st")
        st.markdown(f"Level: {level_badge(stats)}", unsafe_allow_html=True)
        cloud  = st.slider("Cloud (AWS/GCP)",   0, 10, 0, key="cl")
        st.markdown(f"Level: {level_badge(cloud)}", unsafe_allow_html=True)

with col_b:
    with st.expander("🤝 Soft & Business Skills", expanded=True):
        prob   = st.slider("Problem Solving",   0, 10, 0, key="ps")
        st.markdown(f"Level: {level_badge(prob)}", unsafe_allow_html=True)
        comm   = st.slider("Communication",     0, 10, 0, key="co")
        st.markdown(f"Level: {level_badge(comm)}", unsafe_allow_html=True)
        lead   = st.slider("Leadership",        0, 10, 0, key="le")
        st.markdown(f"Level: {level_badge(lead)}", unsafe_allow_html=True)
        biz    = st.slider("Business Sense",    0, 10, 0, key="bi")
        st.markdown(f"Level: {level_badge(biz)}", unsafe_allow_html=True)

user_skills = {
    "python": python, "sql": sql, "ml": ml, "stats": stats,
    "cloud": cloud, "problem_solving": prob,
    "communication": comm, "leadership": lead, "business": biz
}
values_list = [user_skills[f] for f in FEATURES]

# Live mini-metrics
st.markdown("**📈 Your current skill summary:**")
lv1, lv2, lv3, lv4 = st.columns(4)
lv1.metric("Python",   f"{python}/10")
lv2.metric("ML",       f"{ml}/10")
lv3.metric("SQL",      f"{sql}/10")
lv4.metric("Overall",  f"{int(np.mean(values_list))}/10")

st.divider()

# ══════════════════════════════════════════════════════
#  PREDICT BUTTON
# ══════════════════════════════════════════════════════
pred_col, _ = st.columns([2, 4])
with pred_col:
    predict_btn = st.button("🔮 Predict My Career Path", type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════
if predict_btn:
    # Loading animation with tips
    tip_placeholder = st.empty()
    for _ in range(3):
        tip_placeholder.info(f"🤖 Analysing... {random.choice(CAREER_TIPS)}")
        time.sleep(0.5)
    tip_placeholder.empty()

    ml_pred      = model.predict([values_list])[0]
    all_scores   = {r: match_score(user_skills, r) for r in ROLE_NEEDS}
    sorted_roles = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    top_role, top_score = sorted_roles[0]

    st.session_state.last_role   = top_role
    st.session_state.last_name   = user_name
    st.session_state.did_predict = True

    name_str = user_name.strip() if user_name.strip() else "there"
    info     = ROLE_INFO[top_role]

    st.success(f"✅ Done! Hi **{name_str}**, your top career match is **{info['emoji']} {top_role}** with a **{top_score}% match score!**")
    st.balloons()

    voice_speak(
        f"Analysis complete! Hi {name_str}! Your top career match is {top_role} "
        f"with a {top_score} percent match score. "
        f"Expected salary is {info['salary']} per year in India. "
        f"Scroll down to see your full personalised roadmap!"
    )

    # ── TOP METRICS ──
    st.subheader("🏆 Your Career Match Summary")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(f"{info['emoji']} Best Match",  top_role,     f"{top_score}% match")
    m2.metric("💰 Salary Range",              info["salary"], info["growth"] + " growth")
    m3.metric("🌐 Remote Work",               info["remote"])
    m4.metric("🤖 ML Prediction",             ml_pred)
    m5.metric("📍 Your Location",             location or "India")

    st.divider()

    have, need = skill_gap(user_skills, top_role)

    # ════════════════════════════════════
    #  TABBED RESULTS
    # ════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Career Matches",
        "📋 Skill Gap",
        "📅 Learning Timeline",
        "🏫 Institutions",
        "⚖️ Compare Careers"
    ])

    # ── TAB 1 : MATCH CHARTS ──
    with tab1:
        c1, c2 = st.columns([3, 2])
        with c1:
            df_scores = pd.DataFrame(sorted_roles, columns=["Career", "Match %"])
            fig1 = px.bar(
                df_scores, x="Match %", y="Career", orientation="h",
                color="Match %", color_continuous_scale="Purples",
                text="Match %", title="Career Match Scores — All Roles"
            )
            fig1.update_traces(texttemplate="%{text}%", textposition="outside")
            fig1.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="white", showlegend=False, coloraxis_showscale=False,
                height=360, margin=dict(l=10, r=60, t=40, b=10)
            )
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            st.markdown(f"### {info['emoji']} {top_role}")
            st.markdown(f"*{info['desc']}*")
            st.markdown(f"**💰 Salary:** {info['salary']}")
            st.markdown(f"**📈 Job Growth:** {info['growth']}")
            st.markdown(f"**🌐 Remote:** {info['remote']}")
            st.markdown(f"**Match Score:** {top_score}%")
            st.progress(top_score / 100)

        # Radar chart
        st.subheader(f"🕸️ Your Profile vs {top_role}")
        radar_labels = [s.replace("_", " ").title() for s in FEATURES]
        user_vals    = [user_skills[s] for s in FEATURES]
        req_vals     = [ROLE_NEEDS[top_role].get(s, 0) for s in FEATURES]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatterpolar(
            r=req_vals + [req_vals[0]], theta=radar_labels + [radar_labels[0]],
            fill="toself", name="Required",
            line=dict(color="#7F77DD"), fillcolor="rgba(127,119,221,0.15)"
        ))
        fig2.add_trace(go.Scatterpolar(
            r=user_vals + [user_vals[0]], theta=radar_labels + [radar_labels[0]],
            fill="toself", name="Your Skills",
            line=dict(color="#4ade80"), fillcolor="rgba(74,222,128,0.15)"
        ))
        fig2.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white", legend=dict(orientation="h"),
            height=400, margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── TAB 2 : SKILL GAP ──
    with tab2:
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("### ✅ Skills You Already Have")
            if have:
                pills = "".join(
                    f'<span class="pill-green">{s.replace("_"," ").title()}</span>'
                    for s in have
                )
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.info("Rate skills to see what you already have!")

        with g2:
            st.markdown("### 📚 Skills to Learn (Priority Order)")
            if need:
                for skill, needed, current in need:
                    gap = needed - current
                    cname, link = COURSE_LINKS.get(skill, (skill, "#"))
                    label = skill.replace("_", " ").title()
                    st.markdown(
                        f'<span class="pill-amber">{label} (+{gap})</span> '
                        f'<a href="{link}" target="_blank" '
                        f'style="color:#93c5fd;font-size:12px">→ {cname}</a>',
                        unsafe_allow_html=True
                    )
            else:
                st.success("🎉 You meet all requirements for this role!")

        st.markdown(f"\n**Overall Readiness for {top_role}:** {top_score}%")
        st.progress(top_score / 100)

        if top_score < 50:
            voice_speak(f"You are {top_score} percent ready for {top_role}. Focus on the skills shown in red — especially the top 2 or 3. You can do this!")
        elif top_score < 80:
            voice_speak(f"You are {top_score} percent ready for {top_role}. You are getting close! Fill the gaps shown and you will be job-ready soon.")
        else:
            voice_speak(f"Amazing! You are {top_score} percent ready for {top_role}. You are nearly job-ready. Start applying and polish your portfolio!")

    # ── TAB 3 : TIMELINE ──
    with tab3:
        total_weeks   = 0
        timeline_rows = []
        if need:
            base_date  = pd.Timestamp("2025-06-01")
            start_week = 1
            for skill, needed, current in need:
                gap   = needed - current
                weeks = max(1, round((gap * 3) / hrs_week * 7))
                end_week = start_week + weeks - 1
                timeline_rows.append({
                    "Skill": skill.replace("_", " ").title(),
                    "Start": base_date + pd.Timedelta(weeks=start_week - 1),
                    "End":   base_date + pd.Timedelta(weeks=end_week),
                    "Weeks": weeks
                })
                start_week = end_week + 1

            total_weeks  = sum(r["Weeks"] for r in timeline_rows)
            total_months = round(total_weeks / 4.3, 1)

            st.info(
                f"⏱️ At **{hrs_week} hrs/week** you'll be job-ready in "
                f"**~{total_weeks} weeks ({total_months} months)**"
            )
            voice_speak(
                f"Your learning timeline is ready! At {hrs_week} hours per week, "
                f"you will be job-ready in {total_weeks} weeks, "
                f"which is about {total_months} months. Consistency is everything!"
            )

            tl_df = pd.DataFrame(timeline_rows)
            fig3  = px.timeline(
                tl_df, x_start="Start", x_end="End", y="Skill",
                color="Skill", title=f"Your Learning Plan — {hrs_week} hrs/week"
            )
            fig3.update_yaxes(autorange="reversed")
            fig3.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="white", showlegend=False,
                height=350, margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig3, use_container_width=True)

            for row in timeline_rows:
                st.markdown(
                    f"- **{row['Skill']}** — {row['Weeks']} week(s) · "
                    f"{COURSE_LINKS.get(row['Skill'].lower().replace(' ','_'),('Free resources','#'))[0]}"
                )
        else:
            st.success("🎉 No skill gaps! You're already job-ready for this role.")
            voice_speak(f"Congratulations! You have no skill gaps for {top_role}. You are already job-ready! Start applying now.")

    # ── TAB 4 : INSTITUTIONS ──
    with tab4:
        st.markdown("### 🏫 Best Platforms for Your Skill Gaps")
        missing_skills = [s for s, _, _ in need] if need else list(user_skills.keys())
        for inst in INSTITUTIONS:
            inst["score"] = len([s for s in missing_skills if s in inst["skills"]])
        ranked = sorted(INSTITUTIONS, key=lambda x: x["score"], reverse=True)

        ic1, ic2, ic3 = st.columns(3)
        for col, inst in zip([ic1, ic2, ic3], ranked[:3]):
            with col:
                covers = ", ".join(s.replace("_"," ").title() for s in inst["skills"][:3])
                st.markdown(f"""
<div class="inst-card">
  <b style="font-size:15px">🏛️ {inst['name']}</b><br><br>
  <span style="color:#aaa;font-size:12px">⏱️ Duration: {inst['duration']}</span><br>
  <span style="color:#aaa;font-size:12px">💰 Cost: {inst['cost']}</span><br>
  <span style="color:#4ade80;font-size:13px;font-weight:600">✅ Placement: {inst['placement']}</span><br>
  <span style="color:#aaa;font-size:11px">Covers: {covers}...</span><br><br>
  <a href="{inst['link']}" target="_blank"
     style="color:#7F77DD;font-size:13px;font-weight:600;text-decoration:none">
    🔗 Visit Website →
  </a>
</div>
""", unsafe_allow_html=True)

        st.divider()
        st.markdown("### 📊 All Platforms Compared")
        df_inst = pd.DataFrame([{
            "Platform":   i["name"],
            "Duration":   i["duration"],
            "Cost":       i["cost"],
            "Placement":  i["placement"],
            "Relevance":  f"{i['score']}/{len(missing_skills[:5])} skills covered"
        } for i in ranked])
        st.dataframe(df_inst, use_container_width=True, hide_index=True)

    # ── TAB 5 : COMPARE CAREERS ──
    with tab5:
        st.markdown("### ⚖️ Compare Two Career Paths Side by Side")
        cc1, cc2 = st.columns(2)
        with cc1: role_a = st.selectbox("Career A", list(ROLE_NEEDS.keys()), key="ca")
        with cc2: role_b = st.selectbox("Career B", list(ROLE_NEEDS.keys()), index=1, key="cb")

        if role_a != role_b:
            ia = ROLE_INFO[role_a]
            ib = ROLE_INFO[role_b]
            sa = match_score(user_skills, role_a)
            sb = match_score(user_skills, role_b)
            winner = role_a if sa >= sb else role_b

            st.markdown(f"""
<table class="cmp-table">
  <tr>
    <th>Metric</th>
    <th>{ia['emoji']} {role_a}</th>
    <th>{ib['emoji']} {role_b}</th>
  </tr>
  <tr><td>Your Match %</td>
      <td style="color:{'#4ade80' if sa>=sb else '#aaa'};font-weight:600">{sa}%</td>
      <td style="color:{'#4ade80' if sb>sa else '#aaa'};font-weight:600">{sb}%</td></tr>
  <tr><td>Salary Range</td><td>{ia['salary']}</td><td>{ib['salary']}</td></tr>
  <tr><td>Job Growth</td><td>{ia['growth']}</td><td>{ib['growth']}</td></tr>
  <tr><td>Remote Work</td><td>{ia['remote']}</td><td>{ib['remote']}</td></tr>
  <tr><td>Description</td><td style="font-size:12px">{ia['desc']}</td>
      <td style="font-size:12px">{ib['desc']}</td></tr>
</table>
""", unsafe_allow_html=True)

            st.markdown(f"\n**🏆 Better match for you:** {winner} ({max(sa, sb)}%)")
            voice_speak(
                f"Comparing {role_a} versus {role_b}. "
                f"Your match score for {role_a} is {sa} percent, "
                f"and for {role_b} it is {sb} percent. "
                f"{winner} is the better match for your current skills!"
            )

            fig_cmp = go.Figure(data=[
                go.Bar(name=role_a, x=["Match %"], y=[sa], marker_color="#7F77DD"),
                go.Bar(name=role_b, x=["Match %"], y=[sb], marker_color="#4ade80"),
            ])
            fig_cmp.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="white", barmode="group",
                height=250, margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_cmp, use_container_width=True)
        else:
            st.warning("Please select two different careers to compare.")

    st.divider()

    # ── DOWNLOAD ──
    st.subheader("📥 Download Your Full Career Roadmap")
    roadmap = f"""
╔══════════════════════════════════════════════════════════╗
║            AI CAREER PATH ANALYZER — ROADMAP             ║
╚══════════════════════════════════════════════════════════╝

Name         : {user_name or 'User'}
Location     : {location or 'India'}
Background   : {current_role or 'N/A'}
Generated    : 2025

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOP CAREER MATCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Role         : {top_role}
Match Score  : {top_score}%
Salary Range : {info['salary']}
Job Growth   : {info['growth']}
Remote Work  : {info['remote']}
Description  : {info['desc']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALL CAREER MATCH SCORES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""" + "\n".join(f"  {r:<28} {s}%" for r, s in sorted_roles) + f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SKILLS YOU ALREADY HAVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {', '.join(s.replace('_',' ').title() for s in have) or 'Rate your skills to see'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SKILLS TO LEARN — PRIORITY ORDER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""" + "\n".join(
    f"  {s.replace('_',' ').title():<28} need +{n-c} levels → {COURSE_LINKS.get(s,('',))[0]}"
    for s,n,c in need
) + f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LEARNING TIMELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Study pace : {hrs_week} hrs/week
  Total time : ~{total_weeks} weeks ({round(total_weeks/4.3,1)} months)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMMENDED PLATFORMS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""" + "\n".join(
    f"  {i['name']:<30} | {i['duration']} | {i['cost']} | Placement: {i['placement']}"
    for i in (sorted(INSTITUTIONS, key=lambda x: x.get('score',0), reverse=True))[:3]
) + """

══════════════════════════════════════════════════════════
  Generated by AI Career Path Analyzer — Built with ❤️
══════════════════════════════════════════════════════════
"""
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            label="📥 Download Roadmap (.txt)",
            data=roadmap,
            file_name=f"{(user_name or 'career').replace(' ','_')}_roadmap.txt",
            mime="text/plain",
            use_container_width=True
        )
    with dl2:
        if st.button("🔊 Read Full Summary Aloud", use_container_width=True):
            voice_speak(
                f"Career summary for {user_name or 'you'}. "
                f"Top match: {top_role} with {top_score} percent match score. "
                f"Salary: {info['salary']}. Growth: {info['growth']}. "
                f"Skills to learn: {', '.join(s.replace('_',' ') for s,_,_ in need[:3]) if need else 'none, you are ready'}. "
                f"At {hrs_week} hours per week, you will be job-ready in {total_weeks} weeks. Good luck!"
            )

# ══════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════
st.divider()
st.caption("🚀 AI Career Path Analyzer · Streamlit · scikit-learn · Plotly · Web Speech API · Built with ❤️")
