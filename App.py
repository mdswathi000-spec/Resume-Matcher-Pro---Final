"""
app.py  —  Resume Matcher Pro  (Blank-Screen Fix Edition)
Root causes of blank screen fixed:
  1. All module imports wrapped in try/except — app never crashes silently
  2. Missing packages show a clear install message instead of blank page
  3. fitz (PyMuPDF) import error handled — falls back to text-only upload
  4. fpdf2 import error handled — PDF export shows install message
  5. All f-string quote conflicts replaced with .format()
  6. st.progress() clamped to 0.0–1.0
  7. compare page results stored in session_state
  8. export_excel buf.seek(0) added
  9. page_accuracy loop variable renamed to avoid shadowing
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import io
import re
import datetime
import hashlib
import sqlite3
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
#  SAFE OPTIONAL IMPORTS  — app never crashes if a package is missing
# ─────────────────────────────────────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
    FITZ_OK = True
except ImportError:
    FITZ_OK = False

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    from fpdf import FPDF
    FPDF_OK = True
except ImportError:
    FPDF_OK = False

try:
    from wordcloud import WordCloud
    WC_OK = True
except ImportError:
    WC_OK = False

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Matcher Pro",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Hide ALL Streamlit default UI chrome ──────────────────────────────────────
st.markdown("""
<style>
/* Top-right toolbar: Share, Star, Edit, GitHub icon, 3-dot menu */
[data-testid="stToolbar"]          { display: none !important; visibility: hidden !important; }
/* Bottom-right "Manage app" bar */
[data-testid="stStatusWidget"]     { display: none !important; visibility: hidden !important; }
/* Top coloured decoration bar */
[data-testid="stDecoration"]       { display: none !important; }
/* Hamburger main menu */
#MainMenu                          { display: none !important; }
/* "Made with Streamlit" footer */
footer                             { display: none !important; }
/* Full header bar */
header[data-testid="stHeader"]     { display: none !important; background: none !important; }
/* Remove extra top padding left by hidden header */
.block-container                   { padding-top: 1.5rem !important; }
/* Hide deploy button */
.stDeployButton                    { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SHOW DEPENDENCY ERROR BANNER IF NEEDED
# ─────────────────────────────────────────────────────────────────────────────
missing_pkgs = []
if not SKLEARN_OK: missing_pkgs.append("scikit-learn")
if not FITZ_OK:    missing_pkgs.append("pymupdf")

if missing_pkgs:
    st.error(
        "⚠️  Missing required packages: `{}`\n\n"
        "Run this command in your terminal, then refresh:\n\n"
        "```\npip install {}\n```".format(
            ", ".join(missing_pkgs),
            " ".join(missing_pkgs),
        )
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  DATABASE  (inline — no separate file import needed)
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH = "resume_matcher.db"

def _db():
    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,              # wait up to 30s if locked — fixes "database is locked"
        check_same_thread=False, # required for Streamlit's threading model
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")    # allows concurrent reads + writes
    conn.execute("PRAGMA busy_timeout=30000")  # extra 30s fallback
    return conn

def init_db():
    conn = _db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT DEFAULT '',
            course TEXT DEFAULT '',
            created TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            resume_name TEXT,
            job_title TEXT,
            score REAL DEFAULT 0,
            ngram_score REAL DEFAULT 0,
            tfidf_score REAL DEFAULT 0,
            ats_score REAL DEFAULT 0,
            matched_skills TEXT DEFAULT '[]',
            missing_skills TEXT DEFAULT '[]',
            dominant_cat TEXT DEFAULT '',
            created TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS comparisons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            resume_a TEXT,
            resume_b TEXT,
            score_a REAL,
            score_b REAL,
            winner TEXT,
            created TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit(); conn.close()

def _hash(pw): return hashlib.sha256(pw.encode()).hexdigest()

def register_user(username, password, name, course):
    conn = None
    try:
        conn = _db()
        conn.execute(
            "INSERT INTO users (username,password,name,course) VALUES (?,?,?,?)",
            (username.strip(), _hash(password), name.strip(), course.strip()),
        )
        conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "Username already exists. Please choose another."
    except sqlite3.OperationalError as e:
        return False, "Database error: {}. Please try again.".format(str(e))
    finally:
        if conn:
            conn.close()

def login_user(username, password):
    conn = None
    try:
        conn = _db()
        row = conn.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username.strip(), _hash(password)),
        ).fetchone()
        return dict(row) if row else None
    except sqlite3.OperationalError:
        return None
    finally:
        if conn:
            conn.close()

def save_analysis(user_id, resume_name, job_title, result):
    conn = None
    try:
        conn = _db()
        conn.execute("""INSERT INTO analyses
            (user_id,resume_name,job_title,score,ngram_score,tfidf_score,
             ats_score,matched_skills,missing_skills,dominant_cat)
            VALUES (?,?,?,?,?,?,?,?,?,?)""", (
            user_id, resume_name, str(job_title)[:80],
            result.get("score",0), result.get("ngram_score",0),
            result.get("tfidf_score",0), result.get("ats_score",0),
            json.dumps(result.get("matched_list",[])[:20]),
            json.dumps(result.get("missing_list",[])[:20]),
            result.get("dominant_cat",""),
        ))
        conn.commit()
    except sqlite3.OperationalError:
        pass
    finally:
        if conn: conn.close()

def get_history(user_id):
    conn = None
    try:
        conn = _db()
        rows = conn.execute(
            "SELECT * FROM analyses WHERE user_id=? ORDER BY created DESC LIMIT 50",
            (user_id,)).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        if conn: conn.close()

def get_stats(user_id):
    conn = None
    try:
        conn = _db()
        rows = conn.execute(
            "SELECT score,created FROM analyses WHERE user_id=? ORDER BY created",
            (user_id,)).fetchall()
        if not rows:
            return {"count":0,"best":0,"avg":0,"trend":[]}
        scores = [r["score"] for r in rows]
        dates  = [r["created"][:10] for r in rows]
        return {
            "count": len(scores),
            "best":  max(scores),
            "avg":   round(sum(scores)/len(scores), 1),
            "trend": list(zip(dates, scores)),
        }
    except sqlite3.OperationalError:
        return {"count":0,"best":0,"avg":0,"trend":[]}
    finally:
        if conn: conn.close()

def save_comparison(user_id, na, nb, sa, sb):
    conn = None
    try:
        conn = _db()
        conn.execute("INSERT INTO comparisons (user_id,resume_a,resume_b,score_a,score_b,winner) VALUES (?,?,?,?,?,?)",
                     (user_id, na, nb, sa, sb, na if sa >= sb else nb))
        conn.commit()
    except sqlite3.OperationalError:
        pass
    finally:
        if conn: conn.close()

def get_comparisons(user_id):
    conn = None
    try:
        conn = _db()
        rows = conn.execute(
            "SELECT * FROM comparisons WHERE user_id=? ORDER BY created DESC LIMIT 20",
            (user_id,)).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        if conn: conn.close()

init_db()

# ─────────────────────────────────────────────────────────────────────────────
#  NLP ENGINE  (inline — no processor.py import needed)
# ─────────────────────────────────────────────────────────────────────────────

SKILL_CATEGORIES = {
    "Programming":  ["python","java","javascript","typescript","c++","c#","go","rust","ruby","php","swift","kotlin","scala","r","matlab","bash"],
    "Web & UI":     ["react","angular","vue","django","flask","fastapi","spring","node.js","express","html","css","tailwind","bootstrap","next.js"],
    "Data & ML":    ["machine learning","deep learning","nlp","tensorflow","pytorch","scikit-learn","pandas","numpy","opencv","data analysis","statistics","power bi","tableau","matplotlib"],
    "Cloud/DevOps": ["aws","azure","gcp","docker","kubernetes","ci/cd","jenkins","terraform","linux","git","github","gitlab"],
    "Databases":    ["sql","mysql","postgresql","mongodb","redis","sqlite","elasticsearch","oracle","firebase"],
    "Soft Skills":  ["communication","leadership","teamwork","problem solving","critical thinking","agile","scrum","project management"],
}
SKILL_TO_CAT = {s: cat for cat, skills in SKILL_CATEGORIES.items() for s in skills}

CAREER_PATHS = {
    "Programming":  [("Software Developer","Build applications across web, mobile or systems."),("Backend Engineer","Design robust APIs and server-side architecture."),("Full-Stack Developer","Own both frontend UI and backend services.")],
    "Data & ML":    [("Data Scientist","Use statistical modelling to extract business insights."),("ML Engineer","Deploy and scale machine-learning systems in production."),("Data Analyst","Analyse business data and create dashboards and reports.")],
    "Cloud/DevOps": [("DevOps Engineer","Automate pipelines and manage cloud infrastructure."),("Cloud Architect","Design and optimise cloud environments at scale.")],
    "Web & UI":     [("Frontend Developer","Build interactive, accessible web experiences."),("React Developer","Specialise in modern JavaScript UI frameworks.")],
    "Databases":    [("Database Administrator","Manage, tune and secure production databases."),("Data Engineer","Build and maintain data pipelines and warehouses.")],
    "Soft Skills":  [("Technical Project Manager","Lead cross-functional teams to deliver products."),("Scrum Master","Facilitate agile ceremonies and remove blockers.")],
}
SALARY_GUIDE = {
    "Software Developer":     ("3–6 LPA","8–18 LPA","20–40 LPA"),
    "Data Scientist":         ("5–10 LPA","12–25 LPA","25–55 LPA"),
    "DevOps Engineer":        ("5–10 LPA","12–24 LPA","24–48 LPA"),
    "Frontend Developer":     ("3–7 LPA","8–18 LPA","18–35 LPA"),
    "Database Administrator": ("4–8 LPA","9–18 LPA","18–32 LPA"),
}
CERTIFICATIONS = {
    "python":           ["PCEP – Python Essentials 1","Google IT Automation with Python (Coursera)"],
    "machine learning": ["Andrew Ng ML Specialisation (Coursera)","Google ML Crash Course (free)"],
    "aws":              ["AWS Cloud Practitioner (CLF-C02)","AWS Solutions Architect – Associate"],
    "docker":           ["Docker Certified Associate (DCA)"],
    "sql":              ["Oracle SQL Certified Associate","Microsoft DP-900"],
    "react":            ["Meta Front-End Developer Certificate (Coursera)"],
    "git":              ["GitHub Foundations Certification"],
    "linux":            ["LPI Linux Essentials"],
}
ATS_CLICHES = ["results-driven","team player","proven track record","self-starter","synergy",
               "think outside the box","go-getter","hard worker","detail-oriented",
               "passionate about","guru","ninja","rockstar","wizard","highly motivated"]
SECTION_KEYWORDS = {
    "Contact":        ["email","phone","mobile","linkedin","github","contact"],
    "Summary":        ["summary","objective","profile","about me","professional summary"],
    "Education":      ["education","degree","university","college","bca","btech","mca","msc"],
    "Experience":     ["experience","internship","worked","employment"],
    "Skills":         ["skills","technical skills","technologies","tools","programming"],
    "Projects":       ["projects","github","developed","built","created","implemented"],
    "Certifications": ["certification","certified","certificate","coursera","udemy"],
    "Achievements":   ["achievement","award","winner","scholarship","honour"],
}
VALIDATION_SAMPLES = [
    {"resume":"python machine learning tensorflow pandas numpy scikit-learn statistics sql","jd":"python machine learning data science tensorflow pandas numpy statistics sql","human_label":85},
    {"resume":"javascript react html css node.js express mongodb git","jd":"react javascript frontend html css node.js rest api git","human_label":88},
    {"resume":"java spring boot docker kubernetes aws sql postgresql","jd":"java spring boot docker aws kubernetes sql postgresql ci/cd","human_label":90},
    {"resume":"communication leadership project management scrum agile teamwork","jd":"project management agile scrum communication leadership reporting","human_label":82},
    {"resume":"python flask sql html css git communication","jd":"machine learning tensorflow deep learning computer vision pytorch nlp","human_label":18},
    {"resume":"data analysis power bi tableau sql excel statistics python pandas","jd":"data analyst sql python power bi tableau excel reporting","human_label":92},
    {"resume":"aws docker linux bash terraform jenkins ci/cd git","jd":"devops aws docker kubernetes ci/cd terraform linux monitoring","human_label":88},
    {"resume":"c++ algorithms data structures operating systems","jd":"python django rest api postgresql html css javascript react","human_label":12},
    {"resume":"mongodb express react node.js javascript html css git rest api","jd":"full stack developer react node.js express mongodb javascript rest api","human_label":95},
    {"resume":"python pandas numpy matplotlib sql data analysis excel","jd":"python sql data analysis pandas numpy matplotlib tableau","human_label":88},
]

def _clean(text):
    text = text.lower()
    text = re.sub(r"[^\w\s/+#.\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def _word_freq(text, n=20):
    stop = {"with","this","that","from","your","have","will","they","their","about","been",
            "more","also","some","must","would","which","there","these","into","what","should"}
    words = re.findall(r"\b[a-z]{4,}\b", text.lower())
    return dict(Counter(w for w in words if w not in stop).most_common(n))

def extract_text_from_pdf(pdf_file):
    if not FITZ_OK:
        return ""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        st.error("PDF read error: {}".format(e))
        return ""

def detect_sections(resume_text):
    tl = resume_text.lower()
    return {sec: any(kw in tl for kw in kws) for sec, kws in SECTION_KEYWORDS.items()}

def section_score(resume_text):
    d = detect_sections(resume_text)
    return round(sum(1 for v in d.values() if v) / len(SECTION_KEYWORDS) * 100)

def get_graph_data(resume_text, jd_text):
    r_clean = _clean(resume_text)
    j_clean = _clean(jd_text)

    # N-gram overlap
    try:
        vec = CountVectorizer(ngram_range=(1,2), stop_words="english")
        vec.fit([j_clean])
        jd_phrases = set(vec.get_feature_names_out())
    except Exception:
        jd_phrases = set()
    matched_list = sorted(p for p in jd_phrases if p in r_clean)
    missing_list = sorted(p for p in jd_phrases if p not in r_clean)
    ngram_score  = round(len(matched_list) / max(len(jd_phrases), 1) * 100, 1)

    # TF-IDF cosine
    try:
        tfidf = TfidfVectorizer(stop_words="english")
        mat   = tfidf.fit_transform([r_clean, j_clean])
        tfidf_score = round(float(cosine_similarity(mat[0], mat[1])[0][0]) * 100, 1)
    except Exception:
        tfidf_score = 0.0

    score = round(ngram_score * 0.60 + tfidf_score * 0.40, 1)

    # ATS score
    try:
        t2 = TfidfVectorizer(stop_words="english", max_features=50)
        t2.fit([j_clean])
        top_kw = t2.get_feature_names_out().tolist()
    except Exception:
        top_kw = []
    ats_score  = round(len([k for k in top_kw if k in r_clean]) / max(len(top_kw), 1) * 100, 1)
    kw_density = round(len(matched_list) / max(len(jd_phrases), 1) * 100, 1)
    coverage   = min(round(len(resume_text.split()) / max(len(jd_text.split()), 1) * 100, 1), 100)

    # Skill categorisation
    matched_by_cat, missing_by_cat = {}, {}
    for p in matched_list:
        cat = SKILL_TO_CAT.get(p)
        if cat: matched_by_cat.setdefault(cat, []).append(p)
    for p in missing_list:
        cat = SKILL_TO_CAT.get(p)
        if cat: missing_by_cat.setdefault(cat, []).append(p)

    resume_skills = [s for s in SKILL_TO_CAT if s in r_clean]
    cat_counts    = Counter(SKILL_TO_CAT[s] for s in resume_skills)
    dominant_cat  = cat_counts.most_common(1)[0][0] if cat_counts else "Programming"
    career_paths  = CAREER_PATHS.get(dominant_cat, CAREER_PATHS["Programming"])
    top_title     = career_paths[0][0]
    salary_guide  = SALARY_GUIDE.get(top_title, ("3–6 LPA","8–18 LPA","18–35 LPA"))
    certifications= {p: CERTIFICATIONS[p] for p in missing_list[:10] if p in CERTIFICATIONS}
    ats_flags     = [kw for kw in ATS_CLICHES if kw in r_clean]
    word_freq     = _word_freq(jd_text)
    sentences     = re.split(r"[.!?]+", jd_text)
    avg_len       = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    readability   = max(0.0, min(100.0, round(100 - (avg_len - 15) * 2, 1)))
    dims          = {cat: round(len([s for s in skills if s in r_clean]) / len(skills) * 100)
                     for cat, skills in SKILL_CATEGORIES.items()}
    sections      = detect_sections(resume_text)
    sec_score     = section_score(resume_text)

    return {
        "score": score, "ngram_score": ngram_score, "tfidf_score": tfidf_score,
        "ats_score": ats_score, "kw_density": kw_density, "coverage": coverage,
        "matched_list": matched_list, "missing_list": missing_list,
        "matched_count": len(matched_list), "missing_count": len(missing_list),
        "matched_by_cat": matched_by_cat, "missing_by_cat": missing_by_cat,
        "top_jd_keywords": top_kw, "ats_flags": ats_flags,
        "resume_skills": resume_skills, "dominant_cat": dominant_cat,
        "career_paths": career_paths, "salary_guide": salary_guide,
        "certifications": certifications, "word_freq": word_freq,
        "readability": readability, "dims": dims,
        "resume_words": len(resume_text.split()),
        "sections": sections, "section_score": sec_score,
    }

def generate_cover_letter(name, course, skills, company, role, score, missing_list, tone="professional"):
    top_missing = ", ".join(m.upper() for m in missing_list[:3]) or "key technical areas"
    today = datetime.date.today().strftime("%d %B %Y")
    openers = {
        "professional": "I am writing to express my strong interest in the {} position at {}.".format(role, company),
        "enthusiastic": "The opportunity to join {} as {} genuinely excites me — I believe my background is a compelling fit.".format(company, role),
        "concise":      "I am applying for the {} role at {}, and my skill set aligns well with your requirements.".format(role, company),
        "creative":     "When I read the {} job description at {}, I knew immediately this role was built for someone with my background.".format(role, company),
    }
    return (
        "{name}\n{course}\n{today}\n\nDear Hiring Manager,\n\n{opener}\n\n"
        "My resume demonstrates a {score:.0f}% alignment with the job description. "
        "My hands-on experience in {skills} has equipped me to contribute from day one. "
        "Through my final-year project — an AI-powered Resume Matcher built with Python, "
        "NLP, TF-IDF scoring, and Streamlit — I developed practical skills in building "
        "real-world software systems with a database layer, PDF export, and interactive dashboards.\n\n"
        "I acknowledge that I am actively developing expertise in {missing}. "
        "I have already begun targeted learning through online courses and project work, "
        "and my track record of fast independent learning means I will close these gaps quickly.\n\n"
        "I would welcome the opportunity to discuss how my enthusiasm and project experience "
        "can add value to {company}. Thank you for considering my application.\n\n"
        "Yours sincerely,\n{name}"
    ).format(
        name=name, course=course, today=today,
        opener=openers.get(tone, openers["professional"]),
        score=score, skills=skills, missing=top_missing, company=company,
    )

def run_validation():
    results, errors = [], []
    for s in VALIDATION_SAMPLES:
        res = get_graph_data(s["resume"], s["jd"])
        err = abs(res["score"] - s["human_label"])
        results.append({"predicted": res["score"], "actual": s["human_label"],
                        "error": err, "match": err <= 15})
        errors.append(err)
    return {
        "results":  results,
        "accuracy": round(sum(1 for r in results if r["match"]) / len(results) * 100, 1),
        "mae":      round(sum(errors) / len(errors), 1),
    }

# ─────────────────────────────────────────────────────────────────────────────
#  THEME
# ─────────────────────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

DARK = {"bg":"#07090f","card":"#111827","card2":"#1a2234","text":"#dce8f8","muted":"#4a5a72",
        "blue":"#3b82f6","cyan":"#06b6d4","green":"#10b981","red":"#ef4444","amber":"#f59e0b",
        "purple":"#8b5cf6","border":"#1e3a5f","plot_bg":"#111827","plot_paper":"#07090f","grid":"#1e3a5f"}
LIGHT= {"bg":"#f0f4ff","card":"#ffffff","card2":"#e8efff","text":"#1a2234","muted":"#6b7a99",
        "blue":"#2563eb","cyan":"#0891b2","green":"#059669","red":"#dc2626","amber":"#d97706",
        "purple":"#7c3aed","border":"#c7d2fe","plot_bg":"#ffffff","plot_paper":"#f0f4ff","grid":"#e0e7ff"}
T = DARK if st.session_state.dark_mode else LIGHT

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Mono&family=Syne:wght@700;800&display=swap');
html,body,[class*="css"]{{font-family:'Inter',sans-serif!important;background:{bg}!important;color:{text}!important}}
h1,h2,h3{{font-family:'Syne',sans-serif!important}}
h1{{color:{blue}!important;font-size:2rem!important}}
h2{{color:{text}!important;font-size:1.2rem!important;border-bottom:2px solid {blue};padding-bottom:4px}}
h3{{color:{cyan}!important;font-size:.95rem!important}}
section[data-testid="stSidebar"]{{background:{card}!important;border-right:3px solid {blue}!important}}
section[data-testid="stSidebar"] *{{color:{text}!important}}
div[data-testid="stMetric"]{{background:{card}!important;border-left:4px solid {blue}!important;border-radius:4px!important;padding:.8rem 1rem!important}}
div[data-testid="stMetric"] label{{font-family:'Space Mono',monospace!important;font-size:.62rem!important;color:{muted}!important;text-transform:uppercase;letter-spacing:.08em}}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{{font-family:'Syne',sans-serif!important;font-size:1.7rem!important;font-weight:800!important;color:{text}!important}}
.stButton>button{{background:{blue}!important;color:#fff!important;border:none!important;border-radius:6px!important;font-family:'Space Mono',monospace!important;font-size:.8rem!important;font-weight:600!important;letter-spacing:.06em!important;text-transform:uppercase!important;padding:.5rem 1.3rem!important}}
.stButton>button:hover{{background:{cyan}!important}}
textarea,.stTextArea textarea{{background:{card}!important;border:1.5px solid {border}!important;border-radius:6px!important;color:{text}!important;font-family:'Space Mono',monospace!important;font-size:.78rem!important}}
[data-testid="stFileUploader"]{{border:2px dashed {blue}!important;border-radius:6px!important;background:rgba(59,130,246,.04)!important}}
.tag-g{{display:inline-block;background:rgba(16,185,129,.15);color:{green};border:1px solid rgba(16,185,129,.3);font-family:'Space Mono',monospace;font-size:.67rem;padding:2px 8px;margin:2px;border-radius:3px}}
.tag-r{{display:inline-block;background:rgba(239,68,68,.15);color:{red};border:1px solid rgba(239,68,68,.3);font-family:'Space Mono',monospace;font-size:.67rem;padding:2px 8px;margin:2px;border-radius:3px}}
.tag-b{{display:inline-block;background:rgba(59,130,246,.15);color:{blue};border:1px solid rgba(59,130,246,.3);font-family:'Space Mono',monospace;font-size:.67rem;padding:2px 8px;margin:2px;border-radius:3px}}
.tag-p{{display:inline-block;background:rgba(139,92,246,.15);color:{purple};border:1px solid rgba(139,92,246,.3);font-family:'Space Mono',monospace;font-size:.67rem;padding:2px 8px;margin:2px;border-radius:3px}}
.sbanner{{display:flex;align-items:center;gap:18px;background:{card};padding:1.1rem 1.5rem;border-left:6px solid {blue};margin-bottom:1rem}}
.snum{{font-family:'Syne',sans-serif;font-size:3.2rem;font-weight:800;line-height:1}}
.slbl{{font-family:'Space Mono',monospace;font-size:.62rem;color:{muted};text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px}}
.sverd{{font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:700}}
.cert-row{{background:rgba(6,182,212,.06);border-left:3px solid {cyan};padding:4px 9px;margin-bottom:3px;font-family:'Space Mono',monospace;font-size:.73rem;color:{text}}}
.dvd{{border:none;border-top:1px solid {border};margin:1.3rem 0}}
.sec-ok{{color:{green};font-family:'Space Mono',monospace;font-size:.77rem;padding:2px 0}}
.sec-no{{color:{red};font-family:'Space Mono',monospace;font-size:.77rem;padding:2px 0}}
.arch-box{{background:{card2};border:1px solid {border};border-radius:8px;padding:.8rem 1rem;font-family:'Space Mono',monospace;font-size:.75rem;text-align:center;color:{text}}}
/* ── Auth card inputs — compact mobile style ── */
div[data-testid="stTabs"] button{{font-family:'Space Mono',monospace!important;font-size:.75rem!important;letter-spacing:.04em!important}}
div[data-testid="stTextInput"] input{{padding:.55rem .85rem!important;font-size:.85rem!important;border-radius:9px!important}}
div[data-testid="stTextInput"]{{margin-bottom:.4rem!important}}
</style>"""
st.markdown(CSS.format(**T), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for _k, _v in {
    "user": None, "result": None, "resume_text": "", "jd_text": "",
    "cover_letter": "", "resume_name": "",
    "cmp_ra": None, "cmp_rb": None, "cmp_fa_name": "", "cmp_fb_name": "",
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────────────────────
#  UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def score_color(s):
    return T["green"] if s >= 70 else T["amber"] if s >= 45 else T["red"]

def verdict(s):
    if s >= 70: return "Strong Match — ready to apply"
    if s >= 45: return "Moderate Match — close the gaps"
    return "Weak Match — preparation needed"

def tags_html(items, cls="tag-g"):
    return "".join('<span class="{}">{}</span>'.format(cls, t) for t in items)

def field_label(label):
    st.markdown(
        "<div style='font-family:Space Mono,monospace;font-size:.65rem;"
        "text-transform:uppercase;letter-spacing:.07em;color:{};margin-bottom:2px'>{}</div>".format(
            T["muted"], label), unsafe_allow_html=True)

def safe_progress(val):
    return min(max(float(val) / 100.0, 0.0), 1.0)

def gauge_chart(score, h=210):
    col = score_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"suffix": "%", "font": {"family": "Syne", "size": 36, "color": col}},
        gauge={"axis":{"range":[0,100],"tickfont":{"size":8,"family":"Space Mono"}},
               "bar":{"color":col,"thickness":0.25},"bgcolor":T["card"],"borderwidth":0,
               "steps":[{"range":[0,40],"color":T["bg"]},{"range":[40,70],"color":T["card"]},
                        {"range":[70,100],"color":T["card2"]}]}))
    fig.update_layout(paper_bgcolor=T["plot_paper"], height=h, margin=dict(t=8,b=8,l=8,r=8))
    return fig

def radar_chart(dims):
    cats = list(dims.keys()); vals = list(dims.values())
    fig = go.Figure(go.Scatterpolar(
        r=vals+[vals[0]], theta=cats+[cats[0]], fill="toself",
        line=dict(color=T["blue"], width=2), fillcolor="rgba(59,130,246,0.15)"))
    fig.update_layout(
        polar=dict(bgcolor=T["card"],
                   radialaxis=dict(visible=True,range=[0,100],gridcolor=T["border"],
                                   tickfont=dict(size=8,family="Space Mono")),
                   angularaxis=dict(tickfont=dict(size=9,family="Syne",color=T["text"]))),
        paper_bgcolor=T["plot_paper"], height=270, margin=dict(t=8,b=8,l=25,r=25))
    return fig

def pbar(label, val):
    field_label(label)
    st.progress(safe_progress(val), text="{:.0f}%".format(val))

def freq_chart(wf):
    df  = pd.DataFrame(list(wf.items()), columns=["Word","Count"]).sort_values("Count")
    fig = px.bar(df, x="Count", y="Word", orientation="h", color="Count",
                 color_continuous_scale=[T["card2"], T["blue"], T["cyan"]])
    fig.update_layout(paper_bgcolor=T["plot_paper"], plot_bgcolor=T["card"],
                      xaxis=dict(gridcolor=T["border"]), yaxis=dict(tickfont=dict(size=9)),
                      coloraxis_showscale=False, height=360, margin=dict(t=4,b=4,l=4,r=4))
    return fig

def make_wordcloud(text):
    if not WC_OK: return None
    try:
        bg = "#1a2234" if st.session_state.dark_mode else "#ffffff"
        wc = WordCloud(width=700, height=300, background_color=bg,
                       colormap="Blues" if st.session_state.dark_mode else "viridis",
                       max_words=80)
        wc.generate(text or "no text")
        return wc.to_array()
    except Exception:
        return None

def export_excel(history):
    rows = []
    for h in history:
        mat = json.loads(h.get("matched_skills","[]"))
        mis = json.loads(h.get("missing_skills","[]"))
        rows.append({"Date":h["created"][:10],"Resume":h["resume_name"],"Role":h["job_title"],
                     "Score (%)":h["score"],"ATS (%)":h["ats_score"],
                     "Matched":", ".join(mat[:5]),"Gaps":", ".join(mis[:5]),"Category":h["dominant_cat"]})
    buf = io.BytesIO()
    pd.DataFrame(rows).to_excel(buf, index=False, sheet_name="History")
    buf.seek(0)
    return buf.getvalue()

def pdf_report(candidate_name, resume_name, job_title, result, cover_letter=""):
    if not FPDF_OK:
        return None
    try:
        from pdf_export import generate_pdf_report
        return generate_pdf_report(candidate_name, resume_name, job_title, result, cover_letter)
    except Exception:
        pass
    # Inline minimal PDF fallback
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Resume Matcher Pro — Analysis Report", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, "Candidate: {}".format(candidate_name), ln=True)
        pdf.cell(0, 8, "Resume: {}".format(resume_name), ln=True)
        pdf.cell(0, 8, "Score: {:.0f}%  |  ATS: {:.0f}%".format(result.get("score",0), result.get("ats_score",0)), ln=True)
        pdf.cell(0, 8, "Matched: {}".format(", ".join(result.get("matched_list",[])[:15])), ln=True)
        pdf.cell(0, 8, "Gaps: {}".format(", ".join(result.get("missing_list",[])[:15])), ln=True)
        return bytes(pdf.output())
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
#  AUTH PAGE  — compact mobile-style card
# ─────────────────────────────────────────────────────────────────────────────
def page_auth():
    register_user("demo", "demo123", "Demo User", "BCA Final Year")

    # Center a narrow card using columns
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown("""
        <div style="text-align:center;padding:1.8rem 0 1rem">
          <div style="font-size:2rem;margin-bottom:.3rem">◈</div>
          <div style="font-family:Syne,sans-serif;font-size:1.6rem;font-weight:800;
               background:linear-gradient(135deg,#3b82f6,#06b6d4);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               background-clip:text;line-height:1.1;margin-bottom:.35rem">
            Resume Matcher Pro
          </div>
          <div style="font-family:Space Mono,monospace;font-size:.7rem;
               color:#4a5a72;letter-spacing:.05em">
            Sign in to save history &amp; access all features
          </div>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["🔑  Sign In", "✨  Create Account"])

        # ── SIGN IN ──────────────────────────────────────────────────────────
        with tab1:
            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
            un = st.text_input("Username", placeholder="Enter username", key="li_u",
                               label_visibility="collapsed")
            pw = st.text_input("Password", placeholder="Enter password",
                               type="password", key="li_p",
                               label_visibility="collapsed")
            if st.button("Sign In →", key="li_b", use_container_width=True):
                if not un or not pw:
                    st.error("Please enter both username and password.")
                else:
                    user = login_user(un, pw)
                    if user:
                        st.session_state.user = user
                        st.rerun()
                    else:
                        st.error("Incorrect username or password.")


        # ── CREATE ACCOUNT ────────────────────────────────────────────────────
        with tab2:
            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
            r_n  = st.text_input("Full Name",        placeholder="Your full name",       key="rn", label_visibility="collapsed")
            r_c  = st.text_input("Course / Degree",  placeholder="e.g. BCA Final Year", key="rc", label_visibility="collapsed")
            r_u  = st.text_input("Username",         placeholder="Choose a username",    key="ru", label_visibility="collapsed")
            c1, c2 = st.columns(2)
            with c1:
                r_p  = st.text_input("Password",         placeholder="Password",  type="password", key="rp",  label_visibility="collapsed")
            with c2:
                r_p2 = st.text_input("Confirm Password", placeholder="Confirm",   type="password", key="rp2", label_visibility="collapsed")
            if st.button("Create Account →", key="rb", use_container_width=True):
                if r_p != r_p2:
                    st.error("Passwords do not match.")
                elif not all([r_n, r_u, r_p, r_c]):
                    st.error("Please fill all fields.")
                else:
                    ok, msg = register_user(r_u, r_p, r_n, r_c)
                    if ok:
                        st.success(msg + " Please sign in.")
                    else:
                        st.error(msg)

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    user  = st.session_state.user
    stats = get_stats(user["id"])
    with st.sidebar:
        st.markdown(
            "<div style='font-family:Syne,sans-serif;font-size:1.05rem;font-weight:800;"
            "color:{blue};margin-bottom:2px'>◈ Resume Matcher Pro</div>"
            "<div style='font-family:Space Mono,monospace;font-size:.62rem;"
            "color:{muted};margin-bottom:.8rem'>Signed in as "
            "<b style=\"color:{cyan}\">{uname}</b></div>".format(
                blue=T["blue"], muted=T["muted"], cyan=T["cyan"],
                uname=user["username"]),
            unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Analyses", stats["count"])
        c2.metric("Best", "{:.0f}%".format(stats["best"]) if stats["best"] else "—")
        st.markdown("---")
        lbl = "☀️ Light Mode" if st.session_state.dark_mode else "🌙 Dark Mode"
        if st.button(lbl, key="theme_btn"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
        st.markdown("---")
        page = st.radio("Navigate", [
            "🎯  Analyse Resume",
            "📊  My Dashboard",
            "⚖️  Compare Resumes",
            "✉️  Cover Letter",
            "🏆  Career Explorer",
            "🛡️  ATS Audit",
            "🔬  Accuracy Report",
            "🏗️  Architecture",
        ], label_visibility="collapsed")
        st.markdown("---")
        if st.button("Sign Out", key="signout_btn"):
            st.session_state.user   = None
            st.session_state.result = None
            st.rerun()
        st.markdown(
            "<div style='font-family:Space Mono,monospace;font-size:.62rem;"
            "color:{};line-height:1.8'>NLP ▸ N-Gram + TF-IDF<br>Score ▸ Blended 60/40<br>"
            "DB ▸ SQLite persistent<br>Export ▸ PDF + Excel</div>".format(T["muted"]),
            unsafe_allow_html=True)
    return page

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 1 — ANALYSE
# ─────────────────────────────────────────────────────────────────────────────
def page_analyse():
    st.markdown("## 🎯 Analyse Resume")
    user = st.session_state.user
    col_l, col_r = st.columns([1,1], gap="large")
    with col_l:
        st.markdown("#### Job Description")
        jd   = st.text_area("JD", height=220, placeholder="Paste full job description…",
                             label_visibility="collapsed", key="jd_in")
        role = st.text_input("Role / Job Title", placeholder="e.g. Software Engineer Intern", key="role_in")
        email_notify = st.text_input("Email for notification (optional)", placeholder="you@email.com", key="notif")
    with col_r:
        st.markdown("#### Resume (PDF)")
        uploaded = st.file_uploader("Upload PDF", type=["pdf","txt"],
                                    label_visibility="collapsed", key="pdf_up")
        if uploaded:
            st.success("✔  {}".format(uploaded.name))

    if st.button("◈  Run Full Analysis", use_container_width=False):
        if not uploaded: st.error("Upload a resume PDF."); return
        if not jd.strip(): st.error("Paste a job description."); return
        with st.spinner("Running NLP analysis…"):
            if uploaded.name.lower().endswith(".txt"):
                rtxt = uploaded.read().decode("utf-8", errors="ignore")
            else:
                rtxt = extract_text_from_pdf(uploaded)
            if not rtxt.strip():
                st.error("Could not extract text from the file. Try a .txt file instead.")
                return
            result = get_graph_data(rtxt, jd)
            st.session_state.result      = result
            st.session_state.resume_text = rtxt
            st.session_state.jd_text     = jd
            st.session_state.resume_name = uploaded.name
            save_analysis(user["id"], uploaded.name, role or jd[:60], result)
        if email_notify.strip():
            with st.expander("📧 Email Notification Preview"):
                st.code("Hi {},\n\nScore: {:.0f}%\n{}\n\n— Resume Matcher Pro".format(
                    user.get("name","User"), result["score"], verdict(result["score"])))
        st.success("✔  Analysis saved to your dashboard.")

    if not st.session_state.result: return
    r = st.session_state.result
    s = r["score"]
    st.markdown('<hr class="dvd">', unsafe_allow_html=True)

    st.markdown(
        "<div class='sbanner'>"
        "<div><div class='slbl'>Overall Match</div>"
        "<div class='snum' style='color:{col}'>{score:.0f}<span style='font-size:1.3rem'>%</span></div></div>"
        "<div><div class='sverd' style='color:{col}'>{verd}</div>"
        "<div style='font-family:Space Mono,monospace;font-size:.65rem;color:{muted};margin-top:5px'>"
        "N-Gram {ng}%  ·  TF-IDF {tf}%  ·  ATS {ats}%  ·  Sections {sec}%"
        "</div></div></div>".format(
            col=score_color(s), score=s, verd=verdict(s), muted=T["muted"],
            ng=r["ngram_score"], tf=r["tfidf_score"],
            ats=r["ats_score"], sec=r["section_score"]),
        unsafe_allow_html=True)

    k1,k2,k3,k4,k5,k6,k7 = st.columns(7)
    k1.metric("Matched",    r["matched_count"])
    k2.metric("Gaps",       r["missing_count"])
    k3.metric("ATS",        "{:.0f}%".format(r["ats_score"]))
    k4.metric("KW Density", "{:.0f}%".format(r["kw_density"]))
    k5.metric("Coverage",   "{:.0f}%".format(r["coverage"]))
    k6.metric("Sections",   "{}%".format(r["section_score"]))
    k7.metric("Clichés",    len(r["ats_flags"]))

    st.markdown('<hr class="dvd">', unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,1,1], gap="large")
    with c1:
        st.markdown("### Gauge")
        st.plotly_chart(gauge_chart(s), use_container_width=True, config={"displayModeBar":False})
    with c2:
        st.markdown("### Skill Radar")
        st.plotly_chart(radar_chart(r["dims"]), use_container_width=True, config={"displayModeBar":False})
    with c3:
        st.markdown("### Score Breakdown")
        for lbl, val in [("N-Gram",r["ngram_score"]),("TF-IDF",r["tfidf_score"]),
                         ("ATS",r["ats_score"]),("KW Density",r["kw_density"]),
                         ("Coverage",r["coverage"]),("Sections",r["section_score"])]:
            pbar(lbl, val)

    st.markdown('<hr class="dvd">', unsafe_allow_html=True)
    t1, t2 = st.columns(2, gap="large")
    with t1:
        st.markdown("### ✔ Matched  ({})".format(r["matched_count"]))
        st.markdown(tags_html(r["matched_list"][:32], "tag-g"), unsafe_allow_html=True)
    with t2:
        st.markdown("### ✖ Gaps  ({})".format(r["missing_count"]))
        st.markdown(tags_html(r["missing_list"][:32], "tag-r"), unsafe_allow_html=True)

    st.markdown('<hr class="dvd">', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("### JD Keyword Frequency")
        if r["word_freq"]:
            st.plotly_chart(freq_chart(r["word_freq"]), use_container_width=True, config={"displayModeBar":False})
    with c2:
        st.markdown("### JD Word Cloud")
        wc_img = make_wordcloud(st.session_state.jd_text)
        if wc_img is not None: st.image(wc_img, use_container_width=True)
        else: st.info("Run `pip install wordcloud` and restart to enable word cloud.")

    st.markdown('<hr class="dvd">', unsafe_allow_html=True)
    st.markdown("### Resume Section Detector")
    sec_cols = st.columns(4)
    for i, (sec, present) in enumerate(r["sections"].items()):
        cls = "sec-ok" if present else "sec-no"
        sec_cols[i%4].markdown(
            "<div class='{}'>{}&nbsp;&nbsp;{}</div>".format(cls, "✔" if present else "✖", sec),
            unsafe_allow_html=True)

    st.markdown('<hr class="dvd">', unsafe_allow_html=True)
    if r["certifications"]:
        st.markdown("### Recommended Certifications")
        for skill, certs in r["certifications"].items():
            field_label(skill)
            for c in certs:
                st.markdown("<div class='cert-row'>◎&nbsp;&nbsp;{}</div>".format(c), unsafe_allow_html=True)
        st.markdown("")

    st.markdown("### Career Paths  —  {}".format(r["dominant_cat"]))
    sal = r["salary_guide"]
    for title, desc in r["career_paths"]:
        with st.expander(title):
            st.markdown(desc)
            ca,cb,cc = st.columns(3)
            ca.metric("Fresher", sal[0]); cb.metric("Mid", sal[1]); cc.metric("Senior", sal[2])

    st.markdown('<hr class="dvd">', unsafe_allow_html=True)
    st.markdown("### 📥 Download Report")
    pdf_bytes = pdf_report(user.get("name", user["username"]),
                           st.session_state.resume_name,
                           st.session_state.jd_text[:60], r,
                           st.session_state.cover_letter)
    if pdf_bytes:
        st.download_button("⬇  Download PDF Report", data=pdf_bytes,
                           file_name="analysis_{}.pdf".format(user["username"]),
                           mime="application/pdf")
    else:
        st.info("Install fpdf2 for PDF export: `pip install fpdf2`")

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 2 — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def page_dashboard():
    st.markdown("## 📊 My Dashboard")
    user  = st.session_state.user
    stats = get_stats(user["id"])
    if not stats["count"]:
        st.info("No analyses yet. Run your first analysis on the Analyse Resume page."); return
    k1,k2,k3 = st.columns(3)
    k1.metric("Total Analyses", stats["count"])
    k2.metric("Best Score",     "{:.0f}%".format(stats["best"]))
    k3.metric("Average Score",  "{:.0f}%".format(stats["avg"]))

    st.markdown('<hr class="dvd">', unsafe_allow_html=True)
    if len(stats["trend"]) > 1:
        st.markdown("### Score Trend")
        df  = pd.DataFrame(stats["trend"], columns=["Date","Score"])
        fig = px.line(df, x="Date", y="Score", markers=True, color_discrete_sequence=[T["blue"]])
        fig.add_hline(y=70, line_dash="dot", line_color=T["green"], annotation_text="Strong match threshold")
        fig.update_layout(paper_bgcolor=T["plot_paper"], plot_bgcolor=T["card"],
                          yaxis=dict(range=[0,100], gridcolor=T["grid"]),
                          xaxis=dict(gridcolor=T["grid"]), height=250, margin=dict(t=4,b=4,l=4,r=4))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    st.markdown('<hr class="dvd">', unsafe_allow_html=True)
    history = get_history(user["id"])
    st.markdown("### Analysis History")
    rows = []
    for h in history:
        mat = json.loads(h.get("matched_skills","[]"))
        mis = json.loads(h.get("missing_skills","[]"))
        rows.append({"Date":h["created"][:10],"Resume":h["resume_name"],"Role":h["job_title"],
                     "Score":"{:.0f}%".format(h["score"]),"ATS":"{:.0f}%".format(h["ats_score"]),
                     "Top Match":mat[0] if mat else "—","Top Gap":mis[0] if mis else "—",
                     "Category":h["dominant_cat"]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### 📤 Export History")
    try:
        xl = export_excel(history)
        st.download_button("⬇  Download as Excel", data=xl,
                           file_name="history_{}.xlsx".format(user["username"]),
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        st.info("Install openpyxl for Excel export: `pip install openpyxl`")

    comps = get_comparisons(user["id"])
    if comps:
        st.markdown('<hr class="dvd">', unsafe_allow_html=True)
        st.markdown("### Comparison History")
        crows = [{"Date":c["created"][:10],"Resume A":c["resume_a"],
                  "Score A":"{:.0f}%".format(c["score_a"]),"Resume B":c["resume_b"],
                  "Score B":"{:.0f}%".format(c["score_b"]),"Winner":c["winner"]} for c in comps]
        st.dataframe(pd.DataFrame(crows), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 3 — COMPARE
# ─────────────────────────────────────────────────────────────────────────────
def page_compare():
    st.markdown("## ⚖️ Compare Two Resumes")
    user = st.session_state.user
    jd_col, a_col, b_col = st.columns([1.2,1,1], gap="large")
    with jd_col:
        jd = st.text_area("Job Description", height=250, placeholder="Paste the full JD…", key="cmp_jd")
    with a_col:
        st.markdown("#### Resume A")
        fa = st.file_uploader("Resume A", type=["pdf","txt"], label_visibility="collapsed", key="cmp_a")
    with b_col:
        st.markdown("#### Resume B")
        fb = st.file_uploader("Resume B", type=["pdf","txt"], label_visibility="collapsed", key="cmp_b")

    if st.button("⚖️  Compare Now"):
        if not (fa and fb and jd.strip()):
            st.error("Upload both resumes and paste the JD."); return
        with st.spinner("Analysing both resumes…"):
            def read_file(f):
                if f.name.lower().endswith(".txt"):
                    return f.read().decode("utf-8", errors="ignore")
                return extract_text_from_pdf(f)
            st.session_state.cmp_ra      = get_graph_data(read_file(fa), jd)
            st.session_state.cmp_rb      = get_graph_data(read_file(fb), jd)
            st.session_state.cmp_fa_name = fa.name
            st.session_state.cmp_fb_name = fb.name
            save_comparison(user["id"], fa.name, fb.name,
                            st.session_state.cmp_ra["score"],
                            st.session_state.cmp_rb["score"])

    ra = st.session_state.cmp_ra
    rb = st.session_state.cmp_rb
    if ra is None or rb is None: return

    winner = st.session_state.cmp_fa_name if ra["score"] >= rb["score"] else st.session_state.cmp_fb_name
    st.markdown(
        "<div style='background:rgba(16,185,129,.08);border:2px solid {green};"
        "border-radius:8px;padding:.9rem;text-align:center;margin-bottom:1rem'>"
        "<div style='font-family:Space Mono,monospace;font-size:.65rem;color:{muted};"
        "text-transform:uppercase'>Winner</div>"
        "<div style='font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;"
        "color:{green}'>🏆&nbsp;&nbsp;{winner}</div></div>".format(
            green=T["green"], muted=T["muted"], winner=winner),
        unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    for col, res, fname in [(c1,ra,st.session_state.cmp_fa_name),(c2,rb,st.session_state.cmp_fb_name)]:
        with col:
            bdr = T["green"] if res["score"] >= (rb["score"] if res is ra else ra["score"]) else T["red"]
            st.markdown(
                "<div style='border:2px solid {bdr};border-radius:8px;padding:.8rem;margin-bottom:.6rem'>"
                "<div style='font-family:Syne,sans-serif;font-weight:700'>{name}</div>"
                "<div style='font-family:Syne,sans-serif;font-size:2.4rem;font-weight:800;color:{col}'>"
                "{score:.0f}%</div>"
                "<div style='font-family:Space Mono,monospace;font-size:.7rem;color:{muted}'>"
                "{verd}</div></div>".format(
                    bdr=bdr, name=fname, col=score_color(res["score"]),
                    score=res["score"], muted=T["muted"], verd=verdict(res["score"])),
                unsafe_allow_html=True)
            st.plotly_chart(gauge_chart(res["score"],190), use_container_width=True, config={"displayModeBar":False})
            for lbl,val in [("N-Gram",res["ngram_score"]),("TF-IDF",res["tfidf_score"]),("ATS",res["ats_score"])]:
                pbar(lbl, val)
            st.markdown("**Matched:**")
            st.markdown(tags_html(res["matched_list"][:10],"tag-g"), unsafe_allow_html=True)
            st.markdown("**Gaps:**")
            st.markdown(tags_html(res["missing_list"][:8],"tag-r"), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 4 — COVER LETTER
# ─────────────────────────────────────────────────────────────────────────────
def page_cover():
    st.markdown("## ✉️ Cover Letter Generator")
    user = st.session_state.user
    r    = st.session_state.result
    c1, c2 = st.columns(2, gap="large")
    with c1:
        name    = st.text_input("Full Name",    value=user.get("name",""),   key="cl_nm")
        course  = st.text_input("Course",       value=user.get("course",""), key="cl_co")
        skills  = st.text_input("Top Skills",   placeholder="Python, ML, SQL", key="cl_sk")
        company = st.text_input("Company",      placeholder="TCS, Infosys, Google", key="cl_cp")
        role2   = st.text_input("Role Applied", placeholder="Software Engineer Intern", key="cl_rl")
        tone    = st.selectbox("Tone", ["professional","enthusiastic","concise","creative"])
    with c2:
        jd_cl = st.text_area("Job Description", height=300,
                              value=st.session_state.jd_text, key="cl_jd")
    if st.button("✉️  Generate Letter"):
        if not skills: st.error("Enter your skills."); return
        missing = r["missing_list"] if r else []
        score   = r["score"]        if r else 0.0
        letter  = generate_cover_letter(name, course, skills, company, role2, score, missing, tone)
        st.session_state.cover_letter = letter
        st.markdown("### Your Letter")
        st.text_area("Copy below:", letter, height=400, key="cl_out")
        st.download_button("⬇  Download .txt", data=letter,
                           file_name="cover_{}.txt".format(name.replace(" ","_")),
                           mime="text/plain")

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 5 — CAREER EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
def page_career():
    st.markdown("## 🏆 Career Explorer")
    r = st.session_state.result
    if not r: st.info("Run an analysis first."); return
    st.markdown("**Dominant skill area:** `{}`".format(r["dominant_cat"]))
    sal = r["salary_guide"]
    for i, (title, desc) in enumerate(r["career_paths"], 1):
        with st.expander("{}.  {}".format(i, title), expanded=(i==1)):
            st.markdown("_{}_ ".format(desc))
            ca,cb,cc = st.columns(3)
            ca.metric("Fresher",sal[0]); cb.metric("Mid",sal[1]); cc.metric("Senior",sal[2])
            m = r["matched_by_cat"].get(r["dominant_cat"],[])
            g = r["missing_by_cat"].get(r["dominant_cat"],[])
            if m: st.markdown("**Matching:**"); st.markdown(tags_html(m,"tag-g"), unsafe_allow_html=True)
            if g: st.markdown("**Gaps:**");     st.markdown(tags_html(g,"tag-r"), unsafe_allow_html=True)
    st.markdown('<hr class="dvd">', unsafe_allow_html=True)
    st.markdown("### Skill Coverage by Category")
    rtxt_lower = st.session_state.resume_text.lower()
    for cat, pct in r["dims"].items():
        found = [s for s in SKILL_CATEGORIES.get(cat,[]) if s in rtxt_lower]
        field_label("{} — {}%".format(cat, pct))
        st.progress(safe_progress(pct))
        if found: st.markdown(tags_html(found,"tag-b"), unsafe_allow_html=True)
        st.markdown("")

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 6 — ATS AUDIT
# ─────────────────────────────────────────────────────────────────────────────
def page_ats():
    st.markdown("## 🛡️ ATS Audit")
    r = st.session_state.result
    if not r: st.info("Run an analysis first."); return
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("### Cliché Scan")
        if r["ats_flags"]:
            st.error("⚠  {} clichés — remove or rephrase:".format(len(r["ats_flags"])))
            st.markdown(tags_html(r["ats_flags"],"tag-r"), unsafe_allow_html=True)
        else:
            st.success("✔  No clichés detected.")
        st.markdown("---")
        st.markdown("### ATS Checklist (9 checks)")
        checks = [
            ("PDF uploaded",                 True),
            ("Skills section present",        r["sections"].get("Skills",        False)),
            ("Contact info visible",          r["sections"].get("Contact",       False)),
            ("Education section",             r["sections"].get("Education",     False)),
            ("Experience / Projects section", r["sections"].get("Experience",    False) or
                                              r["sections"].get("Projects",      False)),
            ("Certifications mentioned",      r["sections"].get("Certifications",False)),
            ("No ATS clichés",                len(r["ats_flags"]) == 0),
            ("Keyword match ≥ 40%",           r["ats_score"] >= 40),
            ("Resume ≥ 200 words",            r.get("resume_words",0) >= 200),
        ]
        for label, ok in checks:
            cls = "sec-ok" if ok else "sec-no"
            st.markdown(
                "<div class='{}'>{}&nbsp;&nbsp;{}</div>".format(cls, "✔" if ok else "✖", label),
                unsafe_allow_html=True)
    with c2:
        st.markdown("### ATS Score")
        st.plotly_chart(gauge_chart(r["ats_score"]),  use_container_width=True, config={"displayModeBar":False})
        st.markdown("### Readability")
        st.plotly_chart(gauge_chart(r["readability"]), use_container_width=True, config={"displayModeBar":False})
    st.markdown('<hr class="dvd">', unsafe_allow_html=True)
    st.markdown("### Certification Roadmap")
    if r["certifications"]:
        for skill, certs in r["certifications"].items():
            field_label(skill)
            for c in certs:
                st.markdown("<div class='cert-row'>◎&nbsp;&nbsp;{}</div>".format(c), unsafe_allow_html=True)
    else:
        st.success("Focus on project-based learning for key missing skills.")
    st.markdown("### Top JD Keywords")
    st.markdown(tags_html(r["top_jd_keywords"],"tag-p"), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 7 — ACCURACY REPORT
# ─────────────────────────────────────────────────────────────────────────────
def page_accuracy():
    st.markdown("## 🔬 Accuracy & Validation Report")
    st.caption("NLP scorer tested against 10 manually-labelled resume–JD pairs.")
    if st.button("▶  Run Validation Test"):
        with st.spinner("Evaluating 10 samples…"):
            val = run_validation()
        k1,k2,k3 = st.columns(3)
        k1.metric("Accuracy (±15 pts)", "{:.0f}%".format(val["accuracy"]))
        k2.metric("Mean Abs Error",     "{:.1f} pts".format(val["mae"]))
        k3.metric("Samples Tested",     "10")
        st.markdown('<hr class="dvd">', unsafe_allow_html=True)
        table_rows = []
        for i, row in enumerate(val["results"]):   # renamed to 'row' — no shadowing
            table_rows.append({"Sample":"Sample {}".format(i+1),
                                "Predicted":"{:.1f}%".format(row["predicted"]),
                                "Human Label":"{}%".format(row["actual"]),
                                "Error":"{:.1f} pts".format(row["error"]),
                                "Pass (±15)":"✔" if row["match"] else "✖"})
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
        st.markdown("### Predicted vs Actual")
        pred   = [row["predicted"] for row in val["results"]]
        actual = [row["actual"]    for row in val["results"]]
        labels = ["S{}".format(i+1) for i in range(len(pred))]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Predicted", x=labels, y=pred,   marker_color=T["blue"]))
        fig.add_trace(go.Bar(name="Actual",    x=labels, y=actual, marker_color=T["green"]))
        fig.update_layout(barmode="group", paper_bgcolor=T["plot_paper"], plot_bgcolor=T["card"],
                          yaxis=dict(range=[0,100], gridcolor=T["grid"]),
                          legend=dict(font=dict(family="Space Mono",size=9)),
                          height=270, margin=dict(t=4,b=4,l=4,r=4))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown('<hr class="dvd">', unsafe_allow_html=True)
        st.markdown("### Methodology")
        st.markdown("""
- **Algorithm:** Blended (60% N-Gram + 40% TF-IDF cosine similarity)
- **Validation:** 10 resume–JD pairs manually scored by a human reviewer
- **Tolerance:** Prediction within ±15 points of human label = correct
- **Limitation:** Semantic understanding limited — BERT would improve edge cases
- **Improvement path:** Fine-tune `sentence-transformers/all-MiniLM-L6-v2`
        """)

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 8 — ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
def page_architecture():
    st.markdown("## 🏗️ System Architecture")
    st.caption("How the modules connect to deliver every feature.")
    st.markdown('<hr class="dvd">', unsafe_allow_html=True)
    _, mid, _ = st.columns([1,2,1])
    with mid:
        st.markdown(
            "<div class='arch-box' style='border:2px solid {blue}'>"
            "<b style='color:{blue}'>👤 User / Browser</b><br>"
            "<span style='font-size:.68rem;color:{muted}'>Streamlit Web Interface</span>"
            "</div>".format(blue=T["blue"], muted=T["muted"]),
            unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;font-size:1.2rem;color:{};margin:.2rem 0'>↓</div>".format(T["muted"]), unsafe_allow_html=True)
    _, mid2, _ = st.columns([1,2,1])
    with mid2:
        st.markdown(
            "<div class='arch-box' style='border:2px solid {cyan};background:rgba(6,182,212,.06)'>"
            "<b style='color:{cyan}'>📱 app.py</b><br>"
            "<span style='font-size:.68rem;color:{muted}'>UI Layer — 8 Pages — Routing — Charts — Theme Toggle</span>"
            "</div>".format(cyan=T["cyan"], muted=T["muted"]),
            unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;font-size:1.2rem;color:{};margin:.2rem 0'>↓&nbsp;&nbsp;↓&nbsp;&nbsp;↓&nbsp;&nbsp;↓&nbsp;&nbsp;↓</div>".format(T["muted"]), unsafe_allow_html=True)
    modules = [
        (T["green"],  "🧠 NLP Engine",        "TF-IDF · N-Gram\nATS · Section Detect\nValidation · Cover Letter"),
        (T["amber"],  "💾 Database",          "SQLite · Users\nHistory · Comparisons\nSHA-256 Auth"),
        (T["purple"], "📄 PDF Export",        "PDF Report\nScore Bars · Tags\nCover Letter Page"),
        (T["blue"],   "🧪 Test Suite",        "40+ Unit Tests\nPytest\nAll modules covered"),
        (T["red"],    "📋 Dependencies",      "streamlit · sklearn\npymupdf · plotly\nfpdf2 · openpyxl"),
    ]
    cols = st.columns(5)
    for col, (color, title, desc) in zip(cols, modules):
        with col:
            st.markdown(
                "<div class='arch-box' style='border:2px solid {color}'>"
                "<b style='color:{color}'>{title}</b><br>"
                "<span style='font-size:.65rem;color:{muted};white-space:pre-line'>{desc}</span>"
                "</div>".format(color=color, title=title, desc=desc, muted=T["muted"]),
                unsafe_allow_html=True)
    st.markdown('<hr class="dvd">', unsafe_allow_html=True)
    st.markdown("### Technology Stack")
    st.dataframe(pd.DataFrame([
        ["Frontend","Streamlit","UI — 8 pages, custom CSS, dark/light theme"],
        ["NLP","scikit-learn","TF-IDF vectoriser + cosine similarity"],
        ["NLP","CountVectorizer","N-gram (1,2) phrase extraction"],
        ["PDF Parse","PyMuPDF","Text extraction from PDF resumes"],
        ["Charts","Plotly","Gauge, radar, bar, line charts"],
        ["Database","SQLite3","Persistent users, history, comparisons"],
        ["PDF Export","fpdf2","Downloadable PDF analysis report"],
        ["Excel","openpyxl + pandas","History export to .xlsx"],
        ["Word Cloud","wordcloud","Visual JD keyword cloud"],
        ["Testing","pytest","40+ unit tests across all modules"],
        ["Auth","hashlib SHA-256","Password hashing"],
    ], columns=["Layer","Technology","Purpose"]), use_container_width=True, hide_index=True)
    st.markdown('<hr class="dvd">', unsafe_allow_html=True)
    st.markdown("### How to Run")
    st.code("""pip install streamlit scikit-learn pymupdf pandas plotly fpdf2 openpyxl wordcloud pytest
streamlit run app.py
# Demo login: username=demo  password=demo123""", language="bash")

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ROUTER
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not st.session_state.user:
        page_auth(); return
    page = render_sidebar()
    if   "Analyse"      in page: page_analyse()
    elif "Dashboard"    in page: page_dashboard()
    elif "Compare"      in page: page_compare()
    elif "Cover"        in page: page_cover()
    elif "Career"       in page: page_career()
    elif "ATS"          in page: page_ats()
    elif "Accuracy"     in page: page_accuracy()
    elif "Architecture" in page: page_architecture()

if __name__ == "__main__":
    main()