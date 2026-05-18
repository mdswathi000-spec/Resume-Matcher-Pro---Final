"""
processor.py — Resume Matcher Pro
NLP engine: TF-IDF + N-gram blended scoring, skill categorisation,
ATS audit, section detector, certification roadmap, career paths,
salary guide, accuracy validation.
"""

import re, datetime
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


# ──────────────────────────────────────────────────────────────────────────────
#  KNOWLEDGE BASE
# ──────────────────────────────────────────────────────────────────────────────

SKILL_CATEGORIES = {
    "Programming":  ["python","java","javascript","typescript","c++","c#","go","rust","ruby","php","swift","kotlin","scala","r","matlab","perl","bash","shell"],
    "Web & UI":     ["react","angular","vue","django","flask","fastapi","spring","node.js","express","html","css","tailwind","bootstrap","next.js","jquery","sass"],
    "Data & ML":    ["machine learning","deep learning","nlp","tensorflow","pytorch","scikit-learn","pandas","numpy","opencv","hugging face","llm","computer vision","data analysis","statistics","power bi","tableau","matplotlib","seaborn"],
    "Cloud/DevOps": ["aws","azure","gcp","docker","kubernetes","ci/cd","jenkins","terraform","linux","git","github","gitlab","ansible","nginx","redis"],
    "Databases":    ["sql","mysql","postgresql","mongodb","redis","sqlite","elasticsearch","cassandra","oracle","firebase","dynamodb"],
    "Soft Skills":  ["communication","leadership","teamwork","problem solving","critical thinking","agile","scrum","project management","presentation","time management"],
}

SKILL_TO_CAT = {s: cat for cat, skills in SKILL_CATEGORIES.items() for s in skills}

CAREER_PATHS = {
    "Programming":  [("Software Developer","Build applications across web, mobile, or systems."),
                     ("Backend Engineer","Design robust APIs and server-side architecture."),
                     ("Full-Stack Developer","Own both frontend UI and backend services.")],
    "Data & ML":    [("Data Scientist","Use statistical modelling to extract business insights."),
                     ("ML Engineer","Deploy and scale machine-learning systems in production."),
                     ("Data Analyst","Analyse business data and create dashboards and reports.")],
    "Cloud/DevOps": [("DevOps Engineer","Automate pipelines and manage cloud infrastructure."),
                     ("Cloud Architect","Design and optimise cloud environments at scale."),
                     ("SRE Engineer","Ensure uptime and reliability of production systems.")],
    "Web & UI":     [("Frontend Developer","Build interactive, accessible web experiences."),
                     ("UI/UX Engineer","Bridge design and code focused on usability."),
                     ("React Developer","Specialise in modern JavaScript UI frameworks.")],
    "Databases":    [("Database Administrator","Manage, tune and secure production databases."),
                     ("Data Engineer","Build and maintain data pipelines and warehouses.")],
    "Soft Skills":  [("Technical Project Manager","Lead cross-functional teams to deliver products."),
                     ("Scrum Master","Facilitate agile ceremonies and remove blockers.")],
}

SALARY_GUIDE = {
    "Software Developer":        ("3–6 LPA",  "8–18 LPA",  "20–40 LPA"),
    "Backend Engineer":          ("4–8 LPA",  "10–22 LPA", "22–45 LPA"),
    "Full-Stack Developer":      ("4–9 LPA",  "10–22 LPA", "22–45 LPA"),
    "Data Scientist":            ("5–10 LPA", "12–25 LPA", "25–55 LPA"),
    "ML Engineer":               ("6–12 LPA", "14–28 LPA", "28–60 LPA"),
    "Data Analyst":              ("3–7 LPA",  "8–18 LPA",  "18–35 LPA"),
    "DevOps Engineer":           ("5–10 LPA", "12–24 LPA", "24–48 LPA"),
    "Frontend Developer":        ("3–7 LPA",  "8–18 LPA",  "18–35 LPA"),
    "Database Administrator":    ("4–8 LPA",  "9–18 LPA",  "18–32 LPA"),
    "Data Engineer":             ("5–10 LPA", "12–25 LPA", "25–50 LPA"),
    "Technical Project Manager": ("6–10 LPA", "12–22 LPA", "22–40 LPA"),
}

ATS_CLICHES = [
    "results-driven","team player","proven track record","self-starter","synergy",
    "think outside the box","go-getter","hard worker","detail-oriented",
    "passionate about","guru","ninja","rockstar","wizard","dynamic professional",
    "highly motivated","go above and beyond",
]

CERTIFICATIONS = {
    "python":           ["PCEP – Python Essentials 1 (Python Institute)","Google IT Automation with Python (Coursera)"],
    "machine learning": ["Andrew Ng ML Specialisation (Coursera)","Google ML Crash Course (free)"],
    "deep learning":    ["deeplearning.ai Deep Learning Specialisation","fast.ai Practical Deep Learning (free)"],
    "aws":              ["AWS Cloud Practitioner (CLF-C02)","AWS Solutions Architect – Associate"],
    "azure":            ["AZ-900 Microsoft Azure Fundamentals","AZ-204 Developing Solutions for Azure"],
    "docker":           ["Docker Certified Associate (DCA)"],
    "kubernetes":       ["Certified Kubernetes Administrator (CKA)","CKAD – App Developer"],
    "sql":              ["Oracle SQL Certified Associate","Microsoft DP-900 Azure Data Fundamentals"],
    "data analysis":    ["Google Data Analytics Certificate (Coursera)","IBM Data Analyst Professional Certificate"],
    "react":            ["Meta Front-End Developer Certificate (Coursera)"],
    "nlp":              ["Hugging Face NLP Course (free)","Stanford CS224N – NLP with Deep Learning"],
    "git":              ["GitHub Foundations Certification"],
    "linux":            ["LPI Linux Essentials","Red Hat RHCSA"],
    "tableau":          ["Tableau Desktop Specialist"],
    "power bi":         ["Microsoft PL-300 Power BI Data Analyst"],
}

# ── Resume section keywords ───────────────────────────────────────────────────
SECTION_KEYWORDS = {
    "Contact":    ["email","phone","mobile","linkedin","github","address","contact"],
    "Summary":    ["summary","objective","profile","about me","career objective","professional summary"],
    "Education":  ["education","qualification","degree","university","college","school","gpa","cgpa","bca","btech","mca","msc","bsc"],
    "Experience": ["experience","internship","work history","employment","worked at","project experience","job","role"],
    "Skills":     ["skills","technical skills","core competencies","technologies","tools","programming","languages"],
    "Projects":   ["projects","academic projects","personal projects","github","developed","built","created","implemented"],
    "Certifications": ["certification","certified","certificate","course","training","completed","coursera","udemy","nptel"],
    "Achievements":   ["achievement","award","winner","rank","scholarship","honour","honor","recognition"],
}

# ── Validation samples (10 labelled pairs) ────────────────────────────────────
VALIDATION_SAMPLES = [
    {"resume":"python machine learning tensorflow data analysis pandas numpy scikit-learn statistics","jd":"python machine learning data science tensorflow deep learning pandas numpy statistics sql","human_label":85},
    {"resume":"javascript react html css node.js express mongodb git agile","jd":"react javascript frontend html css node.js rest api git teamwork","human_label":88},
    {"resume":"java spring boot microservices docker kubernetes aws rest api sql postgresql","jd":"java spring boot docker aws kubernetes microservices sql postgresql ci/cd","human_label":90},
    {"resume":"communication leadership project management scrum agile teamwork presentation","jd":"project management agile scrum communication leadership stakeholder reporting","human_label":82},
    {"resume":"python flask sql html css git communication problem solving","jd":"machine learning tensorflow deep learning computer vision pytorch nlp","human_label":18},
    {"resume":"data analysis power bi tableau sql excel statistics python pandas","jd":"data analyst sql python power bi tableau excel reporting visualisation","human_label":92},
    {"resume":"aws docker linux bash terraform jenkins ci/cd git devops monitoring","jd":"devops aws docker kubernetes ci/cd terraform linux ansible monitoring","human_label":88},
    {"resume":"c++ algorithms data structures operating systems computer networks","jd":"python django rest api postgresql html css javascript react frontend","human_label":12},
    {"resume":"mongodb express react node.js javascript html css git agile rest api","jd":"full stack developer react node.js express mongodb javascript rest api","human_label":95},
    {"resume":"python pandas numpy matplotlib sql data analysis excel communication","jd":"python sql data analysis pandas numpy matplotlib tableau dashboard","human_label":88},
]


# ──────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_file) -> str:
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


def _clean(text):
    text = text.lower()
    text = re.sub(r"[^\w\s/+#.\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _word_freq(text, n=20):
    stop = {"with","this","that","from","your","have","will","they","their",
            "about","been","more","also","some","must","would","which","there",
            "these","into","what","should","shall","each","able","both","need"}
    words = re.findall(r"\b[a-z]{4,}\b", text.lower())
    return dict(Counter(w for w in words if w not in stop).most_common(n))


# ── Priority 3: Resume section detector ──────────────────────────────────────
def detect_sections(resume_text: str) -> dict:
    """
    Detects which standard resume sections are present.
    Returns {section_name: True/False}
    """
    text_lower = resume_text.lower()
    found = {}
    for section, keywords in SECTION_KEYWORDS.items():
        found[section] = any(kw in text_lower for kw in keywords)
    return found


def section_score(resume_text: str) -> int:
    """Returns 0-100 score based on how many sections are present."""
    detected = detect_sections(resume_text)
    present  = sum(1 for v in detected.values() if v)
    return round(present / len(SECTION_KEYWORDS) * 100)


# ──────────────────────────────────────────────────────────────────────────────
#  CORE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def get_graph_data(resume_text: str, jd_text: str) -> dict:
    r_clean = _clean(resume_text)
    j_clean = _clean(jd_text)

    # 1. N-gram overlap
    vec = CountVectorizer(ngram_range=(1, 2), stop_words="english")
    try:
        vec.fit([j_clean])
        jd_phrases  = set(vec.get_feature_names_out())
    except ValueError:
        jd_phrases  = set()
    matched_list = sorted(p for p in jd_phrases if p in r_clean)
    missing_list = sorted(p for p in jd_phrases if p not in r_clean)
    ngram_score  = round(len(matched_list) / max(len(jd_phrases), 1) * 100, 1)

    # 2. TF-IDF cosine similarity
    try:
        tfidf = TfidfVectorizer(stop_words="english")
        mat   = tfidf.fit_transform([r_clean, j_clean])
        tfidf_score = round(float(cosine_similarity(mat[0], mat[1])[0][0]) * 100, 1)
    except ValueError:
        tfidf_score = 0.0

    score = round(ngram_score * 0.60 + tfidf_score * 0.40, 1)

    # 3. ATS score
    try:
        t2 = TfidfVectorizer(stop_words="english", max_features=50)
        t2.fit([j_clean])
        top_kw = t2.get_feature_names_out().tolist()
    except ValueError:
        top_kw = []
    ats_score  = round(len([k for k in top_kw if k in r_clean]) / max(len(top_kw), 1) * 100, 1)
    kw_density = round(len(matched_list) / max(len(jd_phrases), 1) * 100, 1)
    coverage   = min(round(len(resume_text.split()) / max(len(jd_text.split()), 1) * 100, 1), 100)

    # 4. Skill categorisation
    matched_by_cat: dict = {}
    missing_by_cat: dict = {}
    for p in matched_list:
        cat = SKILL_TO_CAT.get(p)
        if cat: matched_by_cat.setdefault(cat, []).append(p)
    for p in missing_list:
        cat = SKILL_TO_CAT.get(p)
        if cat: missing_by_cat.setdefault(cat, []).append(p)

    # 5. Resume skills + dominant category
    resume_skills = [s for s in SKILL_TO_CAT if s in r_clean]
    cat_counts    = Counter(SKILL_TO_CAT[s] for s in resume_skills)
    dominant_cat  = cat_counts.most_common(1)[0][0] if cat_counts else "Programming"
    career_paths  = CAREER_PATHS.get(dominant_cat, CAREER_PATHS["Programming"])
    top_title     = career_paths[0][0]
    salary_guide  = SALARY_GUIDE.get(top_title, ("3–6 LPA","8–18 LPA","18–35 LPA"))

    # 6. Certifications
    certifications = {p: CERTIFICATIONS[p] for p in missing_list[:12] if p in CERTIFICATIONS}

    # 7. ATS clichés
    ats_flags = [kw for kw in ATS_CLICHES if kw in r_clean]

    # 8. Word frequency
    word_freq = _word_freq(jd_text)

    # 9. Readability
    sentences = re.split(r"[.!?]+", jd_text)
    avg_len   = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    readability = max(0.0, min(100.0, round(100 - (avg_len - 15) * 2, 1)))

    # 10. Dimension scores (radar chart)
    dims = {cat: round(len([s for s in skills if s in r_clean]) / len(skills) * 100)
            for cat, skills in SKILL_CATEGORIES.items()}

    # 11. Section detection (Priority 3)
    sections       = detect_sections(resume_text)
    section_sc     = section_score(resume_text)

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
        "sections": sections, "section_score": section_sc,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  COVER LETTER
# ──────────────────────────────────────────────────────────────────────────────

def generate_cover_letter(name, course, skills, company, role, score, missing_list, tone="professional"):
    top_missing = ", ".join(m.upper() for m in missing_list[:3]) or "key technical areas"
    today = datetime.date.today().strftime("%d %B %Y")
    openers = {
        "professional": f"I am writing to express my strong interest in the {role} position at {company}.",
        "enthusiastic": f"The opportunity to join {company} as {role} genuinely excites me — I believe my background is a compelling fit.",
        "concise":      f"I am applying for the {role} role at {company}, and my skill set aligns well with your requirements.",
        "creative":     f"When I read the {role} job description at {company}, I knew immediately this role was built for someone with my background.",
    }
    return f"""{name}
{course}
{today}

Dear Hiring Manager,

{openers.get(tone, openers['professional'])}

My resume demonstrates a {score:.0f}% alignment with the job description. My hands-on experience in {skills} has equipped me to contribute from day one. Through my final-year project — an AI-powered Resume Matcher built with Python, NLP, TF-IDF scoring, and Streamlit — I developed practical skills in building real-world software systems with a database layer, PDF export, and interactive dashboards.

I acknowledge that I am actively developing expertise in {top_missing}. I have already begun targeted learning through online courses and project work, and my track record of fast independent learning means I will close these gaps quickly.

I would welcome the opportunity to discuss how my enthusiasm and project experience can add value to {company}. Thank you for considering my application.

Yours sincerely,
{name}
"""


# ──────────────────────────────────────────────────────────────────────────────
#  ACCURACY VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

def run_validation() -> dict:
    results, errors = [], []
    for s in VALIDATION_SAMPLES:
        res  = get_graph_data(s["resume"], s["jd"])
        err  = abs(res["score"] - s["human_label"])
        results.append({"predicted": res["score"], "actual": s["human_label"],
                         "error": err, "match": err <= 15})
        errors.append(err)
    return {
        "results":  results,
        "accuracy": round(sum(1 for r in results if r["match"]) / len(results) * 100, 1),
        "mae":      round(sum(errors) / len(errors), 1),
    }
