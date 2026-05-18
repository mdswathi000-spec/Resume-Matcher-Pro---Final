"""
pdf_export.py — Resume Matcher Pro
Generates a professional PDF analysis report using fpdf2.
Includes score bars, skill tags, career paths, certifications, ATS audit.
"""

from fpdf import FPDF
import datetime


class ReportPDF(FPDF):
    def __init__(self, candidate_name="Candidate"):
        super().__init__()
        self.candidate_name = candidate_name
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_fill_color(7, 9, 15)
        self.rect(0, 0, 210, 18, "F")
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(59, 130, 246)
        self.set_xy(10, 5)
        self.cell(0, 8, "Resume Matcher Pro  —  Analysis Report", ln=False)
        self.set_xy(0, 5)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(74, 90, 114)
        self.cell(200, 8, datetime.date.today().strftime("%d %B %Y"), align="R")
        self.ln(14)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(74, 90, 114)
        self.cell(0, 8, f"Resume Matcher Pro  ·  Page {self.page_no()}  ·  {self.candidate_name}", align="C")

    def section_title(self, title, color=(59, 130, 246)):
        self.set_fill_color(*color)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 8, f"  {title}", ln=True, fill=True)
        self.ln(2)
        self.set_text_color(20, 30, 48)

    def score_bar(self, label, value, color=(59, 130, 246)):
        bar_w = 130
        filled = int(bar_w * min(value, 100) / 100)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(60, 80, 110)
        self.cell(55, 6, label)
        self.set_fill_color(220, 230, 245)
        self.rect(self.get_x(), self.get_y() + 1, bar_w, 4, "F")
        self.set_fill_color(*color)
        if filled > 0:
            self.rect(self.get_x(), self.get_y() + 1, filled, 4, "F")
        self.set_x(self.get_x() + bar_w + 3)
        self.set_text_color(*color)
        self.set_font("Helvetica", "B", 9)
        self.cell(0, 6, f"{value:.0f}%", ln=True)
        self.set_text_color(20, 30, 48)

    def tag_row(self, tags, color_rgb=(16, 185, 129)):
        self.set_font("Helvetica", "", 8)
        x_start = self.get_x()
        for tag in tags[:20]:
            tw = self.get_string_width(tag) + 8
            if self.get_x() + tw > 195:
                self.ln(7); self.set_x(x_start)
            rx, ry = self.get_x(), self.get_y()
            self.set_fill_color(*color_rgb)
            self.set_text_color(255, 255, 255)
            self.rect(rx, ry, tw, 5.5, "F")
            self.set_xy(rx + 2, ry)
            self.cell(tw - 2, 5.5, tag)
            self.set_xy(self.get_x() + 2, ry)
        self.ln(8)
        self.set_text_color(20, 30, 48)

    def kv(self, label, value, hl=False):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(80, 100, 130)
        self.cell(55, 7, label)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(16, 185, 129) if hl else self.set_text_color(20, 30, 48)
        self.cell(0, 7, str(value), ln=True)
        self.set_text_color(20, 30, 48)

    def checklist_row(self, label, ok):
        icon  = "✔" if ok else "✖"
        color = (16, 185, 129) if ok else (239, 68, 68)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*color)
        self.cell(0, 6, f"  {icon}  {label}", ln=True)
        self.set_text_color(20, 30, 48)


def generate_pdf_report(candidate_name, resume_name, job_title, result, cover_letter=""):
    pdf = ReportPDF(candidate_name)
    pdf.add_page()

    # ── Summary ───────────────────────────────────────────────────────────────
    pdf.section_title("Analysis Summary")
    pdf.kv("Candidate",   candidate_name)
    pdf.kv("Resume File", resume_name)
    pdf.kv("Role / JD",   job_title[:70])
    pdf.kv("Date",        datetime.date.today().strftime("%d %B %Y"))
    pdf.ln(3)
    s     = result.get("score", 0)
    s_col = (16,185,129) if s>=70 else (245,158,11) if s>=45 else (239,68,68)
    verd  = "Strong Match" if s>=70 else "Moderate Match" if s>=45 else "Weak Match"
    pdf.set_font("Helvetica","B",26)
    pdf.set_text_color(*s_col)
    pdf.cell(0, 13, f"{s:.0f}%  —  {verd}", ln=True, align="C")
    pdf.ln(2); pdf.set_text_color(20,30,48)

    # ── Score Breakdown ───────────────────────────────────────────────────────
    pdf.section_title("Score Breakdown", (30,58,100))
    pdf.score_bar("Overall Match",   result.get("score",0),       (59,130,246))
    pdf.score_bar("N-Gram Score",    result.get("ngram_score",0), (99,102,241))
    pdf.score_bar("TF-IDF Cosine",   result.get("tfidf_score",0), (139,92,246))
    pdf.score_bar("ATS Score",       result.get("ats_score",0),   (16,185,129))
    pdf.score_bar("Keyword Density", result.get("kw_density",0),  (6,182,212))
    pdf.score_bar("Resume Coverage", result.get("coverage",0),    (245,158,11))
    pdf.score_bar("Section Score",   result.get("section_score",0),(236,72,153))
    pdf.ln(3)

    # ── Matched Skills ────────────────────────────────────────────────────────
    pdf.section_title("Matched Skills", (16,100,60))
    pdf.tag_row(result.get("matched_list",[])[:25], (16,185,129))
    pdf.ln(2)

    # ── Skill Gaps ────────────────────────────────────────────────────────────
    pdf.section_title("Skill Gaps", (160,30,30))
    pdf.tag_row(result.get("missing_list",[])[:25], (239,68,68))
    pdf.ln(2)

    # ── Section Detection ─────────────────────────────────────────────────────
    pdf.section_title("Resume Section Audit", (80,50,120))
    sections = result.get("sections", {})
    for sec, present in sections.items():
        pdf.checklist_row(sec, present)
    pdf.ln(2)

    # ── ATS Audit ─────────────────────────────────────────────────────────────
    pdf.section_title("ATS Audit", (120,40,40))
    flags = result.get("ats_flags", [])
    if flags:
        pdf.set_font("Helvetica","",8.5)
        pdf.set_text_color(239,68,68)
        pdf.multi_cell(0, 5.5, f"  ⚠  Clichés detected: {', '.join(flags)}")
    else:
        pdf.set_font("Helvetica","",8.5)
        pdf.set_text_color(16,185,129)
        pdf.cell(0, 5.5, "  ✔  No ATS clichés detected.", ln=True)
    pdf.set_text_color(20,30,48); pdf.ln(2)

    # ── Career Paths ──────────────────────────────────────────────────────────
    pdf.section_title("Recommended Career Paths", (120,53,15))
    sal = result.get("salary_guide", ("—","—","—"))
    for title, desc in result.get("career_paths", []):
        pdf.set_font("Helvetica","B",9); pdf.set_text_color(20,30,48)
        pdf.cell(0,7,f"  {title}",ln=True)
        pdf.set_font("Helvetica","",8.5); pdf.set_text_color(80,100,130)
        pdf.set_x(15); pdf.multi_cell(0,5.5,desc)
    pdf.set_font("Helvetica","",8.5); pdf.set_text_color(60,80,110)
    pdf.cell(0,6,f"  Salary:  Fresher {sal[0]}  |  Mid {sal[1]}  |  Senior {sal[2]}",ln=True)
    pdf.ln(2)

    # ── Certifications ────────────────────────────────────────────────────────
    certs = result.get("certifications",{})
    if certs:
        pdf.section_title("Recommended Certifications", (5,80,120))
        for skill, cl in certs.items():
            pdf.set_font("Helvetica","B",8.5); pdf.set_text_color(6,182,212)
            pdf.cell(0,6,f"  {skill.upper()}",ln=True)
            for c in cl:
                pdf.set_font("Helvetica","",8); pdf.set_text_color(60,80,110)
                pdf.set_x(18); pdf.cell(0,5.5,f"◎  {c}",ln=True)
        pdf.ln(2)

    # ── Cover Letter ──────────────────────────────────────────────────────────
    if cover_letter:
        pdf.add_page()
        pdf.section_title("Generated Cover Letter", (30,60,120))
        pdf.set_font("Helvetica","",9); pdf.set_text_color(30,40,60)
        for line in cover_letter.split("\n"):
            pdf.multi_cell(0, 5.5, line if line else " ")

    return bytes(pdf.output())
