"""
database.py — Resume Matcher Pro
SQLite storage: users (hashed passwords), analysis history,
resume comparisons. Persistent across all sessions.
"""

import sqlite3, json, hashlib, datetime

DB_PATH = "resume_matcher.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn(); c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL, password TEXT NOT NULL,
        name TEXT, course TEXT,
        created TEXT DEFAULT (datetime('now')))""")
    c.execute("""CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, resume_name TEXT, job_title TEXT,
        score REAL, ngram_score REAL, tfidf_score REAL, ats_score REAL,
        matched_skills TEXT, missing_skills TEXT, dominant_cat TEXT,
        created TEXT DEFAULT (datetime('now')),
        FOREIGN KEY(user_id) REFERENCES users(id))""")
    c.execute("""CREATE TABLE IF NOT EXISTS comparisons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, resume_a TEXT, resume_b TEXT,
        score_a REAL, score_b REAL, winner TEXT,
        created TEXT DEFAULT (datetime('now')),
        FOREIGN KEY(user_id) REFERENCES users(id))""")
    conn.commit(); conn.close()


def _hash(pw): return hashlib.sha256(pw.encode()).hexdigest()


def register_user(username, password, name, course):
    try:
        conn = get_conn()
        conn.execute("INSERT INTO users (username,password,name,course) VALUES (?,?,?,?)",
                     (username.strip(), _hash(password), name.strip(), course.strip()))
        conn.commit(); conn.close()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "Username already exists."


def login_user(username, password):
    conn = get_conn()
    row  = conn.execute("SELECT * FROM users WHERE username=? AND password=?",
                        (username.strip(), _hash(password))).fetchone()
    conn.close(); return row


def save_analysis(user_id, resume_name, job_title, result):
    conn = get_conn()
    conn.execute("""INSERT INTO analyses
        (user_id,resume_name,job_title,score,ngram_score,tfidf_score,ats_score,
         matched_skills,missing_skills,dominant_cat)
        VALUES (?,?,?,?,?,?,?,?,?,?)""", (
        user_id, resume_name, job_title[:80],
        result.get("score",0), result.get("ngram_score",0),
        result.get("tfidf_score",0), result.get("ats_score",0),
        json.dumps(result.get("matched_list",[])[:20]),
        json.dumps(result.get("missing_list",[])[:20]),
        result.get("dominant_cat",""),
    ))
    conn.commit(); conn.close()


def get_history(user_id):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM analyses WHERE user_id=? ORDER BY created DESC LIMIT 50",
                        (user_id,)).fetchall()
    conn.close(); return [dict(r) for r in rows]


def get_stats(user_id):
    conn = get_conn()
    rows = conn.execute("SELECT score,created FROM analyses WHERE user_id=? ORDER BY created",
                        (user_id,)).fetchall()
    conn.close()
    if not rows: return {"count":0,"best":0,"avg":0,"trend":[]}
    scores = [r["score"] for r in rows]
    return {"count":len(scores),"best":max(scores),
            "avg":round(sum(scores)/len(scores),1),
            "trend":[(r["created"][:10], r["score"]) for r in rows]}


def save_comparison(user_id, name_a, name_b, score_a, score_b):
    winner = name_a if score_a >= score_b else name_b
    conn   = get_conn()
    conn.execute("INSERT INTO comparisons (user_id,resume_a,resume_b,score_a,score_b,winner) VALUES (?,?,?,?,?,?)",
                 (user_id, name_a, name_b, score_a, score_b, winner))
    conn.commit(); conn.close()


def get_comparisons(user_id):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM comparisons WHERE user_id=? ORDER BY created DESC LIMIT 20",
                        (user_id,)).fetchall()
    conn.close(); return [dict(r) for r in rows]
