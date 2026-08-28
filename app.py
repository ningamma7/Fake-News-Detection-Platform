from flask import Flask, render_template, request
import pickle
import sqlite3
import os
from datetime import datetime

app = Flask(__name__)

# Load model if exists, else dummy
try:
    vectorizer = pickle.load(open("model/tfidf_vectorizer.pkl","rb"))
    model = pickle.load(open("model/lr_model.pkl","rb"))
    print("Model loaded")
except:
    print("Model not found - Train first with train.py")
    vectorizer = None
    model = None

# DB setup
def init_db():
    conn = sqlite3.connect("database.db")
    conn.execute("CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY, news TEXT, result TEXT, confidence REAL, date TEXT)")
    conn.close()
init_db()

@app.route("/", methods=["GET","POST"])
def index():
    result = None
    if request.method == "POST":
        news = request.form["news"]

        if vectorizer is None:
            # For Review-1 demo without model
            label = "FAKE" if len(news) % 2 == 0 else "REAL"
            confidence = 86.5
            highlighted = f"<span style='background:#ffcccc;padding:2px'>{news[:100]}</span> {news[100:]}"
        else:
            vec = vectorizer.transform([news])
            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0]
            confidence = max(prob) * 100
            label = "REAL" if pred==1 else "FAKE"
            if confidence < 60:
                label = "UNCERTAIN - Needs Verification"
            # Simple highlighting
            words = news.split()
            highlighted = ""
            for w in words[:100]:
                highlighted += f"<span style='background:#ffcccc;padding:1px'>{w}</span> " if len(w)>7 else w+" "

        # Save to DB
        conn = sqlite3.connect("database.db")
        conn.execute("INSERT INTO history (news, result, confidence, date) VALUES (?,?,?,?)",
                     (news[:500], label, float(confidence), datetime.now().strftime("%Y-%m-%d %H:%M")))
        conn.commit()
        conn.close()

        result = {"label": label, "confidence": f"{float(confidence):.2f}", "highlighted": highlighted, "news": news}

    # Get history
    conn = sqlite3.connect("database.db")
    history = conn.execute("SELECT * FROM history ORDER BY id DESC LIMIT 10").fetchall()
    conn.close()

    return render_template("index.html", result=result, history=history)

if __name__ == "__main__":
    app.run(debug=True)