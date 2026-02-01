import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import feedparser
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base
import re



# Summarization imports
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# ----------------------------
# Database setup
# ----------------------------
DATABASE_URL = "sqlite:///./news.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class News(Base):
    __tablename__ = "news"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255))
    content = Column(Text)

Base.metadata.create_all(bind=engine)

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="Data Driven Online News Summarizer-ML-NLP")

# ----------------------------
# Utility: Clean + Summarizer
# ----------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def summarize_text(text, sentences_count=3):
    text = clean_text(text)
    if not text or len(text.split()) < 30:  # too short to summarize
        return text if text else "No content available."
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, sentences_count)
        return " ".join([str(sentence) for sentence in summary])
    except Exception as e:
        return f"Could not summarize due to error: {str(e)}"

# ----------------------------
# Fetch News from Public RSS Feed
# ----------------------------
def fetch_news_from_rss():
    url = "http://feeds.bbci.co.uk/news/world/rss.xml"
    feed = feedparser.parse(url)
    if feed.entries:
        articles = []
        for entry in feed.entries[:10]:  # limit to 10 articles
            content = getattr(entry, "summary", None) or getattr(entry, "description", "")
            articles.append({
                "title": entry.title,
                "content": clean_text(content)
            })
        return articles
    return None

def seed_local_db():
    db = SessionLocal()
    if db.query(News).count() == 0:
        sample_articles = [
            {"title": "Local Economy Growth", "content": "The local economy has shown signs of growth with new businesses opening."},
            {"title": "Community Event", "content": "A community event was held downtown with hundreds of people attending."},
            {"title": "Tech Innovation", "content": "Local startups are focusing on AI-driven solutions to improve daily life."}
        ]
        for art in sample_articles:
            db.add(News(title=art["title"], content=art["content"]))
        db.commit()
    db.close()




# ----------------------------
# API Endpoints
# ----------------------------
@app.on_event("startup")
def startup_event():
    articles = fetch_news_from_rss()
    db = SessionLocal()
    if articles:
        db.query(News).delete()
        for art in articles:
            db.add(News(title=art["title"], content=art["content"]))
        db.commit()
    else:
        seed_local_db()
    db.close()


@app.get("/", response_class=HTMLResponse)
def home():
    db = SessionLocal()
    articles = db.query(News).all()
    db.close()

    html_content = "<h1>Data Driven Online News Summarizer-ML-NLP</h1><ul>"
    for a in articles:
        html_content += f"<li><b>{a.title}</b> - <a href='/summarize/{a.id}'>Summarize</a></li>"
    html_content += "</ul>"
    return html_content

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return HTMLResponse(content="", status_code=200)

@app.get("/news")
def get_news():
    db = SessionLocal()
    articles = db.query(News).all()
    db.close()
    return [{"id": a.id, "title": a.title, "content": a.content} for a in articles]


@app.get("/summarize/{news_id}", response_class=HTMLResponse)
def summarize_article(news_id: int):
    db = SessionLocal()
    article = db.query(News).filter(News.id == news_id).first()
    db.close()
    if not article:
        return HTMLResponse("<h2>Article not found</h2>", status_code=404)
    summary = summarize_text(article.content)
    return f"<h2>{article.title}</h2><p><b>Summary:</b> {summary}</p>"