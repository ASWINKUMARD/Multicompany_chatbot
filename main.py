###############################################################
#  MAIN BACKEND – FULL FIXED VERSION
#  Works with Streamlit Frontend (PART-2)
###############################################################

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os, re, json, time, hashlib, requests, shutil, functools
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin
from collections import deque
from typing import Optional, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base
from bs4 import BeautifulSoup

###############################################################
# GLOBAL CONFIG
###############################################################

DATABASE_URL = "sqlite:///./multi_company_chatbots.db"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "kwaipilot/kat-coder-pro:free"

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

###############################################################
# DATABASE TABLES
###############################################################

class Company(Base):
    __tablename__ = "companies"
    id = Column(Integer, primary_key=True)
    company_name = Column(String(255), unique=True)
    company_slug = Column(String(255), unique=True)
    website_url = Column(String(300))
    emails = Column(Text)
    phones = Column(Text)
    address_india = Column(Text)
    address_international = Column(Text)
    pages_scraped = Column(Integer, default=0)
    scraping_status = Column(String(50), default="pending")
    max_pages_to_scrape = Column(Integer, default=40)
    last_scraped = Column(DateTime)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ChatHistory(Base):
    __tablename__="chat_history"
    id=Column(Integer,primary_key=True)
    company_slug=Column(String(255))
    question=Column(Text)
    answer=Column(Text)
    session_id=Column(String(255))
    timestamp=Column(DateTime,default=lambda:datetime.now(timezone.utc))


class UserContact(Base):
    __tablename__="user_contacts"
    id=Column(Integer,primary_key=True)
    company_slug=Column(String(255))
    name=Column(String(255))
    email=Column(String(255))
    phone=Column(String(50))
    timestamp=Column(DateTime,default=lambda:datetime.now(timezone.utc))

Base.metadata.create_all(engine)

###############################################################
# UTIL
###############################################################

def create_slug(name:str)->str:
    slug=re.sub(r'[^a-z0-9]+','-',name.lower()).strip("-")
    return slug

def get_chroma_directory(slug): return f"./chroma_db/{slug}"

# URL CHECK
def validate_url(url):
    try:
        p=urlparse(url)
        return p.scheme in ("http","https") and p.netloc!=""
    except: return False


###############################################################
# EMBEDDINGS (FIXED – NO MORE STUCK)
###############################################################

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

_EMBED_CACHE=None

def get_embeddings():
    global _EMBED_CACHE
    if _EMBED_CACHE: return _EMBED_CACHE

    print("🔄 Loading embeddings...")
    try:
        _EMBED_CACHE=HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device":"cpu"},
            encode_kwargs={"normalize_embeddings":True}
        )
    except:
        print("⚠️ Fallback small model used")
        _EMBED_CACHE=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("✅ Embeddings loaded\n")
    return _EMBED_CACHE


###############################################################
# SMART SCRAPER (Stable + Contact Extractor)
###############################################################

class WebScraper:
    def __init__(self,slug):
        self.slug=slug
        self.session=requests.Session()
        self.info={"emails":set(),"phones":set(),"address_india":None,"address_international":None}

    def extract_contacts(self,text):
        if not text:return
        for m in re.findall(r"[A-Za-z0-9._+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",text):
            self.info["emails"].add(m)

        for m in re.findall(r"\+?\d[\d -]{7,14}",text):
            self.info["phones"].add(m)

    def scrape(self,url,max_pages=30,cb=None)->tuple[list,dict]:
        q=deque([url])
        seen=set(); docs=[]

        while q and len(seen)<max_pages:
            link=q.popleft()
            if link in seen:continue
            seen.add(link)

            try:
                r=self.session.get(link,timeout=15)
                if r.status_code!=200:continue
                soup=BeautifulSoup(r.text,"html.parser")
                text=soup.get_text(" ",strip=True)
                self.extract_contacts(text)

                content="\n".join(t.strip() for t in text.split("\n") if len(t.strip())>40)[:2500]
                if len(content)>300:
                    docs.append(Document(page_content=content,metadata={"source":link}))

                for a in soup.find_all("a",href=True):
                    u=urljoin(url,a['href']).split("#")[0]
                    if urlparse(u).netloc==urlparse(url).netloc and u not in seen:
                        q.append(u)
                if cb:cb(len(seen),max_pages,link)

            except:pass

        return docs,{
            "emails":list(self.info["emails"]),
            "phones":list(self.info["phones"]),
            "address_india":self.info["address_india"],
            "address_international":self.info["address_international"],
            "pages_scraped":len(seen)
        }


###############################################################
# AI ENGINE (Vector DB + Chat)
###############################################################

class CompanyAI:
    def __init__(self,slug):
        self.slug=slug
        self.vector=None; self.retriever=None
        self.data=None; self.ready=False

        self.prompt=PromptTemplate(
            template="""You are AI assistant for {name}.
Use only the following context:

{context}

Question: {q}

Answer shortly (2-4 sentences), no hallucination.""",
            input_variables=["name","context","q"]
        )

    def initialize(self,url,maxp,cb=None):
        docs,info=WebScraper(self.slug).scrape(url,maxp,cb)
        if len(docs)<3:return False

        self.data=info
        text=RecursiveCharacterTextSplitter(chunk_size=900,chunk_overlap=150).split_documents(docs)

        path=get_chroma_directory(self.slug)
        if os.path.exists(path):shutil.rmtree(path)
        os.makedirs(path,exist_ok=True)

        self.vector=Chroma.from_documents(text,get_embeddings(),persist_directory=path)
        self.retriever=self.vector.as_retriever(k=4)
        self.ready=True;return True

    def load(self):
        path=get_chroma_directory(self.slug)
        if not os.path.exists(path):return False
        self.vector=Chroma(persist_directory=path,embedding_function=get_embeddings())
        self.retriever=self.vector.as_retriever(k=4)
        self.ready=True;return True

    def ask(self,q):
        if not self.ready:return "⏳ Loading..."
        docs=self.retriever.get_relevant_documents(q)
        ctx="\n".join(d.page_content[:400] for d in docs[:3])

        payload={
            "model":MODEL,
            "messages":[{"role":"system","content":"Answer using context only"},
                        {"role":"user","content":self.prompt.format(q=q,context=ctx,name=self.slug)}]
        }
        r=requests.post(OPENROUTER_API_BASE,json=payload,
                        headers={"Authorization":f"Bearer {OPENROUTER_API_KEY}"})
        return r.json()["choices"][0]["message"]["content"]


###############################################################
# DB EXPOSE FUNCTIONS FOR STREAMLIT
###############################################################

def create_company(name,url,maxp):
    db=SessionLocal()
    slug=create_slug(name)
    if db.query(Company).filter_by(company_slug=slug).first():return None
    c=Company(company_name=name,company_slug=slug,website_url=url,max_pages_to_scrape=maxp)
    db.add(c);db.commit();return slug

def update_company_after_scraping(slug,info,status="completed"):
    db=SessionLocal();c=db.query(Company).filter_by(company_slug=slug).first()
    c.emails=json.dumps(info["emails"]);c.phones=json.dumps(info["phones"])
    c.address_india=info["address_india"];c.address_india=info["address_india"]
    c.address_international=info["address_international"]
    c.pages_scraped=info["pages_scraped"];c.scraping_status=status
    c.last_scraped=datetime.now(timezone.utc);db.commit()

def get_all_companies():
    db=SessionLocal();return db.query(Company).all()

def get_company_by_slug(s):
    db=SessionLocal();return db.query(Company).filter_by(company_slug=s).first()

###############################################################
# STREAMLIT FRONTEND UI  - PART 2 (Full Stable Working UI)
###############################################################

import streamlit as st
import time, hashlib, os
from urllib.parse import urlparse
from datetime import datetime

# ===== IMPORTS FROM MAIN.PY ===== #
from main import (
    create_company, update_company_after_scraping,
    get_all_companies, get_company_by_slug,
    CompanyAI, validate_url
)

###############################################################
# PAGE SETTINGS
###############################################################

st.set_page_config(page_title="AI Chatbot Generator", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .main { background: #f7f9ff; }
    .header {text-align:center;padding:25px;border-radius:10px;color:white;
        background:linear-gradient(135deg,#5b6cff,#764ba2);}
    .card {background:white;padding:15px;border-radius:10px;margin:10px 0;
        box-shadow:0 2px 8px rgba(0,0,0,.08);}
    .btn {font-weight:600;border-radius:8px;}
</style>
""", unsafe_allow_html=True)

###############################################################
# SESSION INIT
###############################################################

if "page" not in st.session_state: st.session_state.page="home"
if "ai" not in st.session_state: st.session_state.ai=None
if "company" not in st.session_state: st.session_state.company=None
if "chat" not in st.session_state: st.session_state.chat=[]
if "session" not in st.session_state:
    st.session_state.session = hashlib.md5(os.urandom(16)).hexdigest()[:12]


###############################################################
# SIDEBAR NAVIGATION
###############################################################

with st.sidebar:
    st.title("🤖 AI Chatbot")
    if st.button("🏠 Home"): st.session_state.page="home"; st.rerun()
    if st.button("➕ Create New"): st.session_state.page="create"; st.rerun()
    if st.button("📋 All Chatbots"): st.session_state.page="list"; st.rerun()

    st.markdown("---")
    st.caption(f"Session: `{st.session_state.session}`")
    st.caption("Powered by RAG + OpenRouter")


###############################################################
# HOME PAGE
###############################################################

if st.session_state.page=="home":
    st.markdown('<div class="header"><h1>🤖 AI Chatbot Generator</h1>'
                '<p>Create Chatbot From Any Website</p></div>', unsafe_allow_html=True)

    c1,c2=st.columns(2)
    if c1.button("➕ Create Chatbot",use_container_width=True): st.session_state.page="create"; st.rerun()
    if c2.button("📋 View Chatbots",use_container_width=True): st.session_state.page="list"; st.rerun()

    st.info("""
### 🔥 Features
- Automatic Website Scraper  
- Builds Company Knowledge Base  
- AI Answers Using Real Website Content  
- Contact Information Extraction  
- Multi-Company Chat Support  
""")

###############################################################
# CREATE NEW CHATBOT PAGE
###############################################################

elif st.session_state.page=="create":
    st.markdown('<div class="header"><h2>Create New Chatbot</h2></div>', unsafe_allow_html=True)

    with st.form("create_bot"):
        name = st.text_input("Company Name*",placeholder="Example Pvt Ltd")
        url = st.text_input("Website URL*",placeholder="https://example.com")
        pages = st.slider("Scrape Pages",10,60,30)

        submit=st.form_submit_button("🚀 Create Chatbot")

    if submit:
        if not name or not url: st.error("Enter all fields!"); st.stop()
        if not validate_url(url): st.error("Invalid URL"); st.stop()

        slug=create_company(name,url,pages)
        if not slug:
            st.error("Company already exists! Use unique name"); st.stop()

        st.success("Company Created ✔ Starting Web Scraper...")

        prog = st.progress(0); status=st.empty()

        def cb(a,b,u): prog.progress(a/b); status.write(f"Scraping → {u}")

        ai = CompanyAI(slug)
        ok = ai.initialize(url,pages,cb)

        if ok:
            update_company_after_scraping(slug, ai.data, "completed")
            st.success("🎉 Chatbot Ready")
            st.balloons()

            st.session_state.company = slug
            st.session_state.ai = ai
            st.session_state.page="chat"
            st.session_state.chat=[]
            st.rerun()
        else:
            st.error("Failed processing website. Try new domain.")


###############################################################
# LIST BOT PAGE
###############################################################

elif st.session_state.page=="list":
    st.markdown('<div class="header"><h2>All Chatbots</h2></div>', unsafe_allow_html=True)

    bots=get_all_companies()
    if not bots:
        st.warning("No chatbots found. Create one →")
        if st.button("➕ Create Now"): st.session_state.page="create"; st.rerun()
    else:
        for c in bots:
            with st.container():
                st.markdown(f"<div class='card'><h3>{c.company_name}</h3>"
                            f"<p>🌐 {c.website_url}<br>📄 {c.pages_scraped} pages</p>",
                            unsafe_allow_html=True)

                if c.scraping_status=="completed":
                    if st.button(f"💬 Chat With {c.company_name}",key=c.id):
                        st.session_state.company=c.company_slug
                        st.session_state.ai=None
                        st.session_state.page="chat"
                        st.session_state.chat=[]
                        st.rerun()
                else:
                    st.error("Not Ready Yet")


###############################################################
# CHAT PAGE
###############################################################

elif st.session_state.page=="chat":

    if not st.session_state.company:
        st.error("No company selected"); st.stop()

    c=get_company_by_slug(st.session_state.company)
    st.markdown(f"<div class='header'><h2>💬 {c.company_name}</h2></div>", unsafe_allow_html=True)

    # Load AI model if not loaded
    if not st.session_state.ai:
        st.write("Loading knowledge base...")
        ai=CompanyAI(st.session_state.company)
        if not ai.load():
            st.error("Chatbot not built yet. Recreate.")
            st.stop()
        st.session_state.ai=ai

    # Display chat history
    for m in st.session_state.chat:
        with st.chat_message(m["role"]): st.write(m["content"])

    # Input box
    user=st.chat_input("Ask about the company...")
    if user:
        st.session_state.chat.append({"role":"user","content":user})
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = st.session_state.ai.ask(user)
                st.write(reply)

        st.session_state.chat.append({"role":"assistant","content":reply})
        st.rerun()
