###############################################################
#  FULL SINGLE-FILE AI CHATBOT SYSTEM
#  backend + UI + embeddings + RAG + scraper WORKING VERSION
###############################################################

import streamlit as st
import os, re, json, time, hashlib, requests, shutil
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin
from collections import deque
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base
from bs4 import BeautifulSoup

###############################################################
# CONFIG
###############################################################

st.set_page_config(page_title="AI Chatbot Generator", page_icon="🤖", layout="wide")

DATABASE_URL = "sqlite:///./multi_company_chatbots.db"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
MODEL = "kwaipilot/kat-coder-pro:free"
API = "https://openrouter.ai/api/v1/chat/completions"

if not OPENROUTER_API_KEY:
    st.sidebar.error("⚠️ Add OPENROUTER_API_KEY in Secrets!")

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)

###############################################################
# DATABASE
###############################################################

class Company(Base):
    __tablename__="companies"
    id=Column(Integer,primary_key=True)
    name=Column(String(255))
    slug=Column(String(255),unique=True)
    url=Column(String(300))
    pages=Column(Integer,default=30)
    status=Column(String(50),default="pending")
    data=Column(Text)
    created=Column(DateTime,default=lambda:datetime.now(timezone.utc))

class Chat(Base):
    __tablename__="history"
    id=Column(Integer,primary_key=True)
    slug=Column(String(255))
    q=Column(Text)
    a=Column(Text)

Base.metadata.create_all(engine)

###############################################################
# LANGCHAIN SAFE IMPORT SYSTEM
###############################################################

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except:
    from langchain.text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_community.vectorstores import Chroma
except:
    from langchain.vectorstores import Chroma

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except:
        from langchain.embeddings import HuggingFaceEmbeddings

try:
    from langchain_core.documents import Document
except:
    try:
        from langchain.docstore.document import Document
    except:
        from langchain.schema import Document

try:
    from langchain_core.prompts import PromptTemplate
except:
    from langchain.prompts import PromptTemplate

###############################################################
# URL VALIDATOR
###############################################################

def validate(url):
    try:
        p=urlparse(url)
        return p.scheme in ("http","https") and p.netloc
    except:return False

###############################################################
# EMBEDDINGS (no infinite loading)
###############################################################

_EMB=None
def embeddings():
    global _EMB
    if _EMB:return _EMB
    try:
        _EMB=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except:
        _EMB=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _EMB

###############################################################
# SCRAPER
###############################################################

def scrape(url,maxp,cb=None):
    q=deque([url])
    seen=set(); docs=[]

    while q and len(seen)<maxp:
        u=q.popleft()
        if u in seen:continue
        seen.add(u)

        try:
            r=requests.get(u,timeout=15)
            if r.status_code!=200:continue
            soup=BeautifulSoup(r.text,"html.parser")
            text=soup.get_text(" ",strip=True)
            clean="\n".join(t for t in text.split("\n") if len(t)>45)[:2500]
            if len(clean)>200:
                docs.append(Document(page_content=clean,metadata={"src":u}))

            for a in soup.find_all("a",href=True):
                link=urljoin(url,a['href']).split("#")[0]
                if urlparse(link).netloc==urlparse(url).netloc:
                    q.append(link)

            if cb:cb(len(seen),maxp,u)

        except:pass

    return docs,{"pages":len(seen)}

###############################################################
# AI CORE
###############################################################

class AI:
    def __init__(self,slug):self.slug=slug;self.v=None

    def build(self,url,maxp,cb=None):
        docs,data=scrape(url,maxp,cb)
        if len(docs)<3:return False

        path=f"chroma/{self.slug}"
        if os.path.exists(path):shutil.rmtree(path)
        os.makedirs(path,exist_ok=True)

        spl=RecursiveCharacterTextSplitter(chunk_size=900,chunk_overlap=150)
        docs=spl.split_documents(docs)

        self.v=Chroma.from_documents(docs,embeddings(),persist_directory=path)
        return True

    def load(self):
        path=f"chroma/{self.slug}"
        if not os.path.exists(path):return False
        self.v=Chroma(persist_directory=path,embedding_function=embeddings())
        return True

    def ask(self,q):
        ctx="\n".join(d.page_content[:300] for d in self.v.similarity_search(q,k=3))
        prompt=f"Use ONLY info below:\n{ctx}\nQ:{q}\nA:"

        r=requests.post(API,json={
            "model":MODEL,
            "messages":[{"role":"user","content":prompt}]
        },headers={"Authorization":f"Bearer {OPENROUTER_API_KEY}"})
        return r.json()["choices"][0]["message"]["content"]

###############################################################
# STREAMLIT UI
###############################################################

if "page" not in st.session_state: st.session_state.page="home"
if "chat" not in st.session_state: st.session_state.chat=[]
if "slug" not in st.session_state: st.session_state.slug=None
if "ai" not in st.session_state: st.session_state.ai=None

################################ HOME ################################
if st.session_state.page=="home":
    st.title("🤖 AI Chatbot Generator")
    st.write("Create chatbot from any website automatically.")

    if st.button("➕ Create New Chatbot"): st.session_state.page="create"; st.rerun()
    if st.button("📋 View All Chatbots"): st.session_state.page="list"; st.rerun()

################################ CREATE BOT ################################
elif st.session_state.page=="create":
    st.header("Create New Chatbot")

    name=st.text_input("Company Name")
    url=st.text_input("Website URL")
    pages=st.slider("Pages to Scrape",10,60,30)

    if st.button("🚀 Build Chatbot"):
        if not(name and validate(url)): st.error("Invalid input"); st.stop()

        db=Session()
        slug=re.sub(r'[^a-z0-9]+','-',name.lower())
        db.add(Company(name=name,slug=slug,url=url,pages=pages))
        db.commit()

        ai=AI(slug)
        prog=st.progress(0); text=st.empty()
        ok=ai.build(url,pages,lambda a,b,u:(prog.progress(a/b),text.write(u)))

        if ok:
            st.success("Chatbot Ready ✔")
            st.session_state.slug=slug
            st.session_state.ai=ai
            st.session_state.chat=[]
            st.session_state.page="chat"; st.rerun()
        else: st.error("Scraping failed")

################################ LIST ################################
elif st.session_state.page=="list":
    st.header("All Chatbots")
    db=Session(); bots=db.query(Company).all()

    for b in bots:
        st.write(f"### {b.name}")
        if st.button(f"💬 Chat",key=b.slug):
            st.session_state.slug=b.slug; st.session_state.ai=AI(b.slug)
            st.session_state.ai.load()
            st.session_state.page="chat"; st.session_state.chat=[]; st.rerun()

################################ CHAT ################################
elif st.session_state.page=="chat":
    c=Session().query(Company).filter_by(slug=st.session_state.slug).first()
    st.header(f"Chat with {c.name}")

    # show history
    for m in st.session_state.chat:
        with st.chat_message(m["role"]): st.write(m["text"])

    q=st.chat_input("Ask something...")
    if q:
        with st.chat_message("user"): st.write(q)
        with st.chat_message("assistant"):
            ans=st.session_state.ai.ask(q)
            st.write(ans)

        st.session_state.chat.append({"role":"user","text":q})
        st.session_state.chat.append({"role":"assistant","text":ans})
        st.rerun()
