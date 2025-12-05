"""
COMPLETE AI CHATBOT SYSTEM - FIXED VERSION
Multi-company chatbot with web scraping, simple RAG (no embeddings), and Streamlit UI

FIXES APPLIED:
1. Enhanced _call_llm method with better error handling
2. Added detailed debug logging
3. Fixed response parsing logic
4. Added fallback mechanisms
5. Improved timeout handling
6. Better API key validation

Run with:
    streamlit run app.py
    
Set environment variable:
    export OPENROUTER_API_KEY="your_key_here"
"""

# =============================================================================
# OPTIONAL SQLITE FIX
# =============================================================================
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

# =============================================================================
# IMPORTS
# =============================================================================
import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base
import re
import os
import hashlib
import json
import time
from typing import Optional, Dict, List, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRAPER_MAX_SECONDS = 90
SCRAPER_REQUEST_TIMEOUT = 10
SCRAPER_SLEEP_BETWEEN = 0.2

DATABASE_URL = "sqlite:///./chatbot_database.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "meta-llama/llama-3.1-8b-instruct:free"

PRIORITY_PAGES = ["", "about", "services", "solutions", "products", "contact", "team"]

# =============================================================================
# DATABASE MODELS
# =============================================================================
class Company(Base):
    __tablename__ = "companies"
    id = Column(Integer, primary_key=True, index=True)
    company_name = Column(String(255), unique=True, nullable=False, index=True)
    company_slug = Column(String(255), unique=True, nullable=False, index=True)
    website_url = Column(String(500), nullable=False)
    emails = Column(Text, nullable=True)
    phones = Column(Text, nullable=True)
    address_india = Column(Text, nullable=True)
    address_international = Column(Text, nullable=True)
    pages_scraped = Column(Integer, default=0)
    last_scraped = Column(DateTime, nullable=True)
    scraping_status = Column(String(50), default="pending")
    is_active = Column(Boolean, default=True)
    max_pages_to_scrape = Column(Integer, default=40)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, onupdate=lambda: datetime.now(timezone.utc))

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    company_slug = Column(String(255), nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    session_id = Column(String(100), nullable=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class CompanyDocument(Base):
    __tablename__ = "company_documents"
    id = Column(Integer, primary_key=True, index=True)
    company_slug = Column(String(255), nullable=False, index=True)
    url = Column(String(500), nullable=False)
    title = Column(String(500), nullable=True)
    content = Column(Text, nullable=False)

Base.metadata.create_all(bind=engine)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def create_slug(company_name: str) -> str:
    slug = company_name.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")

def normalize_base_url(url: str) -> str:
    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    return url.rstrip("/") + "/"

# =============================================================================
# WEB SCRAPER
# =============================================================================
class WebScraper:
    def __init__(self, company_slug: str):
        self.company_slug = company_slug
        self.company_info = {
            "emails": set(),
            "phones": set(),
            "address_india": None,
            "address_international": None,
        }
        self.debug_info = []

    def clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s.,!?;:()\-\'\"]+", "", text)
        return text.strip()

    def extract_contact_info(self, text: str):
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        emails = re.findall(email_pattern, text)
        for email in emails:
            if not email.lower().endswith((".png", ".jpg", ".gif")):
                self.company_info["emails"].add(email.lower())

        phone_patterns = [
            r"\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
            r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        ]
        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            for phone in phones:
                cleaned = re.sub(r"[^\d+]", "", phone)
                if 7 <= len(cleaned) <= 15:
                    self.company_info["phones"].add(phone.strip())

    def is_valid_url(self, url: str, base_domain: str) -> bool:
        try:
            parsed = urlparse(url)
            if parsed.netloc != base_domain:
                return False
            skip_ext = [".pdf", ".jpg", ".png", ".gif", ".zip"]
            return not any(url.lower().endswith(ext) for ext in skip_ext)
        except:
            return False

    def extract_content(self, soup: BeautifulSoup, url: str) -> Dict:
        content_dict = {"url": url, "title": "", "content": ""}
        try:
            if soup.find("title"):
                content_dict["title"] = soup.find("title").get_text(strip=True)
            
            for tag in soup(["script", "style", "iframe"]):
                tag.decompose()
            
            text = soup.get_text(separator="\n", strip=True)
            self.extract_contact_info(text)
            
            lines = [l for l in text.split("\n") if len(l.strip()) > 20]
            content_dict["content"] = "\n".join(lines[:100])
        except Exception as e:
            self.debug_info.append(f"Error {url}: {str(e)[:80]}")
        
        return content_dict

    def scrape_website(self, base_url: str, max_pages: int = 40, progress_callback=None) -> Tuple[List[Dict], Dict]:
        start_time = time.time()
        max_pages = min(max_pages, 25)
        base_url = normalize_base_url(base_url)
        visited = set()
        queue = deque([urljoin(base_url, p) for p in PRIORITY_PAGES])
        base_domain = urlparse(base_url).netloc
        pages = []

        headers = {"User-Agent": "Mozilla/5.0 ChatbotScraper/1.0"}

        while queue and len(visited) < max_pages:
            if time.time() - start_time > SCRAPER_MAX_SECONDS:
                break

            url = queue.popleft().split("#")[0].split("?")[0]
            if url in visited or not self.is_valid_url(url, base_domain):
                continue

            try:
                response = requests.get(url, headers=headers, timeout=SCRAPER_REQUEST_TIMEOUT)
                if response.status_code != 200:
                    continue

                visited.add(url)
                if progress_callback:
                    progress_callback(len(visited), max_pages, url)

                soup = BeautifulSoup(response.text, "html.parser")
                content_data = self.extract_content(soup, url)
                
                if len(content_data["content"]) > 50:
                    pages.append(content_data)

                for link in soup.find_all("a", href=True):
                    next_url = urljoin(url, link["href"]).split("#")[0].split("?")[0]
                    if next_url not in visited and self.is_valid_url(next_url, base_domain):
                        queue.append(next_url)

                time.sleep(SCRAPER_SLEEP_BETWEEN)
            except:
                continue

        if len(pages) < 1:
            raise Exception(f"Insufficient content: {len(pages)} pages")

        return pages, {
            "emails": list(self.company_info["emails"]),
            "phones": list(self.company_info["phones"]),
            "address_india": self.company_info["address_india"],
            "address_international": self.company_info["address_international"],
            "pages_scraped": len(visited),
        }

# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================
def create_company(company_name: str, website_url: str, max_pages: int = 40) -> Optional[str]:
    db = SessionLocal()
    try:
        slug = create_slug(company_name)
        if db.query(Company).filter(Company.company_slug == slug).first():
            return None
        company = Company(
            company_name=company_name,
            company_slug=slug,
            website_url=website_url,
            max_pages_to_scrape=max_pages,
            scraping_status="pending",
        )
        db.add(company)
        db.commit()
        return slug
    except Exception as e:
        print(f"Error: {e}")
        db.rollback()
        return None
    finally:
        db.close()

def update_company_after_scraping(slug: str, company_info: Dict, status: str = "completed"):
    db = SessionLocal()
    try:
        company = db.query(Company).filter(Company.company_slug == slug).first()
        if company:
            company.emails = json.dumps(company_info.get("emails", []))
            company.phones = json.dumps(company_info.get("phones", []))
            company.pages_scraped = company_info.get("pages_scraped", 0)
            company.scraping_status = status
            company.last_scraped = datetime.now(timezone.utc)
            db.commit()
    except Exception as e:
        print(f"Error: {e}")
        db.rollback()
    finally:
        db.close()

def get_all_companies() -> List[Company]:
    db = SessionLocal()
    try:
        return db.query(Company).order_by(Company.created_at.desc()).all()
    finally:
        db.close()

def get_company_by_slug(slug: str) -> Optional[Company]:
    db = SessionLocal()
    try:
        return db.query(Company).filter(Company.company_slug == slug).first()
    finally:
        db.close()

def save_chat_message(company_slug: str, question: str, answer: str, session_id: str):
    db = SessionLocal()
    try:
        msg = ChatHistory(company_slug=company_slug, question=question, answer=answer, session_id=session_id)
        db.add(msg)
        db.commit()
    except:
        db.rollback()
    finally:
        db.close()

def replace_company_documents(slug: str, pages: List[Dict]):
    db = SessionLocal()
    try:
        db.query(CompanyDocument).filter(CompanyDocument.company_slug == slug).delete()
        db.commit()

        for p in pages:
            content = p["content"]
            if not content:
                continue
            
            sentences = re.split(r"(?<=[.!?])\s+", content)
            chunk = ""
            for sent in sentences:
                if len(chunk) + len(sent) < 800:
                    chunk += " " + sent
                else:
                    if len(chunk.strip()) > 50:
                        doc = CompanyDocument(
                            company_slug=slug,
                            url=p["url"],
                            title=p.get("title", "")[:490],
                            content=chunk.strip(),
                        )
                        db.add(doc)
                    chunk = sent
            
            if len(chunk.strip()) > 50:
                doc = CompanyDocument(
                    company_slug=slug,
                    url=p["url"],
                    title=p.get("title", "")[:490],
                    content=chunk.strip(),
                )
                db.add(doc)

        db.commit()
    except Exception as e:
        print(f"Error: {e}")
        db.rollback()
    finally:
        db.close()

def get_company_documents(slug: str) -> List[CompanyDocument]:
    db = SessionLocal()
    try:
        return db.query(CompanyDocument).filter(CompanyDocument.company_slug == slug).all()
    finally:
        db.close()

# =============================================================================
# KEYWORD RETRIEVER
# =============================================================================
STOPWORDS = set(["the", "is", "are", "a", "an", "of", "and", "to", "for", "on", "in"])

def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

def retrieve_relevant_chunks(company_slug: str, question: str, top_k: int = 5) -> List[CompanyDocument]:
    docs = get_company_documents(company_slug)
    if not docs:
        return []

    q_tokens = set(tokenize(question))
    if not q_tokens:
        return []

    scored = []
    for d in docs:
        c_tokens = set(tokenize(d.content))
        score = len(q_tokens & c_tokens)
        if score > 0:
            scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for score, d in scored[:top_k]]

# =============================================================================
# AI ENGINE - FIXED
# =============================================================================
class CompanyAI:
    def __init__(self, company_slug: str):
        self.company_slug = company_slug
        self.status = {"ready": False, "error": None}
        self.company_data = None

    def initialize(self, website_url: str, max_pages: int = 40, progress_callback=None):
        company_info = {"emails": [], "phones": [], "pages_scraped": 0}
        try:
            if progress_callback:
                progress_callback(0, max_pages, "Starting...")

            scraper = WebScraper(self.company_slug)
            pages, company_info = scraper.scrape_website(website_url, max_pages, progress_callback)
            
            replace_company_documents(self.company_slug, pages)
            self.company_data = company_info
            self.status["ready"] = True
            
            update_company_after_scraping(self.company_slug, company_info, "completed")
            
            if progress_callback:
                progress_callback(company_info["pages_scraped"], max_pages, "✅ Ready!")
            
            return True
        except Exception as e:
            self.status["error"] = str(e)
            update_company_after_scraping(self.company_slug, company_info, "failed")
            return False

    def load_existing(self):
        docs = get_company_documents(self.company_slug)
        if not docs:
            self.status["error"] = "No documents found"
            return False

        company = get_company_by_slug(self.company_slug)
        if company:
            try:
                self.company_data = {
                    "emails": json.loads(company.emails) if company.emails else [],
                    "phones": json.loads(company.phones) if company.phones else [],
                }
            except:
                self.company_data = {}

        self.status["ready"] = True
        return True

    def get_contact_info(self) -> str:
        if not self.company_data:
            return "Contact information not available."
        
        info = self.company_data
        msg = "📞 **CONTACT INFORMATION**\n\n"
        
        if info.get("emails"):
            msg += "📧 **Email:**\n" + "\n".join([f"• {e}" for e in info["emails"][:5]]) + "\n\n"
        
        if info.get("phones"):
            msg += "📱 **Phone:**\n" + "\n".join([f"• {p}" for p in info["phones"][:5]]) + "\n"
        
        return msg.strip()

    def ask(self, question: str, chat_history: List = None, session_id: str = None) -> str:
        try:
            if not self.status.get("ready"):
                return "⚠️ Chatbot not ready"

            q_lower = question.lower().strip()

            if q_lower in ["hi", "hello", "hey"] or len(q_lower) < 5:
                company = get_company_by_slug(self.company_slug)
                name = company.company_name if company else "our company"
                return f"👋 Hello! I'm here to answer questions about {name}. How can I help?"

            if any(k in q_lower for k in ["email", "contact", "phone"]):
                return self.get_contact_info()

            docs = retrieve_relevant_chunks(self.company_slug, question, top_k=5)
            if not docs:
                return "I couldn't find specific information about that. Please try rephrasing."

            context = "\n\n".join([f"{d.content[:400]}" for d in docs])

            company = get_company_by_slug(self.company_slug)
            company_name = company.company_name if company else "the company"

            prompt = f"""You are an AI assistant for {company_name}.

CONTEXT: {context[:2000]}

QUESTION: {question}

Provide a helpful, concise answer (2-3 sentences) based ONLY on the context above."""

            answer = self._call_llm(prompt)

            if answer and not answer.startswith("⚠️"):
                save_chat_message(self.company_slug, question, answer, session_id or "")
                return answer
            
            return answer or "I'm having trouble generating a response."

        except Exception as e:
            print(f"ERROR: {e}")
            return "⚠️ An error occurred. Please try again."

    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """FIXED: Enhanced error handling and response parsing"""
        
        if not OPENROUTER_API_KEY:
            return "⚠️ API key not configured. Please set OPENROUTER_API_KEY environment variable."

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Multi-Company Chatbot"
        }

        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 500,
        }

        for attempt in range(max_retries):
            try:
                print(f"[LLM] Attempt {attempt + 1}/{max_retries}")
                
                resp = requests.post(
                    OPENROUTER_API_BASE,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                
                print(f"[LLM] Status: {resp.status_code}")

                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        
                        # Check for error in response
                        if "error" in data:
                            error_msg = data["error"].get("message", "Unknown error")
                            print(f"[LLM] API Error: {error_msg}")
                            if attempt < max_retries - 1:
                                time.sleep(2 ** attempt)
                                continue
                            return f"⚠️ API Error: {error_msg}"
                        
                        # Extract content
                        if "choices" in data and len(data["choices"]) > 0:
                            content = data["choices"][0].get("message", {}).get("content", "")
                            if content and len(content.strip()) > 10:
                                print(f"[LLM] Success! Response length: {len(content)}")
                                return content.strip()
                        
                        print(f"[LLM] No valid content in response: {data.keys()}")
                        
                    except json.JSONDecodeError as e:
                        print(f"[LLM] JSON decode error: {e}")
                        print(f"[LLM] Raw response: {resp.text[:200]}")
                
                elif resp.status_code == 401:
                    return "⚠️ Invalid API key. Please check your OPENROUTER_API_KEY."
                
                elif resp.status_code == 429:
                    print("[LLM] Rate limit hit")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    return "⚠️ Rate limit reached. Please wait a moment and try again."
                
                elif resp.status_code >= 500:
                    print(f"[LLM] Server error: {resp.status_code}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                
                else:
                    print(f"[LLM] Unexpected status: {resp.status_code}")
                    print(f"[LLM] Response: {resp.text[:200]}")

                # Retry logic
                if attempt < max_retries - 1:
                    sleep_time = 2 ** attempt
                    print(f"[LLM] Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                    continue

            except requests.exceptions.Timeout:
                print("[LLM] Request timeout")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return "⚠️ Request timed out. Please try again."
            
            except requests.exceptions.RequestException as e:
                print(f"[LLM] Request error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
            
            except Exception as e:
                print(f"[LLM] Unexpected error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue

        return "⚠️ Could not get response after multiple attempts. Please check your API key and try again."

# =============================================================================
# STREAMLIT APP
# =============================================================================
def init_session_state():
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = hashlib.md5(str(time.time()).encode()).hexdigest()[:12]
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    if "current_company_slug" not in st.session_state:
        st.session_state["current_company_slug"] = None
    if "ai_engines" not in st.session_state:
        st.session_state["ai_engines"] = {}

def get_ai_for_company(slug: str) -> Optional[CompanyAI]:
    if not slug:
        return None
    if slug not in st.session_state["ai_engines"]:
        ai = CompanyAI(slug)
        ai.load_existing()
        st.session_state["ai_engines"][slug] = ai
    return st.session_state["ai_engines"].get(slug)

def render_sidebar():
    st.sidebar.title("⚙️ Admin Panel")
    
    with st.sidebar.expander("➕ Add New Company", expanded=False):
        name = st.text_input("Company Name", key="new_company_name")
        url = st.text_input("Website URL", key="new_company_url", placeholder="https://example.com")
        max_pages = st.slider("Max pages", 5, 80, 25, key="new_company_max_pages")

        if st.button("Create Company", key="create_company_btn"):
            if not name or not url:
                st.warning("Please provide both name and URL.")
            else:
                slug = create_company(name.strip(), url.strip(), max_pages)
                if slug is None:
                    st.error("Company already exists.")
                else:
                    st.success(f"Created: {slug}")
                    st.session_state["current_company_slug"] = slug
                    st.rerun()

    st.sidebar.subheader("Select Company")
    companies = get_all_companies()

    if not companies:
        st.sidebar.info("No companies yet.")
        return None, None, None

    company_map = {c.company_slug: f"{c.company_name}" for c in companies}
    slug_list = list(company_map.keys())
    label_list = list(company_map.values())

    default_idx = 0
    if st.session_state["current_company_slug"] in slug_list:
        default_idx = slug_list.index(st.session_state["current_company_slug"])

    selected_label = st.sidebar.selectbox("Choose Company", label_list, index=default_idx)
    selected_slug = slug_list[label_list.index(selected_label)]

    if st.session_state["current_company_slug"] != selected_slug:
        st.session_state["current_company_slug"] = selected_slug
        st.session_state["chat_messages"] = []

    selected_company = get_company_by_slug(selected_slug)

    with st.sidebar.expander("Company Details", expanded=True):
        if selected_company:
            st.write(f"**Name:** {selected_company.company_name}")
            st.write(f"**Website:** {selected_company.website_url}")
            st.write(f"**Pages:** {selected_company.pages_scraped or 0}")
            st.write(f"**Status:** {selected_company.scraping_status}")

    reinit = st.sidebar.button("🔄 Re-scrape", key="rescrape_btn")
    return selected_slug, selected_company, reinit

def main():
    st.set_page_config(page_title="Multi-Company AI Chatbot", page_icon="🤖", layout="wide")
    init_session_state()

    st.title("🤖 Multi-Company AI Chatbot")
    st.caption("Web Scraping + Simple RAG + LLM")

    if not OPENROUTER_API_KEY:
        st.error("⚠️ OPENROUTER_API_KEY not set! Set it as an environment variable.")
        st.code("export OPENROUTER_API_KEY='your_key_here'", language="bash")
        return

    selected_slug, selected_company, reinit = render_sidebar()

    col_left, col_right = st.columns([2, 1])

    with col_right:
        st.subheader("⚙️ Status")

        if not selected_company:
            st.info("Create or select a company")
        else:
            ai = get_ai_for_company(selected_slug)

            if reinit:
                progress = st.progress(0)
                status_text = st.empty()

                def callback(done, total, url):
                    progress.progress(min(done / max(total, 1), 1.0))
                    status_text.write(f"Scraping ({done}/{total})")

                ai = CompanyAI(selected_slug)
                st.session_state["ai_engines"][selected_slug] = ai

                ok = ai.initialize(
                    website_url=selected_company.website_url,
                    max_pages=selected_company.max_pages_to_scrape or 25,
                    progress_callback=callback,
                )

                if ok:
                    st.success("✅ Initialized!")
                    st.rerun()
                else:
                    st.error(f"❌ {ai.status.get('error')}")

            else:
                if ai and ai.status.get("ready"):
                    st.success("✅ Ready")
                elif ai and ai.status.get("error"):
                    st.error(f"❌ {ai.status['error']}")
                else:
                    if st.button("Initialize", key="init_btn"):
                        progress = st.progress(0)
                        status_text = st.empty()

                        def callback(done, total, url):
                            progress.progress(min(done / max(total, 1), 1.0))
                            status_text.write(f"Scraping ({done}/{total})")

                        ai = CompanyAI(selected_slug)
                        st.session_state["ai_engines"][selected_slug] = ai

                        ok = ai.initialize(
                            website_url=selected_company.website_url,
                            max_pages=selected_company.max_pages_to_scrape or 25,
                            progress_callback=callback,
                        )

                        if ok:
                            st.success("✅ Initialized!")
                            st.rerun()
                        else:
                            st.error(f"❌ {ai.status.get('error')}")

    with col_left:
        st.subheader("💬 Chat")

        if not selected_company:
            st.info("Select a company to start chatting")
            return

        ai = get_ai_for_company(selected_slug)

        if not ai or not ai.status.get("ready"):
            st.warning("⚠️ Chatbot not initialized")
            return

        for msg in st.session_state["chat_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_input := st.chat_input("Ask about this company..."):
            st.session_state["chat_messages"].append({"role": "user", "content": user_input})

            with st.chat_message("user"):
                st.markdown(user_input)

            recent_history = []
            msgs = st.session_state["chat_messages"]
            for i in range(0, len(msgs) - 1, 2):
                if i + 1 < len(msgs):
                    recent_history.append({
                        "question": msgs[i]["content"],
                        "answer": msgs[i + 1]["content"]
                    })

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = ai.ask(
                        user_input,
                        chat_history=recent_history[-3:],
                        session_id=st.session_state["session_id"],
                    )
                st.markdown(answer)

            st.session_state["chat_messages"].append({"role": "assistant", "content": answer})
            st.rerun()


if __name__ == "__main__":
    main()
