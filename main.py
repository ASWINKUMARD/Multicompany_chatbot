"""
COMPLETE AI CHATBOT SYSTEM - ALL-IN-ONE FILE
Multi-company chatbot with web scraping, RAG (FAISS), and Streamlit UI

Run with:
    streamlit run app.py
"""

# =============================================================================
# OPTIONAL SQLITE PATCH (mostly for local Windows / Colab)
# On Streamlit Cloud this will usually fall back to default sqlite3.
# =============================================================================
import sys

try:
    import pysqlite3  # type: ignore
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    # Use built-in sqlite3
    import sqlite3  # noqa: F401

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
import shutil

# LangChain & FAISS
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS  # older LC versions

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        from langchain.embeddings import HuggingFaceEmbeddings

try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.docstore.document import Document
    except ImportError:
        from langchain.schema import Document

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate


# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

DATABASE_URL = "sqlite:///./chatbot_database.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Company(Base):
    """Database model for companies"""
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
    """Database model for chat history"""
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    company_slug = Column(String(255), nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    session_id = Column(String(100), nullable=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


Base.metadata.create_all(bind=engine)

# =============================================================================
# API CONFIGURATION
# =============================================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "kwaipilot/kat-coder-pro:free"

PRIORITY_PAGES = [
    "",
    "about",
    "services",
    "solutions",
    "products",
    "contact",
    "team",
    "careers",
    "about-us",
    "contact-us",
    "our-services",
    "home",
    "portfolio",
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_slug(company_name: str) -> str:
    """Convert company name to URL-friendly slug"""
    slug = company_name.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


def get_faiss_directory(company_slug: str) -> str:
    """Directory to store FAISS index for this company"""
    base_dir = "./faiss_db"
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, company_slug)


# =============================================================================
# WEB SCRAPER CLASS
# =============================================================================


class WebScraper:
    """Web scraper with intelligent content extraction"""

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
        """Clean and normalize text"""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s.,!?;:()\-\'\"]+", "", text)
        return text.strip()

    def extract_contact_info(self, text: str):
        """Extract contact information from text"""
        # Extract emails
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        emails = re.findall(email_pattern, text)
        for email in emails:
            if not email.lower().endswith((".png", ".jpg", ".gif", ".css", ".js")):
                self.company_info["emails"].add(email.lower())

        # Extract phone numbers
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

        # Extract addresses (very heuristic)
        lines = text.split("\n")
        for i, line in enumerate(lines):
            low = line.lower()

            india_keywords = [
                "india",
                "mumbai",
                "delhi",
                "bangalore",
                "chennai",
                "madurai",
                "pune",
            ]
            if any(city in low for city in india_keywords):
                if not self.company_info["address_india"]:
                    block = " ".join(lines[max(0, i - 2) : min(len(lines), i + 4)])
                    cleaned = self.clean_text(block)
                    if 20 < len(cleaned) < 500:
                        self.company_info["address_india"] = cleaned

            intl_keywords = ["singapore", "usa", "uk", "uae", "dubai", "london"]
            if any(country in low for country in intl_keywords):
                if not self.company_info["address_international"]:
                    block = " ".join(lines[max(0, i - 2) : min(len(lines), i + 4)])
                    cleaned = self.clean_text(block)
                    if 20 < len(cleaned) < 500:
                        self.company_info["address_international"] = cleaned

    def is_valid_url(self, url: str, base_domain: str) -> bool:
        """Check if URL should be scraped"""
        try:
            parsed = urlparse(url)
            if parsed.netloc != base_domain:
                return False

            skip_extensions = [
                ".pdf",
                ".jpg",
                ".jpeg",
                ".png",
                ".gif",
                ".zip",
                ".mp4",
                ".css",
                ".js",
                ".svg",
                ".ico",
            ]
            skip_paths = [
                "/wp-admin/",
                "/admin/",
                "/login",
                "/cart/",
                "/checkout/",
                "/privacy",
                "/terms",
            ]

            url_lower = url.lower()

            for ext in skip_extensions:
                if url_lower.endswith(ext):
                    return False

            for path in skip_paths:
                if path in url_lower:
                    return False

            return True
        except Exception:
            return False

    def extract_content(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract meaningful content from page"""
        content_dict = {"url": url, "title": "", "content": ""}

        try:
            if soup.find("title"):
                content_dict["title"] = soup.find("title").get_text(strip=True)

            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc and meta_desc.get("content"):
                content_dict["content"] += meta_desc["content"] + "\n\n"

            full_text = soup.get_text(separator="\n", strip=True)
            self.extract_contact_info(full_text)

            for tag in soup(["script", "style", "iframe", "nav", "footer", "header"]):
                tag.decompose()

            main_selectors = ["main", "article", "[role='main']", ".content", "#content"]
            texts = []

            for selector in main_selectors:
                elements = soup.select(selector)
                for elem in elements:
                    text = elem.get_text(separator="\n", strip=True)
                    if len(text) > 100:
                        texts.append(text)

            for tag_name in ["h1", "h2", "h3", "p"]:
                for tag in soup.find_all(tag_name):
                    text = tag.get_text(strip=True)
                    if len(text) > 20:
                        texts.append(text)

            if len(texts) < 5:
                body = soup.find("body")
                if body:
                    text = body.get_text(separator="\n", strip=True)
                    if text:
                        texts.append(text)

            if texts:
                combined = "\n".join(texts)
                lines = [
                    self.clean_text(line)
                    for line in combined.split("\n")
                    if len(line.strip()) > 15
                ]

                seen = set()
                unique_lines = []
                for line in lines:
                    line_lower = line.lower()
                    if line_lower not in seen and len(line) > 20:
                        seen.add(line_lower)
                        unique_lines.append(line)

                content_dict["content"] += "\n".join(unique_lines)

        except Exception as e:
            self.debug_info.append(f"Error {url}: {str(e)[:50]}")

        return content_dict

    def scrape_website(
        self,
        base_url: str,
        max_pages: int = 40,
        progress_callback=None,
    ) -> Tuple[List[Document], Dict]:
        """Scrape website and return Documents"""
        visited = set()
        queue = deque()
        base_domain = urlparse(base_url).netloc

        base_url = base_url.rstrip("/") + "/"

        for page in PRIORITY_PAGES:
            for url in [urljoin(base_url, page), urljoin(base_url, page + "/")]:
                if url not in queue:
                    queue.append(url)

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        documents = []

        while queue and len(visited) < max_pages:
            url = queue.popleft()
            url = url.split("#")[0].split("?")[0].rstrip("/")

            if url in visited or not self.is_valid_url(url, base_domain):
                continue

            try:
                response = requests.get(url, headers=headers, timeout=20)

                if response.status_code != 200:
                    continue

                visited.add(url)

                if progress_callback:
                    progress_callback(len(visited), max_pages, url)

                soup = BeautifulSoup(response.text, "html.parser")
                content_data = self.extract_content(soup, url)

                if len(content_data["content"]) > 50:
                    doc = Document(
                        page_content=content_data["content"],
                        metadata={
                            "source": url,
                            "title": content_data["title"],
                            "company": self.company_slug,
                        },
                    )
                    documents.append(doc)

                for link in soup.find_all("a", href=True):
                    try:
                        next_url = urljoin(url, link["href"])
                        next_url = (
                            next_url.split("#")[0].split("?")[0].rstrip("/")
                        )

                        if (
                            next_url not in visited
                            and self.is_valid_url(next_url, base_domain)
                            and next_url not in queue
                        ):
                            queue.append(next_url)
                    except Exception:
                        pass

                time.sleep(0.3)

            except Exception as e:
                self.debug_info.append(f"Error {url}: {str(e)[:50]}")
                continue

        if len(documents) < 3:
            raise Exception(f"Insufficient content: {len(documents)} pages")

        return documents, {
            "emails": list(self.company_info["emails"]),
            "phones": list(self.company_info["phones"]),
            "address_india": self.company_info["address_india"],
            "address_international": self.company_info["address_international"],
            "pages_scraped": len(visited),
        }


# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================


def create_company(
    company_name: str,
    website_url: str,
    max_pages: int = 40,
) -> Optional[str]:
    """Create new company"""
    db = SessionLocal()
    try:
        slug = create_slug(company_name)

        existing = db.query(Company).filter(Company.company_slug == slug).first()
        if existing:
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
        db.refresh(company)

        return slug
    except Exception as e:
        print(f"Error in create_company: {e}")
        db.rollback()
        return None
    finally:
        db.close()


def update_company_after_scraping(
    slug: str,
    company_info: Dict,
    status: str = "completed",
):
    """Update company after scraping"""
    db = SessionLocal()
    try:
        company = db.query(Company).filter(Company.company_slug == slug).first()
        if company:
            company.emails = json.dumps(company_info.get("emails", []))
            company.phones = json.dumps(company_info.get("phones", []))
            company.address_india = company_info.get("address_india")
            company.address_international = company_info.get("address_international")
            company.pages_scraped = company_info.get("pages_scraped", 0)
            company.scraping_status = status
            company.last_scraped = datetime.now(timezone.utc)
            db.commit()
    except Exception as e:
        print(f"Error in update_company_after_scraping: {e}")
        db.rollback()
    finally:
        db.close()


def get_all_companies() -> List[Company]:
    """Get all companies"""
    db = SessionLocal()
    try:
        return db.query(Company).order_by(Company.created_at.desc()).all()
    finally:
        db.close()


def get_company_by_slug(slug: str) -> Optional[Company]:
    """Get company by slug"""
    db = SessionLocal()
    try:
        return db.query(Company).filter(Company.company_slug == slug).first()
    finally:
        db.close()


def save_chat_message(
    company_slug: str,
    question: str,
    answer: str,
    session_id: str,
):
    """Save chat message to DB"""
    db = SessionLocal()
    try:
        msg = ChatHistory(
            company_slug=company_slug,
            question=question,
            answer=answer,
            session_id=session_id,
        )
        db.add(msg)
        db.commit()
    except Exception as e:
        print(f"Error saving chat: {e}")
        db.rollback()
    finally:
        db.close()


def get_chat_history(
    company_slug: str,
    session_id: Optional[str],
    limit: int = 10,
) -> List[Dict]:
    """Load recent chat history for context"""
    db = SessionLocal()
    try:
        q = db.query(ChatHistory).filter(ChatHistory.company_slug == company_slug)
        if session_id:
            q = q.filter(ChatHistory.session_id == session_id)

        rows = q.order_by(ChatHistory.timestamp.desc()).limit(limit).all()
        rows = list(reversed(rows))

        history = []
        for r in rows:
            history.append(
                {
                    "question": r.question,
                    "answer": r.answer,
                    "timestamp": r.timestamp,
                }
            )
        return history
    finally:
        db.close()


# =============================================================================
# AI ENGINE CLASS (FAISS VERSION)
# =============================================================================


class CompanyAI:
    """AI Engine with RAG using FAISS"""

    def __init__(self, company_slug: str):
        self.company_slug = company_slug
        self.vectorstore = None
        self.retriever = None
        self.embeddings = None
        self.status = {"ready": False, "error": None}
        self.company_data = None

        self.qa_template = """You are an intelligent AI assistant for {company_name}. Answer accurately using the provided context.

CONTEXT FROM COMPANY WEBSITE:
{context}

RECENT CONVERSATION:
{chat_history}

USER QUESTION: {question}

INSTRUCTIONS:
1. Answer using ONLY information from the context
2. Be specific, detailed, and helpful
3. If information is not in context, say "I don't have that information. Here's how to contact us: [provide contact details]"
4. Be conversational and friendly
5. Include relevant details when appropriate

ANSWER (2-5 sentences):"""

        self.qa_prompt = PromptTemplate(
            template=self.qa_template,
            input_variables=["company_name", "context", "chat_history", "question"],
        )

    # -------------------------------------------------------------------------
    # INITIALIZATION / LOADING
    # -------------------------------------------------------------------------
    def initialize(
        self,
        website_url: str,
        max_pages: int = 40,
        progress_callback=None,
    ):
        """Initialize by scraping and creating FAISS vector store"""
        try:
            if progress_callback:
                progress_callback(0, max_pages, "Starting scraper...")

            scraper = WebScraper(self.company_slug)
            documents, company_info = scraper.scrape_website(
                website_url,
                max_pages,
                progress_callback,
            )

            if len(documents) < 3:
                self.status["error"] = f"Not enough content: {len(documents)} pages"
                update_company_after_scraping(
                    self.company_slug,
                    {**company_info, "pages_scraped": company_info.get("pages_scraped", 0)},
                    status="failed",
                )
                return False

            self.company_data = company_info

            if progress_callback:
                progress_callback(max_pages, max_pages, "Processing documents...")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "],
                length_function=len,
            )

            split_docs = text_splitter.split_documents(documents)

            if len(split_docs) < 5:
                self.status["error"] = f"Not enough chunks: {len(split_docs)}"
                update_company_after_scraping(
                    self.company_slug,
                    {**company_info, "pages_scraped": company_info.get("pages_scraped", 0)},
                    status="failed",
                )
                return False

            if progress_callback:
                progress_callback(max_pages, max_pages, "Creating embeddings...")

            # CPU-only embeddings to avoid meta-tensor issues
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

            faiss_dir = get_faiss_directory(self.company_slug)
            if os.path.exists(faiss_dir):
                shutil.rmtree(faiss_dir)
            os.makedirs(faiss_dir, exist_ok=True)

            if progress_callback:
                progress_callback(max_pages, max_pages, "Building FAISS index...")

            self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)

            # Persist FAISS index
            self.vectorstore.save_local(faiss_dir)

            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 6, "fetch_k": 15, "lambda_mult": 0.7},
            )

            self.status["ready"] = True

            if progress_callback:
                progress_callback(max_pages, max_pages, "✅ Chatbot Ready!")

            update_company_after_scraping(
                self.company_slug,
                {**company_info, "pages_scraped": company_info.get("pages_scraped", 0)},
                status="completed",
            )

            return True

        except Exception as e:
            self.status["error"] = f"Error: {str(e)}"
            print(f"Initialization error: {e}")
            import traceback

            traceback.print_exc()

            update_company_after_scraping(
                self.company_slug,
                {
                    "emails": [],
                    "phones": [],
                    "address_india": None,
                    "address_international": None,
                    "pages_scraped": 0,
                },
                status="failed",
            )
            return False

    def load_existing(self):
        """Load existing FAISS vector store from disk"""
        try:
            faiss_dir = get_faiss_directory(self.company_slug)
            index_file = os.path.join(faiss_dir, "index.faiss")
            if not os.path.exists(index_file):
                self.status["error"] = "Chatbot data not found (FAISS index missing)"
                return False

            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

            # Compatibility with different LC versions
            try:
                self.vectorstore = FAISS.load_local(
                    faiss_dir,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            except TypeError:
                self.vectorstore = FAISS.load_local(faiss_dir, self.embeddings)

            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 6, "fetch_k": 15, "lambda_mult": 0.7},
            )

            company = get_company_by_slug(self.company_slug)
            if company:
                try:
                    self.company_data = {
                        "emails": json.loads(company.emails) if company.emails else [],
                        "phones": json.loads(company.phones) if company.phones else [],
                        "address_india": company.address_india,
                        "address_international": company.address_international,
                    }
                except Exception:
                    self.company_data = {}

            self.status["ready"] = True
            return True

        except Exception as e:
            self.status["error"] = f"Load error: {str(e)}"
            print(f"Load error: {e}")
            return False

    # -------------------------------------------------------------------------
    # CONTACT INFO
    # -------------------------------------------------------------------------
    def get_contact_info(self):
        """Format contact information"""
        if not self.company_data:
            return "Contact information not available."

        info = self.company_data
        msg = "📞 **CONTACT INFORMATION**\n\n"

        if info.get("address_india"):
            msg += f"🏢 **India Office:**\n{info['address_india']}\n\n"

        if info.get("address_international"):
            msg += f"🌍 **International Office:**\n{info['address_international']}\n\n"

        if info.get("emails"):
            msg += "📧 **Email:**\n" + "\n".join([f"• {e}" for e in info["emails"][:5]]) + "\n\n"

        if info.get("phones"):
            msg += "📱 **Phone:**\n" + "\n".join([f"• {p}" for p in info["phones"][:5]]) + "\n"

        return msg.strip()

    # -------------------------------------------------------------------------
    # MAIN Q&A
    # -------------------------------------------------------------------------
    def ask(
        self,
        question: str,
        chat_history: List = None,
        session_id: str = None,
    ) -> str:
        """Answer question using RAG"""
        try:
            if not self.status.get("ready", False):
                return (
                    "⚠️ System is initializing or not ready. "
                    "Please initialize or re-scrape from the sidebar."
                )

            if not self.retriever:
                return "⚠️ Chatbot not properly initialized. Please try reloading or re-scraping."

            q_lower = question.lower().strip()

            # Simple greeting
            greetings = ["hi", "hello", "hey", "hai"]
            if q_lower in greetings or len(q_lower) < 5:
                company = get_company_by_slug(self.company_slug)
                company_name = company.company_name if company else "our company"
                return f"👋 Hello! I'm here to answer questions about {company_name}. How can I help you today?"

            # Direct contact info requests
            contact_keywords = [
                "email",
                "contact",
                "phone",
                "address",
                "office",
                "location",
                "reach",
            ]
            if any(keyword in q_lower for keyword in contact_keywords):
                return self.get_contact_info()

            print(f"\n=== Processing question: {question} ===")

            # Retrieve relevant docs
            print("Retrieving relevant documents...")
            try:
                if hasattr(self.retriever, "invoke"):
                    relevant_docs = self.retriever.invoke(question)
                else:
                    relevant_docs = self.retriever.get_relevant_documents(question)
            except Exception as e:
                print(f"Retrieval error: {e}")
                return "⚠️ Error retrieving information. Please try again."

            print(f"Found {len(relevant_docs)} relevant documents")

            if not relevant_docs or len(relevant_docs) == 0:
                return (
                    "I couldn't find specific information about that in the website content. "
                    "Could you rephrase your question or ask something else?"
                )

            # Build context
            context_parts = []
            for i, doc in enumerate(relevant_docs[:5]):
                if hasattr(doc, "page_content") and doc.page_content:
                    source = doc.metadata.get("source", "unknown")
                    title = doc.metadata.get("title", "")

                    context_part = f"[From: {title if title else source}]\n{doc.page_content[:500]}"
                    context_parts.append(context_part)
                    print(f"Doc {i+1}: {len(doc.page_content)} chars from {source}")

            if not context_parts:
                return "I found documents but couldn't extract content. Please try again."

            context = "\n\n---\n\n".join(context_parts)
            print(f"Total context length: {len(context)} chars")

            history_text = ""
            if chat_history:
                history_text = "\n".join(
                    [
                        f"User: {msg['question']}\nAssistant: {msg['answer']}"
                        for msg in chat_history[-3:]
                    ]
                )

            company = get_company_by_slug(self.company_slug)
            company_name = company.company_name if company else "the company"

            prompt = self.qa_prompt.format(
                company_name=company_name,
                context=context[:4000],
                chat_history=history_text,
                question=question,
            )

            print(f"Prompt length: {len(prompt)} chars")
            print("Calling LLM...")

            answer = self._call_llm(prompt)

            print(f"Answer received: {answer[:100] if answer else 'None'}...")

            if answer and not answer.startswith("⚠️"):
                save_chat_message(self.company_slug, question, answer, session_id or "")
                return answer
            elif answer:
                return answer
            else:
                return "I'm having trouble generating a response. Please try again."

        except Exception as e:
            print(f"ERROR in ask(): {str(e)}")
            import traceback

            traceback.print_exc()
            return "⚠️ An internal error occurred. Please try again."

    # -------------------------------------------------------------------------
    # LLM CALL
    # -------------------------------------------------------------------------
    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Call LLM API"""
        if not OPENROUTER_API_KEY:
            print("ERROR: OPENROUTER_API_KEY not set!")
            return (
                "⚠️ API key not configured. Please set OPENROUTER_API_KEY environment variable "
                "on the server or Streamlit Cloud."
            )

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "AI Chatbot",
        }

        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Answer accurately based on the provided context only.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.5,
            "max_tokens": 600,
        }

        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1}/{max_retries} - Calling LLM...")

                response = requests.post(
                    OPENROUTER_API_BASE,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )

                print(f"Response status: {response.status_code}")

                if response.status_code == 200:
                    result = response.json()

                    if "choices" in result and len(result["choices"]) > 0:
                        answer = result["choices"][0]["message"]["content"].strip()

                        if answer and len(answer) > 10:
                            return answer
                        else:
                            print(f"Answer too short: {answer}")
                    else:
                        print(f"No choices in response: {result}")

                elif response.status_code == 429:
                    print("Rate limited, waiting before retry...")
                    if attempt < max_retries - 1:
                        time.sleep(3)
                        continue
                    else:
                        return "⚠️ API rate limit reached. Please wait a bit and try again."

                elif response.status_code == 401:
                    print("Authentication failed!")
                    return "⚠️ Invalid API key. Please check your OPENROUTER_API_KEY."

                else:
                    error_text = response.text[:300]
                    print(f"Error response ({response.status_code}): {error_text}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        return (
                            "⚠️ The language model API returned an error. "
                            "Please try again later."
                        )

            except requests.exceptions.Timeout:
                print(f"Request timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    return "⚠️ Request to the AI model timed out. Please try again."

            except Exception as e:
                print(f"LLM error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    return (
                        "⚠️ An unexpected error occurred while contacting the AI model. "
                        "Please try again."
                    )

        return "⚠️ Could not get a response from the AI model after multiple attempts."


# =============================================================================
# STREAMLIT APP
# =============================================================================


def init_session_state():
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = hashlib.md5(
            str(time.time()).encode("utf-8")
        ).hexdigest()[:12]

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    if "current_company_slug" not in st.session_state:
        st.session_state["current_company_slug"] = None

    if "ai_engines" not in st.session_state:
        st.session_state["ai_engines"] = {}  # slug -> CompanyAI instance


def get_ai_for_company(slug: str) -> Optional[CompanyAI]:
    if slug is None:
        return None

    if slug not in st.session_state["ai_engines"]:
        ai = CompanyAI(slug)
        loaded = ai.load_existing()
        st.session_state["ai_engines"][slug] = ai
        if not loaded:
            print(f"FAISS index not yet built for {slug}")

    return st.session_state["ai_engines"].get(slug)


def render_sidebar():
    st.sidebar.title("⚙️ Admin Panel")

    st.sidebar.subheader("Add / Configure Company")

    with st.sidebar.expander("➕ Add New Company", expanded=False):
        name = st.text_input("Company Name", key="new_company_name")
        url = st.text_input(
            "Website URL",
            key="new_company_url",
            placeholder="https://example.com",
        )
        max_pages = st.slider(
            "Max pages to scrape",
            5,
            80,
            40,
            key="new_company_max_pages",
        )

        if st.button("Create Company", key="create_company_btn"):
            if not name or not url:
                st.warning("Please provide both company name and website URL.")
            else:
                slug = create_company(
                    name.strip(),
                    url.strip(),
                    max_pages=max_pages,
                )
                if slug is None:
                    st.error("A company with that name already exists.")
                else:
                    st.success(f"Company created with slug: {slug}")
                    st.session_state["current_company_slug"] = slug
                    if slug in st.session_state["ai_engines"]:
                        del st.session_state["ai_engines"][slug]

    st.sidebar.subheader("Existing Companies")
    companies = get_all_companies()
    if not companies:
        st.sidebar.info("No companies added yet.")
        return None, None, None

    slug_to_label = {
        c.company_slug: f"{c.company_name} ({c.company_slug})" for c in companies
    }
    slug_list = list(slug_to_label.keys())
    label_list = [slug_to_label[s] for s in slug_list]

    default_idx = 0
    if st.session_state["current_company_slug"] in slug_list:
        default_idx = slug_list.index(st.session_state["current_company_slug"])

    selected_label = st.sidebar.selectbox(
        "Select Company",
        label_list,
        index=default_idx if label_list else 0,
    )
    selected_slug = slug_list[label_list.index(selected_label)]
    st.session_state["current_company_slug"] = selected_slug

    selected_company = get_company_by_slug(selected_slug)

    with st.sidebar.expander("Company Details", expanded=True):
        if selected_company:
            st.write(f"**Name:** {selected_company.company_name}")
            st.write(f"**Slug:** `{selected_company.company_slug}`")
            st.write(f"**Website:** {selected_company.website_url}")
            st.write(f"**Pages Scraped:** {selected_company.pages_scraped or 0}")
            st.write(f"**Status:** {selected_company.scraping_status}")
            if selected_company.last_scraped:
                st.write(f"**Last Scraped:** {selected_company.last_scraped}")
        else:
            st.info("Company not found in DB")

    reinit = st.sidebar.button(
        "🔄 Re-scrape & Rebuild Chatbot",
        key="rescrape_btn",
    )

    return selected_slug, selected_company, reinit


def main():
    st.set_page_config(
        page_title="Multi-Company AI Chatbot (FAISS)",
        page_icon="🤖",
        layout="wide",
    )
    init_session_state()

    st.title("🤖 Multi-Company AI Chatbot (FAISS)")
    st.caption("Web scraping + RAG (FAISS) + OpenRouter LLM + Streamlit UI")

    if not OPENROUTER_API_KEY:
        st.warning(
            "OPENROUTER_API_KEY environment variable is not set. "
            "The chatbot will not be able to call the LLM API."
        )

    selected_slug, selected_company, reinit = render_sidebar()

    col_left, col_right = st.columns([2, 1])

    # -------------------------------------------------------------------------
    # RIGHT COLUMN: INIT / STATUS
    # -------------------------------------------------------------------------
    with col_right:
        st.subheader("⚙️ Chatbot Status")

        if not selected_company:
            st.info("Select or create a company to get started.")
        else:
            ai = get_ai_for_company(selected_slug)

            if reinit:
                progress = st.progress(0)
                status_text = st.empty()

                def callback(done, total, url):
                    ratio = min(done / max(total, 1), 1.0)
                    progress.progress(ratio)
                    status_text.write(f"Scraping ({done}/{total}) - {url}")

                status_text.write("Starting scraping and initialization...")
                ai = CompanyAI(selected_slug)
                st.session_state["ai_engines"][selected_slug] = ai

                update_company_after_scraping(
                    selected_slug,
                    {
                        "emails": [],
                        "phones": [],
                        "address_india": None,
                        "address_international": None,
                        "pages_scraped": 0,
                    },
                    status="running",
                )

                ok = ai.initialize(
                    website_url=selected_company.website_url,
                    max_pages=selected_company.max_pages_to_scrape or 40,
                    progress_callback=callback,
                )

                if ok:
                    st.success("Chatbot initialized successfully!")
                else:
                    st.error(f"Initialization failed: {ai.status.get('error')}")

            else:
                if ai and ai.status.get("ready", False):
                    st.success("Chatbot is ready to answer questions.")
                elif ai and ai.status.get("error"):
                    st.error(f"Chatbot error: {ai.status['error']}")
                    if st.button("Retry Load Existing Data", key="retry_load_btn"):
                        loaded = ai.load_existing()
                        if loaded:
                            st.success("Loaded existing FAISS index successfully.")
                        else:
                            st.error(f"Load failed: {ai.status.get('error')}")
                else:
                    if st.button(
                        "Initialize from Website Content",
                        key="init_from_website_btn",
                    ):
                        progress = st.progress(0)
                        status_text = st.empty()

                        def callback(done, total, url):
                            ratio = min(done / max(total, 1), 1.0)
                            progress.progress(ratio)
                            status_text.write(f"Scraping ({done}/{total}) - {url}")

                        status_text.write("Starting scraping and initialization...")
                        ai = CompanyAI(selected_slug)
                        st.session_state["ai_engines"][selected_slug] = ai

                        update_company_after_scraping(
                            selected_slug,
                            {
                                "emails": [],
                                "phones": [],
                                "address_india": None,
                                "address_international": None,
                                "pages_scraped": 0,
                            },
                            status="running",
                        )

                        ok = ai.initialize(
                            website_url=selected_company.website_url,
                            max_pages=selected_company.max_pages_to_scrape or 40,
                            progress_callback=callback,
                        )

                        if ok:
                            st.success("Chatbot initialized successfully!")
                        else:
                            st.error(f"Initialization failed: {ai.status.get('error')}")

            if ai and ai.company_data:
                with st.expander("📞 Contact Info (Detected)", expanded=False):
                    st.markdown(ai.get_contact_info())

    # -------------------------------------------------------------------------
    # LEFT COLUMN: CHAT UI
    # -------------------------------------------------------------------------
    with col_left:
        st.subheader("💬 Chat Interface")

        if not selected_company:
            st.info("Please create or select a company from the sidebar to start chatting.")
            return

        ai = get_ai_for_company(selected_slug)

        if not ai or not ai.status.get("ready", False):
            st.warning(
                "Chatbot is not yet initialized for this company. "
                "Please initialize or re-scrape from the sidebar."
            )
            return

        # Load DB history for this session if chat is empty
        db_history = get_chat_history(
            selected_slug,
            st.session_state["session_id"],
            limit=20,
        )

        if not st.session_state["chat_messages"] and db_history:
            for msg in db_history:
                st.session_state["chat_messages"].append(
                    {"role": "user", "content": msg["question"]}
                )
                st.session_state["chat_messages"].append(
                    {"role": "assistant", "content": msg["answer"]}
                )

        # Display conversation
        for msg in st.session_state["chat_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        user_input = st.chat_input("Ask something about this company...")
        if user_input:
            st.session_state["chat_messages"].append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.markdown(user_input)

            # Build recent history pairs (user -> assistant)
            recent_history = []
            msgs = st.session_state["chat_messages"]
            for i in range(len(msgs) - 1):
                if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                    recent_history.append(
                        {
                            "question": msgs[i]["content"],
                            "answer": msgs[i + 1]["content"],
                        }
                    )

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = ai.ask(
                        user_input,
                        chat_history=recent_history,
                        session_id=st.session_state["session_id"],
                    )
                    st.markdown(answer)

            st.session_state["chat_messages"].append(
                {"role": "assistant", "content": answer}
            )


if __name__ == "__main__":
    main()
