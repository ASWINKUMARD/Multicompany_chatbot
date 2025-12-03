__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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

# FIXED: Compatible imports for all LangChain versions
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain.vectorstores import Chroma

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        from langchain.embeddings import HuggingFaceEmbeddings

# FIXED: Correct Document import
try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.docstore.document import Document
    except ImportError:
        from langchain.schema import Document

DATABASE_URL = "sqlite:///./multi_company_chatbots.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Company(Base):
    """Stores information about each company"""
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
    """Stores chat conversations"""
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True)
    company_slug = Column(String(255), nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    session_id = Column(String(100), nullable=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class UserContact(Base):
    """Stores user contact information"""
    __tablename__ = "user_contacts"
    
    id = Column(Integer, primary_key=True, index=True)
    company_slug = Column(String(255), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False)
    phone = Column(String(20), nullable=False)
    session_id = Column(String(100), nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


Base.metadata.create_all(bind=engine)


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.0-flash-exp:free"

PRIORITY_PAGES = [
    "", "about", "services", "solutions", "products", "contact", "team",
    "careers", "about-us", "contact-us", "our-services", "home"
]



def create_slug(company_name: str) -> str:
    """Convert company name to URL-friendly slug"""
    slug = company_name.lower()
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    return slug.strip('-')


def get_chroma_directory(company_slug: str) -> str:
    """Get ChromaDB directory path"""
    return f"./chroma_db/{company_slug}"


class WebScraper:
    """Web scraper with content extraction"""
    
    def __init__(self, company_slug: str):
        self.company_slug = company_slug
        self.company_info = {
            'emails': set(),
            'phones': set(),
            'address_india': None,
            'address_international': None
        }
        self.debug_info = []
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"]+', '', text)
        return text.strip()
    
    def extract_contact_info(self, text: str):
        """Extract contact information"""
        # Emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        for email in emails:
            if not email.lower().endswith(('.png', '.jpg', '.gif')):
                self.company_info['emails'].add(email.lower())
        
        # Phones
        phone_patterns = [
            r'\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        ]
        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            for phone in phones:
                cleaned = re.sub(r'[^\d+]', '', phone)
                if 7 <= len(cleaned) <= 15:
                    self.company_info['phones'].add(phone.strip())
        
        # Addresses
        lines = text.split('\n')
        for i, line in enumerate(lines):
            low = line.lower()
            
            india_keywords = ['india', 'mumbai', 'delhi', 'bangalore', 'chennai', 'madurai']
            if any(city in low for city in india_keywords):
                if not self.company_info['address_india']:
                    block = " ".join(lines[max(0, i-2):min(len(lines), i+4)])
                    cleaned = self.clean_text(block)
                    if 20 < len(cleaned) < 500:
                        self.company_info['address_india'] = cleaned
            
            intl_keywords = ['singapore', 'usa', 'uk', 'uae', 'dubai']
            if any(country in low for country in intl_keywords):
                if not self.company_info['address_international']:
                    block = " ".join(lines[max(0, i-2):min(len(lines), i+4)])
                    cleaned = self.clean_text(block)
                    if 20 < len(cleaned) < 500:
                        self.company_info['address_international'] = cleaned
    
    def is_valid_url(self, url: str, base_domain: str) -> bool:
        """Check if URL should be scraped"""
        try:
            parsed = urlparse(url)
            if parsed.netloc != base_domain:
                return False
            
            skip_extensions = ['.pdf', '.jpg', '.png', '.zip', '.mp4', '.css', '.js']
            skip_paths = ['/wp-admin/', '/admin/', '/login', '/cart/']
            
            url_lower = url.lower()
            
            for ext in skip_extensions:
                if url_lower.endswith(ext):
                    return False
            
            for path in skip_paths:
                if path in url_lower:
                    return False
            
            return True
        except:
            return False
    
    def extract_content(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract content from page"""
        content_dict = {'url': url, 'title': '', 'content': ''}
        
        try:
            if soup.find('title'):
                content_dict['title'] = soup.find('title').get_text(strip=True)
            
            meta_desc = soup.find('meta', attrs={"name": "description"})
            if meta_desc and meta_desc.get("content"):
                content_dict['content'] += meta_desc["content"] + "\n\n"
            
            full_text = soup.get_text(separator="\n", strip=True)
            self.extract_contact_info(full_text)
            
            for tag in soup(['script', 'style', 'iframe', 'nav', 'footer', 'header']):
                tag.decompose()
            
            main_selectors = ["main", "article", "[role='main']", ".content", "#content"]
            texts = []
            
            for selector in main_selectors:
                elements = soup.select(selector)
                for elem in elements:
                    text = elem.get_text(separator="\n", strip=True)
                    if len(text) > 100:
                        texts.append(text)
            
            for tag_name in ['h1', 'h2', 'h3', 'p']:
                for tag in soup.find_all(tag_name):
                    text = tag.get_text(strip=True)
                    if len(text) > 20:
                        texts.append(text)
            
            if len(texts) < 5:
                body = soup.find('body')
                if body:
                    text = body.get_text(separator="\n", strip=True)
                    if text:
                        texts.append(text)
            
            if texts:
                combined = "\n".join(texts)
                lines = [self.clean_text(line) for line in combined.split("\n") if len(line.strip()) > 15]
                
                seen = set()
                unique_lines = []
                for line in lines:
                    line_lower = line.lower()
                    if line_lower not in seen and len(line) > 20:
                        seen.add(line_lower)
                        unique_lines.append(line)
                
                content_dict['content'] = "\n".join(unique_lines)
        
        except Exception as e:
            self.debug_info.append(f"Error {url}: {str(e)[:50]}")
        
        return content_dict
    
def scrape_website(self, base_url: str, max_pages: int = 40, 
                      progress_callback=None) -> Tuple[List[Document], Dict]:
        """Scrape website and return Documents"""
        visited = set()
        queue = deque()
        base_domain = urlparse(base_url).netloc
        
        base_url = base_url.rstrip('/') + '/'
        
        for page in PRIORITY_PAGES:
            for url in [urljoin(base_url, page), urljoin(base_url, page + '/')]:
                if url not in queue:
                    queue.append(url)
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        documents = []
        
        while queue and len(visited) < max_pages:
            url = queue.popleft()
            url = url.split("#")[0].split("?")[0].rstrip('/')
            
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
                
                if len(content_data['content']) > 50:
                    doc = Document(
                        page_content=content_data['content'],
                        metadata={
                            'source': url,
                            'title': content_data['title'],
                            'company': self.company_slug
                        }
                    )
                    documents.append(doc)
                
                for link in soup.find_all("a", href=True):
                    try:
                        next_url = urljoin(url, link['href'])
                        next_url = next_url.split("#")[0].split("?")[0].rstrip('/')
                        
                        if next_url not in visited and self.is_valid_url(next_url, base_domain):
                            if next_url not in queue:
                                queue.append(next_url)
                    except:
                        pass
                
                time.sleep(0.3)
            
            except Exception as e:
                self.debug_info.append(f"Error {url}: {str(e)[:50]}")
                continue
        
        if len(documents) < 3:
            raise Exception(f"Insufficient content: {len(documents)} pages")
        
        return documents, {
            'emails': list(self.company_info['emails']),
            'phones': list(self.company_info['phones']),
            'address_india': self.company_info['address_india'],
            'address_international': self.company_info['address_international'],
            'pages_scraped': len(visited)
        }


def create_company(company_name: str, website_url: str, max_pages: int = 40) -> Optional[str]:
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
            scraping_status="pending"
        )
        db.add(company)
        db.commit()
        db.refresh(company)
        
        return slug
    except Exception as e:
        print(f"Error: {e}")
        db.rollback()
        return None
    finally:
        db.close()


def update_company_after_scraping(slug: str, company_info: Dict, status: str = "completed"):
    """Update company after scraping"""
    db = SessionLocal()
    try:
        company = db.query(Company).filter(Company.company_slug == slug).first()
        if company:
            company.emails = json.dumps(company_info.get('emails', []))
            company.phones = json.dumps(company_info.get('phones', []))
            company.address_india = company_info.get('address_india')
            company.address_international = company_info.get('address_international')
            company.pages_scraped = company_info.get('pages_scraped', 0)
            company.scraping_status = status
            company.last_scraped = datetime.now(timezone.utc)
            db.commit()
    except Exception as e:
        print(f"Error: {e}")
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
"""
MULTI-COMPANY AI CHATBOT - PART 2: AI ENGINE + UI
Fixed duplicate element IDs
"""

import streamlit as st
import time
import requests
import json
import os
from typing import List
import shutil
import hashlib

# FIXED: Compatible imports for all LangChain versions
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain.vectorstores import Chroma

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        from langchain.embeddings import HuggingFaceEmbeddings

# FIXED: Correct Document import
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

# Import from main.py (Part 1)
from main import (
    Company, ChatHistory, UserContact,
    WebScraper, create_company, update_company_after_scraping,
    get_all_companies, get_company_by_slug, get_chroma_directory,
    SessionLocal, OPENROUTER_API_KEY, OPENROUTER_API_BASE, MODEL
)


# ============================================================================
# AI ENGINE WITH RAG
# ============================================================================

class CompanyAI:
    """AI Engine with RAG for answering questions"""
    
    def __init__(self, company_slug: str):
        self.company_slug = company_slug
        self.vectorstore = None
        self.retriever = None
        self.embeddings = None
        self.status = {"ready": False, "error": None}
        self.company_data = None
        
        self.qa_template = """You are an intelligent AI assistant for {company_name}. Answer questions accurately using the provided context.

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
            input_variables=["company_name", "context", "chat_history", "question"]
        )
    
    def initialize(self, website_url: str, max_pages: int = 40, progress_callback=None):
        """Initialize by scraping and creating vector store"""
        try:
            # Scrape website
            if progress_callback:
                progress_callback(0, max_pages, "Starting...")
            
            scraper = WebScraper(self.company_slug)
            documents, company_info = scraper.scrape_website(
                website_url, 
                max_pages, 
                progress_callback
            )
            
            if len(documents) < 3:
                self.status["error"] = f"Not enough content: {len(documents)} pages"
                return False
            
            self.company_data = company_info
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "],
                length_function=len
            )
            
            split_docs = text_splitter.split_documents(documents)
            
            if len(split_docs) < 5:
                self.status["error"] = f"Not enough chunks: {len(split_docs)}"
                return False
            
            # Create embeddings (using smaller model for memory efficiency)
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Create vector store
            chroma_dir = get_chroma_directory(self.company_slug)
            
            if os.path.exists(chroma_dir):
                shutil.rmtree(chroma_dir)
            
            os.makedirs(chroma_dir, exist_ok=True)
            
            self.vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory=chroma_dir,
                collection_name="company_knowledge"
            )
            
            if self.vectorstore._collection.count() == 0:
                raise Exception("Vectorstore is empty")
            
            # Create retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 6,
                    "fetch_k": 15,
                    "lambda_mult": 0.7
                }
            )
            
            self.status["ready"] = True
            
            if progress_callback:
                progress_callback(max_pages, max_pages, "✅ Ready!")
            
            return True
        
        except Exception as e:
            self.status["error"] = f"Error: {str(e)}"
            print(f"Initialization error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_existing(self):
        """Load existing vector store"""
        try:
            chroma_dir = get_chroma_directory(self.company_slug)
            
            if not os.path.exists(chroma_dir):
                self.status["error"] = "Not found"
                return False
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            self.vectorstore = Chroma(
                persist_directory=chroma_dir,
                embedding_function=self.embeddings,
                collection_name="company_knowledge"
            )
            
            if self.vectorstore._collection.count() == 0:
                self.status["error"] = "Empty vectorstore"
                return False
            
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 6,
                    "fetch_k": 15,
                    "lambda_mult": 0.7
                }
            )
            
            # Load company data
            company = get_company_by_slug(self.company_slug)
            if company:
                try:
                    self.company_data = {
                        'emails': json.loads(company.emails) if company.emails else [],
                        'phones': json.loads(company.phones) if company.phones else [],
                        'address_india': company.address_india,
                        'address_international': company.address_international
                    }
                except:
                    self.company_data = {}
            
            self.status["ready"] = True
            return True
        
        except Exception as e:
            self.status["error"] = f"Load error: {str(e)}"
            print(f"Load error: {e}")
            return False
    
    def get_contact_info(self):
        """Format contact information"""
        if not self.company_data:
            return "Contact information not available."
        
        info = self.company_data
        msg = "📞 **CONTACT INFORMATION**\n\n"
        
        if info.get('address_india'):
            msg += f"🇮🇳 **India Office:**\n{info['address_india']}\n\n"
        
        if info.get('address_international'):
            msg += f"🌍 **International Office:**\n{info['address_international']}\n\n"
        
        if info.get('emails'):
            msg += "📧 **Email:**\n" + "\n".join([f"• {e}" for e in info['emails'][:5]]) + "\n\n"
        
        if info.get('phones'):
            msg += "☎️ **Phone:**\n" + "\n".join([f"• {p}" for p in info['phones'][:5]]) + "\n"
        
        return msg.strip()
    
    def ask(self, question: str, chat_history: List = None, session_id: str = None) -> str:
        """Answer question using RAG"""
        if not self.status["ready"]:
            return "⚠️ System is initializing. Please wait..."
        
        q_lower = question.lower().strip()
        
        # Handle greetings
        greetings = ["hi", "hello", "hey", "hai"]
        if q_lower in greetings or len(q_lower) < 5:
            company = get_company_by_slug(self.company_slug)
            company_name = company.company_name if company else "our company"
            return f"Hello! 👋 I'm here to answer questions about {company_name}. How can I help you today?"
        
        # Handle contact requests
        contact_keywords = ["email", "contact", "phone", "address", "office", "location", "reach"]
        if any(keyword in q_lower for keyword in contact_keywords):
            return self.get_contact_info()
        
        try:
            # Retrieve relevant documents
            relevant_docs = self.retriever.get_relevant_documents(question)
            
            if not relevant_docs or len(relevant_docs) == 0:
                return "I couldn't find specific information about that. Could you rephrase your question?"
            
            # Build context
            context_parts = []
            for doc in relevant_docs[:5]:
                if hasattr(doc, 'page_content') and doc.page_content:
                    source = doc.metadata.get('source', 'unknown')
                    title = doc.metadata.get('title', '')
                    
                    context_part = f"[From: {title if title else source}]\n{doc.page_content}"
                    context_parts.append(context_part)
            
            if not context_parts:
                return "I found documents but couldn't extract content. Please try again."
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Format chat history
            history_text = ""
            if chat_history:
                history_text = "\n".join([
                    f"User: {msg['question']}\nAssistant: {msg['answer']}" 
                    for msg in chat_history[-3:]
                ])
            
            # Get company name
            company = get_company_by_slug(self.company_slug)
            company_name = company.company_name if company else "the company"
            
            # Build prompt
            prompt = self.qa_prompt.format(
                company_name=company_name,
                context=context[:5000],
                chat_history=history_text,
                question=question
            )
            
            # Call LLM
            answer = self._call_llm(prompt)
            
            if answer:
                self.save_chat(question, answer, session_id)
                return answer
            else:
                return "I'm having trouble generating a response. Please try again."
        
        except Exception as e:
            print(f"Error in ask(): {e}")
            return "An error occurred. Please try again."
    
    def _call_llm(self, prompt: str, max_retries: int = 2) -> str:
        """Call LLM API"""
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Answer accurately based on context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.5,
            "max_tokens": 600,
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    OPENROUTER_API_BASE,
                    headers=headers,
                    json=payload,
                    timeout=45
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if "choices" in result and len(result["choices"]) > 0:
                        answer = result["choices"][0]["message"]["content"].strip()
                        
                        if answer and len(answer) > 20:
                            return answer
                
                elif response.status_code == 429:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
            
            except Exception as e:
                print(f"LLM error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
        
        return None
    
    def save_chat(self, question: str, answer: str, session_id: str = None):
        """Save chat to database"""
        db = SessionLocal()
        try:
            chat = ChatHistory(
                company_slug=self.company_slug,
                question=question,
                answer=answer,
                session_id=session_id
            )
            db.add(chat)
            db.commit()
        except Exception as e:
            print(f"Error saving chat: {e}")
            db.rollback()
        finally:
            db.close()


# ============================================================================
# STREAMLIT UI - FIXED WITH UNIQUE KEYS
# ============================================================================

st.set_page_config(
    page_title="AI Chatbot Generator",
    page_icon="🤖",
    layout="wide"
)

# Minimal CSS for better performance
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .header-title { color: white; font-size: 2.5rem; font-weight: bold; }
    .company-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if "page" not in st.session_state:
    st.session_state.page = "home"
if "current_company" not in st.session_state:
    st.session_state.current_company = None
if "ai_instance" not in st.session_state:
    st.session_state.ai_instance = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:16]

# Sidebar - FIXED: Added unique keys
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    
    if st.button("🏠 Home", use_container_width=True, key="nav_home"):
        st.session_state.page = "home"
        st.rerun()
    
    if st.button("➕ Create New", use_container_width=True, key="nav_create"):
        st.session_state.page = "create"
        st.rerun()
    
    if st.button("📋 View All", use_container_width=True, key="nav_list"):
        st.session_state.page = "list"
        st.rerun()

# Pages
if st.session_state.page == "home":
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🤖 AI Chatbot Generator</h1>
        <p style="color: white; font-size: 1.1rem;">Create intelligent chatbots from any website!</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("➕ Create New Chatbot", use_container_width=True, type="primary", key="home_create"):
            st.session_state.page = "create"
            st.rerun()
    
    with col2:
        if st.button("📋 View All Chatbots", use_container_width=True, key="home_list"):
            st.session_state.page = "list"
            st.rerun()
    
    st.info("💡 **How it works:** Enter a company name and website URL. The AI will scrape the website and create an intelligent chatbot!")

elif st.session_state.page == "create":
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">➕ Create New Chatbot</h1>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("create_form", clear_on_submit=False):
        name = st.text_input("Company Name *", placeholder="e.g., Acme Corporation", key="form_name")
        url = st.text_input("Website URL *", placeholder="e.g., https://example.com", key="form_url")
        pages = st.slider("Max Pages to Scrape", 10, 60, 40, 10, key="form_pages")
        
        submitted = st.form_submit_button("🚀 Create Chatbot", type="primary", use_container_width=True)
        
        if submitted:
            if name and url:
                if not url.startswith('http'):
                    url = 'https://' + url
                
                slug = create_company(name, url, pages)
                
                if slug:
                    st.success(f"✅ Created: {name}")
                    
                    prog = st.progress(0, text="Initializing...")
                    status = st.empty()
                    
                    def progress_cb(current, total, url_text):
                        prog.progress(min(current/total, 1.0), text=f"📄 Scraped {current}/{total} pages")
                        status.caption(f"Current: {url_text[:50]}...")
                    
                    ai = CompanyAI(slug)
                    if ai.initialize(url, pages, progress_cb):
                        update_company_after_scraping(slug, ai.company_data, "completed")
                        st.success("✅ Chatbot is ready!")
                        st.balloons()
                        time.sleep(2)
                        
                        st.session_state.current_company = slug
                        st.session_state.ai_instance = ai
                        st.session_state.messages = []
                        st.session_state.chat_history = []
                        st.session_state.page = "chat"
                        st.rerun()
                    else:
                        st.error(f"❌ Failed: {ai.status.get('error', 'Unknown error')}")
                        update_company_after_scraping(slug, {}, "failed")
                else:
                    st.error("⚠️ Company already exists!")
            else:
                st.warning("⚠️ Please fill in all fields")

elif st.session_state.page == "list":
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">📋 All Chatbots</h1>
    </div>
    """, unsafe_allow_html=True)
    
    companies = get_all_companies()
    
    if not companies:
        st.info("No chatbots yet. Create your first one!")
    else:
        for c in companies:
            st.markdown('<div class="company-card">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"### {c.company_name}")
                st.caption(f"🌐 {c.website_url}")
                st.caption(f"📄 {c.pages_scraped} pages | Status: {c.scraping_status}")
            
            with col2:
                if c.scraping_status == "completed":
                    st.success("✅ Ready")
                elif c.scraping_status == "failed":
                    st.error("❌ Failed")
                else:
                    st.warning("⏳ Pending")
            
            with col3:
                if c.scraping_status == "completed":
                    # FIXED: Added unique key using company ID
                    if st.button("💬 Chat", key=f"chat_btn_{c.id}", use_container_width=True):
                        st.session_state.current_company = c.company_slug
                        st.session_state.page = "chat"
                        st.session_state.ai_instance = None
                        st.session_state.messages = []
                        st.session_state.chat_history = []
                        st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "chat":
    if not st.session_state.current_company:
        st.error("No company selected")
        st.stop()
    
    c = get_company_by_slug(st.session_state.current_company)
    
    if not c:
        st.error("Company not found")
        st.stop()
    
    if not st.session_state.ai_instance:
        with st.spinner("Loading chatbot..."):
            ai = CompanyAI(st.session_state.current_company)
            if not ai.load_existing():
                st.error(f"Failed to load: {ai.status.get('error', 'Unknown error')}")
                st.stop()
            st.session_state.ai_instance = ai
    
    st.markdown(f"""
    <div class="header-container">
        <h1 class="header-title">💬 {c.company_name}</h1>
        <p style="color: white;">Ask me anything about the company!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input - FIXED: Added unique key
    if prompt := st.chat_input("Ask a question...", key="chat_input_main"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.ai_instance.ask(
                    prompt,
                    st.session_state.chat_history,
                    st.session_state.session_id
                )
            st.markdown(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_history.append({"question": prompt, "answer": answer})
        update_company_after_scraping(slug, {}, "failed")
                else:
                    st.error("⚠️ Company already exists!")
            else:
                st.warning("⚠️ Please fill in all fields")

elif st.session_state.page == "list":
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">📋 All Chatbots</h1>
    </div>
    """, unsafe_allow_html=True)
    
    companies = get_all_companies()
    
    if not companies:
        st.info("No chatbots yet. Create your first one!")
    else:
        for c in companies:
            st.markdown('<div class="company-card">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"### {c.company_name}")
                st.caption(f"🌐 {c.website_url}")
                st.caption(f"📄 {c.pages_scraped} pages | Status: {c.scraping_status}")
            
            with col2:
                if c.scraping_status == "completed":
                    st.success("✅ Ready")
                elif c.scraping_status == "failed":
                    st.error("❌ Failed")
                else:
                    st.warning("⏳ Pending")
            
            with col3:
                if c.scraping_status == "completed":
                    if st.button("💬 Chat", key=f"chat_{c.id}", use_container_width=True):
                        st.session_state.current_company = c.company_slug
                        st.session_state.page = "chat"
                        st.session_state.ai_instance = None
                        st.session_state.messages = []
                        st.session_state.chat_history = []
                        st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "chat":
    if not st.session_state.current_company:
        st.error("No company selected")
        st.stop()
    
    c = get_company_by_slug(st.session_state.current_company)
    
    if not c:
        st.error("Company not found")
        st.stop()
    
    if not st.session_state.ai_instance:
        with st.spinner("Loading chatbot..."):
            ai = CompanyAI(st.session_state.current_company)
            if not ai.load_existing():
                st.error(f"Failed to load: {ai.status.get('error', 'Unknown error')}")
                st.stop()
            st.session_state.ai_instance = ai
    
    st.markdown(f"""
    <div class="header-container">
        <h1 class="header-title">💬 {c.company_name}</h1>
        <p style="color: white;">Ask me anything about the company!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.ai_instance.ask(
                    prompt,
                    st.session_state.chat_history,
                    st.session_state.session_id
                )
            st.markdown(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_history.append({"question": prompt, "answer": answer})
