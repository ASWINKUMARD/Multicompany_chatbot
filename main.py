"""
AI Chatbot Generator - Backend (main.py)
Fixed and optimized version
"""

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

# Compatible imports for all LangChain versions
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

# ============================================================================
# DATABASE MODELS
# ============================================================================

DATABASE_URL = "sqlite:///./multi_company_chatbots.db"
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
    pool_recycle=3600
)
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


# Create all tables
Base.metadata.create_all(bind=engine)

# ============================================================================
# CONFIGURATION
# ============================================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.0-flash-exp:free"

PRIORITY_PAGES = [
    "", "about", "services", "solutions", "products", "contact", "team",
    "careers", "about-us", "contact-us", "our-services", "home"
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_slug(company_name: str) -> str:
    """Convert company name to URL-friendly slug"""
    slug = company_name.lower()
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    return slug.strip('-')


def get_chroma_directory(company_slug: str) -> str:
    """Get ChromaDB directory path"""
    base_dir = "./chroma_db"
    os.makedirs(base_dir, exist_ok=True)
    return f"{base_dir}/{company_slug}"


# ============================================================================
# WEB SCRAPER
# ============================================================================

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
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"]+', '', text)
        return text.strip()
    
    def extract_contact_info(self, text: str):
        """Extract contact information"""
        if not text:
            return
            
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        for email in emails:
            if not email.lower().endswith(('.png', '.jpg', '.gif', '.svg')):
                self.company_info['emails'].add(email.lower())
        
        # Extract phones
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
        
        # Extract addresses
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if not line.strip():
                continue
                
            low = line.lower()
            
            # India addresses
            india_keywords = ['india', 'mumbai', 'delhi', 'bangalore', 'chennai', 'madurai', 'pune', 'hyderabad']
            if any(city in low for city in india_keywords):
                if not self.company_info['address_india']:
                    start_idx = max(0, i-2)
                    end_idx = min(len(lines), i+4)
                    block = " ".join(lines[start_idx:end_idx])
                    cleaned = self.clean_text(block)
                    if 20 < len(cleaned) < 500:
                        self.company_info['address_india'] = cleaned
            
            # International addresses
            intl_keywords = ['singapore', 'usa', 'uk', 'uae', 'dubai', 'london', 'new york']
            if any(country in low for country in intl_keywords):
                if not self.company_info['address_international']:
                    start_idx = max(0, i-2)
                    end_idx = min(len(lines), i+4)
                    block = " ".join(lines[start_idx:end_idx])
                    cleaned = self.clean_text(block)
                    if 20 < len(cleaned) < 500:
                        self.company_info['address_international'] = cleaned
    
    def is_valid_url(self, url: str, base_domain: str) -> bool:
        """Check if URL should be scraped"""
        if not url:
            return False
            
        try:
            parsed = urlparse(url)
            if parsed.netloc != base_domain:
                return False
            
            skip_extensions = ['.pdf', '.jpg', '.png', '.zip', '.mp4', '.css', '.js', '.gif', '.svg', '.ico']
            skip_paths = ['/wp-admin/', '/admin/', '/login', '/cart/', '/checkout/']
            
            url_lower = url.lower()
            
            # Check extensions
            for ext in skip_extensions:
                if url_lower.endswith(ext):
                    return False
            
            # Check paths
            for path in skip_paths:
                if path in url_lower:
                    return False
            
            return True
        except Exception as e:
            self.debug_info.append(f"URL validation error: {str(e)}")
            return False
    
    def extract_content(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract content from page"""
        content_dict = {'url': url, 'title': '', 'content': ''}
        
        try:
            # Extract title
            if soup.find('title'):
                content_dict['title'] = soup.find('title').get_text(strip=True)
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={"name": "description"})
            if meta_desc and meta_desc.get("content"):
                content_dict['content'] += meta_desc["content"] + "\n\n"
            
            # Extract contact info from full text
            full_text = soup.get_text(separator="\n", strip=True)
            self.extract_contact_info(full_text)
            
            # Remove unwanted elements
            for tag in soup(['script', 'style', 'iframe', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()
            
            # Extract main content
            main_selectors = ["main", "article", "[role='main']", ".content", "#content", ".main-content"]
            texts = []
            
            for selector in main_selectors:
                elements = soup.select(selector)
                for elem in elements:
                    text = elem.get_text(separator="\n", strip=True)
                    if len(text) > 100:
                        texts.append(text)
            
            # Extract from important tags
            for tag_name in ['h1', 'h2', 'h3', 'p', 'li']:
                for tag in soup.find_all(tag_name):
                    text = tag.get_text(strip=True)
                    if len(text) > 20:
                        texts.append(text)
            
            # Fallback to body
            if len(texts) < 5:
                body = soup.find('body')
                if body:
                    text = body.get_text(separator="\n", strip=True)
                    if text:
                        texts.append(text)
            
            # Process and deduplicate
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
                
                content_dict['content'] = "\n".join(unique_lines[:200])  # Limit lines
        
        except Exception as e:
            self.debug_info.append(f"Content extraction error for {url}: {str(e)[:100]}")
        
        return content_dict
    
    def scrape_website(self, base_url: str, max_pages: int = 40, 
                       progress_callback=None) -> Tuple[List[Document], Dict]:
        """Scrape website and return Documents"""
        visited = set()
        queue = deque()
        
        try:
            base_domain = urlparse(base_url).netloc
        except Exception as e:
            raise Exception(f"Invalid URL: {str(e)}")
        
        base_url = base_url.rstrip('/') + '/'
        
        # Add priority pages first
        for page in PRIORITY_PAGES:
            for url in [urljoin(base_url, page), urljoin(base_url, page + '/')]:
                if url not in queue:
                    queue.append(url)
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        documents = []
        
        while queue and len(visited) < max_pages:
            url = queue.popleft()
            
            # Normalize URL
            try:
                url = url.split("#")[0].split("?")[0].rstrip('/')
            except:
                continue
            
            if url in visited or not self.is_valid_url(url, base_domain):
                continue
            
            try:
                response = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
                
                if response.status_code != 200:
                    continue
                
                visited.add(url)
                
                if progress_callback:
                    progress_callback(len(visited), max_pages, url)
                
                soup = BeautifulSoup(response.text, "html.parser")
                content_data = self.extract_content(soup, url)
                
                # Only add documents with substantial content
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
                
                # Find and queue new links
                for link in soup.find_all("a", href=True):
                    try:
                        next_url = urljoin(url, link['href'])
                        next_url = next_url.split("#")[0].split("?")[0].rstrip('/')
                        
                        if (next_url not in visited and 
                            self.is_valid_url(next_url, base_domain) and 
                            next_url not in queue):
                            queue.append(next_url)
                    except:
                        pass
                
                # Polite delay
                time.sleep(0.3)
            
            except requests.exceptions.Timeout:
                self.debug_info.append(f"Timeout: {url}")
                continue
            except Exception as e:
                self.debug_info.append(f"Error {url}: {str(e)[:100]}")
                continue
        
        # Validate results
        if len(documents) < 3:
            raise Exception(f"Insufficient content scraped: only {len(documents)} pages with content")
        
        return documents, {
            'emails': list(self.company_info['emails'])[:10],  # Limit to 10
            'phones': list(self.company_info['phones'])[:10],
            'address_india': self.company_info['address_india'],
            'address_international': self.company_info['address_international'],
            'pages_scraped': len(visited)
        }


# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def create_company(company_name: str, website_url: str, max_pages: int = 40) -> Optional[str]:
    """Create new company"""
    db = SessionLocal()
    try:
        slug = create_slug(company_name)
        
        # Check if exists
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
        print(f"Database error: {e}")
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
        print(f"Update error: {e}")
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
AI Chatbot Generator - Frontend (app.py)
Fixed and optimized Streamlit application
"""

import streamlit as st
import time
import requests
import json
import os
import shutil
import hashlib
from typing import List, Optional

# Compatible imports
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

# Import from main.py
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
            if progress_callback:
                progress_callback(0, max_pages, "Starting scraper...")
            
            # Scrape website
            scraper = WebScraper(self.company_slug)
            documents, company_info = scraper.scrape_website(
                website_url, 
                max_pages, 
                progress_callback
            )
            
            if len(documents) < 3:
                self.status["error"] = f"Insufficient content: {len(documents)} pages"
                return False
            
            self.company_data = company_info
            
            if progress_callback:
                progress_callback(max_pages, max_pages, "Processing documents...")
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "],
                length_function=len
            )
            
            split_docs = text_splitter.split_documents(documents)
            
            if len(split_docs) < 5:
                self.status["error"] = f"Insufficient chunks: {len(split_docs)}"
                return False
            
            if progress_callback:
                progress_callback(max_pages, max_pages, "Creating embeddings...")
            
            # Create embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            chroma_dir = get_chroma_directory(self.company_slug)
            
            # Clean existing directory
            if os.path.exists(chroma_dir):
                shutil.rmtree(chroma_dir)
            
            os.makedirs(chroma_dir, exist_ok=True)
            
            if progress_callback:
                progress_callback(max_pages, max_pages, "Building vector database...")
            
            # Create vectorstore
            self.vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory=chroma_dir,
                collection_name="company_knowledge"
            )
            
            # Verify vectorstore
            if self.vectorstore._collection.count() == 0:
                raise Exception("Vectorstore is empty after creation")
            
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
                progress_callback(max_pages, max_pages, "✅ Chatbot Ready!")
            
            return True
        
        except Exception as e:
            self.status["error"] = f"Initialization error: {str(e)}"
            print(f"Initialization error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_existing(self):
        """Load existing vector store"""
        try:
            chroma_dir = get_chroma_directory(self.company_slug)
            
            if not os.path.exists(chroma_dir):
                self.status["error"] = "Chatbot data not found"
                return False
            
            # Create embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Load vectorstore
            self.vectorstore = Chroma(
                persist_directory=chroma_dir,
                embedding_function=self.embeddings,
                collection_name="company_knowledge"
            )
            
            # Verify vectorstore
            if self.vectorstore._collection.count() == 0:
                self.status["error"] = "Empty vectorstore"
                return False
            
            # Create retriever
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
                except Exception as e:
                    print(f"Error loading company data: {e}")
                    self.company_data = {}
            
            self.status["ready"] = True
            return True
        
        except Exception as e:
            self.status["error"] = f"Load error: {str(e)}"
            print(f"Load error: {e}")
            import traceback
            traceback.print_exc()
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
            msg += "📱 **Phone:**\n" + "\n".join([f"• {p}" for p in info['phones'][:5]]) + "\n"
        
        return msg.strip()
    
    def ask(self, question: str, chat_history: List = None, session_id: str = None) -> str:
        """Answer question using RAG"""
        try:
            # Check system status
            if not self.status.get("ready", False):
                return "⚠️ System is initializing. Please wait..."
            
            if not self.retriever:
                return "⚠️ Chatbot not properly initialized. Please try reloading."
            
            q_lower = question.lower().strip()
            
            # Handle greetings
            greetings = ["hi", "hello", "hey", "hai", "hola"]
            if q_lower in greetings or len(q_lower) < 5:
                company = get_company_by_slug(self.company_slug)
                company_name = company.company_name if company else "our company"
                return f"👋 Hello! I'm here to answer questions about {company_name}. How can I help you today?"
            
            # Handle contact requests
            contact_keywords = ["email", "contact", "phone", "address", "office", "location", "reach", "call"]
            if any(keyword in q_lower for keyword in contact_keywords):
                return self.get_contact_info()
            
            print(f"\n=== Processing question: {question} ===")
            
            # Retrieve relevant documents
            print("Retrieving relevant documents...")
            try:
                # Use invoke() for newer LangChain versions, fallback to old method
                if hasattr(self.retriever, 'invoke'):
                    relevant_docs = self.retriever.invoke(question)
                else:
                    relevant_docs = self.retriever.get_relevant_documents(question)
            except Exception as e:
                print(f"Retrieval error: {e}")
                return "⚠️ Error retrieving information. Please try again."
            
            print(f"Found {len(relevant_docs)} relevant documents")
            
            if not relevant_docs or len(relevant_docs) == 0:
                return "I couldn't find specific information about that. Could you rephrase your question or ask something else?"
            
            # Build context from documents
            context_parts = []
            for i, doc in enumerate(relevant_docs[:5]):
                if hasattr(doc, 'page_content') and doc.page_content:
                    source = doc.metadata.get('source', 'unknown')
                    title = doc.metadata.get('title', '')
                    
                    # Truncate long content
                    content = doc.page_content[:600]
                    context_part = f"[Source: {title if title else source}]\n{content}"
                    context_parts.append(context_part)
                    print(f"Doc {i+1}: {len(doc.page_content)} chars from {source}")
            
            if not context_parts:
                return "I found documents but couldn't extract relevant content. Please try rephrasing your question."
            
            context = "\n\n---\n\n".join(context_parts)
            print(f"Total context length: {len(context)} chars")
            
            # Format chat history
            history_text = ""
            if chat_history and len(chat_history) > 0:
                recent_history = chat_history[-3:]  # Last 3 exchanges
                history_text = "\n".join([
                    f"User: {msg['question']}\nAssistant: {msg['answer']}" 
                    for msg in recent_history
                ])
            
            # Get company name
            company = get_company_by_slug(self.company_slug)
            company_name = company.company_name if company else "the company"
            
            # Build prompt
            prompt = self.qa_prompt.format(
                company_name=company_name,
                context=context[:4500],  # Limit context size
                chat_history=history_text[:500],  # Limit history
                question=question
            )
            
            print(f"Prompt length: {len(prompt)} chars")
            print("Calling LLM...")
            
            # Call LLM
            answer = self._call_llm(prompt)
            
            print(f"Answer received: {answer[:100] if answer else 'None'}...")
            
            # Save successful conversations
            if answer and not answer.startswith("⚠️") and not answer.startswith("❌"):
                self.save_chat(question, answer, session_id)
                return answer
            else:
                return answer if answer else "I'm having trouble generating a response. Please try again."
        
        except Exception as e:
            print(f"ERROR in ask(): {str(e)}")
            import traceback
            traceback.print_exc()
            return f"⚠️ An error occurred: {str(e)[:100]}. Please try again or reload the page."
    
    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Call LLM API with retry logic"""
        # Validate API key
        if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "":
            print("ERROR: OPENROUTER_API_KEY not set!")
            return "⚠️ API key not configured. Please set OPENROUTER_API_KEY environment variable."
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "AI Chatbot Generator"
        }
        
        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Answer accurately and concisely based on the provided context."
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
                print(f"LLM API call attempt {attempt + 1}/{max_retries}...")
                
                response = requests.post(
                    OPENROUTER_API_BASE,
                    headers=headers,
                    json=payload,
                    timeout=60
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
                        print(f"No choices in response: {str(result)[:200]}")
                
                elif response.status_code == 429:
                    print("Rate limited, waiting...")
                    if attempt < max_retries - 1:
                        time.sleep(3)
                        continue
                    else:
                        return "⚠️ API rate limit reached. Please wait a moment and try again."
                
                elif response.status_code == 401:
                    print("Authentication failed!")
                    return "⚠️ Invalid API key. Please check your OPENROUTER_API_KEY."
                
                elif response.status_code == 400:
                    error_text = response.text[:300]
                    print(f"Bad request: {error_text}")
                    return "⚠️ Invalid request to AI service. Please try rephrasing your question."
                
                else:
                    error_text = response.text[:300]
                    print(f"Error response ({response.status_code}): {error_text}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
            
            except requests.exceptions.Timeout:
                print(f"Request timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    return "⚠️ Request timed out. Please try again."
            
            except requests.exceptions.ConnectionError:
                print(f"Connection error on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    return "⚠️ Connection error. Please check your internet connection."
            
            except Exception as e:
                print(f"LLM error on attempt {attempt + 1}: {str(e)}")
                import traceback
                traceback.print_exc()
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
        
        return "❌ Failed to get response from AI after multiple attempts. Please try again."
    
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
# STREAMLIT UI
# ============================================================================

st.set_page_config(
    page_title="AI Chatbot Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header-title { 
        color: white; 
        font-size: 2.5rem; 
        font-weight: bold; 
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .company-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .company-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
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

# Sidebar Navigation
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
    
    st.markdown("---")
    st.caption("🤖 Powered by AI & RAG Technology")
    st.caption(f"Session: {st.session_state.session_id[:8]}")

# ============================================================================
# PAGE ROUTING
# ============================================================================

# HOME PAGE
if st.session_state.page == "home":
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🤖 AI Chatbot Generator</h1>
        <p style="color: white; font-size: 1.1rem; margin-top: 1rem;">
            Create intelligent chatbots from any website in minutes!
        </p>
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
    
    st.info("ℹ️ **How it works:** Enter a company name and website URL. The AI will scrape the website, extract information, and create an intelligent chatbot that can answer questions about the company!")
    
    st.markdown("### ✨ Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🌐 Web Scraping**\n\nAutomatically extracts content from websites")
    with col2:
        st.markdown("**🧠 AI-Powered**\n\nUses advanced RAG technology for accurate answers")
    with col3:
        st.markdown("**💬 Natural Chat**\n\nConversational interface for easy interaction")

# CREATE PAGE
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
        
        st.caption("⏱️ This process may take 2-5 minutes depending on website size")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            submitted = st.form_submit_button("🚀 Create Chatbot", type="primary", use_container_width=True)
        with col2:
            if st.form_submit_button("❌ Cancel", use_container_width=True):
                st.session_state.page = "home"
                st.rerun()
        
        if submitted:
            if name and url:
                # Validate URL
                if not url.startswith('http'):
                    url = 'https://' + url
                
                with st.spinner("Creating company record..."):
                    slug = create_company(name, url, pages)
                
                if slug:
                    st.success(f"✅ Created: {name}")
                    
                    prog_bar = st.progress(0, text="Initializing scraper...")
                    status_text = st.empty()
                    
                    def progress_cb(current, total, url_text):
                        progress = min(current / total, 1.0)
                        prog_bar.progress(progress, text=f"🔍 Scraped {current}/{total} pages")
                        status_text.caption(f"Current: {url_text[:80]}...")
                    
                    ai = CompanyAI(slug)
                    success = ai.initialize(url, pages, progress_cb)
                    
                    if success:
                        update_company_after_scraping(slug, ai.company_data, "completed")
                        prog_bar.progress(1.0, text="✅ Chatbot is ready!")
                        st.success("🎉 Chatbot created successfully!")
                        st.balloons()
                        time.sleep(2)
                        
                        # Navigate to chat
                        st.session_state.current_company = slug
                        st.session_state.ai_instance = ai
                        st.session_state.messages = []
                        st.session_state.chat_history = []
                        st.session_state.page = "chat"
                        st.rerun()
                    else:
                        error_msg = ai.status.get('error', 'Unknown error')
                        st.error(f"❌ Failed to create chatbot: {error_msg}")
                        update_company_after_scraping(slug, {}, "failed")
                else:
                    st.error("⚠️ Company already exists with this name!")
            else:
                st.warning("⚠️ Please fill in all required fields")

# LIST PAGE
elif st.session_state.page == "list":
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">📋 All Chatbots</h1>
    </div>
    """, unsafe_allow_html=True)
    
    companies = get_all_companies()
    
    if not companies:
        st.info("No chatbots created yet. Create your first one!")
        if st.button("➕ Create First Chatbot", type="primary"):
            st.session_state.page = "create"
            st.rerun()
    else:
        st.markdown(f"**Total Chatbots: {len(companies)}**")
        
        for c in companies:
            st.markdown('<div class="company-card">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"### {c.company_name}")
                st.caption(f"🌐 {c.website_url}")
                st.caption(f"📄 {c.pages_scraped} pages scraped | 📅 {c.created_at.strftime('%Y-%m-%d %H:%M')}")
            
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
                elif c.scraping_status == "failed":
                    st.button("🔄 Retry", key=f"retry_{c.id}", use_container_width=True, disabled=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

# CHAT PAGE
elif st.session_state.page == "chat":
    if not st.session_state.current_company:
        st.error("⚠️ No company selected")
        if st.button("← Go Back"):
            st.session_state.page = "list"
            st.rerun()
        st.stop()
    
    c = get_company_by_slug(st.session_state.current_company)
    
    if not c:
        st.error("⚠️ Company not found")
        if st.button("← Go Back"):
            st.session_state.page = "list"
            st.rerun()
        st.stop()
    
    # Load AI instance if not already loaded
    if not st.session_state.ai_instance:
        with st.spinner("Loading chatbot..."):
            try:
                ai = CompanyAI(st.session_state.current_company)
                if not ai.load_existing():
                    st.error(f"❌ Failed to load chatbot: {ai.status.get('error', 'Unknown error')}")
                    if st.button("← Go Back"):
                        st.session_state.page = "list"
                        st.rerun()
                    st.stop()
                st.session_state.ai_instance = ai
                st.success("✅ Chatbot loaded successfully!")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading chatbot: {str(e)}")
                if st.button("← Go Back"):
                    st.session_state.page = "list"
                    st.rerun()
                st.stop()
    
    st.markdown(f"""
    <div class="header-container">
        <h1 class="header-title">💬 Chat with {c.company_name}</h1>
        <p style="color: white; margin-top: 0.5rem;">Ask me anything about the company!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Back button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back"):
            st.session_state.page = "list"
            st.session_state.current_company = None
            st.session_state.ai_instance = None
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question...", key="chat_input_main"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Ensure AI instance is still valid
                    if not st.session_state.ai_instance:
                        answer = "⚠️ Chatbot disconnected. Please reload the page."
                    elif not st.session_state.ai_instance.status.get("ready", False):
                        answer = "⚠️ Chatbot not ready. Please try again."
                    else:
                        answer = st.session_state.ai_instance.ask(
                            prompt,
                            st.session_state.chat_history,
                            st.session_state.session_id
                        )
                except Exception as e:
                    print(f"Error in chat: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    answer = "⚠️ An error occurred. Please try reloading the page."
            
            st.markdown(answer)
        
        # Save messages
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # Only add to chat history if it's a successful response
        if not answer.startswith("⚠️") and not answer.startswith("❌"):
            st.session_state.chat_history.append({"question": prompt, "answer": answer})
        
        st.rerun()
    
    # Clear chat button in sidebar
    if len(st.session_state.messages) > 0:
        st.sidebar.markdown("---")
        if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
        
        st.sidebar.caption(f"💬 Messages: {len(st.session_state.messages)}")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, LangChain, and ChromaDB")
