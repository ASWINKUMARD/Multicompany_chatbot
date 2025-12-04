"""
OPTIMIZED MULTI-COMPANY AI CHATBOT - PART 1 (main.py)
Save this as: main.py
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

# Database setup
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

# API Configuration
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
    """Web scraper with content extraction - OPTIMIZED"""
    
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
        """Extract content from page - IMPROVED"""
        content_dict = {'url': url, 'title': '', 'content': ''}
        
        try:
            # Get title
            if soup.find('title'):
                content_dict['title'] = soup.find('title').get_text(strip=True)[:200]
            
            # Get meta description
            meta_desc = soup.find('meta', attrs={"name": "description"})
            if meta_desc and meta_desc.get("content"):
                content_dict['content'] += meta_desc["content"] + "\n\n"
            
            # Extract contact info from full text BEFORE removing tags
            full_text = soup.get_text(separator="\n", strip=True)
            self.extract_contact_info(full_text)
            
            # Remove unwanted tags
            for tag in soup(['script', 'style', 'iframe', 'nav', 'footer', 'header', 'aside', 'noscript']):
                tag.decompose()
            
            # Try multiple extraction strategies
            texts = []
            
            # Strategy 1: Main content areas
            main_selectors = ["main", "article", "[role='main']", ".content", "#content", ".main-content", "#main"]
            for selector in main_selectors:
                elements = soup.select(selector)
                for elem in elements:
                    text = elem.get_text(separator="\n", strip=True)
                    if len(text) > 100:
                        texts.append(text)
                        print(f"Found content in {selector}: {len(text)} chars")
            
            # Strategy 2: Headings and paragraphs
            if len(texts) == 0:
                print("No main content found, extracting headings and paragraphs...")
                for tag_name in ['h1', 'h2', 'h3', 'h4', 'p', 'li', 'div']:
                    for tag in soup.find_all(tag_name):
                        text = tag.get_text(strip=True)
                        if len(text) > 20:
                            texts.append(text)
            
            # Strategy 3: Body fallback
            if len(texts) < 5:
                print("Limited content, using body fallback...")
                body = soup.find('body')
                if body:
                    text = body.get_text(separator="\n", strip=True)
                    if text and len(text) > 100:
                        texts.append(text)
            
            # Process and clean content
            if texts:
                combined = "\n".join(texts)
                lines = [self.clean_text(line) for line in combined.split("\n") if len(line.strip()) > 15]
                
                # Deduplicate while preserving order
                seen = set()
                unique_lines = []
                for line in lines:
                    line_lower = line.lower()
                    if line_lower not in seen and len(line) > 20:
                        seen.add(line_lower)
                        unique_lines.append(line)
                
                content_dict['content'] += "\n".join(unique_lines)
                print(f"Extracted {len(unique_lines)} unique lines from {url}")
            else:
                print(f"WARNING: No content extracted from {url}")
        
        except Exception as e:
            print(f"Extract error for {url}: {str(e)}")
            self.debug_info.append(f"Error {url}: {str(e)[:50]}")
        
        return content_dict
    
    def scrape_website(self, base_url: str, max_pages: int = 40, 
                       progress_callback=None) -> Tuple[List[Document], Dict]:
        """Scrape website and return Documents - IMPROVED"""
        visited = set()
        queue = deque()
        
        # Parse and validate base URL
        try:
            parsed = urlparse(base_url)
            base_domain = parsed.netloc
            if not base_domain:
                raise Exception(f"Invalid URL: {base_url}")
            print(f"Starting scrape of domain: {base_domain}")
        except Exception as e:
            raise Exception(f"URL parsing error: {str(e)}")
        
        # Normalize base URL
        if not base_url.endswith('/'):
            base_url = base_url + '/'
        
        # Add priority pages
        queue.append(base_url)  # Always start with base URL
        for page in PRIORITY_PAGES:
            if page:  # Skip empty string
                url = urljoin(base_url, page)
                if url not in queue:
                    queue.append(url)
        
        print(f"Initial queue size: {len(queue)}")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        }
        
        documents = []
        failed_urls = []
        
        while queue and len(visited) < max_pages:
            url = queue.popleft()
            
            # Normalize URL
            url = url.split("#")[0].split("?")[0].rstrip('/')
            
            if url in visited or not self.is_valid_url(url, base_domain):
                continue
            
            try:
                print(f"\nScraping: {url}")
                response = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
                
                print(f"Status: {response.status_code}")
                
                if response.status_code != 200:
                    print(f"Skipping {url} - Status: {response.status_code}")
                    failed_urls.append(url)
                    continue
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' not in content_type:
                    print(f"Skipping {url} - Not HTML: {content_type}")
                    continue
                
                visited.add(url)
                
                if progress_callback:
                    progress_callback(len(visited), max_pages, url)
                
                # Parse content
                soup = BeautifulSoup(response.text, "html.parser")
                content_data = self.extract_content(soup, url)
                
                # More lenient content check
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
                    print(f"✓ Added document {len(documents)} - {len(content_data['content'])} chars")
                else:
                    print(f"✗ Content too short: {len(content_data['content'])} chars")
                
                # Find new links
                if len(visited) < max_pages:
                    links_found = 0
                    for link in soup.find_all("a", href=True):
                        if links_found >= 50:
                            break
                        try:
                            next_url = urljoin(url, link['href'])
                            next_url = next_url.split("#")[0].split("?")[0].rstrip('/')
                            
                            if next_url not in visited and self.is_valid_url(next_url, base_domain):
                                if next_url not in queue:
                                    queue.append(next_url)
                                    links_found += 1
                        except:
                            pass
                    
                    if links_found > 0:
                        print(f"Added {links_found} new links to queue")
                
                time.sleep(0.2)
            
            except requests.exceptions.Timeout:
                print(f"✗ Timeout: {url}")
                failed_urls.append(url)
                continue
            except requests.exceptions.RequestException as e:
                print(f"✗ Request error for {url}: {str(e)[:100]}")
                failed_urls.append(url)
                continue
            except Exception as e:
                print(f"✗ Error processing {url}: {str(e)[:100]}")
                self.debug_info.append(f"Error {url}: {str(e)[:50]}")
                continue
        
        print(f"\n=== Scraping Summary ===")
        print(f"Pages visited: {len(visited)}")
        print(f"Documents created: {len(documents)}")
        print(f"Failed URLs: {len(failed_urls)}")
        
        if len(documents) == 0:
            error_msg = f"No content extracted from {len(visited)} pages visited."
            if failed_urls:
                error_msg += f" Failed URLs: {failed_urls[:3]}"
            raise Exception(error_msg)
        
        if len(documents) < 3:
            print(f"WARNING: Only {len(documents)} documents created")
            # Don't fail if we got at least 1 document
            if len(documents) == 0:
                raise Exception(f"No valid content found on website")
        
        return documents, {
            'emails': list(self.company_info['emails'])[:10],
            'phones': list(self.company_info['phones'])[:10],
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
OPTIMIZED MULTI-COMPANY AI CHATBOT - PART 2 (app.py)
Save this as: app.py

Run with: streamlit run app.py
"""

import streamlit as st
import time
import requests
import json
import os
from typing import List
import shutil
import hashlib

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


class CompanyAI:
    """AI Engine with RAG for answering questions - OPTIMIZED"""
    
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
        """Initialize by scraping and creating vector store - IMPROVED"""
        try:
            if progress_callback:
                progress_callback(0, max_pages, "Starting scraper...")
            
            scraper = WebScraper(self.company_slug)
            documents, company_info = scraper.scrape_website(
                website_url, 
                max_pages, 
                progress_callback
            )
            
            # More lenient check - allow even 1 document
            if len(documents) == 0:
                self.status["error"] = "No content could be extracted from website"
                return False
            
            print(f"✓ Scraped {len(documents)} documents")
            
            self.company_data = company_info
            
            if progress_callback:
                progress_callback(max_pages, max_pages, "Processing documents...")
            
            # CHANGED: Reduced chunk size and overlap
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,  # Changed from 1000
                chunk_overlap=150,  # Changed from 200
                separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "],
                length_function=len
            )
            
            # Filter out empty/short documents
            valid_docs = [doc for doc in documents if doc.page_content and len(doc.page_content.strip()) > 100]
            
            if len(valid_docs) == 0:
                self.status["error"] = "No valid content after filtering"
                return False
            
            print(f"✓ {len(valid_docs)} valid documents")
            
            split_docs = text_splitter.split_documents(valid_docs)
            
            # More lenient - allow as few as 3 chunks
            if len(split_docs) < 3:
                self.status["error"] = f"Not enough text chunks: {len(split_docs)} (need at least 3)"
                return False
            
            print(f"✓ Split into {len(split_docs)} chunks")
            
            if progress_callback:
                progress_callback(max_pages, max_pages, "Creating embeddings...")
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            chroma_dir = get_chroma_directory(self.company_slug)
            
            if os.path.exists(chroma_dir):
                try:
                    shutil.rmtree(chroma_dir)
                    print(f"✓ Cleaned old data")
                except Exception as e:
                    print(f"Warning: Could not remove old data: {e}")
            
            os.makedirs(chroma_dir, exist_ok=True)
            
            if progress_callback:
                progress_callback(max_pages, max_pages, "Building vector database...")
            
            self.vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory=chroma_dir,
                collection_name="company_knowledge"
            )
            
            # Verify vectorstore
            count = self.vectorstore._collection.count()
            if count == 0:
                raise Exception("Vectorstore is empty after creation")
            
            print(f"✓ Vectorstore created with {count} embeddings")
            
            # CHANGED: Simplified retriever for faster search
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",  # Changed from "mmr"
                search_kwargs={"k": min(5, count)}  # Don't try to retrieve more than we have
            )
            
            self.status["ready"] = True
            
            if progress_callback:
                progress_callback(max_pages, max_pages, "✅ Chatbot Ready!")
            
            return True
        
        except Exception as e:
            self.status["error"] = f"Initialization error: {str(e)}"
            print(f"INITIALIZATION ERROR: {e}")
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
            
            # CHANGED: Simplified retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
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
        try:
            # Check if system is ready
            if not self.status.get("ready", False):
                return "⚠️ System is initializing. Please wait..."
            
            # Check if retriever exists
            if not self.retriever:
                return "⚠️ Chatbot not properly initialized. Please try reloading."
            
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
            
            print(f"\n=== Processing question: {question} ===")
            
            # Retrieve relevant documents - FIXED: Use invoke() for newer LangChain versions
            print("Retrieving relevant documents...")
            try:
                # Try the new invoke method first
                relevant_docs = self.retriever.invoke(question)
            except AttributeError:
                # Fallback to old method if invoke doesn't exist
                try:
                    relevant_docs = self.retriever.get_relevant_documents(question)
                except Exception as e:
                    print(f"Retrieval error: {e}")
                    return "⚠️ Error retrieving information. Please try again."
            
            print(f"Found {len(relevant_docs)} relevant documents")
            
            if not relevant_docs or len(relevant_docs) == 0:
                return "I couldn't find specific information about that. Could you rephrase your question?"
            
            # Build context - CHANGED: Reduced from 5 to 4 docs, 500 to 400 chars
            context_parts = []
            for i, doc in enumerate(relevant_docs[:4]):  # Changed from [:5]
                if hasattr(doc, 'page_content') and doc.page_content:
                    title = doc.metadata.get('title', '')
                    content = doc.page_content[:400]  # Changed from [:500]
                    
                    context_part = f"[{title}]\n{content}" if title else content
                    context_parts.append(context_part)
                    print(f"Doc {i+1}: {len(doc.page_content)} chars")
            
            if not context_parts:
                return "I found documents but couldn't extract content. Please try again."
            
            context = "\n\n---\n\n".join(context_parts)
            print(f"Total context length: {len(context)} chars")
            
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
            
            # Build prompt - CHANGED: Reduced context size
            prompt = self.qa_prompt.format(
                company_name=company_name,
                context=context[:3000],  # Changed from [:4000]
                chat_history=history_text,
                question=question
            )
            
            print(f"Prompt length: {len(prompt)} chars")
            print("Calling LLM...")
            
            # Call LLM
            answer = self._call_llm(prompt)
            
            print(f"Answer received: {answer[:100] if answer else 'None'}...")
            
            if answer and not answer.startswith("⚠️"):
                self.save_chat(question, answer, session_id)
                return answer
            elif answer:
                return answer  # Return error message
            else:
                return "I'm having trouble generating a response. Please try again."
        
        except Exception as e:
            print(f"ERROR in ask(): {str(e)}")
            import traceback
            traceback.print_exc()
            return f"⚠️ An error occurred. Please try again or reload the page."
    
    def _call_llm(self, prompt: str, max_retries: int = 2) -> str:  # CHANGED: From 3 to 2
        """Call LLM API"""
        if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "":
            print("ERROR: OPENROUTER_API_KEY not set!")
            return "⚠️ API key not configured. Please set OPENROUTER_API_KEY environment variable."
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "AI Chatbot"
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
                print(f"Attempt {attempt + 1}/{max_retries} - Calling LLM...")
                
                response = requests.post(
                    OPENROUTER_API_BASE,
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                print(f"Response status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"Response received: {str(result)[:200]}...")
                    
                    if "choices" in result and len(result["choices"]) > 0:
                        answer = result["choices"][0]["message"]["content"].strip()
                        
                        if answer and len(answer) > 10:
                            return answer
                        else:
                            print(f"Answer too short: {answer}")
                    else:
                        print(f"No choices in response: {result}")
                
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
                
                else:
                    error_text = response.text[:300]
                    print(f"Error response: {error_text}")
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
            
            except Exception as e:
                print(f"LLM error on attempt {attempt + 1}: {str(e)}")
                import traceback
                traceback.print_exc()
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
        
        return "⚠️ Failed to get response from AI. Please try again."
    
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
    layout="wide"
)

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
    .header-title { color: white; font-size: 2.5rem; font-weight: bold; margin: 0; }
    .company-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
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
    st.caption("💡 Powered by AI & RAG")

# HOME PAGE
if st.session_state.page == "home":
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🤖 AI Chatbot Generator</h1>
        <p style="color: white; font-size: 1.1rem; margin-top: 1rem;">Create intelligent chatbots from any website in minutes!</p>
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
    
    st.info("💡 **How it works:** Enter a company name and website URL. The AI will scrape the website, extract information, and create an intelligent chatbot that can answer questions about the company!")

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
        
        st.caption("⚠️ This process may take 1-2 minutes depending on website size")
        
        submitted = st.form_submit_button("🚀 Create Chatbot", type="primary", use_container_width=True)
        
        if submitted:
            if name and url:
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
                        prog_bar.progress(progress, text=f"📄 Scraped {current}/{total} pages")
                        status_text.caption(f"Current: {url_text[:60]}...")
                    
                    ai = CompanyAI(slug)
                    success = ai.initialize(url, pages, progress_cb)
                    
                    if success:
                        update_company_after_scraping(slug, ai.company_data, "completed")
                        prog_bar.progress(1.0, text="✅ Chatbot is ready!")
                        st.success("🎉 Chatbot created successfully!")
                        st.balloons()
                        time.sleep(2)
                        
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
        for c in companies:
            st.markdown('<div class="company-card">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"### {c.company_name}")
                st.caption(f"🌐 {c.website_url}")
                st.caption(f"📄 {c.pages_scraped} pages scraped")
            
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

# CHAT PAGE
elif st.session_state.page == "chat":
    if not st.session_state.current_company:
        st.error("No company selected")
        if st.button("← Go Back"):
            st.session_state.page = "list"
            st.rerun()
        st.stop()
    
    c = get_company_by_slug(st.session_state.current_company)
    
    if not c:
        st.error("Company not found")
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
                    st.error(f"Failed to load chatbot: {ai.status.get('error', 'Unknown error')}")
                    if st.button("← Go Back"):
                        st.session_state.page = "list"
                        st.rerun()
                    st.stop()
                st.session_state.ai_instance = ai
                st.success("✅ Chatbot loaded successfully!")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"Error loading chatbot: {str(e)}")
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
    if st.button("← Back to List"):
        st.session_state.page = "list"
        st.session_state.current_company = None
        st.session_state.ai_instance = None
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
    
    # Display chat messages
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
        st.session_state.chat_history.append({"question": prompt, "answer": answer})
        
        st.rerun()
    
    # Clear chat button
    if len(st.session_state.messages) > 0:
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🗑️ Clear Chat", key="clear_chat"):
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.rerun()
