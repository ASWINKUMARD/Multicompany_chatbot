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
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools

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

import streamlit as st

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

DATABASE_URL = "sqlite:///./multi_company_chatbots.db"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "kwaipilot/kat-coder-pro:free"

# Global embeddings cache to avoid reloading
_EMBEDDINGS_CACHE = None

PRIORITY_PAGES = [
    "", "about", "services", "solutions", "products", "contact", "team",
    "careers", "about-us", "contact-us", "our-services", "home"
]

# ============================================================================
# DATABASE SETUP
# ============================================================================

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, pool_pre_ping=True)
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
    return f"./chroma_db/{company_slug}"


def get_embeddings():
    """Get cached embeddings model"""
    global _EMBEDDINGS_CACHE
    
    if _EMBEDDINGS_CACHE is None:
        print("🔄 Loading embeddings model (first time only)...")
        _EMBEDDINGS_CACHE = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )
        print("✅ Embeddings model loaded and cached")
    
    return _EMBEDDINGS_CACHE


@functools.lru_cache(maxsize=1000)
def clean_text_cached(text: str) -> str:
    """Clean and normalize text with caching"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:()\-\'\"]+', '', text)
    return text.strip()


# ============================================================================
# WEB SCRAPER (OPTIMIZED)
# ============================================================================

class WebScraper:
    """Optimized web scraper with parallel requests"""
    
    def __init__(self, company_slug: str):
        self.company_slug = company_slug
        self.company_info = {
            'emails': set(),
            'phones': set(),
            'address_india': None,
            'address_international': None
        }
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
    
    def extract_contact_info(self, text: str):
        """Extract contact information from text"""
        if not text:
            return
        
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        for email in emails:
            email_lower = email.lower()
            if not email_lower.endswith(('.png', '.jpg', '.gif', '.jpeg', '.svg')):
                self.company_info['emails'].add(email_lower)
        
        # Extract phone numbers
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
        
        # Extract addresses (optimized)
        if not self.company_info['address_india']:
            india_match = re.search(r'(.{0,100}(?:india|mumbai|delhi|bangalore|chennai|madurai).{0,200})', 
                                   text, re.IGNORECASE)
            if india_match:
                addr = clean_text_cached(india_match.group(1))
                if 20 < len(addr) < 500:
                    self.company_info['address_india'] = addr
        
        if not self.company_info['address_international']:
            intl_match = re.search(r'(.{0,100}(?:singapore|usa|uk|uae|dubai|london).{0,200})', 
                                  text, re.IGNORECASE)
            if intl_match:
                addr = clean_text_cached(intl_match.group(1))
                if 20 < len(addr) < 500:
                    self.company_info['address_international'] = addr
    
    def is_valid_url(self, url: str, base_domain: str) -> bool:
        """Check if URL should be scraped"""
        if not url:
            return False
        
        try:
            parsed = urlparse(url)
            
            if parsed.netloc != base_domain:
                return False
            
            skip_extensions = ['.pdf', '.jpg', '.png', '.zip', '.mp4', '.css', 
                             '.js', '.gif', '.svg', '.ico', '.woff', '.ttf']
            skip_paths = ['/wp-admin/', '/admin/', '/login', '/cart/', '/checkout/']
            
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
    
    def fetch_page(self, url: str) -> Optional[Dict]:
        """Fetch single page (for parallel execution)"""
        try:
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Extract title
                title = soup.find('title')
                title_text = title.get_text(strip=True) if title else ""
                
                # Extract meta description
                content = ""
                meta_desc = soup.find('meta', attrs={"name": "description"})
                if meta_desc and meta_desc.get("content"):
                    content += meta_desc["content"] + "\n\n"
                
                # Extract contact info from full text
                full_text = soup.get_text(separator="\n", strip=True)
                self.extract_contact_info(full_text)
                
                # Remove unwanted tags
                for tag in soup(['script', 'style', 'iframe', 'nav', 'footer', 
                               'header', 'aside', 'noscript']):
                    tag.decompose()
                
                # Extract main content efficiently
                text_parts = []
                
                # Priority: main content areas
                for selector in ["main", "article", "[role='main']", ".content", "#content"]:
                    for elem in soup.select(selector):
                        text = elem.get_text(separator="\n", strip=True)
                        if len(text) > 100:
                            text_parts.append(text)
                            break
                    if text_parts:
                        break
                
                # Fallback: headers and paragraphs
                if not text_parts:
                    for tag in soup.find_all(['h1', 'h2', 'h3', 'p']):
                        text = tag.get_text(strip=True)
                        if len(text) > 20:
                            text_parts.append(text)
                
                # Join and clean
                if text_parts:
                    combined = "\n".join(text_parts)
                    lines = [clean_text_cached(line) for line in combined.split("\n") 
                            if len(line.strip()) > 15]
                    
                    # Deduplicate
                    seen = set()
                    unique_lines = []
                    for line in lines:
                        line_lower = line.lower()
                        if line_lower not in seen and len(line) > 20:
                            seen.add(line_lower)
                            unique_lines.append(line)
                    
                    content += "\n".join(unique_lines)
                
                # Find new links
                new_links = []
                for link in soup.find_all("a", href=True):
                    try:
                        next_url = urljoin(url, link['href'])
                        next_url = next_url.split("#")[0].split("?")[0].rstrip('/')
                        new_links.append(next_url)
                    except:
                        pass
                
                return {
                    'url': url,
                    'title': title_text,
                    'content': content,
                    'links': new_links,
                    'success': True
                }
            
            return {'url': url, 'success': False}
            
        except Exception as e:
            return {'url': url, 'success': False, 'error': str(e)[:100]}
    
    def scrape_website(self, base_url: str, max_pages: int = 40, 
                       progress_callback=None) -> Tuple[List[Document], Dict]:
        """Scrape website with parallel requests"""
        visited = set()
        queue = deque()
        base_domain = urlparse(base_url).netloc
        
        base_url = base_url.rstrip('/') + '/'
        
        # Add priority pages
        for page in PRIORITY_PAGES:
            for url_variant in [urljoin(base_url, page), urljoin(base_url, page + '/')]:
                if url_variant not in queue:
                    queue.append(url_variant)
        
        documents = []
        
        # Use ThreadPoolExecutor for parallel scraping
        with ThreadPoolExecutor(max_workers=5) as executor:
            while queue and len(visited) < max_pages:
                # Get batch of URLs
                batch = []
                while queue and len(batch) < 5:
                    url = queue.popleft()
                    url = url.split("#")[0].split("?")[0].rstrip('/')
                    
                    if url not in visited and self.is_valid_url(url, base_domain):
                        batch.append(url)
                        visited.add(url)
                
                if not batch:
                    break
                
                # Fetch batch in parallel
                futures = {executor.submit(self.fetch_page, url): url for url in batch}
                
                for future in as_completed(futures):
                    url = futures[future]
                    
                    try:
                        result = future.result()
                        
                        if result.get('success') and len(result.get('content', '')) > 100:
                            doc = Document(
                                page_content=result['content'][:3000],  # Limit size
                                metadata={
                                    'source': url,
                                    'title': result['title'],
                                    'company': self.company_slug
                                }
                            )
                            documents.append(doc)
                            
                            # Add new links to queue
                            for link in result.get('links', []):
                                if (link not in visited and 
                                    self.is_valid_url(link, base_domain) and
                                    link not in queue):
                                    queue.append(link)
                        
                        if progress_callback:
                            progress_callback(len(visited), max_pages, url)
                    
                    except Exception as e:
                        print(f"Error processing {url}: {e}")
                
                time.sleep(0.1)  # Small delay between batches
        
        if len(documents) < 3:
            raise Exception(f"Insufficient content: {len(documents)} pages")
        
        return documents, {
            'emails': list(self.company_info['emails'])[:10],
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
        print(f"Error creating company: {e}")
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
        print(f"Error updating company: {e}")
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


# ============================================================================
# AI ENGINE (OPTIMIZED)
# ============================================================================

class CompanyAI:
    """Optimized AI Engine with RAG"""
    
    def __init__(self, company_slug: str):
        self.company_slug = company_slug
        self.vectorstore = None
        self.retriever = None
        self.status = {"ready": False, "error": None}
        self.company_data = None
        
        self.qa_template = """You are an AI assistant for {company_name}. Answer using the provided context.

CONTEXT:
{context}

RECENT CHAT:
{chat_history}

QUESTION: {question}

INSTRUCTIONS:
- Use ONLY information from context
- Be specific and helpful
- If info not available, say so and provide contact details
- Keep answers 2-4 sentences
- Be conversational

ANSWER:"""

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
                self.status["error"] = f"Not enough content: {len(documents)} pages"
                return False
            
            self.company_data = company_info
            
            if progress_callback:
                progress_callback(max_pages, max_pages, "Creating chunks...")
            
            # Split documents (optimized settings)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=150,
                separators=["\n\n", "\n", ". ", " "],
                length_function=len
            )
            
            split_docs = text_splitter.split_documents(documents)
            
            if len(split_docs) < 5:
                self.status["error"] = f"Not enough chunks: {len(split_docs)}"
                return False
            
            if progress_callback:
                progress_callback(max_pages, max_pages, "Loading embeddings...")
            
            # Use cached embeddings
            embeddings = get_embeddings()
            
            # Setup ChromaDB
            chroma_dir = get_chroma_directory(self.company_slug)
            
            if os.path.exists(chroma_dir):
                import shutil
                shutil.rmtree(chroma_dir)
            
            os.makedirs(chroma_dir, exist_ok=True)
            
            if progress_callback:
                progress_callback(max_pages, max_pages, "Building database...")
            
            # Create vectorstore with batch processing
            self.vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=embeddings,
                persist_directory=chroma_dir,
                collection_name="company_knowledge"
            )
            
            if self.vectorstore._collection.count() == 0:
                raise Exception("Vectorstore is empty")
            
            # Create retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.7}
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
                self.status["error"] = "Data not found"
                return False
            
            # Use cached embeddings
            embeddings = get_embeddings()
            
            # Load vectorstore
            self.vectorstore = Chroma(
                persist_directory=chroma_dir,
                embedding_function=embeddings,
                collection_name="company_knowledge"
            )
            
            if self.vectorstore._collection.count() == 0:
                self.status["error"] = "Empty database"
                return False
            
            # Create retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.7}
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
            return "📞 Contact information not available."
        
        info = self.company_data
        msg = "📞 **CONTACT INFORMATION**\n\n"
        
        if info.get('address_india'):
            msg += f"🏢 **India Office:**\n{info['address_india']}\n\n"
        
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
            if not self.status.get("ready", False):
                return "⚠️ System initializing..."
            
            if not self.retriever:
                return "⚠️ Not initialized properly"
            
            q_lower = question.lower().strip()
            
            # Handle greetings
            if q_lower in ["hi", "hello", "hey", "hai"] or len(q_lower) < 5:
                company = get_company_by_slug(self.company_slug)
                company_name = company.company_name if company else "our company"
                return f"👋 Hello! I'm here to answer questions about {company_name}. How can I help?"
            
            # Handle contact requests
            contact_keywords = ["email", "contact", "phone", "address", "office", "location", "reach"]
            if any(keyword in q_lower for keyword in contact_keywords):
                return self.get_contact_info()
            
            # Retrieve documents
            try:
                relevant_docs = self.retriever.invoke(question)
            except AttributeError:
                relevant_docs = self.retriever.get_relevant_documents(question)
            
            if not relevant_docs:
                return "I couldn't find information about that. Could you rephrase?"
            
            # Build context (optimized size)
            context_parts = []
            for doc in relevant_docs[:3]:  # Use top 3 docs only
                if hasattr(doc, 'page_content') and doc.page_content:
                    content = doc.page_content[:400]  # Limit each doc
                    context_parts.append(content)
            
            if not context_parts:
                return "Couldn't extract content. Please rephrase."
            
            context = "\n\n".join(context_parts)
            
            # Format chat history
            history_text = ""
            if chat_history and len(chat_history) > 0:
                history_text = "\n".join([
                    f"Q: {msg['question']}\nA: {msg['answer']}" 
                    for msg in chat_history[-2:]  # Last 2 only
                ])
            
            # Get company name
            company = get_company_by_slug(self.company_slug)
            company_name = company.company_name if company else "the company"
            
            # Build prompt
            prompt = self.qa_prompt.format(
                company_name=company_name,
                context=context[:2000],  # Hard limit
                chat_history=history_text[:300],
                question=question
            )
            
            # Call LLM
            answer = self._call_llm(prompt)
            
            if answer and not answer.startswith("⚠️") and not answer.startswith("❌"):
                self.save_chat(question, answer, session_id)
                return answer
            elif answer:
                return answer
            else:
                return "❌ Failed to generate response."
        
        except Exception as e:
            print(f"Error in ask(): {e}")
            return "❌ An error occurred. Please try again."
    
    def _call_llm(self, prompt: str, max_retries: int = 2) -> str:
        """Call LLM API"""
        if not OPENROUTER_API_KEY:
            return "⚠️ API key not configured."
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
        }
        
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.4,
            "max_tokens": 400,
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    OPENROUTER_API_BASE,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        answer = result["choices"][0]["message"]["content"].strip()
                        if answer and len(answer) > 10:
                            return answer
                
                elif response.status_code == 429:
                    if attempt < max_retries - 1:
                        time.sleep(3)
                        continue
                    else:
                        return "⚠️ Rate limit reached. Please wait."
                
                elif response.status_code == 401:
                    return "❌ Invalid API key."
                
                else:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
            
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    return "⚠️ Request timed out."
            
            except Exception as e:
                print(f"LLM error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
        
        return "❌ Failed to get response."
    
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
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
    div[data-testid="stChatMessage"] {
        border-radius: 10px;
        padding: 1rem;
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
    
    if st.session_state.current_company:
        company = get_company_by_slug(st.session_state.current_company)
        if company:
            st.info(f"📌 Current: {company.company_name}")
    
    st.markdown("---")
    st.caption("🤖 Powered by AI & RAG")
    st.caption(f"Session: {st.session_state.session_id[:8]}")

# ============================================================================
# HOME PAGE
# ============================================================================

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
    
    st.markdown("### 🚀 How it works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**1️⃣ Enter Details**\n\nProvide company name and website URL")
    
    with col2:
        st.info("**2️⃣ AI Scrapes**\n\nOur AI extracts all content")
    
    with col3:
        st.info("**3️⃣ Chat Away!**\n\nGet instant answers")
    
    st.markdown("### ✨ Features")
    
    features = [
        "🔍 Automatic website scraping",
        "🧠 Intelligent Q&A",
        "📞 Contact extraction",
        "💬 Natural conversations",
        "📊 Multiple companies",
        "⚡ Fast responses"
    ]
    
    col1, col2 = st.columns(2)
    
    for i, feature in enumerate(features):
        if i % 2 == 0:
            col1.markdown(f"- {feature}")
        else:
            col2.markdown(f"- {feature}")

# ============================================================================
# CREATE PAGE
# ============================================================================

elif st.session_state.page == "create":
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">➕ Create New Chatbot</h1>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("create_form", clear_on_submit=False):
        name = st.text_input(
            "Company Name *", 
            placeholder="e.g., Acme Corporation", 
            key="form_name"
        )
        
        url = st.text_input(
            "Website URL *", 
            placeholder="e.g., https://example.com", 
            key="form_url"
        )
        
        pages = st.slider(
            "Max Pages to Scrape", 
            10, 50, 30, 10, 
            key="form_pages",
            help="Recommended: 30 pages for faster setup"
        )
        
        st.info("⏱️ Optimized process: 1-3 minutes")
        
        submitted = st.form_submit_button("🚀 Create Chatbot", type="primary", use_container_width=True)
        
        if submitted:
            if not name or not url:
                st.warning("⚠️ Please fill all fields")
            else:
                if not url.startswith('http'):
                    url = 'https://' + url
                
                try:
                    parsed = urlparse(url)
                    if not parsed.netloc:
                        st.error("❌ Invalid URL format")
                        st.stop()
                except:
                    st.error("❌ Invalid URL")
                    st.stop()
                
                with st.spinner("Creating company..."):
                    slug = create_company(name, url, pages)
                
                if slug:
                    st.success(f"✅ Created: {name}")
                    
                    prog_bar = st.progress(0, text="Initializing...")
                    status_text = st.empty()
                    
                    def progress_cb(current, total, url_text):
                        progress = min(current / total, 1.0)
                        prog_bar.progress(progress, text=f"📄 Scraped {current}/{total} pages")
                        status_text.caption(f"Current: {url_text[:70]}...")
                    
                    ai = CompanyAI(slug)
                    success = ai.initialize(url, pages, progress_cb)
                    
                    if success:
                        update_company_after_scraping(slug, ai.company_data, "completed")
                        prog_bar.progress(1.0, text="✅ Ready!")
                        st.success("🎉 Chatbot created!")
                        st.balloons()
                        time.sleep(1)
                        
                        st.session_state.current_company = slug
                        st.session_state.ai_instance = ai
                        st.session_state.messages = []
                        st.session_state.chat_history = []
                        st.session_state.page = "chat"
                        st.rerun()
                    else:
                        error_msg = ai.status.get('error', 'Unknown error')
                        st.error(f"❌ Failed: {error_msg}")
                        update_company_after_scraping(slug, {}, "failed")
                else:
                    st.error("❌ Company already exists!")
    
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

# ============================================================================
# LIST PAGE
# ============================================================================

elif st.session_state.page == "list":
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">📋 All Chatbots</h1>
    </div>
    """, unsafe_allow_html=True)
    
    companies = get_all_companies()
    
    if not companies:
        st.info("📭 No chatbots yet")
        if st.button("➕ Create First Chatbot", type="primary"):
            st.session_state.page = "create"
            st.rerun()
    else:
        total = len(companies)
        completed = sum(1 for c in companies if c.scraping_status == "completed")
        failed = sum(1 for c in companies if c.scraping_status == "failed")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total", total)
        col2.metric("Active", completed)
        col3.metric("Failed", failed)
        
        st.markdown("---")
        
        for c in companies:
            st.markdown('<div class="company-card">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"### {c.company_name}")
                st.caption(f"🌐 {c.website_url}")
                st.caption(f"📄 {c.pages_scraped} pages")
                if c.last_scraped:
                    st.caption(f"🕒 {c.last_scraped.strftime('%Y-%m-%d %H:%M')}")
            
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
    
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

# ============================================================================
# CHAT PAGE
# ============================================================================

elif st.session_state.page == "chat":
    if not st.session_state.current_company:
        st.error("❌ No company selected")
        if st.button("← Go Back"):
            st.session_state.page = "list"
            st.rerun()
        st.stop()
    
    c = get_company_by_slug(st.session_state.current_company)
    
    if not c:
        st.error("❌ Company not found")
        if st.button("← Go Back"):
            st.session_state.page = "list"
            st.rerun()
        st.stop()
    
    if not st.session_state.ai_instance:
        with st.spinner("Loading chatbot..."):
            try:
                ai = CompanyAI(st.session_state.current_company)
                if not ai.load_existing():
                    st.error(f"❌ Failed to load: {ai.status.get('error', 'Unknown')}")
                    if st.button("← Go Back"):
                        st.session_state.page = "list"
                        st.rerun()
                    st.stop()
                st.session_state.ai_instance = ai
                st.success("✅ Loaded!")
                time.sleep(0.3)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                if st.button("← Go Back"):
                    st.session_state.page = "list"
                    st.rerun()
                st.stop()
    
    st.markdown(f"""
    <div class="header-container">
        <h1 class="header-title">💬 Chat with {c.company_name}</h1>
        <p style="color: white; margin-top: 0.5rem;">Ask me anything!</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        if st.button("← Back to List"):
            st.session_state.page = "list"
            st.session_state.current_company = None
            st.session_state.ai_instance = None
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if len(st.session_state.messages) > 0:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.rerun()
    
    with col3:
        if st.button("📞 Contact"):
            contact_info = st.session_state.ai_instance.get_contact_info()
            st.info(contact_info)
    
    st.markdown("---")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Ask a question...", key="chat_input_main"):
        if len(prompt.strip()) == 0:
            st.warning("⚠️ Please enter a question")
            st.stop()
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    if not st.session_state.ai_instance:
                        answer = "⚠️ Chatbot disconnected. Please reload."
                    else:
                        answer = st.session_state.ai_instance.ask(
                            prompt,
                            st.session_state.chat_history,
                            st.session_state.session_id
                        )
                except Exception as e:
                    print(f"Chat error: {e}")
                    answer = "❌ Error occurred. Please reload."
            
            st.markdown(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_history.append({"question": prompt, "answer": answer})
        
        st.rerun()
    
    if len(st.session_state.messages) == 0:
        st.markdown("### 💡 Sample Questions")
        
        sample_questions = [
            "What does this company do?",
            "What services do you offer?",
            "How can I contact you?",
            "Tell me about your team",
        ]
        
        cols = st.columns(2)
        for i, q in enumerate(sample_questions):
            if cols[i % 2].button(q, key=f"sample_{i}"):
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()
