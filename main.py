"""
AI Chatbot Generator - Complete Fixed Backend (main.py)
Thoroughly analyzed and optimized version
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
from sqlalchemy.exc import SQLAlchemyError
import re
import os
import json
import time
import logging
from typing import Optional, Dict, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
# DATABASE CONFIGURATION
# ============================================================================

DATABASE_URL = "sqlite:///./multi_company_chatbots.db"

try:
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False, "timeout": 30},
        pool_pre_ping=True,
        pool_recycle=3600,
        pool_size=10,
        max_overflow=20,
        echo=False
    )
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    Base = declarative_base()
    logger.info("Database engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")
    raise


# ============================================================================
# DATABASE MODELS
# ============================================================================

class Company(Base):
    """Stores information about each company"""
    __tablename__ = "companies"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    company_name = Column(String(255), unique=True, nullable=False, index=True)
    company_slug = Column(String(255), unique=True, nullable=False, index=True)
    website_url = Column(String(500), nullable=False)
    
    emails = Column(Text, nullable=True, default='[]')
    phones = Column(Text, nullable=True, default='[]')
    address_india = Column(Text, nullable=True)
    address_international = Column(Text, nullable=True)
    
    pages_scraped = Column(Integer, default=0)
    last_scraped = Column(DateTime, nullable=True)
    scraping_status = Column(String(50), default="pending")
    error_message = Column(Text, nullable=True)
    
    is_active = Column(Boolean, default=True)
    max_pages_to_scrape = Column(Integer, default=40)
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class ChatHistory(Base):
    """Stores chat conversations"""
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    company_slug = Column(String(255), nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    session_id = Column(String(100), nullable=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


class UserContact(Base):
    """Stores user contact information"""
    __tablename__ = "user_contacts"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    company_slug = Column(String(255), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False)
    phone = Column(String(20), nullable=False)
    session_id = Column(String(100), nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# Create all tables
try:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
except Exception as e:
    logger.error(f"Failed to create tables: {e}")
    raise


# ============================================================================
# CONFIGURATION
# ============================================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.0-flash-exp:free"

PRIORITY_PAGES = [
    "", "about", "about-us", "services", "solutions", "products", 
    "contact", "contact-us", "team", "careers", "our-services", "home"
]

# Scraping configuration
REQUEST_TIMEOUT = 20
MAX_RETRIES = 3
DELAY_BETWEEN_REQUESTS = 0.5
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_slug(company_name: str) -> str:
    """Convert company name to URL-friendly slug"""
    if not company_name:
        raise ValueError("Company name cannot be empty")
    
    slug = company_name.lower().strip()
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    slug = slug.strip('-')
    
    if not slug:
        raise ValueError("Invalid company name - cannot create slug")
    
    return slug[:200]  # Limit length


def get_chroma_directory(company_slug: str) -> str:
    """Get ChromaDB directory path"""
    if not company_slug:
        raise ValueError("Company slug cannot be empty")
    
    base_dir = "./chroma_db"
    os.makedirs(base_dir, exist_ok=True)
    
    # Sanitize slug for filesystem
    safe_slug = re.sub(r'[^a-z0-9\-_]', '', company_slug.lower())
    return os.path.join(base_dir, safe_slug)


def validate_url(url: str) -> bool:
    """Validate URL format"""
    if not url:
        return False
    
    try:
        result = urlparse(url)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except Exception:
        return False


# ============================================================================
# WEB SCRAPER CLASS
# ============================================================================

class WebScraper:
    """Advanced web scraper with robust error handling"""
    
    def __init__(self, company_slug: str):
        self.company_slug = company_slug
        self.company_info = {
            'emails': set(),
            'phones': set(),
            'address_india': None,
            'address_international': None
        }
        self.debug_info = []
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': USER_AGENT})
    
    def __del__(self):
        """Cleanup session on deletion"""
        try:
            if hasattr(self, 'session'):
                self.session.close()
        except:
            pass
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        try:
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            # Remove special characters but keep punctuation
            text = re.sub(r'[^\w\s.,!?;:()\-\'\"@+]+', '', text)
            return text.strip()
        except Exception as e:
            logger.warning(f"Error cleaning text: {e}")
            return text.strip()
    
    def extract_contact_info(self, text: str):
        """Extract contact information with validation"""
        if not text:
            return
        
        try:
            # Extract emails with validation
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
            emails = re.findall(email_pattern, text)
            
            for email in emails:
                email_lower = email.lower()
                # Filter out image files and invalid emails
                if (not email_lower.endswith(('.png', '.jpg', '.gif', '.svg', '.jpeg')) and
                    '@' in email_lower and 
                    len(email_lower) > 5 and
                    len(email_lower) < 100):
                    self.company_info['emails'].add(email_lower)
            
            # Extract phone numbers with validation
            phone_patterns = [
                r'\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
                r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
                r'\d{3}[-.\s]\d{3}[-.\s]\d{4}'
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
                india_keywords = [
                    'india', 'mumbai', 'delhi', 'bangalore', 'bengaluru',
                    'chennai', 'madurai', 'pune', 'hyderabad', 'kolkata'
                ]
                
                if any(keyword in low for keyword in india_keywords):
                    if not self.company_info['address_india']:
                        start_idx = max(0, i - 2)
                        end_idx = min(len(lines), i + 4)
                        block = " ".join(lines[start_idx:end_idx])
                        cleaned = self.clean_text(block)
                        if 20 < len(cleaned) < 500:
                            self.company_info['address_india'] = cleaned
                
                # International addresses
                intl_keywords = [
                    'singapore', 'usa', 'uk', 'uae', 'dubai', 
                    'london', 'new york', 'california'
                ]
                
                if any(keyword in low for keyword in intl_keywords):
                    if not self.company_info['address_international']:
                        start_idx = max(0, i - 2)
                        end_idx = min(len(lines), i + 4)
                        block = " ".join(lines[start_idx:end_idx])
                        cleaned = self.clean_text(block)
                        if 20 < len(cleaned) < 500:
                            self.company_info['address_international'] = cleaned
        
        except Exception as e:
            logger.warning(f"Error extracting contact info: {e}")
    
    def is_valid_url(self, url: str, base_domain: str) -> bool:
        """Check if URL should be scraped with comprehensive validation"""
        if not url:
            return False
        
        try:
            parsed = urlparse(url)
            
            # Must be same domain
            if parsed.netloc != base_domain:
                return False
            
            # Skip file extensions
            skip_extensions = [
                '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico',
                '.zip', '.rar', '.tar', '.gz', '.mp4', '.mp3', '.avi',
                '.mov', '.wmv', '.flv', '.css', '.js', '.xml', '.json'
            ]
            
            url_lower = url.lower()
            if any(url_lower.endswith(ext) for ext in skip_extensions):
                return False
            
            # Skip problematic paths
            skip_paths = [
                '/wp-admin/', '/admin/', '/login', '/logout', '/signin',
                '/signup', '/register', '/cart/', '/checkout/', '/account/',
                '/dashboard/', '/wp-content/', '/wp-includes/', '/api/'
            ]
            
            if any(path in url_lower for path in skip_paths):
                return False
            
            return True
        
        except Exception as e:
            logger.debug(f"URL validation error for {url}: {e}")
            return False
    
    def extract_content(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract content from page with robust error handling"""
        content_dict = {'url': url, 'title': '', 'content': ''}
        
        try:
            # Extract title
            title_tag = soup.find('title')
            if title_tag:
                content_dict['title'] = self.clean_text(title_tag.get_text(strip=True))
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={"name": "description"})
            if not meta_desc:
                meta_desc = soup.find('meta', attrs={"property": "og:description"})
            
            if meta_desc and meta_desc.get("content"):
                content_dict['content'] += self.clean_text(meta_desc["content"]) + "\n\n"
            
            # Extract contact info from full page
            full_text = soup.get_text(separator="\n", strip=True)
            self.extract_contact_info(full_text)
            
            # Remove unwanted elements
            for tag in soup(['script', 'style', 'iframe', 'nav', 'footer', 
                           'header', 'aside', 'noscript', 'svg']):
                tag.decompose()
            
            # Extract main content with priority selectors
            main_selectors = [
                "main", "article", "[role='main']", 
                ".content", "#content", ".main-content",
                ".post-content", ".entry-content"
            ]
            
            texts = []
            
            # Try main selectors first
            for selector in main_selectors:
                elements = soup.select(selector)
                for elem in elements:
                    text = elem.get_text(separator="\n", strip=True)
                    if len(text) > 100:
                        texts.append(text)
            
            # Extract from important tags
            for tag_name in ['h1', 'h2', 'h3', 'h4', 'p', 'li', 'td']:
                for tag in soup.find_all(tag_name):
                    text = tag.get_text(strip=True)
                    if len(text) > 20:
                        texts.append(text)
            
            # Fallback to body if needed
            if len(texts) < 5:
                body = soup.find('body')
                if body:
                    text = body.get_text(separator="\n", strip=True)
                    if text:
                        texts.append(text)
            
            # Process and deduplicate content
            if texts:
                combined = "\n".join(texts)
                lines = []
                
                for line in combined.split("\n"):
                    cleaned = self.clean_text(line)
                    if len(cleaned) > 20:  # Minimum line length
                        lines.append(cleaned)
                
                # Remove duplicates while preserving order
                seen = set()
                unique_lines = []
                for line in lines:
                    line_lower = line.lower()
                    if line_lower not in seen:
                        seen.add(line_lower)
                        unique_lines.append(line)
                
                # Limit total lines
                content_dict['content'] += "\n".join(unique_lines[:300])
        
        except Exception as e:
            logger.warning(f"Content extraction error for {url}: {str(e)[:100]}")
            self.debug_info.append(f"Extract error {url}: {str(e)[:50]}")
        
        return content_dict
    
    def scrape_website(self, base_url: str, max_pages: int = 40, 
                      progress_callback=None) -> Tuple[List[Document], Dict]:
        """Scrape website with comprehensive error handling and retry logic"""
        
        # Validate base URL
        if not validate_url(base_url):
            raise ValueError(f"Invalid URL format: {base_url}")
        
        visited = set()
        queue = deque()
        failed_urls = set()
        
        try:
            parsed_base = urlparse(base_url)
            base_domain = parsed_base.netloc
        except Exception as e:
            raise ValueError(f"Invalid URL: {str(e)}")
        
        # Normalize base URL
        base_url = base_url.rstrip('/') + '/'
        
        # Add priority pages first
        for page in PRIORITY_PAGES:
            for url_variant in [urljoin(base_url, page), urljoin(base_url, page + '/')]:
                normalized = url_variant.rstrip('/')
                if normalized not in queue:
                    queue.append(normalized)
        
        documents = []
        retry_count = {}
        
        logger.info(f"Starting scrape of {base_url} (max {max_pages} pages)")
        
        while queue and len(visited) < max_pages:
            url = queue.popleft()
            
            # Normalize URL
            try:
                url = url.split("#")[0].split("?")[0].rstrip('/')
            except Exception:
                continue
            
            # Skip if already visited or invalid
            if url in visited or url in failed_urls:
                continue
            
            if not self.is_valid_url(url, base_domain):
                continue
            
            # Retry logic
            retry_attempts = retry_count.get(url, 0)
            if retry_attempts >= MAX_RETRIES:
                failed_urls.add(url)
                continue
            
            try:
                # Make request with timeout
                response = self.session.get(
                    url,
                    timeout=REQUEST_TIMEOUT,
                    allow_redirects=True,
                    verify=True
                )
                
                # Check status
                if response.status_code != 200:
                    logger.debug(f"Non-200 status for {url}: {response.status_code}")
                    retry_count[url] = retry_attempts + 1
                    if retry_attempts < MAX_RETRIES - 1:
                        queue.append(url)
                    continue
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' not in content_type:
                    logger.debug(f"Skipping non-HTML content: {url}")
                    continue
                
                # Mark as visited
                visited.add(url)
                
                # Update progress
                if progress_callback:
                    try:
                        progress_callback(len(visited), max_pages, url)
                    except Exception as e:
                        logger.warning(f"Progress callback error: {e}")
                
                # Parse content
                soup = BeautifulSoup(response.text, "html.parser")
                content_data = self.extract_content(soup, url)
                
                # Only add documents with substantial content
                if len(content_data['content']) > 100:
                    doc = Document(
                        page_content=content_data['content'],
                        metadata={
                            'source': url,
                            'title': content_data['title'],
                            'company': self.company_slug,
                            'scraped_at': datetime.now(timezone.utc).isoformat()
                        }
                    )
                    documents.append(doc)
                    logger.debug(f"Added document from {url} ({len(content_data['content'])} chars)")
                
                # Find and queue new links
                for link_tag in soup.find_all("a", href=True):
                    try:
                        href = link_tag['href']
                        next_url = urljoin(url, href)
                        next_url = next_url.split("#")[0].split("?")[0].rstrip('/')
                        
                        if (next_url not in visited and 
                            next_url not in failed_urls and
                            self.is_valid_url(next_url, base_domain) and
                            next_url not in queue):
                            queue.append(next_url)
                    except Exception:
                        continue
                
                # Polite delay
                time.sleep(DELAY_BETWEEN_REQUESTS)
            
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout for {url}")
                retry_count[url] = retry_attempts + 1
                if retry_attempts < MAX_RETRIES - 1:
                    queue.append(url)
                self.debug_info.append(f"Timeout: {url}")
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error for {url}: {str(e)[:100]}")
                retry_count[url] = retry_attempts + 1
                if retry_attempts < MAX_RETRIES - 1:
                    queue.append(url)
                self.debug_info.append(f"Request error {url}: {str(e)[:50]}")
            
            except Exception as e:
                logger.error(f"Unexpected error for {url}: {str(e)[:100]}")
                self.debug_info.append(f"Error {url}: {str(e)[:50]}")
                failed_urls.add(url)
        
        # Validate results
        logger.info(f"Scraping complete: {len(documents)} documents from {len(visited)} pages")
        
        if len(documents) < 3:
            raise Exception(
                f"Insufficient content scraped: only {len(documents)} pages with content. "
                f"Visited {len(visited)} pages total. "
                f"Please check if the website is accessible and has enough content."
            )
        
        # Return results
        return documents, {
            'emails': list(self.company_info['emails'])[:10],
            'phones': list(self.company_info['phones'])[:10],
            'address_india': self.company_info['address_india'],
            'address_international': self.company_info['address_international'],
            'pages_scraped': len(visited),
            'pages_failed': len(failed_urls),
            'documents_created': len(documents)
        }


# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def create_company(company_name: str, website_url: str, max_pages: int = 40) -> Optional[str]:
    """Create new company with validation"""
    db = SessionLocal()
    try:
        # Validate inputs
        if not company_name or not company_name.strip():
            raise ValueError("Company name cannot be empty")
        
        if not validate_url(website_url):
            raise ValueError("Invalid website URL")
        
        if max_pages < 10 or max_pages > 100:
            raise ValueError("Max pages must be between 10 and 100")
        
        slug = create_slug(company_name)
        
        # Check if exists
        existing = db.query(Company).filter(Company.company_slug == slug).first()
        if existing:
            logger.warning(f"Company already exists with slug: {slug}")
            return None
        
        # Create company
        company = Company(
            company_name=company_name.strip(),
            company_slug=slug,
            website_url=website_url.strip(),
            max_pages_to_scrape=max_pages,
            scraping_status="pending"
        )
        
        db.add(company)
        db.commit()
        db.refresh(company)
        
        logger.info(f"Created company: {company_name} (slug: {slug})")
        return slug
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        db.rollback()
        raise
    
    except SQLAlchemyError as e:
        logger.error(f"Database error creating company: {e}")
        db.rollback()
        return None
    
    except Exception as e:
        logger.error(f"Unexpected error creating company: {e}")
        db.rollback()
        return None
    
    finally:
        db.close()


def update_company_after_scraping(slug: str, company_info: Dict, status: str = "completed"):
    """Update company after scraping with error handling"""
    db = SessionLocal()
    try:
        company = db.query(Company).filter(Company.company_slug == slug).first()
        
        if not company:
            logger.error(f"Company not found for slug: {slug}")
            return False
        
        # Update fields
        company.emails = json.dumps(company_info.get('emails', []))
        company.phones = json.dumps(company_info.get('phones', []))
        company.address_india = company_info.get('address_india')
        company.address_international = company_info.get('address_international')
        company.pages_scraped = company_info.get('pages_scraped', 0)
        company.scraping_status = status
        company.last_scraped = datetime.now(timezone.utc)
        company.updated_at = datetime.now(timezone.utc)
        
        if status == "failed":
            company.error_message = company_info.get('error_message', 'Unknown error')
        
        db.commit()
        logger.info(f"Updated company {slug} with status: {status}")
        return True
    
    except SQLAlchemyError as e:
        logger.error(f"Database error updating company: {e}")
        db.rollback()
        return False
    
    except Exception as e:
        logger.error(f"Unexpected error updating company: {e}")
        db.rollback()
        return False
    
    finally:
        db.close()


def get_all_companies() -> List[Company]:
    """Get all companies with error handling"""
    db = SessionLocal()
    try:
        companies = db.query(Company).order_by(Company.created_at.desc()).all()
        return companies
    
    except SQLAlchemyError as e:
        logger.error(f"Database error getting companies: {e}")
        return []
    
    finally:
        db.close()


def get_company_by_slug(slug: str) -> Optional[Company]:
    """Get company by slug with validation"""
    if not slug:
        return None
    
    db = SessionLocal()
    try:
        company = db.query(Company).filter(Company.company_slug == slug).first()
        return company
    
    except SQLAlchemyError as e:
        logger.error(f"Database error getting company: {e}")
        return None
    
    finally:
        db.close()


# Initialize on import
logger.info("Backend initialized successfully")

"""
AI Chatbot Generator - Complete Fixed Frontend & AI Engine (app.py)
Thoroughly analyzed and optimized Streamlit application
"""

import streamlit as st
import time
import requests
import json
import os
import shutil
import hashlib
import logging
from typing import List, Optional, Dict
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Import from main.py with error handling
try:
    from main import (
        Company, ChatHistory, UserContact,
        WebScraper, create_company, update_company_after_scraping,
        get_all_companies, get_company_by_slug, get_chroma_directory,
        SessionLocal, OPENROUTER_API_KEY, OPENROUTER_API_BASE, MODEL,
        validate_url
    )
    logger.info("Successfully imported from main.py")
except ImportError as e:
    st.error(f"❌ Failed to import from main.py: {e}")
    st.stop()


# ============================================================================
# AI ENGINE WITH RAG
# ============================================================================

class CompanyAI:
    """AI Engine with RAG for answering questions - Production Ready"""
    
    def __init__(self, company_slug: str):
        if not company_slug:
            raise ValueError("Company slug cannot be empty")
        
        self.company_slug = company_slug
        self.vectorstore = None
        self.retriever = None
        self.embeddings = None
        self.status = {"ready": False, "error": None, "loading": False}
        self.company_data = None
        
        # QA Template with improved prompting
        self.qa_template = """You are an intelligent AI assistant for {company_name}. Answer questions accurately and helpfully using the provided context.

CONTEXT FROM COMPANY WEBSITE:
{context}

RECENT CONVERSATION:
{chat_history}

USER QUESTION: {question}

INSTRUCTIONS:
1. Answer using ONLY information from the context above
2. Be specific, detailed, and helpful
3. If the information is not in the context, politely say you don't have that information and offer to help with contact details
4. Be conversational and friendly
5. Keep your answer concise (2-5 sentences)
6. Do not make up information

ANSWER:"""

        self.qa_prompt = PromptTemplate(
            template=self.qa_template,
            input_variables=["company_name", "context", "chat_history", "question"]
        )
    
    def initialize(self, website_url: str, max_pages: int = 40, progress_callback=None):
        """Initialize by scraping and creating vector store with comprehensive error handling"""
        self.status["loading"] = True
        
        try:
            logger.info(f"Initializing AI for {self.company_slug}")
            
            # Validate URL
            if not validate_url(website_url):
                raise ValueError(f"Invalid website URL: {website_url}")
            
            if progress_callback:
                progress_callback(0, max_pages, "🚀 Starting web scraper...")
            
            # Scrape website
            scraper = WebScraper(self.company_slug)
            documents, company_info = scraper.scrape_website(
                website_url, 
                max_pages, 
                progress_callback
            )
            
            logger.info(f"Scraped {len(documents)} documents")
            
            if len(documents) < 3:
                self.status["error"] = f"Insufficient content: only {len(documents)} pages scraped"
                return False
            
            self.company_data = company_info
            
            if progress_callback:
                progress_callback(max_pages, max_pages, "📄 Processing documents...")
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
                length_function=len,
                is_separator_regex=False
            )
            
            split_docs = text_splitter.split_documents(documents)
            logger.info(f"Split into {len(split_docs)} chunks")
            
            if len(split_docs) < 5:
                self.status["error"] = f"Insufficient content chunks: only {len(split_docs)} created"
                return False
            
            if progress_callback:
                progress_callback(max_pages, max_pages, "🧠 Creating embeddings...")
            
            # Create embeddings with error handling
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info("Embeddings model loaded")
            except Exception as e:
                self.status["error"] = f"Failed to load embeddings model: {str(e)}"
                return False
            
            chroma_dir = get_chroma_directory(self.company_slug)
            
            # Clean existing directory
            if os.path.exists(chroma_dir):
                try:
                    shutil.rmtree(chroma_dir)
                    logger.info(f"Cleaned existing ChromaDB directory: {chroma_dir}")
                except Exception as e:
                    logger.warning(f"Could not clean directory: {e}")
            
            os.makedirs(chroma_dir, exist_ok=True)
            
            if progress_callback:
                progress_callback(max_pages, max_pages, "💾 Building vector database...")
            
            # Create vectorstore with error handling
            try:
                self.vectorstore = Chroma.from_documents(
                    documents=split_docs,
                    embedding=self.embeddings,
                    persist_directory=chroma_dir,
                    collection_name="company_knowledge"
                )
                logger.info("Vector store created")
            except Exception as e:
                self.status["error"] = f"Failed to create vector store: {str(e)}"
                return False
            
            # Verify vectorstore
            try:
                doc_count = self.vectorstore._collection.count()
                logger.info(f"Vector store contains {doc_count} documents")
                
                if doc_count == 0:
                    raise Exception("Vector store is empty after creation")
            except Exception as e:
                self.status["error"] = f"Vector store verification failed: {str(e)}"
                return False
            
            # Create retriever with optimal settings
            try:
                self.retriever = self.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 6,
                        "fetch_k": 20,
                        "lambda_mult": 0.7
                    }
                )
                logger.info("Retriever created")
            except Exception as e:
                self.status["error"] = f"Failed to create retriever: {str(e)}"
                return False
            
            self.status["ready"] = True
            self.status["loading"] = False
            
            if progress_callback:
                progress_callback(max_pages, max_pages, "✅ Chatbot Ready!")
            
            logger.info(f"AI initialization complete for {self.company_slug}")
            return True
        
        except Exception as e:
            self.status["error"] = f"Initialization failed: {str(e)}"
            self.status["loading"] = False
            logger.error(f"Initialization error: {e}", exc_info=True)
            return False
    
    def load_existing(self):
        """Load existing vector store with comprehensive error handling"""
        self.status["loading"] = True
        
        try:
            logger.info(f"Loading existing AI for {self.company_slug}")
            
            chroma_dir = get_chroma_directory(self.company_slug)
            
            if not os.path.exists(chroma_dir):
                self.status["error"] = "Chatbot data not found. Please create the chatbot first."
                self.status["loading"] = False
                return False
            
            # Create embeddings
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info("Embeddings model loaded")
            except Exception as e:
                self.status["error"] = f"Failed to load embeddings: {str(e)}"
                self.status["loading"] = False
                return False
            
            # Load vectorstore
            try:
                self.vectorstore = Chroma(
                    persist_directory=chroma_dir,
                    embedding_function=self.embeddings,
                    collection_name="company_knowledge"
                )
                logger.info("Vector store loaded")
            except Exception as e:
                self.status["error"] = f"Failed to load vector store: {str(e)}"
                self.status["loading"] = False
                return False
            
            # Verify vectorstore
            try:
                doc_count = self.vectorstore._collection.count()
                logger.info(f"Loaded vector store with {doc_count} documents")
                
                if doc_count == 0:
                    self.status["error"] = "Vector store is empty"
                    self.status["loading"] = False
                    return False
            except Exception as e:
                self.status["error"] = f"Vector store verification failed: {str(e)}"
                self.status["loading"] = False
                return False
            
            # Create retriever
            try:
                self.retriever = self.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 6,
                        "fetch_k": 20,
                        "lambda_mult": 0.7
                    }
                )
                logger.info("Retriever created")
            except Exception as e:
                self.status["error"] = f"Failed to create retriever: {str(e)}"
                self.status["loading"] = False
                return False
            
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
                    logger.info("Company data loaded")
                except Exception as e:
                    logger.warning(f"Error loading company data: {e}")
                    self.company_data = {}
            
            self.status["ready"] = True
            self.status["loading"] = False
            logger.info(f"AI loaded successfully for {self.company_slug}")
            return True
        
        except Exception as e:
            self.status["error"] = f"Load failed: {str(e)}"
            self.status["loading"] = False
            logger.error(f"Load error: {e}", exc_info=True)
            return False
    
    def get_contact_info(self):
        """Format contact information with validation"""
        if not self.company_data:
            return "📞 Contact information is not available yet."
        
        info = self.company_data
        msg_parts = ["📞 **CONTACT INFORMATION**\n"]
        
        if info.get('address_india'):
            msg_parts.append(f"🇮🇳 **India Office:**\n{info['address_india']}\n")
        
        if info.get('address_international'):
            msg_parts.append(f"🌍 **International Office:**\n{info['address_international']}\n")
        
        emails = info.get('emails', [])
        if emails:
            msg_parts.append("📧 **Email:**\n" + "\n".join([f"• {e}" for e in emails[:5]]) + "\n")
        
        phones = info.get('phones', [])
        if phones:
            msg_parts.append("📱 **Phone:**\n" + "\n".join([f"• {p}" for p in phones[:5]]))
        
        return "\n".join(msg_parts).strip()
    
    def ask(self, question: str, chat_history: List = None, session_id: str = None) -> str:
        """Answer question using RAG with comprehensive error handling"""
        try:
            # Validate inputs
            if not question or not question.strip():
                return "⚠️ Please ask a question."
            
            question = question.strip()
            
            # Check system status
            if self.status.get("loading", False):
                return "⏳ System is still loading. Please wait a moment..."
            
            if not self.status.get("ready", False):
                return "⚠️ System is not ready. Please try reloading the page."
            
            if not self.retriever:
                return "⚠️ Chatbot not properly initialized. Please try reloading."
            
            q_lower = question.lower()
            
            # Handle greetings
            greetings = ["hi", "hello", "hey", "hai", "hola", "good morning", "good afternoon"]
            if q_lower in greetings or (len(q_lower) < 10 and any(g in q_lower for g in ["hi", "hello", "hey"])):
                company = get_company_by_slug(self.company_slug)
                company_name = company.company_name if company else "our company"
                return f"👋 Hello! I'm here to answer questions about **{company_name}**. How can I help you today?"
            
            # Handle contact requests
            contact_keywords = ["email", "contact", "phone", "address", "office", "location", "reach", "call", "write"]
            if any(keyword in q_lower for keyword in contact_keywords):
                return self.get_contact_info()
            
            logger.info(f"Processing question: {question[:50]}...")
            
            # Retrieve relevant documents
            try:
                # Use invoke() for newer LangChain versions, fallback to old method
                if hasattr(self.retriever, 'invoke'):
                    relevant_docs = self.retriever.invoke(question)
                elif hasattr(self.retriever, 'get_relevant_documents'):
                    relevant_docs = self.retriever.get_relevant_documents(question)
                else:
                    return "⚠️ Retriever method not available. Please reload."
            except Exception as e:
                logger.error(f"Retrieval error: {e}")
                return "⚠️ Error retrieving information. Please try again."
            
            logger.info(f"Found {len(relevant_docs)} relevant documents")
            
            if not relevant_docs or len(relevant_docs) == 0:
                return "I couldn't find specific information about that in our knowledge base. Could you try rephrasing your question or ask about something else?"
            
            # Build context from documents
            context_parts = []
            for i, doc in enumerate(relevant_docs[:5]):
                if hasattr(doc, 'page_content') and doc.page_content:
                    source = doc.metadata.get('source', 'unknown')
                    title = doc.metadata.get('title', '')
                    
                    # Truncate content
                    content = doc.page_content[:700]
                    context_part = f"[Source: {title if title else source}]\n{content}"
                    context_parts.append(context_part)
                    logger.debug(f"Doc {i+1}: {len(doc.page_content)} chars")
            
            if not context_parts:
                return "I found documents but couldn't extract relevant content. Please try rephrasing your question."
            
            context = "\n\n---\n\n".join(context_parts)
            logger.info(f"Context built: {len(context)} chars")
            
            # Format chat history
            history_text = ""
            if chat_history and len(chat_history) > 0:
                recent_history = chat_history[-3:]  # Last 3 exchanges
                history_items = []
                for msg in recent_history:
                    q = msg.get('question', '')
                    a = msg.get('answer', '')
                    if q and a:
                        history_items.append(f"User: {q[:100]}\nAssistant: {a[:200]}")
                history_text = "\n".join(history_items)
            
            # Get company name
            company = get_company_by_slug(self.company_slug)
            company_name = company.company_name if company else "the company"
            
            # Build prompt
            prompt = self.qa_prompt.format(
                company_name=company_name,
                context=context[:5000],  # Limit context
                chat_history=history_text[:800],  # Limit history
                question=question
            )
            
            logger.info(f"Calling LLM (prompt: {len(prompt)} chars)")
            
            # Call LLM
            answer = self._call_llm(prompt)
            
            if not answer:
                return "⚠️ I'm having trouble generating a response. Please try again."
            
            logger.info(f"Answer generated: {answer[:50]}...")
            
            # Save successful conversations
            if answer and not answer.startswith("⚠️") and not answer.startswith("❌"):
                try:
                    self.save_chat(question, answer, session_id)
                except Exception as e:
                    logger.warning(f"Failed to save chat: {e}")
                
                return answer
            else:
                return answer
        
        except Exception as e:
            logger.error(f"Error in ask(): {e}", exc_info=True)
            return f"⚠️ An unexpected error occurred. Please try again or reload the page."
    
    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Call LLM API with retry logic and comprehensive error handling"""
        
        # Validate API key
        if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "":
            logger.error("OPENROUTER_API_KEY not set")
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
                    "content": "You are a helpful AI assistant. Answer accurately and concisely based on the provided context. Do not make up information."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.5,
            "max_tokens": 600,
            "top_p": 0.9
        }
        
        for attempt in range(max_retries):
            try:
                logger.info(f"LLM API call attempt {attempt + 1}/{max_retries}")
                
                response = requests.post(
                    OPENROUTER_API_BASE,
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                logger.info(f"Response status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if "choices" in result and len(result["choices"]) > 0:
                        answer = result["choices"][0]["message"]["content"].strip()
                        
                        if answer and len(answer) > 10:
                            logger.info("LLM response received successfully")
                            return answer
                        else:
                            logger.warning(f"Answer too short: {answer}")
                    else:
                        logger.warning("No choices in response")
                
                elif response.status_code == 429:
                    logger.warning("Rate limited")
                    if attempt < max_retries - 1:
                        time.sleep(3 * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        return "⚠️ API rate limit reached. Please wait a moment and try again."
                
                elif response.status_code == 401:
                    logger.error("Authentication failed")
                    return "⚠️ Invalid API key. Please check your OPENROUTER_API_KEY."
                
                elif response.status_code == 400:
                    error_text = response.text[:300]
                    logger.error(f"Bad request: {error_text}")
                    return "⚠️ Invalid request. Please try rephrasing your question."
                
                elif response.status_code >= 500:
                    logger.error(f"Server error: {response.status_code}")
                    if attempt < max_retries - 1:
                        time.sleep(2 * (attempt + 1))
                        continue
                    else:
                        return "⚠️ API server error. Please try again later."
                
                else:
                    error_text = response.text[:300]
                    logger.error(f"Unexpected status {response.status_code}: {error_text}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
            
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                else:
                    return "⚠️ Request timed out. Please try again."
            
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(3 * (attempt + 1))
                    continue
                else:
                    return "⚠️ Connection error. Please check your internet connection."
            
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    return f"⚠️ Request error: {str(e)[:100]}"
            
            except Exception as e:
                logger.error(f"Unexpected LLM error: {e}", exc_info=True)
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
        
        return "❌ Failed to get response from AI after multiple attempts. Please try again."
    
    def save_chat(self, question: str, answer: str, session_id: str = None):
        """Save chat to database with error handling"""
        db = SessionLocal()
        try:
            chat = ChatHistory(
                company_slug=self.company_slug,
                question=question[:1000],  # Limit length
                answer=answer[:2000],  # Limit length
                session_id=session_id
            )
            db.add(chat)
            db.commit()
            logger.debug("Chat saved to database")
        except Exception as e:
            logger.error(f"Error saving chat: {e}")
            db.rollback()
        finally:
            db.close()


# ============================================================================
# STREAMLIT UI
# ============================================================================

# Page configuration
st.set_page_config(
    page_title="AI Chatbot Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with modern design
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .header-title {
        color: white;
        font-size: 2.8rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .company-card {
        background: white;
        padding: 1.8rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
    }
    .company-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.2);
    }
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with validation
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
    st.session_state.session_id = hashlib.md5(
        (str(time.time()) + str(os.urandom(8))).encode()
    ).hexdigest()[:16]

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
    st.caption("🤖 **Powered by**")
    st.caption("• AI & RAG Technology")
    st.caption("• LangChain & ChromaDB")
    st.caption(f"• Session: `{st.session_state.session_id}`")
    
    # API Key status
    st.markdown("---")
    if OPENROUTER_API_KEY:
        st.success("✅ API Key Configured")
    else:
        st.error("❌ API Key Not Set")
        st.caption("Set OPENROUTER_API_KEY environment variable")


# ============================================================================
# HOME PAGE
# ============================================================================

if st.session_state.page == "home":
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🤖 AI Chatbot Generator</h1>
        <p style="color: white; font-size: 1.2rem; margin-top: 1rem; font-weight: 500;">
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
    
    st.info("ℹ️ **How it works:** Enter a company name and website URL. The AI will scrape the website, extract information, and create an intelligent chatbot that can answer questions!")
    
    st.markdown("### ✨ Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🌐 Intelligent Scraping**
        
        Automatically extracts content from websites with smart filtering
        """)
    
    with col2:
        st.markdown("""
        **🧠 AI-Powered RAG**
        
        Uses advanced Retrieval Augmented Generation for accurate answers
        """)
    
    with col3:
        st.markdown("""
        **💬 Natural Conversations**
        
        Chat naturally with context-aware responses
        """)
    
    st.markdown("### 🚀 Quick Start")
    with st.expander("📖 Getting Started Guide"):
        st.markdown("""
        1. **Create a Chatbot**: Click "Create New Chatbot" and enter company details
        2. **Wait for Processing**: The system will scrape and process the website (2-5 minutes)
        3. **Start Chatting**: Once ready, click "Chat" to interact with your chatbot
        4. **Ask Questions**: Ask anything about the company and get instant answers!
        """)


# ============================================================================
# CREATE PAGE
# ============================================================================

elif st.session_state.page == "create":
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🆕 Create New Chatbot</h1>
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
            min_value=10,
            max_value=60,
            value=40,
            step=10,
            key="form_pages"
        )
        
        st.caption("⏱ This process may take 2–5 minutes depending on website size and number of pages.")
        
        submitted = st.form_submit_button(
            "🚀 Create Chatbot",
            type="primary",
            use_container_width=True
        )
        
        if submitted:
            if not name or not url:
                st.warning("⚠️ Please fill in all required fields (Company Name and Website URL).")
            else:
                # Normalize URL
                if not url.startswith("http://") and not url.startswith("https://"):
                    url = "https://" + url.strip()
                
                # Validate URL using shared validator
                if not validate_url(url):
                    st.error("❌ The provided URL is not valid. Please check and try again.")
                else:
                    with st.spinner("🧾 Creating company record..."):
                        slug = create_company(name.strip(), url.strip(), pages)
                    
                    if not slug:
                        st.error("⚠️ A company with this name (or slug) already exists. Please use a different name.")
                    else:
                        st.success(f"✅ Company created: **{name}**")
                        
                        prog_bar = st.progress(0, text="Initializing scraper...")
                        status_text = st.empty()
                        
                        def progress_cb(current, total, url_text):
                            progress = min(current / max(total, 1), 1.0)
                            prog_bar.progress(
                                progress,
                                text=f"🔍 Scraped {current}/{total} pages"
                            )
                            status_text.caption(f"Current page: {url_text[:80]}")
                        
                        ai = CompanyAI(slug)
                        success = ai.initialize(url, pages, progress_cb)
                        
                        if success:
                            update_company_after_scraping(slug, ai.company_data, "completed")
                            prog_bar.progress(1.0, text="🤖 Chatbot is ready!")
                            st.success("🎉 Chatbot created successfully! Redirecting to chat...")
                            st.balloons()
                            time.sleep(2)
                            
                            st.session_state.current_company = slug
                            st.session_state.ai_instance = ai
                            st.session_state.messages = []
                            st.session_state.chat_history = []
                            st.session_state.page = "chat"
                            st.rerun()
                        else:
                            error_msg = ai.status.get("error", "Unknown error")
                            st.error(f"❌ Failed to create chatbot: {error_msg}")
                            update_company_after_scraping(slug, {}, "failed")


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
        st.info("ℹ️ No chatbots created yet. Click below to create your first one!")
        if st.button("➕ Create First Chatbot", type="primary", use_container_width=True):
            st.session_state.page = "create"
            st.rerun()
    else:
        for c in companies:
            st.markdown('<div class="company-card">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"### {c.company_name}")
                st.caption(f"🌐 {c.website_url}")
                st.caption(f"📄 {c.pages_scraped or 0} pages scraped")
                if c.last_scraped:
                    dt_str = c.last_scraped.strftime("%Y-%m-%d %H:%M")
                    st.caption(f"🕒 Last scraped: {dt_str} UTC")
            
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
                else:
                    st.button("⛔ Unavailable", key=f"chat_disabled_{c.id}", use_container_width=True, disabled=True)
            
            st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# CHAT PAGE
# ============================================================================

elif st.session_state.page == "chat":
    # Ensure a company is selected
    if not st.session_state.current_company:
        st.error("⚠️ No company selected. Please choose a chatbot from the list.")
        if st.button("← Go Back to List", use_container_width=True):
            st.session_state.page = "list"
            st.rerun()
        st.stop()
    
    c = get_company_by_slug(st.session_state.current_company)
    
    if not c:
        st.error("❌ Company not found. It might have been removed.")
        if st.button("← Go Back to List", use_container_width=True):
            st.session_state.page = "list"
            st.session_state.current_company = None
            st.rerun()
        st.stop()
    
    # Load AI instance if not already loaded
    if not st.session_state.ai_instance:
        with st.spinner("⚙️ Loading chatbot..."):
            try:
                ai = CompanyAI(st.session_state.current_company)
                if not ai.load_existing():
                    st.error(f"❌ Failed to load chatbot: {ai.status.get('error', 'Unknown error')}")
                    if st.button("← Go Back to List", use_container_width=True):
                        st.session_state.page = "list"
                        st.session_state.current_company = None
                        st.rerun()
                    st.stop()
                st.session_state.ai_instance = ai
                st.success("✅ Chatbot loaded successfully!")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading chatbot: {str(e)}")
                if st.button("← Go Back to List", use_container_width=True):
                    st.session_state.page = "list"
                    st.session_state.current_company = None
                    st.rerun()
                st.stop()
    
    # Header
    st.markdown(f"""
    <div class="header-container">
        <h1 class="header-title">💬 Chat with {c.company_name}</h1>
        <p style="color: white; margin-top: 0.5rem; font-weight: 500;">
            Ask me anything about this company!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Top controls
    top_col1, top_col2 = st.columns([1, 1])
    with top_col1:
        if st.button("← Back to List", use_container_width=True):
            st.session_state.page = "list"
            st.session_state.current_company = None
            st.session_state.ai_instance = None
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
    with top_col2:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
    
    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    prompt = st.chat_input(f"Ask a question about {c.company_name}...")
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    if not st.session_state.ai_instance:
                        answer = "⚠️ Chatbot is not available. Please reload the page."
                    else:
                        answer = st.session_state.ai_instance.ask(
                            prompt,
                            st.session_state.chat_history,
                            st.session_state.session_id
                        )
                except Exception as e:
                    logger.error(f"Error in chat: {e}", exc_info=True)
                    answer = "⚠️ An unexpected error occurred while generating a response. Please try again."
            
            st.markdown(answer)
        
        # Save messages & history
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_history.append({
            "question": prompt,
            "answer": answer,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        st.rerun()
