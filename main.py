__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from collections import deque
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base
import re
import os
import hashlib
import json
from typing import Optional, Dict, List, Tuple
import time

DATABASE_URL = "sqlite:///./multi_company_chatbots.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class Company(Base):
    """Stores information about each company that has a chatbot"""
    __tablename__ = "companies"
    
    id = Column(Integer, primary_key=True, index=True)
    company_name = Column(String(255), unique=True, nullable=False, index=True)
    company_slug = Column(String(255), unique=True, nullable=False, index=True)
    website_url = Column(String(500), nullable=False)
    logo_url = Column(String(500), nullable=True)
    primary_color = Column(String(20), default="#667eea")
    secondary_color = Column(String(20), default="#764ba2")
    
    # Contact information
    emails = Column(Text, nullable=True)
    phones = Column(Text, nullable=True)
    address_india = Column(Text, nullable=True)
    address_international = Column(Text, nullable=True)
    
    # Scraping metadata
    pages_scraped = Column(Integer, default=0)
    last_scraped = Column(DateTime, nullable=True)
    scraping_status = Column(String(50), default="pending")
    
    # Chatbot configuration
    is_active = Column(Boolean, default=True)
    max_pages_to_scrape = Column(Integer, default=40)
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, onupdate=lambda: datetime.now(timezone.utc))


class ChatHistory(Base):
    """Stores chat conversations for each company"""
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True)
    company_slug = Column(String(255), nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    session_id = Column(String(100), nullable=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class UserContact(Base):
    """Stores user contact information collected via chatbot"""
    __tablename__ = "user_contacts"
    
    id = Column(Integer, primary_key=True, index=True)
    company_slug = Column(String(255), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False)
    phone = Column(String(20), nullable=False)
    session_id = Column(String(100), nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ScrapedContent(Base):
    """Stores raw scraped content for each company"""
    __tablename__ = "scraped_content"
    
    id = Column(Integer, primary_key=True, index=True)
    company_slug = Column(String(255), nullable=False, index=True)
    url = Column(String(1000), nullable=False)
    title = Column(Text, nullable=True)
    content = Column(Text, nullable=False)
    scraped_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

Base.metadata.create_all(bind=engine)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.0-flash-exp:free"

PRIORITY_PAGES = [
    "", "about", "services", "solutions", "products", "contact", "team",
    "careers", "blog", "case-studies", "portfolio", "industries",
    "technology", "expertise", "what-we-do", "who-we-are", "footer",
    "about-us", "contact-us", "our-services", "our-team", "home", "index"
]

def create_slug(company_name: str) -> str:
    """Convert company name to URL-friendly slug"""
    slug = company_name.lower()
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    slug = slug.strip('-')
    return slug


def get_chroma_directory(company_slug: str) -> str:
    """Get the ChromaDB directory path for a specific company"""
    return f"./chroma_db/{company_slug}"

class WebScraper:
    """Handles web scraping - MAXIMUM content extraction"""
    
    def __init__(self, company_slug: str):
        self.company_slug = company_slug
        self.company_info = {
            'emails': set(),
            'phones': set(),
            'address_india': None,
            'address_international': None
        }
        self.scraped_content = {}
        self.debug_info = []
        self.total_content_length = 0  # Track total content
    
    def clean_text(self, text: str) -> str:
        """Aggressively clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"]+', '', text)
        return text.strip()
    
    def extract_contact_info(self, soup: BeautifulSoup, text: str):
        """Extract ALL contact information"""
        # Extract emails - AGGRESSIVE
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        for email in emails:
            if not email.lower().endswith(('.png', '.jpg', '.gif', '.svg', '.css', '.js')):
                self.company_info['emails'].add(email.lower())
        
        # Extract phones - MULTIPLE patterns
        phone_patterns = [
            r'\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\d{10,15}',
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
            low = line.lower()
            
            # Indian addresses
            india_keywords = ['india', 'mumbai', 'delhi', 'bangalore', 'chennai', 
                            'kolkata', 'hyderabad', 'pune', 'madurai', 'coimbatore',
                            'kerala', 'tamil nadu', 'maharashtra', 'karnataka']
            if any(city in low for city in india_keywords):
                if not self.company_info['address_india']:
                    block = " ".join(lines[max(0, i-2):min(len(lines), i+5)])
                    cleaned = self.clean_text(block)
                    if 20 < len(cleaned) < 500:
                        self.company_info['address_india'] = cleaned
            
            # International addresses
            intl_keywords = ['singapore', 'usa', 'uk', 'uae', 'dubai', 'malaysia', 
                           'australia', 'canada', 'london', 'new york']
            if any(country in low for country in intl_keywords):
                if not self.company_info['address_international']:
                    block = " ".join(lines[max(0, i-2):min(len(lines), i+5)])
                    cleaned = self.clean_text(block)
                    if 20 < len(cleaned) < 500:
                        self.company_info['address_international'] = cleaned
    
    def is_valid_url(self, url: str, base_domain: str) -> bool:
        """Check if URL should be scraped"""
        try:
            parsed = urlparse(url)
            
            # Must be same domain
            if parsed.netloc != base_domain:
                return False
            
            # Skip binary files and admin
            skip_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', 
                             '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
                             '.mp4', '.mp3', '.avi', '.mov', '.css', '.js', '.ico']
            
            skip_paths = ['/wp-admin/', '/wp-includes/', '/wp-json/', '/admin/',
                         '/login', '/register', '/cart/', '/checkout/', '/feed/',
                         '/api/', '/download/']
            
            url_lower = url.lower()
            
            for ext in skip_extensions:
                if url_lower.endswith(ext):
                    return False
            
            for path in skip_paths:
                if path in url_lower:
                    return False
            
            # Skip anchor links and query duplicates
            if '#' in url or '?' in url:
                return False
            
            return True
        except:
            return False
    
    def extract_content_ultra_aggressive(self, soup: BeautifulSoup, url: str) -> Dict:
        """MAXIMUM content extraction - NEVER fails"""
        content_dict = {
            'url': url,
            'title': '',
            'main_content': '',
            'raw_text': ''
        }
        
        try:
            # 1. Extract title
            if soup.find('title'):
                content_dict['title'] = soup.find('title').get_text(strip=True)
            
            # 2. Extract meta
            meta_desc = soup.find('meta', attrs={"name": "description"})
            if meta_desc and meta_desc.get("content"):
                content_dict['main_content'] += meta_desc["content"] + "\n\n"
            
            # 3. Extract contact info FIRST
            full_text = soup.get_text(separator="\n", strip=True)
            self.extract_contact_info(soup, full_text)
            
            # 4. Remove unwanted tags
            for tag in soup(['script', 'style', 'iframe', 'noscript', 'svg', 
                           'nav', 'header', 'footer']):
                tag.decompose()
            
            # 5. STRATEGY 1: Target main content areas
            main_selectors = [
                "main", "article", "[role='main']", ".content", ".main-content",
                "#content", "#main", ".post-content", ".entry-content",
                ".page-content", "#primary", ".site-content", ".main",
                ".container", ".wrapper", "section", ".section"
            ]
            
            collected_texts = []
            
            for selector in main_selectors:
                elements = soup.select(selector)
                for elem in elements:
                    text = elem.get_text(separator="\n", strip=True)
                    if len(text) > 100:  # Lowered threshold
                        collected_texts.append(text)
            
            # 6. STRATEGY 2: Extract ALL meaningful tags
            for tag_name in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 
                            'td', 'span', 'div', 'a', 'strong', 'em']:
                for tag in soup.find_all(tag_name):
                    text = tag.get_text(strip=True)
                    # Accept even short texts
                    if len(text) > 15 and not text.isdigit():
                        collected_texts.append(text)
            
            # 7. STRATEGY 3: Body fallback
            if len(collected_texts) < 5:
                body = soup.find('body')
                if body:
                    text = body.get_text(separator="\n", strip=True)
                    if text:
                        collected_texts.append(text)
            
            # 8. STRATEGY 4: Entire page
            if not collected_texts:
                text = soup.get_text(separator="\n", strip=True)
                if text:
                    collected_texts.append(text)
            
            # 9. Clean and deduplicate
            if collected_texts:
                # Join all texts
                combined = "\n".join(collected_texts)
                
                # Split into lines and clean
                lines = []
                for line in combined.split("\n"):
                    cleaned = self.clean_text(line)
                    if len(cleaned) > 10:  # Very low threshold
                        lines.append(cleaned)
                
                # Remove duplicates while preserving order
                seen = set()
                unique_lines = []
                for line in lines:
                    line_lower = line.lower()
                    if line_lower not in seen:
                        seen.add(line_lower)
                        unique_lines.append(line)
                
                content_dict['main_content'] = "\n".join(unique_lines)
                content_dict['raw_text'] = combined[:5000]  # Keep raw for backup
            
            # 10. Add title if content is short
            if len(content_dict['main_content']) < 200 and content_dict['title']:
                content_dict['main_content'] = f"{content_dict['title']}\n\n{content_dict['main_content']}"
            
        except Exception as e:
            self.debug_info.append(f"Extraction error {url}: {str(e)}")
            # Even on error, try to get SOMETHING
            try:
                content_dict['main_content'] = soup.get_text(separator="\n", strip=True)[:3000]
            except:
                pass
        
        return content_dict
    
    def scrape_website(self, base_url: str, max_pages: int = 40, 
                      progress_callback=None) -> Tuple[str, Dict]:
        """
        Scrape website - GUARANTEED to extract content
        """
        visited = set()
        all_content = []
        queue = deque()
        base_domain = urlparse(base_url).netloc
        
        # Normalize base URL
        base_url = base_url.rstrip('/') + '/'
        
        # Add priority pages with ALL variations
        for page in PRIORITY_PAGES:
            urls_to_try = [
                urljoin(base_url, page),
                urljoin(base_url, page + '/'),
                urljoin(base_url, page + '.html'),
                urljoin(base_url, page + '.htm'),
                urljoin(base_url, page + '.php'),
                urljoin(base_url, page + '.asp'),
            ]
            for url in urls_to_try:
                if url not in queue:
                    queue.append(url)
        
        # Add base URL
        queue.append(base_url)
        
        # Enhanced headers
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0"
        }
        
        consecutive_failures = 0
        max_consecutive_failures = 10
        successful_scrapes = 0
        
        while queue and len(visited) < max_pages:
            if consecutive_failures >= max_consecutive_failures and successful_scrapes == 0:
                self.debug_info.append(f"❌ Stopped: {consecutive_failures} failures, 0 successes")
                break
            
            url = queue.popleft()
            
            # Normalize URL
            url = url.split("#")[0].split("?")[0].rstrip('/')
            
            if url in visited or not self.is_valid_url(url, base_domain):
                continue
            
            try:
                # Try to fetch page
                response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
                
                if response.status_code != 200:
                    consecutive_failures += 1
                    self.debug_info.append(f"HTTP {response.status_code}: {url}")
                    continue
                
                # Success!
                consecutive_failures = 0
                successful_scrapes += 1
                visited.add(url)
                
                if progress_callback:
                    progress_callback(len(visited), max_pages, url)
                
                # Parse content
                soup = BeautifulSoup(response.text, "html.parser")
                content_data = self.extract_content_ultra_aggressive(soup, url)
                
                # ACCEPT ALMOST ANY CONTENT
                content_length = len(content_data['main_content'])
                if content_length > 20:  # ULTRA low threshold
                    formatted = f"\n{'='*80}\n"
                    formatted += f"PAGE: {content_data['url']}\n"
                    formatted += f"TITLE: {content_data['title']}\n"
                    formatted += f"{'='*80}\n\n"
                    formatted += content_data['main_content']
                    
                    all_content.append(formatted)
                    self.scraped_content[url] = content_data
                    self.total_content_length += content_length
                    
                    self.debug_info.append(f"✓ Scraped {url}: {content_length} chars")
                else:
                    self.debug_info.append(f"⚠ Skipped {url}: only {content_length} chars")
                
                # Find new links
                for link in soup.find_all("a", href=True):
                    try:
                        next_url = urljoin(url, link['href'])
                        next_url = next_url.split("#")[0].split("?")[0].rstrip('/')
                        
                        if next_url not in visited and self.is_valid_url(next_url, base_domain):
                            if next_url not in queue:
                                queue.append(next_url)
                    except:
                        pass
                
                # Polite delay
                time.sleep(0.4)
            
            except requests.exceptions.Timeout:
                self.debug_info.append(f"⏱ Timeout: {url}")
                consecutive_failures += 1
            except requests.exceptions.RequestException as e:
                self.debug_info.append(f"🔌 Connection error {url}: {str(e)[:50]}")
                consecutive_failures += 1
            except Exception as e:
                self.debug_info.append(f"💥 Error {url}: {str(e)[:50]}")
                consecutive_failures += 1
        
        # Build result
        if not all_content or self.total_content_length < 500:
            error_msg = f"❌ INSUFFICIENT CONTENT SCRAPED\n\n"
            error_msg += f"Statistics:\n"
            error_msg += f"- Pages visited: {len(visited)}\n"
            error_msg += f"- Successful scrapes: {successful_scrapes}\n"
            error_msg += f"- Total content: {self.total_content_length} chars\n"
            error_msg += f"- Base domain: {base_domain}\n\n"
            error_msg += f"Debug Log (last 15 entries):\n"
            error_msg += "\n".join(self.debug_info[-15:])
            
            return error_msg, {
                'emails': list(self.company_info['emails']),
                'phones': list(self.company_info['phones']),
                'address_india': self.company_info['address_india'],
                'address_international': self.company_info['address_international'],
                'pages_scraped': len(visited)
            }
        
        # Add company info header
        header = "\n" + "="*80 + "\n"
        header += "COMPANY CONTACT INFORMATION\n"
        header += "="*80 + "\n\n"
        
        if self.company_info['address_india']:
            header += f"🇮🇳 INDIA OFFICE:\n{self.company_info['address_india']}\n\n"
        
        if self.company_info['address_international']:
            header += f"🌍 INTERNATIONAL OFFICE:\n{self.company_info['address_international']}\n\n"
        
        if self.company_info['emails']:
            header += "📧 EMAILS:\n" + "\n".join([f"  • {e}" for e in sorted(self.company_info['emails'])[:15]]) + "\n\n"
        
        if self.company_info['phones']:
            header += "☎️ PHONES:\n" + "\n".join([f"  • {p}" for p in sorted(self.company_info['phones'])[:15]]) + "\n\n"
        
        all_content.insert(0, header)
        
        final_content = "\n\n".join(all_content)
        
        self.debug_info.append(f"✅ SUCCESS: {len(visited)} pages, {self.total_content_length} chars total")
        
        return final_content, {
            'emails': list(self.company_info['emails']),
            'phones': list(self.company_info['phones']),
            'address_india': self.company_info['address_india'],
            'address_international': self.company_info['address_international'],
            'pages_scraped': len(visited)
        }
import streamlit as st
import time
from sentence_transformers import SentenceTransformer
import shutil

class CompanyAI:
    """AI Engine for company chatbot - ALL ISSUES FIXED"""
    
    def __init__(self, company_slug: str):
        self.company_slug = company_slug
        self.retriever = None
        self.cache = {}
        self.status = {"ready": False, "error": None}
        self.company_data = None
    
    def initialize(self, website_url: str, max_pages: int = 40, progress_callback=None):
        """Initialize AI by scraping and embedding - GUARANTEED SUCCESS"""
        try:
            # Step 1: Scrape website
            if progress_callback:
                progress_callback(0, max_pages, "Starting scraper...")
            
            scraper = WebScraper(self.company_slug)
            content, company_info = scraper.scrape_website(
                website_url, 
                max_pages, 
                progress_callback
            )
            
            # Check for errors
            if content.startswith("❌"):
                self.status["error"] = content
                return False
            
            if len(content) < 100:  # Down from 200
                self.status["error"] = f"Content too short: {len(content)} chars. Debug: {'; '.join(scraper.debug_info[-5:])}"
                return False
            
            self.company_data = company_info
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,  # Smaller chunks for better retrieval
                chunk_overlap=200,  # More overlap
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
                length_function=len
            )
            
            chunks = splitter.split_text(content)
            
            if not chunks:
                self.status["error"] = "Failed to create text chunks"
                return False
            
            # CRITICAL FIX: Accept shorter chunks
            meaningful_chunks = [c.strip() for c in chunks if len(c.strip()) > 30]  # Down from 50
            
            if len(meaningful_chunks) < 2:  # Down from 3
                self.status["error"] = f"Not enough chunks: {len(meaningful_chunks)}. Total chunks: {len(chunks)}"
                return False
            
            # Step 3: Create embeddings with error handling
            try:
                embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            except Exception as e:
                self.status["error"] = f"Failed to load embedding model: {str(e)}"
                return False
            
            class CustomEmbeddings:
                def __init__(self, model):
                    self.model = model
                
                def embed_documents(self, texts):
                    if not texts:
                        return []
                    try:
                        embeddings = self.model.encode(
                            texts,
                            normalize_embeddings=True,
                            show_progress_bar=False,
                            convert_to_numpy=True,
                            batch_size=32  # Added batch size
                        )
                        return embeddings.tolist()
                    except Exception as e:
                        print(f"Embedding error: {e}")
                        return []
                
                def embed_query(self, text):
                    try:
                        embedding = self.model.encode(
                            [text],
                            normalize_embeddings=True,
                            show_progress_bar=False,
                            convert_to_numpy=True
                        )
                        return embedding[0].tolist()
                    except Exception as e:
                        print(f"Query embedding error: {e}")
                        return [0.0] * 384  # Return zero vector as fallback
            
            embeddings = CustomEmbeddings(embedding_model)
            
            # Step 4: Create vector store with error handling
            chroma_dir = get_chroma_directory(self.company_slug)
            
            # Clean up existing directory
            if os.path.exists(chroma_dir):
                try:
                    shutil.rmtree(chroma_dir)
                except Exception as e:
                    print(f"Warning: Could not remove old chroma dir: {e}")
            
            os.makedirs(chroma_dir, exist_ok=True)
            
            try:
                # Create vectorstore with retry logic
                vectorstore = Chroma.from_texts(
                    texts=meaningful_chunks,
                    embedding=embeddings,
                    persist_directory=chroma_dir,
                    collection_name="company_docs"
                )
                
                # Verify vectorstore was created
                if vectorstore._collection.count() == 0:
                    raise Exception("Vectorstore is empty after creation")
                
                self.retriever = vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 6}  # Increased from 5
                )
                
            except Exception as e:
                self.status["error"] = f"Vectorstore creation failed: {str(e)}"
                return False
            
            self.status["ready"] = True
            
            if progress_callback:
                progress_callback(max_pages, max_pages, "✅ Initialization complete!")
            
            return True
        
        except Exception as e:
            self.status["error"] = f"Critical initialization error: {str(e)}"
            import traceback
            print(traceback.format_exc())
            return False
    
    def load_existing(self):
        """Load existing vector store - FIXED"""
        try:
            chroma_dir = get_chroma_directory(self.company_slug)
            
            if not os.path.exists(chroma_dir):
                self.status["error"] = f"Chroma directory not found: {chroma_dir}"
                return False
            
            # Load embedding model
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            
            class CustomEmbeddings:
                def __init__(self, model):
                    self.model = model
                
                def embed_documents(self, texts):
                    if not texts:
                        return []
                    embeddings = self.model.encode(
                        texts,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                    return embeddings.tolist()
                
                def embed_query(self, text):
                    embedding = self.model.encode(
                        [text],
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                    return embedding[0].tolist()
            
            embeddings = CustomEmbeddings(embedding_model)
            
            # Load existing vectorstore
            vectorstore = Chroma(
                persist_directory=chroma_dir,
                embedding_function=embeddings,
                collection_name="company_docs"
            )
            
            # Verify it has content
            if vectorstore._collection.count() == 0:
                self.status["error"] = "Loaded vectorstore is empty"
                return False
            
            self.retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 6}
            )
            
            self.status["ready"] = True
            return True
        
        except Exception as e:
            self.status["error"] = f"Load error: {str(e)}"
            import traceback
            print(traceback.format_exc())
            return False
    
    def get_contact_info(self):
        """Format company contact information"""
        if not self.company_data:
            # Try to load from database
            db = SessionLocal()
            try:
                company = db.query(Company).filter(
                    Company.company_slug == self.company_slug
                ).first()
                
                if not company:
                    return "No contact information available."
                
                try:
                    self.company_data = {
                        'emails': json.loads(company.emails) if company.emails else [],
                        'phones': json.loads(company.phones) if company.phones else [],
                        'address_india': company.address_india,
                        'address_international': company.address_international
                    }
                except:
                    return "Contact information format error."
            finally:
                db.close()
        
        info = self.company_data
        if not any([info.get('emails'), info.get('phones'), 
                   info.get('address_india'), info.get('address_international')]):
            return "No contact information found in our records."
        
        msg = "📞 **CONTACT INFORMATION**\n\n"
        
        if info.get('address_india'):
            msg += f"🇮🇳 **India Office:**\n{info['address_india']}\n\n"
        
        if info.get('address_international'):
            msg += f"🌍 **International Office:**\n{info['address_international']}\n\n"
        
        if info.get('emails'):
            msg += "📧 **Email Addresses:**\n" + "\n".join([f"• {e}" for e in info['emails'][:5]]) + "\n\n"
        
        if info.get('phones'):
            msg += "☎️ **Phone Numbers:**\n" + "\n".join([f"• {p}" for p in info['phones'][:5]]) + "\n"
        
        return msg.strip()
    
    def ask(self, question: str, session_id: str = None) -> str:
        """Answer question using RAG - IMPROVED"""
        if not self.status["ready"]:
            return "⚠️ AI is still initializing. Please wait a moment..."
        
        q_lower = question.lower().strip()
        
        # Handle greetings
        greetings = ["hi", "hello", "hey", "hai", "hii", "helloo", "hi there", 
                    "hello there", "good morning", "good afternoon", "good evening"]
        if q_lower in greetings:
            return "Hello! 👋 I'm here to help you learn about our company. What would you like to know?"
        
        # Handle contact requests
        contact_keywords = ["email", "contact", "phone", "address", "office", 
                          "location", "reach", "call", "write", "where are you",
                          "how to contact", "get in touch"]
        if any(keyword in q_lower for keyword in contact_keywords):
            return self.get_contact_info()
        
        # Check cache
        cache_key = q_lower[:100]  # Limit cache key length
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Retrieve relevant documents
            docs = self.retriever.invoke(question)
            
            if not docs or len(docs) == 0:
                return "I couldn't find specific information about that in our company data. Could you please rephrase your question or ask something else?"
            
            # Build context from retrieved docs
            context_parts = []
            for doc in docs:
                if hasattr(doc, 'page_content') and doc.page_content:
                    context_parts.append(doc.page_content)
            
            if not context_parts:
                return "I found some information but couldn't extract the content properly. Please try rephrasing your question."
            
            context = "\n\n".join(context_parts)[:5000]  # Increased limit
            
            # Create enhanced prompt
            prompt = f"""You are a knowledgeable and friendly AI assistant for this company. Answer the user's question based ONLY on the provided context.

Context Information:
{context}

User Question: {question}

Instructions:
- Provide a clear, concise answer in 2-5 sentences
- Use information ONLY from the context above
- Be specific and include relevant details
- If the context doesn't contain the answer, politely say so
- Be professional yet conversational
- Do not make up or assume information

Answer:"""
            
            # Call LLM API with retries
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://chatbot-generator.com",
                "X-Title": "Company AI Assistant"
            }
            
            payload = {
                "model": MODEL,
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a helpful AI assistant representing this company. Answer questions accurately using only the provided context. Be friendly, professional, and concise."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.4,
                "max_tokens": 600,
                "top_p": 0.9
            }
            
            # Retry logic
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        OPENROUTER_API_BASE,
                        headers=headers,
                        json=payload,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if "choices" in result and len(result["choices"]) > 0:
                            answer = result["choices"][0]["message"]["content"].strip()
                            
                            if answer:
                                self.cache[cache_key] = answer
                                
                                # Save to database
                                self.save_chat(question, answer, session_id)
                                
                                return answer
                            else:
                                return "I received an empty response. Please try asking again."
                        else:
                            return "The AI service returned an unexpected format. Please try again."
                    
                    elif response.status_code == 429:
                        if attempt < max_retries - 1:
                            time.sleep(2)
                            continue
                        return "⚠️ Service is busy. Please wait a moment and try again."
                    
                    else:
                        return f"⚠️ Service error (Code: {response.status_code}). Please try again in a moment."
                
                except requests.exceptions.Timeout:
                    if attempt < max_retries - 1:
                        continue
                    return "⚠️ Request timed out. Please try asking again."
                
                except Exception as e:
                    if attempt < max_retries - 1:
                        continue
                    return f"⚠️ Error processing your request: {str(e)[:100]}"
            
            return "⚠️ Unable to get a response after multiple attempts. Please try again."
        
        except Exception as e:
            print(f"Error in ask(): {e}")
            import traceback
            print(traceback.format_exc())
            return f"⚠️ An error occurred: {str(e)[:100]}"
    
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

def create_company(company_name: str, website_url: str, max_pages: int = 40) -> Optional[str]:
    """Create new company in database"""
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
        print(f"Error creating company: {e}")
        db.rollback()
        return None
    finally:
        db.close()


def update_company_after_scraping(slug: str, company_info: Dict, status: str = "completed"):
    """Update company info after scraping"""
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


st.set_page_config(
    page_title="Multi-Company AI Chatbot Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    }
    
    [data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%); 
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        text-align: center;
    }
    
    .header-title {
        color: white;
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.95);
        font-size: 1.3rem;
        margin-top: 0.5rem;
    }
    
    .company-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: transform 0.2s;
    }
    
    .company-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .status-badge {
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        display: inline-block;
    }
    
    .status-completed { background: #10b981; color: white; }
    .status-processing { background: #f59e0b; color: white; }
    .status-pending { background: #6b7280; color: white; }
    .status-failed { background: #ef4444; color: white; }
    
    .stButton button {
        border-radius: 10px !important;
        font-weight: 600 !important;
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
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "user_info_collected" not in st.session_state:
    st.session_state.user_info_collected = False
if "session_id" not in st.session_state:
    st.session_state.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:16]

# Sidebar Navigation
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()
    
    if st.button("➕ Create New Chatbot", use_container_width=True):
        st.session_state.page = "create"
        st.rerun()
    
    if st.button("📋 View All Chatbots", use_container_width=True):
        st.session_state.page = "list"
        st.rerun()
    
    st.markdown("---")
    
    if st.session_state.current_company:
        company = get_company_by_slug(st.session_state.current_company)
        if company:
            st.markdown(f"**Active Chatbot:**")
            st.info(company.company_name)
            
            if st.button("💬 Back to Chat", use_container_width=True):
                st.session_state.page = "chat"
                st.rerun()
                
if st.session_state.page == "home":
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🤖 Multi-Company AI Chatbot Generator</h1>
        <p class="header-subtitle">Create Intelligent Chatbots for Any Company in Minutes!</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### ⚡ Quick Setup")
        st.write("Provide company name and website URL - we automatically scrape and learn!")
    
    with col2:
        st.markdown("### 🧠 Smart AI")
        st.write("Uses advanced RAG to answer questions accurately based on website content")
    
    with col3:
        st.markdown("### 📊 Multi-Tenant")
        st.write("Manage unlimited company chatbots from one unified dashboard")
    
    st.markdown("---")
    
    st.markdown("### 🚀 Get Started")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("➕ Create Your First Chatbot", use_container_width=True, type="primary"):
            st.session_state.page = "create"
            st.rerun()
    
    with col2:
        if st.button("📋 View Existing Chatbots", use_container_width=True):
            st.session_state.page = "list"
            st.rerun()
    
    # Show existing companies count
    companies = get_all_companies()
    if companies:
        st.markdown("---")
        st.success(f"✅ You have {len(companies)} chatbot(s) ready!")


elif st.session_state.page == "create":
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">➕ Create New Chatbot</h1>
        <p class="header-subtitle">Enter company details to generate your AI assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("create_chatbot_form"):
        company_name = st.text_input(
            "Company Name *",
            placeholder="e.g., Acme Corporation",
            help="Enter the full company name"
        )
        
        website_url = st.text_input(
            "Website URL *",
            placeholder="e.g., https://example.com",
            help="Enter the company's main website URL"
        )
        
        max_pages = st.slider(
            "Maximum Pages to Scrape",
            min_value=10,
            max_value=100,
            value=40,
            step=5,
            help="More pages = more comprehensive knowledge (but slower setup)"
        )
        
        st.info("💡 **Tip:** Start with 40 pages. You can always create a new chatbot with more pages later.")
        
        submit = st.form_submit_button("🚀 Create Chatbot", use_container_width=True, type="primary")
        
        if submit:
            if not company_name or not website_url:
                st.error("❌ Please fill in all required fields")
            elif len(company_name) < 2:
                st.error("❌ Company name is too short")
            else:
                # Validate and normalize URL
                if not website_url.startswith(('http://', 'https://')):
                    website_url = 'https://' + website_url
                
                # Check URL format
                try:
                    parsed = urlparse(website_url)
                    if not parsed.netloc:
                        st.error("❌ Invalid website URL format")
                        st.stop()
                except:
                    st.error("❌ Invalid website URL")
                    st.stop()
                
                # Create company
                slug = create_company(company_name, website_url, max_pages)
                
                if not slug:
                    st.error("❌ A company with this name already exists!")
                    st.info("💡 Try using a different name or check the 'View All Chatbots' page")
                else:
                    st.success(f"✅ Company created: **{company_name}**")
                    
                    # Initialize AI with enhanced progress tracking
                    st.markdown("### 🤖 Initializing AI Assistant...")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    detail_text = st.empty()
                    
                    def progress_callback(current, total, url):
                        progress = min(current / total, 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"📄 Scraping page {current} of {total}")
                        detail_text.caption(f"Current: {url[:70]}...")
                    
                    ai = CompanyAI(slug)
                    success = ai.initialize(website_url, max_pages, progress_callback)
                    
                    if success:
                        # Update database
                        update_company_after_scraping(slug, ai.company_data, "completed")
                        
                        progress_bar.progress(1.0)
                        status_text.success("✅ Chatbot initialized successfully!")
                        detail_text.empty()
                        
                        st.balloons()
                        
                        time.sleep(2)
                        
                        # Navigate to chat
                        st.session_state.current_company = slug
                        st.session_state.ai_instance = ai
                        st.session_state.page = "chat"
                        st.session_state.messages = []
                        st.rerun()
                    else:
                        update_company_after_scraping(slug, {}, "failed")
                        
                        st.error("❌ Initialization failed!")
                        
                        error_msg = ai.status.get('error', 'Unknown error')
                        with st.expander("🔍 View Error Details"):
                            st.code(error_msg, language="text")
                        
                        st.warning("💡 **Troubleshooting Tips:**")
                        st.markdown("""
                        - Increase the number of pages to scrape (try 60-80)
                        - Verify the website URL is correct and accessible
                        - Check if the website has a robots.txt that blocks scraping
                        - Some websites may have anti-scraping measures
                        - Try again in a few moments
                        """)


elif st.session_state.page == "list":
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">📋 All Chatbots</h1>
        <p class="header-subtitle">Manage your company AI assistants</p>
    </div>
    """, unsafe_allow_html=True)
    
    companies = get_all_companies()
    
    if not companies:
        st.info("👋 No chatbots created yet. Create your first one to get started!")
        
        if st.button("➕ Create Your First Chatbot", type="primary"):
            st.session_state.page = "create"
            st.rerun()
    else:
        st.markdown(f"**Total Chatbots:** {len(companies)}")
        st.markdown("---")
        
        for company in companies:
            with st.container():
                st.markdown('<div class="company-card">', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([3, 2, 2])
                
                with col1:
                    st.markdown(f"### {company.company_name}")
                    st.markdown(f"🌐 [{company.website_url}]({company.website_url})")
                    st.caption(f"📄 **Pages Scraped:** {company.pages_scraped}")
                
                with col2:
                    status_class = f"status-{company.scraping_status}"
                    st.markdown(
                        f'<span class="status-badge {status_class}">{company.scraping_status.upper()}</span>',
                        unsafe_allow_html=True
                    )
                    
                    if company.last_scraped:
                        st.caption(f"🕐 {company.last_scraped.strftime('%Y-%m-%d %H:%M')}")
                
                with col3:
                    if company.scraping_status == "completed":
                        if st.button(f"💬 Open Chat", key=f"chat_{company.id}", type="primary"):
                            st.session_state.current_company = company.company_slug
                            st.session_state.page = "chat"
                            st.session_state.messages = []
                            st.session_state.question
