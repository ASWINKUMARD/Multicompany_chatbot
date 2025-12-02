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

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================
DATABASE_URL = "sqlite:///./multi_company_chatbots.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# ============================================================================
# DATABASE MODELS
# ============================================================================

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
    "careers", "blog", "case-studies", "portfolio", "industries",
    "technology", "expertise", "what-we-do", "who-we-are", "footer",
    "about-us", "contact-us", "our-services", "our-team", "home", "index"
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_slug(company_name: str) -> str:
    """Convert company name to URL-friendly slug"""
    slug = company_name.lower()
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    slug = slug.strip('-')
    return slug


def get_chroma_directory(company_slug: str) -> str:
    """Get the ChromaDB directory path for a specific company"""
    return f"./chroma_db/{company_slug}"


# ============================================================================
# ENHANCED WEB SCRAPER CLASS (FIXED ALL ISSUES)
# ============================================================================

class WebScraper:
    """Handles web scraping for company websites - FULLY FIXED VERSION"""
    
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
    
    def clean_address(self, text: str) -> str:
        """Clean and format address text"""
        text = ' '.join(text.split())
        text = re.sub(r'(Corporate Office|Branch Office|Head Office|Registered Office)', 
                     '', text, flags=re.IGNORECASE)
        return re.sub(r'\s+', ' ', text).strip()
    
    def extract_contact_info(self, soup: BeautifulSoup, text: str):
        """Extract emails, phones, and addresses from page content"""
        # Extract emails
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', text)
        for email in emails:
            if not email.lower().endswith(('.png', '.jpg', '.gif', '.svg')):
                self.company_info['emails'].add(email.lower())
        
        # Extract phone numbers with better patterns
        phone_patterns = [
            r'\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\d{10,}',
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
            if any(city in low for city in ['india', 'mumbai', 'delhi', 'bangalore', 
                                             'chennai', 'kolkata', 'hyderabad', 'pune',
                                             'madurai', 'coimbatore', 'kerala', 'tamil nadu']):
                if not self.company_info['address_india'] and len(line) > 15:
                    block = " ".join(lines[max(0, i-1):min(len(lines), i+4)])
                    cleaned = self.clean_address(block)
                    if 15 < len(cleaned) < 400:
                        self.company_info['address_india'] = cleaned
            
            # International addresses
            if any(country in low for country in ['singapore', 'usa', 'uk', 'uae', 
                                                   'malaysia', 'australia', 'canada']):
                if not self.company_info['address_international'] and len(line) > 15:
                    block = " ".join(lines[max(0, i-1):min(len(lines), i+4)])
                    cleaned = self.clean_address(block)
                    if 15 < len(cleaned) < 400:
                        self.company_info['address_international'] = cleaned
    
    def is_valid_url(self, url: str, base_domain: str) -> bool:
        """Check if URL should be scraped"""
        try:
            parsed = urlparse(url)
            
            # Must be same domain
            if parsed.netloc != base_domain:
                return False
            
            # Skip file downloads and admin pages
            skip_patterns = [
                r'\.pdf$', r'\.jpg$', r'\.png$', r'\.gif$', r'\.zip$',
                r'\.doc$', r'\.docx$', r'\.xls$', r'\.xlsx$', r'\.mp4$',
                r'/wp-admin/', r'/wp-includes/', r'/wp-json/', r'/admin/',
                r'/login', r'/register', r'/signup', r'/signin',
                r'/cart/', r'/checkout/', r'/feed/', r'/rss/', r'/api/',
                r'/download/', r'\.xml$', r'\.json$', r'#', r'\?.*page=\d+'
            ]
            
            for pattern in skip_patterns:
                if re.search(pattern, url.lower()):
                    return False
            
            return True
        except:
            return False
    
    def extract_content(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract meaningful content from a webpage - AGGRESSIVE EXTRACTION"""
        content_dict = {
            'url': url,
            'title': '',
            'main_content': '',
            'metadata': {}
        }
        
        try:
            # Extract title
            title_tag = soup.find('title')
            if title_tag:
                content_dict['title'] = title_tag.get_text(strip=True)
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={"name": "description"})
            if meta_desc and meta_desc.get("content"):
                content_dict['metadata']['description'] = meta_desc["content"]
            
            # Extract contact info BEFORE removing elements
            full_text = soup.get_text(separator="\n", strip=True)
            self.extract_contact_info(soup, full_text)
            
            # Remove unwanted elements
            for tag in soup(['script', 'style', 'iframe', 'noscript']):
                tag.decompose()
            
            # AGGRESSIVE MULTI-STRATEGY CONTENT EXTRACTION
            content_parts = []
            
            # Strategy 1: Main content containers
            main_selectors = [
                "main", "article", "[role='main']", ".content", ".main-content",
                "#content", "#main", ".post-content", ".entry-content",
                ".page-content", "#primary", ".site-content", ".container"
            ]
            
            for selector in main_selectors:
                try:
                    elements = soup.select(selector)
                    for elem in elements:
                        text = elem.get_text(separator="\n", strip=True)
                        if len(text) > 50:
                            content_parts.append(text)
                except:
                    pass
            
            # Strategy 2: All meaningful tags
            if len(content_parts) < 2:
                for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'li', 'td', 'span', 'div']):
                    try:
                        text = tag.get_text(strip=True)
                        if len(text) > 20 and not text.isdigit():
                            content_parts.append(text)
                    except:
                        pass
            
            # Strategy 3: Body fallback
            if len(content_parts) < 2:
                body = soup.find('body')
                if body:
                    text = body.get_text(separator="\n", strip=True)
                    if len(text) > 50:
                        content_parts.append(text)
            
            # Strategy 4: Entire page as last resort
            if not content_parts:
                text = soup.get_text(separator="\n", strip=True)
                if len(text) > 30:
                    content_parts.append(text)
            
            # Combine and clean content
            if content_parts:
                combined = "\n\n".join(content_parts)
                lines = [line.strip() for line in combined.split("\n") if line.strip()]
                
                # Remove duplicates while preserving order
                seen = set()
                unique_lines = []
                for line in lines:
                    line_lower = line.lower()
                    if len(line) > 10 and line_lower not in seen:
                        seen.add(line_lower)
                        unique_lines.append(line)
                
                content_dict['main_content'] = "\n".join(unique_lines)
            
            # Add metadata if content is too short
            if len(content_dict['main_content']) < 100:
                if content_dict['title']:
                    content_dict['main_content'] = f"{content_dict['title']}\n\n{content_dict['main_content']}"
                if content_dict['metadata'].get('description'):
                    content_dict['main_content'] += f"\n\n{content_dict['metadata']['description']}"
            
        except Exception as e:
            self.debug_info.append(f"Content extraction error for {url}: {str(e)}")
        
        return content_dict
    
    def scrape_website(self, base_url: str, max_pages: int = 40, 
                      progress_callback=None) -> Tuple[str, Dict]:
        """
        Scrape website and return formatted content - FULLY FIXED VERSION
        """
        visited = set()
        all_content = []
        queue = deque()
        base_domain = urlparse(base_url).netloc
        
        # Normalize base URL
        if not base_url.endswith('/'):
            base_url = base_url + '/'
        
        # Add priority pages with variations
        for page in PRIORITY_PAGES:
            variations = [
                urljoin(base_url, page),
                urljoin(base_url, page + '/'),
                urljoin(base_url, page + '.html'),
                urljoin(base_url, page + '.php'),
            ]
            for var in variations:
                queue.append(var)
        
        # Add base URL variations
        queue.append(base_url)
        queue.append(base_url.rstrip('/'))
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        consecutive_failures = 0
        max_consecutive_failures = 8
        
        while queue and len(visited) < max_pages:
            if consecutive_failures >= max_consecutive_failures:
                self.debug_info.append(f"Stopped after {consecutive_failures} consecutive failures")
                break
            
            url = queue.popleft()
            
            # Normalize URL
            url = url.split("#")[0].split("?")[0]
            
            if url in visited or not self.is_valid_url(url, base_domain):
                continue
            
            try:
                response = requests.get(url, headers=headers, timeout=25, allow_redirects=True)
                
                if response.status_code != 200:
                    consecutive_failures += 1
                    self.debug_info.append(f"HTTP {response.status_code}: {url}")
                    continue
                
                consecutive_failures = 0
                visited.add(url)
                
                if progress_callback:
                    progress_callback(len(visited), max_pages, url)
                
                soup = BeautifulSoup(response.text, "html.parser")
                content_data = self.extract_content(soup, url)
                
                # LOWERED threshold - accept ANY content
                if len(content_data['main_content']) > 30:
                    formatted = f"PAGE URL: {content_data['url']}\n"
                    formatted += f"PAGE TITLE: {content_data['title']}\n"
                    
                    if "description" in content_data['metadata']:
                        formatted += f"PAGE DESCRIPTION: {content_data['metadata']['description']}\n"
                    
                    formatted += f"\nPAGE CONTENT:\n{content_data['main_content']}\n"
                    all_content.append(formatted)
                    
                    self.scraped_content[url] = content_data
                else:
                    self.debug_info.append(f"Minimal content {url}: {len(content_data['main_content'])} chars")
                
                # Find new links
                for link in soup.find_all("a", href=True):
                    try:
                        next_url = urljoin(url, link['href'])
                        next_url = next_url.split("#")[0].split("?")[0]
                        
                        if next_url not in visited and self.is_valid_url(next_url, base_domain):
                            queue.append(next_url)
                    except:
                        pass
                
                # Respectful delay
                time.sleep(0.3)
            
            except requests.exceptions.Timeout:
                self.debug_info.append(f"Timeout: {url}")
                consecutive_failures += 1
            except requests.exceptions.RequestException as e:
                self.debug_info.append(f"Request error {url}: {str(e)}")
                consecutive_failures += 1
            except Exception as e:
                self.debug_info.append(f"Unexpected error {url}: {str(e)}")
                consecutive_failures += 1
        
        # Build final content
        if not all_content:
            error_msg = f"❌ SCRAPING FAILED\n"
            error_msg += f"Visited {len(visited)} pages\n"
            error_msg += f"Base domain: {base_domain}\n"
            error_msg += f"Debug info:\n" + "\n".join(self.debug_info[:10])
            
            return error_msg, {
                'emails': [],
                'phones': [],
                'address_india': None,
                'address_international': None,
                'pages_scraped': len(visited)
            }
        
        # Add contact information header
        header = "="*80 + "\n"
        header += "COMPANY INFORMATION\n"
        header += "="*80 + "\n\n"
        
        if self.company_info['address_india']:
            header += f"INDIA OFFICE:\n{self.company_info['address_india']}\n\n"
        
        if self.company_info['address_international']:
            header += f"INTERNATIONAL OFFICE:\n{self.company_info['address_international']}\n\n"
        
        if self.company_info['emails']:
            header += "CONTACT EMAILS:\n"
            for email in sorted(self.company_info['emails'])[:10]:
                header += f"  - {email}\n"
            header += "\n"
        
        if self.company_info['phones']:
            header += "CONTACT PHONES:\n"
            for phone in sorted(self.company_info['phones'])[:10]:
                header += f"  - {phone}\n"
            header += "\n"
        
        all_content.insert(0, header)
        
        formatted_content = ("\n\n" + "="*80 + "\n\n").join(all_content)
        
        return formatted_content, {
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
    """AI Engine for a specific company's chatbot - FULLY FIXED"""
    
    def __init__(self, company_slug: str):
        self.company_slug = company_slug
        self.retriever = None
        self.cache = {}
        self.status = {"ready": False, "error": None}
        self.company_data = None
    
    def initialize(self, website_url: str, max_pages: int = 40, progress_callback=None):
        """Initialize the AI by scraping and embedding company website"""
        try:
            # Step 1: Scrape website
            scraper = WebScraper(self.company_slug)
            content, company_info = scraper.scrape_website(
                website_url, 
                max_pages, 
                progress_callback
            )
            
            # FIXED: More lenient content check
            if len(content) < 200:
                self.status["error"] = f"Insufficient content scraped. Only {len(content)} characters. Debug: {'; '.join(scraper.debug_info[:3])}"
                return False
            
            # Check for error message
            if content.startswith("❌ SCRAPING FAILED"):
                self.status["error"] = content
                return False
            
            self.company_data = company_info
            
            # Step 2: Split content into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = splitter.split_text(content)
            
            if not chunks or len(chunks) == 0:
                self.status["error"] = "No text chunks created from content"
                return False
            
            # FIXED: Ensure we have meaningful chunks
            meaningful_chunks = [c for c in chunks if len(c.strip()) > 50]
            if len(meaningful_chunks) < 3:
                self.status["error"] = f"Insufficient meaningful chunks: {len(meaningful_chunks)}"
                return False
            
            # Step 3: Create embeddings
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
            
            # Step 4: Create vector store
            chroma_dir = get_chroma_directory(self.company_slug)
            
            if os.path.exists(chroma_dir):
                shutil.rmtree(chroma_dir)
            
            os.makedirs(chroma_dir, exist_ok=True)
            
            vectorstore = Chroma.from_texts(
                texts=meaningful_chunks,
                embedding=embeddings,
                persist_directory=chroma_dir
            )
            
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            self.status["ready"] = True
            
            return True
        
        except Exception as e:
            self.status["error"] = f"Initialization error: {str(e)}"
            return False
    
    def load_existing(self):
        """Load existing vector store for company - FIXED TYPO"""
        try:
            chroma_dir = get_chroma_directory(self.company_slug)
            
            if not os.path.exists(chroma_dir):
                return False
            
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            
            class CustomEmbeddings:
                def __init__(self, model):
                    self.model = model
                
                def embed_documents(self, texts):
                    if not texts:
                        return []
                    embeddings = self.model.encode(
                        texts,
                        normalize_embeddings=True,  # FIXED: Was normalize_embedings
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                    return embeddings.tolist()
                
                def embed_query(self, text):
                    embedding = self.model.encode(
                        [text],
                        normalize_embeddings=True,  # FIXED: Consistent spelling
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                    return embedding[0].tolist()
            
            embeddings = CustomEmbeddings(embedding_model)
            
            vectorstore = Chroma(
                persist_directory=chroma_dir,
                embedding_function=embeddings
            )
            
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            self.status["ready"] = True
            
            return True
        
        except Exception as e:
            self.status["error"] = f"Load error: {str(e)}"
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
                
                self.company_data = {
                    'emails': json.loads(company.emails) if company.emails else [],
                    'phones': json.loads(company.phones) if company.phones else [],
                    'address_india': company.address_india,
                    'address_international': company.address_international
                }
            finally:
                db.close()
        
        info = self.company_data
        if not any([info.get('emails'), info.get('phones'), 
                   info.get('address_india'), info.get('address_international')]):
            return "No contact information found."
        
        msg = "📞 **CONTACT INFORMATION**\n\n"
        
        if info.get('address_india'):
            msg += f"🇮🇳 **India Office:**\n{info['address_india']}\n\n"
        
        if info.get('address_international'):
            msg += f"🌍 **International Office:**\n{info['address_international']}\n\n"
        
        if info.get('emails'):
            msg += "📧 **Emails:**\n" + "\n".join([f"• {e}" for e in info['emails'][:5]]) + "\n\n"
        
        if info.get('phones'):
            msg += "☎️ **Phones:**\n" + "\n".join([f"• {p}" for p in info['phones'][:5]]) + "\n"
        
        return msg.strip()
    
    def ask(self, question: str, session_id: str = None) -> str:
        """Answer a question using RAG"""
        if not self.status["ready"]:
            return "⚠️ AI is still initializing. Please wait..."
        
        q_lower = question.lower().strip()
        
        # Handle greetings
        greetings = ["hi", "hello", "hey", "hai", "hii", "helloo", "hi there", "hello there", "good morning", "good afternoon"]
        if q_lower in greetings:
            return "Hello! 👋 I'm here to help you learn more about our company. What would you like to know?"
        
        # Handle contact requests
        contact_keywords = ["email", "contact", "phone", "address", "office", 
                          "location", "reach", "call", "write", "where are you"]
        if any(keyword in q_lower for keyword in contact_keywords):
            return self.get_contact_info()
        
        # Check cache
        if q_lower in self.cache:
            return self.cache[q_lower]
        
        try:
            # Retrieve relevant documents
            docs = self.retriever.invoke(question)
            
            if not docs:
                return "I couldn't find relevant information about that. Could you rephrase your question or ask something else about our company?"
            
            context = "\n\n".join([doc.page_content for doc in docs])[:4500]
            
            # Create prompt
            prompt = f"""You are a helpful AI assistant for this company. Answer the question using ONLY the context provided below. Be concise, professional, and helpful.

Context:
{context}

Question: {question}

Instructions:
- Answer in 2-4 sentences maximum
- Be specific and accurate based on the context
- If the context doesn't contain the answer, say "I don't have information about that in our company data"
- Don't make up information
- Be friendly and professional

Answer:"""
            
            # Call LLM API
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://chatbot-generator.com",
                "X-Title": "Multi-Company Chatbot"
            }
            
            payload = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful company assistant. Answer questions accurately using only the provided context. Be concise and friendly."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 500
            }
            
            response = requests.post(
                OPENROUTER_API_BASE,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                answer = response.json()["choices"][0]["message"]["content"].strip()
                self.cache[q_lower] = answer
                
                # Save to database
                self.save_chat(question, answer, session_id)
                
                return answer
            else:
                return f"⚠️ API Error ({response.status_code}). Please try again."
        
        except requests.exceptions.Timeout:
            return "⚠️ Request timeout. Please try again."
        except Exception as e:
            return f"⚠️ Error: {str(e)}"
    
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
        finally:
            db.close()

def create_company(company_name: str, website_url: str, max_pages: int = 40) -> Optional[str]:
    """Create a new company in database"""
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
    finally:
        db.close()


def get_all_companies() -> List[Company]:
    """Get all companies"""
    db = SessionLocal()
    try:
        return db.query(Company).all()
    finally:
        db.close()


def get_company_by_slug(slug: str) -> Optional[Company]:
    """Get company by slug"""
    db = SessionLocal()
    try:
        return db.query(Company).filter(Company.company_slug == slug).first()
    finally:
        db.close()


# Page config
st.set_page_config(
    page_title="Multi-Company Chatbot Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    
    [data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%); 
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        text-align: center;
    }
    
    .header-title {
        color: white;
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    .company-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .status-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: bold;
        display: inline-block;
    }
    
    .status-completed { background: #10b981; color: white; }
    .status-processing { background: #f59e0b; color: white; }
    .status-pending { background: #6b7280; color: white; }
    .status-failed { background: #ef4444; color: white; }
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
        st.markdown(f"**Active:** {st.session_state.current_company}")
        
        if st.button("💬 Back to Chat", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()


if st.session_state.page == "home":
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🤖 Multi-Company Chatbot Generator</h1>
        <p class="header-subtitle">Create AI-Powered Chatbots for Any Company in Minutes!</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### ⚡ Quick Setup")
        st.write("Just provide company name and website URL - we handle the rest!")
    
    with col2:
        st.markdown("### 🧠 Smart AI")
        st.write("Automatically learns from website content and answers questions accurately")
    
    with col3:
        st.markdown("### 📊 Multi-Tenant")
        st.write("Manage chatbots for unlimited companies from one dashboard")
    
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

elif st.session_state.page == "create":
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">➕ Create New Chatbot</h1>
        <p class="header-subtitle">Enter company details to generate AI chatbot</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("create_chatbot_form"):
        company_name = st.text_input(
            "Company Name *",
            placeholder="e.g., Syngrid Technologies"
        )
        
        website_url = st.text_input(
            "Website URL *",
            placeholder="e.g., https://syngrid.com"
        )
        
        max_pages = st.slider(
            "Maximum Pages to Scrape",
            min_value=10,
            max_value=100,
            value=40,
            help="More pages = more comprehensive knowledge, but slower initialization"
        )
        
        submit = st.form_submit_button("🚀 Create Chatbot", use_container_width=True, type="primary")
        
        if submit:
            if not company_name or not website_url:
                st.error("❌ Please fill in all required fields")
            else:
                # Validate URL
                if not website_url.startswith(('http://', 'https://')):
                    website_url = 'https://' + website_url
                
                # Create company
                slug = create_company(company_name, website_url, max_pages)
                
                if not slug:
                    st.error("❌ Company already exists with this name!")
                else:
                    st.success(f"✅ Company created: {slug}")
                    
                    # Initialize AI
                    with st.spinner("🤖 Initializing AI chatbot..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        def progress_callback(current, total, url):
                            progress = min(current / total, 1.0)
                            progress_bar.progress(progress)
                            status_text.text(f"📄 Scraping {current}/{total}: {url[:60]}...")
                        
                        ai = CompanyAI(slug)
                        success = ai.initialize(website_url, max_pages, progress_callback)
                        
                        if success:
                            # Update database
                            update_company_after_scraping(slug, ai.company_data, "completed")
                            
                            progress_bar.progress(1.0)
                            status_text.success("✅ Chatbot ready!")
                            
                            time.sleep(2)
                            
                            st.session_state.current_company = slug
                            st.session_state.ai_instance = ai
                            st.session_state.page = "chat"
                            st.session_state.messages = []
                            st.rerun()
                        else:
                            update_company_after_scraping(slug, {}, "failed")
                            st.error(f"❌ Initialization failed: {ai.status.get('error')}")
                            st.warning("💡 Try increasing the number of pages to scrape, or check if the website URL is correct.")

elif st.session_state.page == "list":
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">📋 All Chatbots</h1>
        <p class="header-subtitle">Manage your company chatbots</p>
    </div>
    """, unsafe_allow_html=True)
    
    companies = get_all_companies()
    
    if not companies:
        st.info("No chatbots created yet. Create your first one!")
        if st.button("➕ Create Chatbot"):
            st.session_state.page = "create"
            st.rerun()
    else:
        for company in companies:
            with st.container():
                st.markdown('<div class="company-card">', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([3, 2, 2])
                
                with col1:
                    st.markdown(f"### {company.company_name}")
                    st.markdown(f"🌐 {company.website_url}")
                    st.markdown(f"📄 Pages: {company.pages_scraped}")
                
                with col2:
                    status_class = f"status-{company.scraping_status}"
                    st.markdown(
                        f'<span class="status-badge {status_class}">{company.scraping_status.upper()}</span>',
                        unsafe_allow_html=True
                    )
                    
                    if company.last_scraped:
                        st.caption(f"Last scraped: {company.last_scraped.strftime('%Y-%m-%d %H:%M')}")
                
                with col3:
                    if company.scraping_status == "completed":
                        if st.button(f"💬 Open Chat", key=f"chat_{company.id}"):
                            st.session_state.current_company = company.company_slug
                            st.session_state.page = "chat"
                            st.session_state.messages = []
                            st.session_state.question_count = 0
                            st.session_state.user_info_collected = False
                            st.rerun()
                    else:
                        st.button("⏳ Not Ready", key=f"wait_{company.id}", disabled=True)
                
                st.markdown('</div>', unsafe_allow_html=True)


elif st.session_state.page == "chat":
    if not st.session_state.current_company:
        st.warning("No company selected")
        st.session_state.page = "list"
        st.rerun()
    
    company = get_company_by_slug(st.session_state.current_company)
    
    if not company:
        st.error("Company not found")
        st.session_state.page = "list"
        st.rerun()
    
    # Initialize AI if needed
    if not st.session_state.ai_instance or \
       st.session_state.ai_instance.company_slug != st.session_state.current_company:
        with st.spinner("Loading AI..."):
            ai = CompanyAI(st.session_state.current_company)
            if ai.load_existing():
                st.session_state.ai_instance = ai
                # Load company data
                ai.company_data = {
                    'emails': json.loads(company.emails) if company.emails else [],
                    'phones': json.loads(company.phones) if company.phones else [],
                    'address_india': company.address_india,
                    'address_international': company.address_international
                }
            else:
                st.error("Failed to load AI")
                st.stop()
    
    # Header
    st.markdown(f"""
    <div class="header-container">
        <h1 class="header-title">💬 {company.company_name}</h1>
        <p class="header-subtitle">AI Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if user_input := st.chat_input(f"Ask anything about {company.company_name}..."):
        # User message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        st.session_state.question_count += 1
        
        # Show contact form after 3 questions
        if st.session_state.question_count == 3 and not st.session_state.user_info_collected:
            # AI response first
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    answer = st.session_state.ai_instance.ask(
                        user_input,
                        st.session_state.session_id
                    )
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Contact form
            st.markdown("---")
            st.markdown("### 📋 Help us serve you better!")
            st.info("Share your contact details to receive personalized assistance")
            
            with st.form("contact_form"):
                col1, col2 = st.columns(2)
                with col1:
                    name = st.text_input("Name *")
                    email = st.text_input("Email *")
                with col2:
                    phone = st.text_input("Phone *")
                
                submit_contact = st.form_submit_button("Submit", use_container_width=True)
                
                if submit_contact:
                    if name and email and phone:
                        db = SessionLocal()
                        try:
                            contact = UserContact(
                                company_slug=st.session_state.current_company,
                                name=name,
                                email=email,
                                phone=phone,
                                session_id=st.session_state.session_id
                            )
                            db.add(contact)
                            db.commit()
                            
                            st.session_state.user_info_collected = True
                            st.success("✅ Thank you! Continue chatting...")
                            time.sleep(1)
                            st.rerun()
                        finally:
                            db.close()
                    else:
                        st.error("Please fill all fields")
            
            st.stop()
        
        # Normal response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                answer = st.session_state.ai_instance.ask(
                    user_input,
                    st.session_state.session_id
                )
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; opacity: 0.8; padding: 20px; color: white;'>
    🤖 <strong>Multi-Company Chatbot Generator</strong> — Built with Streamlit, LangChain & OpenRouter
    <br>© 2025 All Rights Reserved
</div>
""", unsafe_allow_html=True)
