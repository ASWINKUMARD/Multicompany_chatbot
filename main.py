"""
MULTI-COMPANY AI CHATBOT - PART 1: BACKEND
Fixed RAG implementation with proper LangChain integration
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

# LangChain imports - PROPER SETUP
from langchain_community.text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document

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
    """Stores information about each company"""
    __tablename__ = "companies"
    
    id = Column(Integer, primary_key=True, index=True)
    company_name = Column(String(255), unique=True, nullable=False, index=True)
    company_slug = Column(String(255), unique=True, nullable=False, index=True)
    website_url = Column(String(500), nullable=False)
    
    # Contact information
    emails = Column(Text, nullable=True)
    phones = Column(Text, nullable=True)
    address_india = Column(Text, nullable=True)
    address_international = Column(Text, nullable=True)
    
    # Scraping metadata
    pages_scraped = Column(Integer, default=0)
    last_scraped = Column(DateTime, nullable=True)
    scraping_status = Column(String(50), default="pending")
    
    # Configuration
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


class ScrapedContent(Base):
    """Stores raw scraped content"""
    __tablename__ = "scraped_content"
    
    id = Column(Integer, primary_key=True, index=True)
    company_slug = Column(String(255), nullable=False, index=True)
    url = Column(String(1000), nullable=False)
    title = Column(Text, nullable=True)
    content = Column(Text, nullable=False)
    scraped_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# Create tables
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
    return f"./chroma_db/{company_slug}"


# ============================================================================
# ENHANCED WEB SCRAPER
# ============================================================================

class WebScraper:
    """Advanced web scraper with better content extraction"""
    
    def __init__(self, company_slug: str):
        self.company_slug = company_slug
        self.company_info = {
            'emails': set(),
            'phones': set(),
            'address_india': None,
            'address_international': None
        }
        self.scraped_pages = []
        self.debug_info = []
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"]+', '', text)
        return text.strip()
    
    def extract_contact_info(self, soup: BeautifulSoup, text: str):
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
            
            india_keywords = ['india', 'mumbai', 'delhi', 'bangalore', 'chennai', 
                            'madurai', 'tamil nadu', 'maharashtra']
            if any(city in low for city in india_keywords):
                if not self.company_info['address_india']:
                    block = " ".join(lines[max(0, i-2):min(len(lines), i+4)])
                    cleaned = self.clean_text(block)
                    if 20 < len(cleaned) < 500:
                        self.company_info['address_india'] = cleaned
            
            intl_keywords = ['singapore', 'usa', 'uk', 'uae', 'dubai', 'malaysia']
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
            
            if '#' in url or '?' in url:
                return False
            
            return True
        except:
            return False
    
    def extract_content(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract meaningful content from page"""
        content_dict = {
            'url': url,
            'title': '',
            'content': ''
        }
        
        try:
            # Title
            if soup.find('title'):
                content_dict['title'] = soup.find('title').get_text(strip=True)
            
            # Meta description
            meta_desc = soup.find('meta', attrs={"name": "description"})
            if meta_desc and meta_desc.get("content"):
                content_dict['content'] += meta_desc["content"] + "\n\n"
            
            # Extract contact info
            full_text = soup.get_text(separator="\n", strip=True)
            self.extract_contact_info(soup, full_text)
            
            # Remove unwanted elements
            for tag in soup(['script', 'style', 'iframe', 'nav', 'footer', 'header']):
                tag.decompose()
            
            # Extract main content
            main_selectors = [
                "main", "article", "[role='main']", ".content", ".main-content",
                "#content", "section", ".section"
            ]
            
            texts = []
            
            # Try main content areas first
            for selector in main_selectors:
                elements = soup.select(selector)
                for elem in elements:
                    text = elem.get_text(separator="\n", strip=True)
                    if len(text) > 100:
                        texts.append(text)
            
            # Extract from semantic tags
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
            
            # Clean and deduplicate
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
            self.debug_info.append(f"Extraction error {url}: {str(e)}")
        
        return content_dict
    
    def scrape_website(self, base_url: str, max_pages: int = 40, 
                      progress_callback=None) -> Tuple[List[Document], Dict]:
        """Scrape website and return LangChain Documents"""
        visited = set()
        queue = deque()
        base_domain = urlparse(base_url).netloc
        
        base_url = base_url.rstrip('/') + '/'
        
        # Add priority pages
        for page in PRIORITY_PAGES:
            for url in [urljoin(base_url, page), urljoin(base_url, page + '/')]:
                if url not in queue:
                    queue.append(url)
        
        queue.append(base_url)
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        
        documents = []
        successful_scrapes = 0
        
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
                successful_scrapes += 1
                
                if progress_callback:
                    progress_callback(len(visited), max_pages, url)
                
                soup = BeautifulSoup(response.text, "html.parser")
                content_data = self.extract_content(soup, url)
                
                if len(content_data['content']) > 50:
                    # Create LangChain Document
                    doc = Document(
                        page_content=content_data['content'],
                        metadata={
                            'source': url,
                            'title': content_data['title'],
                            'company': self.company_slug
                        }
                    )
                    documents.append(doc)
                    self.debug_info.append(f"✓ Scraped: {url} ({len(content_data['content'])} chars)")
                
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
                
                time.sleep(0.3)
            
            except Exception as e:
                self.debug_info.append(f"Error {url}: {str(e)[:50]}")
                continue
        
        if len(documents) < 3:
            raise Exception(f"Insufficient content: Only {len(documents)} pages scraped successfully")
        
        return documents, {
            'emails': list(self.company_info['emails']),
            'phones': list(self.company_info['phones']),
            'address_india': self.company_info['address_india'],
            'address_international': self.company_info['address_international'],
            'pages_scraped': len(visited)
        }


# ============================================================================
# DATABASE HELPER FUNCTIONS
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
"""
MULTI-COMPANY AI CHATBOT - PART 2: AI ENGINE + STREAMLIT UI
Fixed RAG with proper answer generation
"""

import streamlit as st
import time
import requests
import json
import os
from typing import List
import shutil

# Import Part 1 components
from part1_backend import (
    Company, ChatHistory, UserContact, ScrapedContent,
    WebScraper, create_company, update_company_after_scraping,
    get_all_companies, get_company_by_slug, get_chroma_directory,
    SessionLocal, OPENROUTER_API_KEY, OPENROUTER_API_BASE, MODEL
)

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.prompts import PromptTemplate


# ============================================================================
# ENHANCED AI ENGINE WITH PROPER RAG
# ============================================================================

class CompanyAI:
    """Fixed AI Engine with proper context retrieval and answering"""
    
    def __init__(self, company_slug: str):
        self.company_slug = company_slug
        self.vectorstore = None
        self.retriever = None
        self.embeddings = None
        self.status = {"ready": False, "error": None}
        self.company_data = None
        
        # Enhanced prompt template
        self.qa_template = """You are an intelligent AI assistant for {company_name}. Your role is to answer questions accurately based on the provided context from the company's website.

CONTEXT FROM COMPANY WEBSITE:
{context}

CHAT HISTORY:
{chat_history}

USER QUESTION: {question}

INSTRUCTIONS:
1. Answer the question using ONLY information from the context above
2. Be specific, detailed, and helpful
3. If you cannot find the answer in the context, say "I don't have specific information about that in our knowledge base, but I can help you get in touch with our team."
4. Be conversational and friendly
5. Include relevant details like services, products, or contact information when appropriate
6. If the question is about contact details, provide all available information

ANSWER (2-5 sentences, be specific and helpful):"""

        self.qa_prompt = PromptTemplate(
            template=self.qa_template,
            input_variables=["company_name", "context", "chat_history", "question"]
        )
    
    def initialize(self, website_url: str, max_pages: int = 40, progress_callback=None):
        """Initialize AI by scraping and creating vector store"""
        try:
            # Step 1: Scrape website
            if progress_callback:
                progress_callback(0, max_pages, "Starting scraper...")
            
            scraper = WebScraper(self.company_slug)
            documents, company_info = scraper.scrape_website(
                website_url, 
                max_pages, 
                progress_callback
            )
            
            if len(documents) < 3:
                self.status["error"] = f"Not enough content scraped: {len(documents)} pages"
                return False
            
            self.company_data = company_info
            
            # Step 2: Create better text chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "],
                length_function=len
            )
            
            # Split documents
            split_docs = text_splitter.split_documents(documents)
            
            if len(split_docs) < 5:
                self.status["error"] = f"Not enough chunks created: {len(split_docs)}"
                return False
            
            # Step 3: Create embeddings with better model
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Step 4: Create vector store
            chroma_dir = get_chroma_directory(self.company_slug)
            
            # Clean existing directory
            if os.path.exists(chroma_dir):
                shutil.rmtree(chroma_dir)
            
            os.makedirs(chroma_dir, exist_ok=True)
            
            # Create Chroma vectorstore
            self.vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory=chroma_dir,
                collection_name="company_knowledge"
            )
            
            # Verify creation
            if self.vectorstore._collection.count() == 0:
                raise Exception("Vectorstore is empty")
            
            # Create retriever with better search
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance for diverse results
                search_kwargs={
                    "k": 8,  # Retrieve more documents
                    "fetch_k": 20,  # Fetch more for MMR
                    "lambda_mult": 0.7  # Balance between relevance and diversity
                }
            )
            
            self.status["ready"] = True
            
            if progress_callback:
                progress_callback(max_pages, max_pages, "✅ Ready!")
            
            return True
        
        except Exception as e:
            self.status["error"] = f"Initialization error: {str(e)}"
            import traceback
            print(traceback.format_exc())
            return False
    
    def load_existing(self):
        """Load existing vector store"""
        try:
            chroma_dir = get_chroma_directory(self.company_slug)
            
            if not os.path.exists(chroma_dir):
                self.status["error"] = "Vector store not found"
                return False
            
            # Load embeddings
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
            
            if self.vectorstore._collection.count() == 0:
                self.status["error"] = "Vectorstore is empty"
                return False
            
            # Create retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 8,
                    "fetch_k": 20,
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
        """Answer question using enhanced RAG"""
        if not self.status["ready"]:
            return "⚠️ System is still initializing. Please wait..."
        
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
                return "I couldn't find specific information about that. Could you rephrase your question or ask about something else?"
            
            # Build context from documents
            context_parts = []
            for doc in relevant_docs[:6]:  # Use top 6 documents
                if hasattr(doc, 'page_content') and doc.page_content:
                    # Add source info for better context
                    source = doc.metadata.get('source', 'unknown')
                    title = doc.metadata.get('title', '')
                    
                    context_part = f"[From: {title if title else source}]\n{doc.page_content}"
                    context_parts.append(context_part)
            
            if not context_parts:
                return "I found some documents but couldn't extract the content. Please try again."
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Format chat history
            history_text = ""
            if chat_history:
                history_text = "\n".join([
                    f"User: {msg['question']}\nAssistant: {msg['answer']}" 
                    for msg in chat_history[-3:]  # Last 3 exchanges
                ])
            
            # Get company name
            company = get_company_by_slug(self.company_slug)
            company_name = company.company_name if company else "the company"
            
            # Build enhanced prompt
            prompt = self.qa_prompt.format(
                company_name=company_name,
                context=context[:6000],  # Limit context size
                chat_history=history_text,
                question=question
            )
            
            # Call LLM API
            answer = self._call_llm(prompt, question, context)
            
            if answer:
                # Save to database
                self.save_chat(question, answer, session_id)
                return answer
            else:
                return "I'm having trouble generating a response. Please try again."
        
        except Exception as e:
            print(f"Error in ask(): {e}")
            return "An error occurred while processing your question. Please try again."
    
    def _call_llm(self, prompt: str, question: str, context: str, max_retries: int = 2) -> str:
        """Call LLM API with retry logic"""
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://chatbot.com",
        }
        
        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Answer questions accurately based on the provided context. Be specific, friendly, and professional."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.5,
            "max_tokens": 800,
            "top_p": 0.9
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
                    return None
            
            except Exception as e:
                print(f"LLM call error (attempt {attempt + 1}): {e}")
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
# STREAMLIT UI
# ============================================================================

st.set_page_config(
    page_title="AI Chatbot Generator",
    page_icon="🤖",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%); }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        text-align: center;
    }
    .header-title { color: white; font-size: 2.5rem; font-weight: bold; margin: 0; }
    .header-subtitle { color: rgba(255,255,255,0.9); font-size: 1.1rem; margin-top: 0.5rem; }
    .company-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .status-completed { background: #10b981; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; }
    .status-pending { background: #6b7280; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; }
    .status-failed { background: #ef4444; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; }
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

# Sidebar
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()
    
    if st.button("➕ Create Chatbot", use_container_width=True):
        st.session_state.page = "create"
        st.rerun()
    
    if st.button("📋 All Chatbots", use_container_width=True):
        st.session_state.page = "list"
        st.rerun()

# Page routing
if st.session_state.page == "home":
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🤖 AI Chatbot Generator</h1>
        <p class="header-subtitle">Create intelligent chatbots in minutes!</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("➕ Create New Chatbot", use_container_width=True, type="primary"):
            st.session_state.page = "create"
            st.rerun()
    
    with col2:
        if st.button("📋 View Chatbots", use_container_width=True):
            st.session_state.page = "list"
            st.rerun()

elif st.session_state.page == "create":
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">➕ Create New Chatbot</h1>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("create_form"):
        company_name = st.text_input("Company Name *", placeholder="Acme Corp")
        website_url = st.text_input("Website URL *", placeholder="https://example.com")
        max_pages = st.slider("Pages to Scrape", 10, 80, 40, 10)
        
        submit = st.form_submit_button("🚀 Create", use_container_width=True, type="primary")
        
        if submit:
            if not company_name or not website_url:
                st.error("Fill all fields!")
            else:
                if not website_url.startswith('http'):
                    website_url = 'https://' + website_url
                
                slug = create_company(company_name, website_url, max_pages)
                
                if not slug:
                    st.error("Company already exists!")
                else:
                    st.success(f"✅ Created: {company_name}")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def progress_callback(current, total, url):
                        progress_bar.progress(min(current / total, 1.0))
                        status_text.text(f"📄 Page {current}/{total}")
                    
                    ai = CompanyAI(slug)
                    success = ai.initialize(website_url, max_pages, progress_callback)
                    
                    if success:
                        update_company_after_scraping(slug, ai.company_data, "completed")
                        progress_bar.progress(1.0)
                        st.success("✅ Ready!")
                        st.balloons()
                        time.sleep(2)
                        
                        st.session_state.current_company = slug
                        st.session_state.ai_instance = ai
                        st.session_state.page = "chat"
                        st.session_state.messages = []
                        st.session_state.chat_history = []
                        st.rerun()
                    else:
                        update_company_after_scraping(slug, {}, "failed")
                        st.error("❌ Initialization failed")
                        with st.expander("Error Details"):
                            st.code(ai.status.get('error', 'Unknown'))

elif st.session_state.page == "list":
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">📋 All Chatbots</h1>
    </div>
    """, unsafe_allow_html=True)
    
    companies = get_all_companies()
    
    if not companies:
        st.info("No chatbots yet. Create one!")
    else:
        for company in companies:
            st.markdown('<div class="company-card">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([3, 2, 2])
            
            with col1:
                st.markdown(f"### {company.company_name}")
                st.caption(f"🌐 {company.website_url}")
            
            with col2:
                st.markdown(f'<span class="status-{company.scraping_status}">{company.scraping_status.upper()}</span>', unsafe_allow_html=True)
            
            with col3:
                if company.scraping_status == "completed":
                    if st.button("💬 Chat", key=f"chat_{company.id}", type="primary"):
                        st.session_state.current_company = company.company_slug
                        st.session_state.page = "chat"
                        st.session_state.messages = []
                        st.session_state.chat_history = []
                        st.session_state.ai_instance = None
                        st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "chat":
    if not st.session_state.current_company:
        st.warning("No company selected")
        st.stop()
    
    company = get_company_by_slug(st.session_state.current_company)
    
    if not company or company.scraping_status != "completed":
        st.error("Chatbot not ready")
        st.stop()
    
    # Load AI
    if not st.session_state.ai_instance:
        with st.spinner("Loading AI..."):
            ai = CompanyAI(st.session_state.current_company)
            if not ai.load_existing():
                st.error("Failed to load AI")
                st.stop()
            st.session_state.ai_instance = ai
    
    st.markdown(f"""
    <div class="header-container">
        <h1 class="header-title">💬 Chat with {company.company_name}</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if user_input := st.chat_input("Ask anything..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.ai_instance.ask(
                    user_input,
                    st.session_state.chat_history
                )
            st.markdown(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_history.append({"question": user_input, "answer": answer})
