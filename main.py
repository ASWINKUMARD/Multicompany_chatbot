# app.py - Stunning Chatbot UI matching React sample design
import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import os
import hashlib
import time
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "kwaipilot/kat-coder-pro:free"

# Enhanced CSS matching the React sample design
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 0;
    }
    
    .block-container {
        max-width: 1200px;
        padding: 2rem 1rem;
    }
    
    /* Glassmorphism card effect */
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 
            0 8px 32px 0 rgba(31, 38, 135, 0.15),
            0 2px 8px 0 rgba(31, 38, 135, 0.1);
        padding: 0;
        overflow: hidden;
        max-width: 900px;
        margin: 0 auto;
    }
    
    /* Chat header - matching React design */
    .chat-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.25rem 1.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .bot-avatar {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        position: relative;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    .status-indicator {
        position: absolute;
        bottom: 2px;
        right: 2px;
        width: 12px;
        height: 12px;
        background: #10b981;
        border: 2px solid white;
        border-radius: 50%;
        animation: pulse-dot 2s cubic-bezier(0.455, 0.03, 0.515, 0.955) infinite;
    }
    
    @keyframes pulse-dot {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .header-text {
        flex: 1;
        color: white;
    }
    
    .header-text h3 {
        margin: 0;
        font-size: 1.125rem;
        font-weight: 600;
        line-height: 1.2;
    }
    
    .header-text p {
        margin: 0.25rem 0 0 0;
        font-size: 0.75rem;
        opacity: 0.9;
        display: flex;
        align-items: center;
        gap: 0.375rem;
    }
    
    .online-dot {
        width: 6px;
        height: 6px;
        background: #10b981;
        border-radius: 50%;
        display: inline-block;
    }
    
    /* Messages area with subtle gradient */
    .messages-container {
        background: linear-gradient(180deg, #fafbfc 0%, #f5f7fa 100%);
        padding: 1.5rem;
        min-height: 500px;
        max-height: 600px;
        overflow-y: auto;
    }
    
    /* Message bubbles - matching React design exactly */
    .stChatMessage {
        background: transparent !important;
        padding: 0.75rem 0 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* User message styling */
    div[data-testid="stChatMessageContent"] {
        padding: 0 !important;
        background: transparent !important;
    }
    
    .stChatMessage[data-testid*="user-message"] {
        display: flex;
        flex-direction: row-reverse;
    }
    
    .stChatMessage[data-testid*="user-message"] > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 18px 18px 4px 18px !important;
        padding: 0.875rem 1.25rem !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
        max-width: 80% !important;
        margin-left: auto !important;
        font-size: 0.9375rem !important;
        line-height: 1.6 !important;
    }
    
    /* Assistant message styling */
    .stChatMessage[data-testid*="assistant-message"] > div {
        background: white !important;
        color: #1e293b !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 18px 18px 18px 4px !important;
        padding: 0.875rem 1.25rem !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06) !important;
        max-width: 80% !important;
        font-size: 0.9375rem !important;
        line-height: 1.6 !important;
    }
    
    /* Avatar styling for messages */
    .stChatMessage > div:first-child {
        width: 32px !important;
        height: 32px !important;
        border-radius: 50% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        flex-shrink: 0 !important;
        font-size: 16px !important;
    }
    
    /* User avatar */
    .stChatMessage[data-testid*="user-message"] > div:first-child {
        background: white !important;
        border: 1px solid #e2e8f0 !important;
        color: #64748b !important;
    }
    
    /* Assistant avatar */
    .stChatMessage[data-testid*="assistant-message"] > div:first-child {
        background: rgba(102, 126, 234, 0.1) !important;
        color: #667eea !important;
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: flex;
        gap: 0.25rem;
        padding: 1rem;
        background: white;
        border-radius: 18px 18px 18px 4px;
        border: 1px solid #e2e8f0;
        width: fit-content;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    
    .typing-dot {
        width: 6px;
        height: 6px;
        background: rgba(102, 126, 234, 0.5);
        border-radius: 50%;
        animation: typing 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-dot:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-8px); }
    }
    
    /* Input area - matching React design */
    .input-container {
        padding: 1.25rem 1.5rem;
        background: rgba(248, 250, 252, 0.8);
        backdrop-filter: blur(10px);
        border-top: 1px solid #e2e8f0;
    }
    
    .stChatInputContainer {
        background: white !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 16px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.3s ease !important;
        padding: 0 !important;
    }
    
    .stChatInputContainer:focus-within {
        border-color: rgba(102, 126, 234, 0.4) !important;
        box-shadow: 
            0 2px 8px rgba(0, 0, 0, 0.04),
            0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    .stChatInputContainer input {
        padding: 0.875rem 3rem 0.875rem 1.25rem !important;
        font-size: 0.9375rem !important;
        border: none !important;
        background: transparent !important;
    }
    
    .stChatInputContainer input::placeholder {
        color: #94a3b8 !important;
    }
    
    /* Send button styling */
    .stChatInputContainer button {
        position: absolute !important;
        right: 4px !important;
        top: 4px !important;
        width: 40px !important;
        height: 40px !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        color: white !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 0 10px -3px #667eea !important;
    }
    
    .stChatInputContainer button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 0 15px -2px #667eea !important;
    }
    
    /* Footer branding - matching React design */
    .chat-footer {
        text-align: center;
        padding: 0.75rem 1.5rem;
        background: rgba(248, 250, 252, 0.6);
        border-top: 1px solid rgba(226, 232, 240, 0.6);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        font-size: 0.625rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
    }
    
    .sparkles-icon {
        width: 12px;
        height: 12px;
        opacity: 0.5;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        color: white !important;
        font-weight: 500 !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255, 255, 255, 0.3) !important;
        transform: translateY(-1px);
    }
    
    section[data-testid="stSidebar"] .stTextInput input {
        background: rgba(255, 255, 255, 0.9) !important;
        color: #1e293b !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }
    
    section[data-testid="stSidebar"] .stTextInput input::placeholder {
        color: #64748b !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Success/Error/Info messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 12px !important;
        border: none !important;
        padding: 1rem !important;
        font-weight: 500 !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    
    /* Smooth animations */
    * {
        transition: background-color 0.2s ease, border-color 0.2s ease;
    }
</style>
"""

class FastScraper:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        self.timeout = 6
        
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"@+/#]', '', text)
        return text.strip()
    
    def extract_contact_info(self, text: str) -> Dict:
        emails = set()
        phones = set()
        
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for email in re.findall(email_pattern, text):
            if not email.lower().endswith(('.png', '.jpg', '.gif', '.css', '.js')):
                emails.add(email.lower())
        
        phone_patterns = [
            r'\+\d{1,3}[\s.-]?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{4}',
            r'\d{4}[\s.-]\d{4}',
            r'\(\d{2,4}\)\s*\d{4}[\s.-]\d{4}',
        ]
        for pattern in phone_patterns:
            for phone in re.findall(pattern, text):
                cleaned = re.sub(r'[^\d+()]', '', phone)
                if 7 <= len(cleaned) <= 20:
                    phones.add(phone.strip())
        
        return {
            "emails": sorted(list(emails))[:5],
            "phones": sorted(list(phones))[:5]
        }
    
    def scrape_page(self, url: str) -> Optional[Dict]:
        try:
            resp = requests.get(url, headers=self.headers, timeout=self.timeout, allow_redirects=True)
            if resp.status_code != 200:
                return None
            
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'iframe', 'noscript']):
                tag.decompose()
            
            title = soup.find('title').get_text(strip=True) if soup.find('title') else ""
            
            content = ""
            for selector in ['main', 'article', '[role="main"]', '.main-content', '#main']:
                elements = soup.select(selector)
                if elements:
                    content = elements[0].get_text(separator='\n', strip=True)
                    if len(content) > 200:
                        break
            
            if len(content) < 200:
                texts = []
                for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'li']):
                    text = tag.get_text(strip=True)
                    if len(text) > 20:
                        texts.append(text)
                content = '\n'.join(texts)
            
            lines = []
            seen = set()
            for line in content.split('\n'):
                line = self.clean_text(line)
                if len(line) > 25 and line.lower() not in seen:
                    lines.append(line)
                    seen.add(line.lower())
                if len(lines) >= 50:
                    break
            
            content = '\n'.join(lines)
            
            if len(content) < 100:
                return None
            
            return {
                "url": url,
                "title": title[:200],
                "content": content[:4000]
            }
            
        except Exception as e:
            return None
    
    def get_urls_to_scrape(self, base_url: str) -> List[str]:
        if not base_url.startswith('http'):
            base_url = 'https://' + base_url
        base_url = base_url.rstrip('/')
        
        paths = ['', '/about', '/about-us', '/services', '/products',
                '/contact', '/contact-us', '/pricing', '/solutions']
        
        urls = [f"{base_url}{path}" for path in paths]
        
        try:
            resp = requests.get(base_url, headers=self.headers, timeout=self.timeout)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                domain = urlparse(base_url).netloc
                
                for link in soup.find_all('a', href=True)[:60]:
                    href = link['href']
                    full_url = urljoin(base_url, href)
                    
                    if (urlparse(full_url).netloc == domain and 
                        not any(full_url.lower().endswith(ext) for ext in ['.pdf', '.jpg', '.png'])):
                        if full_url not in urls:
                            urls.append(full_url)
        except:
            pass
        
        return urls[:50]
    
    def scrape_website(self, base_url: str, progress_callback=None) -> Tuple[List[Dict], Dict]:
        urls = self.get_urls_to_scrape(base_url)
        pages = []
        all_text = ""
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {executor.submit(self.scrape_page, url): url for url in urls}
            
            completed = 0
            for future in as_completed(future_to_url):
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(urls), future_to_url[future])
                
                try:
                    result = future.result()
                    if result:
                        pages.append(result)
                        all_text += "\n" + result['content']
                except:
                    pass
        
        contact_info = self.extract_contact_info(all_text)
        
        if len(pages) == 0:
            raise Exception(f"Could not scrape any content from {base_url}")
        
        return pages, contact_info

class SmartAI:
    def __init__(self):
        self.response_cache = {}
        
    def call_llm(self, prompt: str) -> str:
        if not OPENROUTER_API_KEY:
            return "⚠️ API key not set. Please configure OPENROUTER_API_KEY."
        
        cache_key = hashlib.md5(prompt.encode()).hexdigest()[:16]
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        
        for attempt in range(2):
            try:
                payload = {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 400,
                }
                
                resp = requests.post(OPENROUTER_API_BASE, headers=headers, json=payload, timeout=45)
                
                if resp.status_code == 200:
                    data = resp.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        content = data["choices"][0].get("message", {}).get("content", "")
                        if content:
                            self.response_cache[cache_key] = content.strip()
                            return content.strip()
                elif resp.status_code == 429:
                    time.sleep(3)
                    continue
                    
            except:
                if attempt < 1:
                    time.sleep(2)
                    continue
        
        return "I'm having trouble connecting right now. Try asking about contact information!"

class UniversalChatbot:
    def __init__(self, company_name: str, website_url: str):
        self.company_name = company_name
        self.website_url = website_url
        self.pages = []
        self.contact_info = {"emails": [], "phones": []}
        self.ready = False
        self.error = None
        self.ai = SmartAI()
        
    def initialize(self, progress_callback=None):
        try:
            scraper = FastScraper()
            self.pages, self.contact_info = scraper.scrape_website(self.website_url, progress_callback)
            self.ready = True
            return True
        except Exception as e:
            self.error = str(e)
            return False
    
    def get_context(self, question: str) -> str:
        if not self.pages:
            return ""
        
        question_words = set(re.findall(r'\w+', question.lower()))
        question_words = {w for w in question_words if len(w) > 3}
        
        scored_pages = []
        for page in self.pages:
            content_words = set(re.findall(r'\w+', page['content'].lower()))
            score = len(question_words & content_words)
            if score > 0:
                scored_pages.append((score, page))
        
        scored_pages.sort(reverse=True, key=lambda x: x[0])
        
        context_parts = []
        for score, page in scored_pages[:5]:
            context_parts.append(page['content'][:1000])
        
        return "\n\n---\n\n".join(context_parts)
    
    def ask(self, question: str) -> str:
        if not self.ready:
            return "⚠️ Chatbot not initialized yet."
        
        question_lower = question.lower().strip()
        
        greeting_words = ['hi', 'hello', 'hey', 'hai']
        if any(question_lower == g or question_lower.startswith(g + ' ') for g in greeting_words):
            return f"👋 Hi there! I'm the AI assistant for {self.company_name}. How can I help you today?"
        
        contact_keywords = ['email', 'contact', 'phone', 'call', 'reach']
        if any(kw in question_lower for kw in contact_keywords):
            msg = f"📞 **Contact Information for {self.company_name}**\n\n"
            
            if self.contact_info['emails']:
                msg += "📧 **Email:**\n" + "\n".join([f"• {e}" for e in self.contact_info['emails']]) + "\n\n"
            
            if self.contact_info['phones']:
                msg += "📱 **Phone:**\n" + "\n".join([f"• {p}" for p in self.contact_info['phones']]) + "\n\n"
            
            if self.website_url:
                msg += f"🌐 **Website:** {self.website_url}"
            
            return msg.strip()
        
        context = self.get_context(question)
        
        if not context or len(context) < 50:
            all_content = "\n".join([p['content'][:500] for p in self.pages[:3]])
            if all_content:
                context = all_content
        
        prompt = f"""You are a helpful AI assistant for {self.company_name}.

Based on the following information, answer the user's question clearly.

INFORMATION:
{context[:2500]}

USER QUESTION: {question}

Provide a helpful, conversational answer in 2-4 sentences. Be specific and friendly.

Answer:"""

        return self.ai.call_llm(prompt)

def init_session():
    if 'chatbots' not in st.session_state:
        st.session_state.chatbots = {}
    if 'current_company' not in st.session_state:
        st.session_state.current_company = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def main():
    st.set_page_config(
        page_title="AutoBot AI",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    init_session()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🏢 Company Management")
        
        with st.expander("➕ Add New Company", expanded=True):
            company_name = st.text_input("Company Name", placeholder="e.g., Acme Corp")
            website_url = st.text_input("Website URL", placeholder="https://example.com")
            
            if st.button("🚀 Create Chatbot", type="primary", use_container_width=True):
                if not company_name or not website_url:
                    st.warning("Please fill in all fields")
                else:
                    slug = re.sub(r'[^a-z0-9]+', '-', company_name.lower()).strip('-')
                    
                    with st.spinner(f"Analyzing {company_name}..."):
                        progress = st.progress(0)
                        status = st.empty()
                        
                        def callback(done, total, url):
                            progress.progress(done / max(total, 1))
                            status.text(f"Scraping {done}/{total} pages...")
                        
                        chatbot = UniversalChatbot(company_name, website_url)
                        success = chatbot.initialize(callback)
                        
                        if success:
                            st.session_state.chatbots[slug] = chatbot
                            st.session_state.current_company = slug
                            st.session_state.chat_history = []
                            st.success("✅ Chatbot ready!")
                            st.rerun()
                        else:
                            st.error(f"❌ {chatbot.error}")
        
        if st.session_state.chatbots:
            st.markdown("### 💬 Active Chatbots")
            
            for slug, bot in st.session_state.chatbots.items():
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(bot.company_name, key=f"select_{slug}", use_container_width=True):
                        st.session_state.current_company = slug
                        st.session_state.chat_history = []
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"delete_{slug}"):
                        del st.session_state.chatbots[slug]
                        if st.session_state.current_company == slug:
                            st.session_state.current_company = None
                        st.rerun()
    
    # Main Content
    if st.session_state.current_company:
        chatbot = st.session_state.chatbots[st.session_state.current_company]
        
        # Chat container with header
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Chat header
        st.markdown(f"""
        <div class="chat-header">
            <div class="bot-avatar">
                🤖
                <span class="status-indicator"></span>
            </div>
            <div class="header-text">
                <h3>{chatbot.company_name} Assistant</h3>
                <p><span class="online-dot"></span> Replies instantly</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
# Messages container
        st.markdown('<div class="messages-container">', unsafe_allow_html=True)
        
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
                st.markdown(msg["content"])
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Input container
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        # Chat input
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Get bot response
            with st.spinner(""):
                response = chatbot.ask(user_input)
            
            # Add assistant response to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })
            
            # Rerun to display new messages
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat footer
        st.markdown("""
        <div class="chat-footer">
            <svg class="sparkles-icon" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0L14.5 8.5L23 11L14.5 13.5L12 22L9.5 13.5L1 11L9.5 8.5L12 0Z"/>
            </svg>
            <span>Powered by AI</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional info below chat
        with st.expander("ℹ️ Chatbot Information"):
            st.write(f"**Company:** {chatbot.company_name}")
            st.write(f"**Website:** {chatbot.website_url}")
            st.write(f"**Pages Scraped:** {len(chatbot.pages)}")
            st.write(f"**Emails Found:** {len(chatbot.contact_info['emails'])}")
            st.write(f"**Phones Found:** {len(chatbot.contact_info['phones'])}")
            
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
    
    else:
        # Welcome screen when no chatbot is selected
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem;">
            <div style="font-size: 5rem; margin-bottom: 1rem;">🤖</div>
            <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       font-size: 3rem; font-weight: 800; margin-bottom: 1rem;">
                AutoBot AI
            </h1>
            <p style="font-size: 1.25rem; color: #64748b; margin-bottom: 2rem;">
                Create intelligent chatbots for any company website in seconds
            </p>
            <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                        border-radius: 16px; padding: 2rem; max-width: 600px; margin: 0 auto;
                        border: 1px solid rgba(102, 126, 234, 0.2);">
                <h3 style="color: #1e293b; margin-bottom: 1rem;">✨ Features</h3>
                <ul style="text-align: left; color: #475569; line-height: 2;">
                    <li>🚀 Instant chatbot creation from any website</li>
                    <li>🔍 Automatic content extraction and analysis</li>
                    <li>📞 Smart contact information detection</li>
                    <li>💬 Natural conversation with AI</li>
                    <li>⚡ Lightning-fast responses</li>
                </ul>
            </div>
            <p style="margin-top: 2rem; color: #94a3b8; font-size: 0.875rem;">
                👈 Get started by creating a new chatbot in the sidebar
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
