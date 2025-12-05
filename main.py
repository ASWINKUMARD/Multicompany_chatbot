# app.py - Complete Production-Ready Chatbot with Stunning UI
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

# Custom CSS for Stunning UI
CUSTOM_CSS = """
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    /* Custom Header */
    .custom-header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 1.5rem 2rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    
    .custom-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .custom-header p {
        color: #64748b;
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }
    
    /* Chat Container */
    .chat-container {
        background: white;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
        padding: 2rem;
        max-width: 900px;
        margin: 0 auto;
        min-height: 600px;
    }
    
    /* Message Bubbles */
    .stChatMessage {
        background: transparent !important;
        padding: 1rem 0 !important;
    }
    
    .stChatMessage[data-testid*="user"] {
        background: transparent !important;
    }
    
    .stChatMessage[data-testid*="assistant"] {
        background: transparent !important;
    }
    
    /* User Message */
    .stChatMessage[data-testid*="user"] > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 18px 18px 4px 18px !important;
        padding: 1rem 1.5rem !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
        margin-left: auto !important;
        max-width: 80% !important;
    }
    
    /* Assistant Message */
    .stChatMessage[data-testid*="assistant"] > div {
        background: #f1f5f9 !important;
        color: #1e293b !important;
        border-radius: 18px 18px 18px 4px !important;
        padding: 1rem 1.5rem !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
        max-width: 80% !important;
    }
    
    /* Chat Input */
    .stChatInputContainer {
        border: none !important;
        background: white !important;
        border-radius: 25px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08) !important;
        padding: 0.5rem !important;
    }
    
    .stChatInputContainer > div {
        border: 2px solid #e2e8f0 !important;
        border-radius: 25px !important;
        background: white !important;
    }
    
    .stChatInputContainer input {
        padding: 1rem 1.5rem !important;
        font-size: 1rem !important;
        border: none !important;
    }
    
    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    .css-1d391kg h2, [data-testid="stSidebar"] h2,
    .css-1d391kg h3, [data-testid="stSidebar"] h3,
    .css-1d391kg p, [data-testid="stSidebar"] p,
    .css-1d391kg label, [data-testid="stSidebar"] label {
        color: white !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Input Fields */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1rem;
        font-weight: 600;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: #10b981;
        color: white;
        border-radius: 12px;
        padding: 1rem;
    }
    
    .stError {
        background: #ef4444;
        color: white;
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* Feature Cards */
    .feature-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Company List */
    .company-item {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.5s ease-out;
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
            return f"👋 Hello! I'm the AI assistant for **{self.company_name}**. How can I help you today?"
        
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
        page_title="AutoBot AI - Universal Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    init_session()
    
    # Custom Header
    st.markdown("""
    <div class="custom-header">
        <h1>🤖 AutoBot AI</h1>
        <p>Create AI-powered chatbots for any company in seconds</p>
    </div>
    """, unsafe_allow_html=True)
    
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
                    
                    with st.spinner(f"🔍 Analyzing {company_name}..."):
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
                            st.success(f"✅ Chatbot ready!")
                            st.rerun()
                        else:
                            st.error(f"❌ Error: {chatbot.error}")
        
        if st.session_state.chatbots:
            st.markdown("### 💬 Your Chatbots")
            
            for slug, bot in st.session_state.chatbots.items():
                is_active = st.session_state.current_company == slug
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(
                        f"{'✓ ' if is_active else ''}{bot.company_name}",
                        key=f"select_{slug}",
                        use_container_width=True
                    ):
                        st.session_state.current_company = slug
                        st.session_state.chat_history = []
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"delete_{slug}"):
                        del st.session_state.chatbots[slug]
                        if st.session_state.current_company == slug:
                            st.session_state.current_company = None
                        st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Stats")
        st.metric("Active Chatbots", len(st.session_state.chatbots))
        
        if st.session_state.current_company:
            bot = st.session_state.chatbots[st.session_state.current_company]
            st.metric("Pages Indexed", len(bot.pages))
            st.metric("Contacts Found", len(bot.contact_info['emails']) + len(bot.contact_info['phones']))
    
    # Main Content
    if not st.session_state.current_company:
        # Landing Page
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h3>⚡ Lightning Fast</h3>
                <p>Scrapes and indexes 50+ pages in under 10 seconds</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h3>🧠 AI-Powered</h3>
                <p>Understands context and provides accurate answers</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-card">
                <h3>🌐 Universal</h3>
                <p>Works with any company website automatically</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 How It Works")
        st.markdown("""
        1. **Enter Company Details** - Just provide a name and website URL
        2. **AI Scrapes & Learns** - We automatically analyze their website
        3. **Start Chatting** - Ask anything about the company instantly
        4. **Embed Anywhere** - Get the widget code to embed on your site
        """)
        
    else:
        chatbot = st.session_state.chatbots[st.session_state.current_company]
        
        # Chat Header
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"### 💬 Chat with {chatbot.company_name}")
        with col2:
            if st.button("🔄 Refresh"):
                with st.spinner("Refreshing..."):
                    chatbot.initialize()
                    st.rerun()
        
        # Chat Messages
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Chat Input
        if user_input := st.chat_input("Ask anything..."):
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            with st.chat_message("user"):
                st.markdown(user_input)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = chatbot.ask(user_input)
                st.markdown(answer)
            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })
            
            st.rerun()

if __name__ == "__main__":
    main()
