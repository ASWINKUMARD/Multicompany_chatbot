"""
UNIVERSAL AI CHATBOT - READY FOR ANY COMPANY
Fast, intelligent chatbot that works with ANY company website automatically

FEATURES:
1. No pre-configuration needed - works with any company
2. Fast parallel scraping (5-10 seconds)
3. Smart content extraction and caching
4. Instant AI responses
5. Automatic contact info detection
6. In-memory storage (no database issues)

Run with:
    streamlit run app.py
    
Set environment variable:
    export OPENROUTER_API_KEY="your_key_here"
"""

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
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"

# Fast, reliable models
MODEL = "kwaipilot/kat-coder-pro:free"

# =============================================================================
# FAST PARALLEL WEB SCRAPER
# =============================================================================
class FastScraper:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        self.timeout = 8
        
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"@+/#]', '', text)
        return text.strip()
    
    def extract_contact_info(self, text: str) -> Dict:
        """Extract emails and phones from text"""
        emails = set()
        phones = set()
        
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for email in re.findall(email_pattern, text):
            if not email.lower().endswith(('.png', '.jpg', '.gif', '.css', '.js')):
                emails.add(email.lower())
        
        # Extract phones - multiple patterns
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
            "emails": sorted(list(emails))[:2],  # Limit to 5
            "phones": sorted(list(phones))[:2]
        }
    
    def scrape_page(self, url: str) -> Optional[Dict]:
        """Scrape a single page - fast and efficient"""
        try:
            resp = requests.get(url, headers=self.headers, timeout=self.timeout, allow_redirects=True)
            if resp.status_code != 200:
                return None
            
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # Remove noise
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'iframe', 'noscript']):
                tag.decompose()
            
            # Get title
            title = ""
            if soup.find('title'):
                title = soup.find('title').get_text(strip=True)
            
            # Get main content - try multiple strategies
            content = ""
            
            # Strategy 1: Look for main content containers
            for selector in ['main', 'article', '[role="main"]', '.main-content', '#main', '.content', '#content']:
                elements = soup.select(selector)
                if elements:
                    content = elements[0].get_text(separator='\n', strip=True)
                    if len(content) > 200:
                        break
            
            # Strategy 2: Get all paragraphs and headings
            if len(content) < 200:
                texts = []
                for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'li']):
                    text = tag.get_text(strip=True)
                    if len(text) > 20:
                        texts.append(text)
                content = '\n'.join(texts)
            
            # Strategy 3: Fallback to body
            if len(content) < 200:
                body = soup.find('body')
                if body:
                    content = body.get_text(separator='\n', strip=True)
            
            # Clean and filter
            lines = []
            seen = set()
            for line in content.split('\n'):
                line = self.clean_text(line)
                if len(line) > 25 and line.lower() not in seen:
                    lines.append(line)
                    seen.add(line.lower())
                if len(lines) >= 40:  # Limit lines
                    break
            
            content = '\n'.join(lines)
            
            if len(content) < 100:
                return None
            
            return {
                "url": url,
                "title": title[:200],
                "content": content[:3000]  # Limit content size
            }
            
        except Exception as e:
            print(f"[SCRAPER] Error {url}: {str(e)[:100]}")
            return None
    
    def get_urls_to_scrape(self, base_url: str) -> List[str]:
        """Get list of important URLs to scrape"""
        if not base_url.startswith('http'):
            base_url = 'https://' + base_url
        base_url = base_url.rstrip('/')
        
        # Important paths to check
        paths = [
            '', '/about', '/about-us', '/services', '/products',
            '/contact', '/contact-us', '/pricing', '/solutions',
            '/home', '/index.html', '/company', '/team'
        ]
        
        urls = [f"{base_url}{path}" for path in paths]
        
        # Try to discover more URLs from homepage
        try:
            resp = requests.get(base_url, headers=self.headers, timeout=self.timeout)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                domain = urlparse(base_url).netloc
                
                for link in soup.find_all('a', href=True)[:30]:  # Limit to 30 links
                    href = link['href']
                    full_url = urljoin(base_url, href)
                    
                    # Only same domain, no files
                    if (urlparse(full_url).netloc == domain and 
                        not any(full_url.lower().endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.zip', '.gif'])):
                        if full_url not in urls:
                            urls.append(full_url)
        except:
            pass
        
        return urls[:40]  # Limit to 15 URLs max
    
    def scrape_website(self, base_url: str, progress_callback=None) -> Tuple[List[Dict], Dict]:
        """Scrape website in parallel - FAST!"""
        start_time = time.time()
        
        urls = self.get_urls_to_scrape(base_url)
        print(f"[SCRAPER] Scraping {len(urls)} URLs from {base_url}")
        
        pages = []
        all_text = ""
        
        # Parallel scraping with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=5) as executor:
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
                except Exception as e:
                    print(f"[SCRAPER] Future error: {e}")
        
        # Extract contact info from all text
        contact_info = self.extract_contact_info(all_text)
        
        elapsed = time.time() - start_time
        print(f"[SCRAPER] Completed in {elapsed:.1f}s - {len(pages)} pages scraped")
        
        if len(pages) == 0:
            raise Exception(f"Could not scrape any content from {base_url}")
        
        return pages, contact_info

# =============================================================================
# SMART AI RESPONDER WITH CACHING
# =============================================================================
class SmartAI:
    def __init__(self):
        self.response_cache = {}  # Cache responses
        
    def call_llm(self, prompt: str) -> str:
        """Call LLM with smart fallbacks and better error handling"""
        
        if not OPENROUTER_API_KEY:
            return "⚠️ API key not set. Please configure OPENROUTER_API_KEY."
        
        # Check cache
        cache_key = hashlib.md5(prompt.encode()).hexdigest()[:16]
        if cache_key in self.response_cache:
            print("[AI] Using cached response")
            return self.response_cache[cache_key]
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Universal Chatbot"
        }
        
        # Try multiple models
        for model_idx, model in enumerate(MODELS):
            for attempt in range(2):  # 2 attempts per model
                try:
                    print(f"[AI] Model {model_idx+1}/{len(MODELS)}: {model} (attempt {attempt+1})")
                    
                    payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "max_tokens": 400,
                    }
                    
                    resp = requests.post(
                        OPENROUTER_API_BASE,
                        headers=headers,
                        json=payload,
                        timeout=45
                    )
                    
                    print(f"[AI] Status: {resp.status_code}")
                    
                    if resp.status_code == 200:
                        try:
                            data = resp.json()
                            print(f"[AI] Response keys: {list(data.keys())}")
                            
                            # Check for API error in response
                            if "error" in data:
                                error_msg = data["error"].get("message", str(data["error"]))
                                print(f"[AI] API Error: {error_msg}")
                                # Try next model
                                break
                            
                            if "choices" in data and len(data["choices"]) > 0:
                                content = data["choices"][0].get("message", {}).get("content", "")
                                if content and len(content.strip()) > 5:
                                    print(f"[AI] ✅ Success! Length: {len(content)}")
                                    # Cache the response
                                    self.response_cache[cache_key] = content.strip()
                                    return content.strip()
                                else:
                                    print(f"[AI] Empty content received")
                            else:
                                print(f"[AI] No choices in response: {data}")
                        
                        except Exception as e:
                            print(f"[AI] JSON parse error: {e}")
                            print(f"[AI] Raw response: {resp.text[:300]}")
                    
                    elif resp.status_code == 401:
                        return "⚠️ Invalid API key. Get one from https://openrouter.ai/keys"
                    
                    elif resp.status_code == 402:
                        print(f"[AI] No credits remaining")
                        # Try next model
                        break
                    
                    elif resp.status_code == 429:
                        print(f"[AI] Rate limited, waiting...")
                        time.sleep(3)
                        continue
                    
                    else:
                        print(f"[AI] Error {resp.status_code}: {resp.text[:200]}")
                    
                    # Retry with same model
                    if attempt < 1:
                        time.sleep(2)
                        continue
                    
                except requests.exceptions.Timeout:
                    print(f"[AI] Timeout with {model}")
                    if attempt < 1:
                        time.sleep(2)
                        continue
                
                except Exception as e:
                    print(f"[AI] Exception with {model}: {type(e).__name__}: {str(e)[:100]}")
                    if attempt < 1:
                        time.sleep(2)
                        continue
            
            # Small delay before trying next model
            if model_idx < len(MODELS) - 1:
                time.sleep(1)
        
        # If all models fail, return helpful fallback
        return "I'm having trouble connecting to the AI service right now. However, I can still help! Try asking about contact information, or visit the company website for more details."

# =============================================================================
# UNIVERSAL COMPANY CHATBOT
# =============================================================================
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
        """Initialize by scraping the website"""
        try:
            scraper = FastScraper()
            self.pages, self.contact_info = scraper.scrape_website(
                self.website_url,
                progress_callback
            )
            self.ready = True
            return True
        except Exception as e:
            self.error = str(e)
            print(f"[CHATBOT] Init error: {e}")
            return False
    
    def get_context(self, question: str) -> str:
        """Get relevant context for the question"""
        if not self.pages:
            return ""
        
        # Simple keyword matching
        question_words = set(re.findall(r'\w+', question.lower()))
        question_words = {w for w in question_words if len(w) > 3}
        
        scored_pages = []
        for page in self.pages:
            content_words = set(re.findall(r'\w+', page['content'].lower()))
            score = len(question_words & content_words)
            if score > 0:
                scored_pages.append((score, page))
        
        scored_pages.sort(reverse=True, key=lambda x: x[0])
        
        # Get top 3 pages
        context_parts = []
        for score, page in scored_pages[:3]:
            context_parts.append(page['content'][:800])
        
        return "\n\n---\n\n".join(context_parts)
    
    def ask(self, question: str) -> str:
        """Answer a question about the company"""
        
        if not self.ready:
            return "⚠️ Chatbot not initialized yet."
        
        question_lower = question.lower().strip()
        
        # Handle greetings
        greeting_words = ['hi', 'hello', 'hey', 'hai', 'good morning', 'good afternoon', 'good evening']
        if any(question_lower == g or question_lower.startswith(g + ' ') for g in greeting_words):
            return f"👋 Hello! I'm an AI assistant for **{self.company_name}**. I can help you with:\n\n• Information about their services/products\n• Contact details\n• Pricing and offerings\n• Any questions about the company\n\nWhat would you like to know?"
        
        # Handle contact info requests
        contact_keywords = ['email', 'contact', 'phone', 'call', 'reach', 'address', 'location', 'office']
        if any(kw in question_lower for kw in contact_keywords):
            msg = f"📞 **Contact Information for {self.company_name}**\n\n"
            
            if self.contact_info['emails']:
                msg += "📧 **Email:**\n" + "\n".join([f"• {e}" for e in self.contact_info['emails']]) + "\n\n"
            
            if self.contact_info['phones']:
                msg += "📱 **Phone:**\n" + "\n".join([f"• {p}" for p in self.contact_info['phones']]) + "\n\n"
            
            if self.website_url:
                msg += f"🌐 **Website:** {self.website_url}"
            
            if not self.contact_info['emails'] and not self.contact_info['phones']:
                msg += f"Visit their website at {self.website_url} for contact details."
            
            return msg.strip()
        
        # Get relevant context
        context = self.get_context(question)
        
        if not context or len(context) < 50:
            # Try to give a general answer from all content
            all_content = "\n".join([p['content'][:500] for p in self.pages[:3]])
            if all_content:
                context = all_content
            else:
                return f"I don't have specific information about that yet. Please visit {self.website_url} or try asking about their services, contact info, or general company information."
        
        # Build prompt
        prompt = f"""You are a helpful AI assistant for {self.company_name}.

Based on the following information from their website, answer the user's question clearly and naturally.

COMPANY INFORMATION:
{context[:2500]}

USER QUESTION: {question}

Instructions:
- Provide a helpful, conversational answer in 2-4 sentences
- Be specific and use details from the context
- If the exact information isn't available, provide related information that might help
- Don't say "based on the context" - just answer naturally
- Be friendly and professional

Answer:"""

        # Get AI response
        answer = self.ai.call_llm(prompt)
        
        # If AI failed but we have context, provide a simple fallback
        if answer.startswith("⚠️") or answer.startswith("I'm having trouble"):
            # Try simple keyword-based answer
            if "service" in question_lower or "offer" in question_lower or "do" in question_lower:
                # Extract key sentences about services
                service_keywords = ['service', 'offer', 'provide', 'solution', 'specialize', 'expert']
                relevant_lines = []
                for page in self.pages[:3]:
                    for line in page['content'].split('\n'):
                        if any(kw in line.lower() for kw in service_keywords) and len(line) > 40:
                            relevant_lines.append(line.strip())
                            if len(relevant_lines) >= 3:
                                break
                    if len(relevant_lines) >= 3:
                        break
                
                if relevant_lines:
                    return f"Based on their website, {self.company_name} offers:\n\n" + "\n\n".join(relevant_lines[:3])
            
            # For other questions, provide what we know
            if self.pages:
                summary = self.pages[0]['content'][:400]
                return f"Here's what I found about {self.company_name}:\n\n{summary}\n\nFor more details, visit {self.website_url}"
        
        return answer

# =============================================================================
# STREAMLIT APP
# =============================================================================
def init_session():
    """Initialize session state"""
    if 'chatbots' not in st.session_state:
        st.session_state.chatbots = {}
    if 'current_company' not in st.session_state:
        st.session_state.current_company = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def main():
    st.set_page_config(
        page_title="Universal AI Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    init_session()
    
    st.title("🤖 Universal AI Chatbot")
    st.caption("Ready-made AI chatbot that works with ANY company - no configuration needed!")
    
    # Check API key
    if not OPENROUTER_API_KEY:
        st.error("⚠️ OPENROUTER_API_KEY not set!")
        st.info("**How to set:**")
        st.code("export OPENROUTER_API_KEY='your_key_here'", language="bash")
        st.info("Get a free key from: https://openrouter.ai/keys")
        
        with st.expander("🔧 Set API Key in App (Temporary)"):
            temp_key = st.text_input("Paste your API key here:", type="password")
            if temp_key and st.button("Use This Key"):
                os.environ['OPENROUTER_API_KEY'] = temp_key
                st.success("✅ Key set! Refresh the page.")
        return
    
    # API Key Test
    with st.expander("🧪 Test API Key", expanded=False):
        st.write(f"**API Key Set:** ✅")
        st.write(f"**Key Preview:** `{OPENROUTER_API_KEY[:10]}...{OPENROUTER_API_KEY[-4:]}`")
        
        if st.button("Test API Connection"):
            with st.spinner("Testing..."):
                test_ai = SmartAI()
                result = test_ai.call_llm("Say 'API is working!' in exactly 3 words.")
                
                if result.startswith("⚠️"):
                    st.error(f"❌ Test failed: {result}")
                    st.info("**Troubleshooting:**\n1. Check your API key at https://openrouter.ai/keys\n2. Verify you have credits\n3. Check console logs for details")
                else:
                    st.success(f"✅ API is working! Response: {result}")
                    st.info("Your chatbot should work properly now!")
    
    # Sidebar - Company Management
    st.sidebar.title("🏢 Company Management")
    
    with st.sidebar.expander("➕ Add New Company", expanded=True):
        company_name = st.text_input("Company Name", placeholder="e.g., Natoma Singapore")
        website_url = st.text_input("Website URL", placeholder="https://example.com")
        
        if st.button("🚀 Create Chatbot", type="primary"):
            if not company_name or not website_url:
                st.warning("Please provide both name and website URL.")
            else:
                slug = re.sub(r'[^a-z0-9]+', '-', company_name.lower()).strip('-')
                
                # Create chatbot
                with st.spinner(f"Creating chatbot for {company_name}..."):
                    progress = st.progress(0)
                    status = st.empty()
                    
                    def callback(done, total, url):
                        progress.progress(done / max(total, 1))
                        status.text(f"Scraping {done}/{total}: {url[:50]}...")
                    
                    chatbot = UniversalChatbot(company_name, website_url)
                    success = chatbot.initialize(callback)
                    
                    if success:
                        st.session_state.chatbots[slug] = chatbot
                        st.session_state.current_company = slug
                        st.session_state.chat_history = []
                        st.success(f"✅ Chatbot ready for {company_name}!")
                        st.rerun()
                    else:
                        st.error(f"❌ Failed: {chatbot.error}")
    
    # Show existing chatbots
    if st.session_state.chatbots:
        st.sidebar.subheader("📋 Your Chatbots")
        
        for slug, bot in st.session_state.chatbots.items():
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                if st.button(f"💬 {bot.company_name}", key=f"select_{slug}"):
                    st.session_state.current_company = slug
                    st.session_state.chat_history = []
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"delete_{slug}"):
                    del st.session_state.chatbots[slug]
                    if st.session_state.current_company == slug:
                        st.session_state.current_company = None
                    st.rerun()
    
    # Main Chat Interface
    if not st.session_state.current_company:
        st.info("👈 Create a new chatbot to get started!")
        
        st.markdown("### 🎯 Features:")
        st.markdown("""
        - ✨ **Instant Setup**: Just provide company name and website
        - 🚀 **Fast Scraping**: Intelligent parallel scraping in 5-10 seconds
        - 🧠 **Smart AI**: Understands context and provides accurate answers
        - 📞 **Auto Contact**: Automatically extracts contact information
        - 💾 **No Database**: Everything in memory, no setup needed
        - 🌐 **Universal**: Works with ANY company website
        """)
        
        st.markdown("### 📝 Example Companies to Try:")
        st.markdown("""
        - Natoma Singapore (cleaning services)
        - Any tech company
        - Any restaurant or cafe
        - Any retail store
        - Any service provider
        """)
        
    else:
        chatbot = st.session_state.chatbots[st.session_state.current_company]
        
        # Header
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"💬 Chat with {chatbot.company_name}")
        with col2:
            if st.button("🔄 Refresh Data"):
                with st.spinner("Refreshing..."):
                    progress = st.progress(0)
                    status = st.empty()
                    
                    def callback(done, total, url):
                        progress.progress(done / max(total, 1))
                        status.text(f"Scraping {done}/{total}...")
                    
                    chatbot.initialize(callback)
                    st.success("✅ Data refreshed!")
                    st.rerun()
        
        # Info panel
        with st.expander("ℹ️ Chatbot Info", expanded=False):
            st.write(f"**Company:** {chatbot.company_name}")
            st.write(f"**Website:** {chatbot.website_url}")
            st.write(f"**Pages Scraped:** {len(chatbot.pages)}")
            st.write(f"**Emails Found:** {len(chatbot.contact_info['emails'])}")
            st.write(f"**Phones Found:** {len(chatbot.contact_info['phones'])}")
        
        # Chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Chat input
        if user_input := st.chat_input("Ask anything about this company..."):
            # Add user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = chatbot.ask(user_input)
                st.markdown(answer)
            
            # Add assistant message
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })
            
            st.rerun()

if __name__ == "__main__":
    main()
