# app.py - Ultra Stunning Chatbot UI with Advanced Visuals
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

# Ultra-Enhanced CSS with stunning visuals
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Animated gradient background */
    .main {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        padding: 0;
        min-height: 100vh;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .block-container {
        max-width: 1200px;
        padding: 2rem 1rem;
    }
    
    /* Floating particles effect */
    .particles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 0;
    }
    
    .particle {
        position: absolute;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        animation: float 20s infinite ease-in-out;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0) translateX(0) rotate(0deg); }
        25% { transform: translateY(-100px) translateX(50px) rotate(90deg); }
        50% { transform: translateY(-200px) translateX(-50px) rotate(180deg); }
        75% { transform: translateY(-100px) translateX(-100px) rotate(270deg); }
    }
    
    /* Advanced glassmorphism card */
    .chat-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(30px) saturate(180%);
        -webkit-backdrop-filter: blur(30px) saturate(180%);
        border-radius: 32px;
        border: 2px solid rgba(255, 255, 255, 0.3);
        box-shadow: 
            0 20px 60px 0 rgba(31, 38, 135, 0.37),
            inset 0 1px 0 rgba(255, 255, 255, 0.5),
            0 1px 2px rgba(0, 0, 0, 0.1);
        padding: 0;
        overflow: hidden;
        max-width: 950px;
        margin: 0 auto;
        position: relative;
        z-index: 1;
        transform: translateY(0);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .chat-container:hover {
        transform: translateY(-4px);
        box-shadow: 
            0 30px 80px 0 rgba(31, 38, 135, 0.45),
            inset 0 1px 0 rgba(255, 255, 255, 0.6),
            0 2px 4px rgba(0, 0, 0, 0.15);
    }
    
    /* Premium chat header with mesh gradient */
    .chat-header {
        background: linear-gradient(135deg, 
            rgba(102, 126, 234, 0.95) 0%, 
            rgba(118, 75, 162, 0.95) 50%,
            rgba(240, 147, 251, 0.95) 100%);
        padding: 1.75rem 2rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        display: flex;
        align-items: center;
        gap: 1.25rem;
        position: relative;
        overflow: hidden;
    }
    
    .chat-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* 3D rotating bot avatar */
    .bot-avatar {
        width: 56px;
        height: 56px;
        border-radius: 50%;
        background: linear-gradient(135deg, #fff 0%, #f0f0f0 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        position: relative;
        box-shadow: 
            0 8px 24px rgba(0, 0, 0, 0.3),
            inset 0 2px 4px rgba(255, 255, 255, 0.8);
        animation: avatarFloat 3s ease-in-out infinite;
        transform-style: preserve-3d;
        z-index: 2;
    }
    
    @keyframes avatarFloat {
        0%, 100% { transform: translateY(0) rotateY(0deg); }
        50% { transform: translateY(-5px) rotateY(360deg); }
    }
    
    .bot-avatar::before {
        content: '';
        position: absolute;
        inset: -3px;
        border-radius: 50%;
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #4facfe);
        background-size: 300% 300%;
        animation: rotateBorder 4s linear infinite;
        z-index: -1;
        filter: blur(8px);
    }
    
    @keyframes rotateBorder {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .status-indicator {
        position: absolute;
        bottom: 2px;
        right: 2px;
        width: 14px;
        height: 14px;
        background: #10b981;
        border: 3px solid white;
        border-radius: 50%;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.6);
        animation: pulse-dot 2s cubic-bezier(0.455, 0.03, 0.515, 0.955) infinite;
    }
    
    @keyframes pulse-dot {
        0%, 100% { 
            transform: scale(1); 
            box-shadow: 0 0 10px rgba(16, 185, 129, 0.6);
        }
        50% { 
            transform: scale(1.15); 
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.9);
        }
    }
    
    .header-text {
        flex: 1;
        color: white;
        z-index: 2;
    }
    
    .header-text h3 {
        margin: 0;
        font-size: 1.25rem;
        font-weight: 700;
        line-height: 1.2;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .header-text p {
        margin: 0.375rem 0 0 0;
        font-size: 0.8125rem;
        opacity: 0.95;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
    }
    
    .online-dot {
        width: 7px;
        height: 7px;
        background: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px rgba(16, 185, 129, 0.8);
        animation: pulse-dot 2s ease-in-out infinite;
    }
    
    /* Messages area with mesh gradient */
    .messages-container {
        background: 
            linear-gradient(180deg, rgba(255, 255, 255, 0.95) 0%, rgba(249, 250, 251, 0.95) 100%),
            url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(102,126,234,0.05)" stroke-width="0.5"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
        padding: 2rem;
        min-height: 550px;
        max-height: 650px;
        overflow-y: auto;
        position: relative;
    }
    
    /* Premium scrollbar */
    .messages-container::-webkit-scrollbar {
        width: 10px;
    }
    
    .messages-container::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        margin: 8px 0;
    }
    
    .messages-container::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    
    .messages-container::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Enhanced message bubbles with 3D effect */
    .stChatMessage {
        background: transparent !important;
        padding: 0.875rem 0 !important;
        margin-bottom: 0.75rem !important;
        animation: messageSlideIn 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    @keyframes messageSlideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* User message with premium gradient */
    .stChatMessage[data-testid*="user-message"] > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 20px 20px 4px 20px !important;
        padding: 1rem 1.5rem !important;
        box-shadow: 
            0 8px 24px rgba(102, 126, 234, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        max-width: 75% !important;
        margin-left: auto !important;
        font-size: 0.9375rem !important;
        line-height: 1.6 !important;
        position: relative;
        transform: translateZ(0);
        transition: all 0.3s ease;
    }
    
    .stChatMessage[data-testid*="user-message"] > div:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 
            0 12px 32px rgba(102, 126, 234, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Assistant message with glassmorphism */
    .stChatMessage[data-testid*="assistant-message"] > div {
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(10px) !important;
        color: #1e293b !important;
        border: 2px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 20px 20px 20px 4px !important;
        padding: 1rem 1.5rem !important;
        box-shadow: 
            0 4px 16px rgba(0, 0, 0, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
        max-width: 75% !important;
        font-size: 0.9375rem !important;
        line-height: 1.6 !important;
        position: relative;
        transition: all 0.3s ease;
    }
    
    .stChatMessage[data-testid*="assistant-message"] > div:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 
            0 8px 24px rgba(0, 0, 0, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.9) !important;
        border-color: rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Premium avatar styling */
    .stChatMessage > div:first-child {
        width: 38px !important;
        height: 38px !important;
        border-radius: 50% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        flex-shrink: 0 !important;
        font-size: 18px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
        transition: all 0.3s ease !important;
    }
    
    .stChatMessage > div:first-child:hover {
        transform: scale(1.1) rotate(5deg);
    }
    
    .stChatMessage[data-testid*="user-message"] > div:first-child {
        background: linear-gradient(135deg, #fff 0%, #f8f9fa 100%) !important;
        border: 2px solid rgba(102, 126, 234, 0.3) !important;
        color: #667eea !important;
    }
    
    .stChatMessage[data-testid*="assistant-message"] > div:first-child {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Advanced typing indicator */
    .typing-indicator {
        display: flex;
        gap: 0.375rem;
        padding: 1.25rem 1.5rem;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px 20px 20px 4px;
        border: 2px solid rgba(102, 126, 234, 0.2);
        width: fit-content;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 50%;
        animation: typing 1.4s infinite ease-in-out;
        box-shadow: 0 0 8px rgba(102, 126, 234, 0.4);
    }
    
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing {
        0%, 60%, 100% { 
            transform: translateY(0) scale(1);
            opacity: 0.6;
        }
        30% { 
            transform: translateY(-12px) scale(1.2);
            opacity: 1;
        }
    }
    
    /* Premium input area */
    .input-container {
        padding: 1.5rem 2rem;
        background: rgba(248, 250, 252, 0.95);
        backdrop-filter: blur(20px);
        border-top: 1px solid rgba(102, 126, 234, 0.2);
        position: relative;
    }
    
    .input-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, 
            transparent 0%, 
            rgba(102, 126, 234, 0.5) 50%, 
            transparent 100%);
    }
    
    .stChatInputContainer {
        background: white !important;
        border: 2px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 20px !important;
        box-shadow: 
            0 4px 16px rgba(0, 0, 0, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        padding: 0 !important;
    }
    
    .stChatInputContainer:focus-within {
        border-color: rgba(102, 126, 234, 0.6) !important;
        box-shadow: 
            0 8px 24px rgba(102, 126, 234, 0.2),
            0 0 0 4px rgba(102, 126, 234, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.9) !important;
        transform: translateY(-2px);
    }
    
    .stChatInputContainer input {
        padding: 1rem 4rem 1rem 1.5rem !important;
        font-size: 0.9375rem !important;
        border: none !important;
        background: transparent !important;
        font-weight: 500 !important;
    }
    
    .stChatInputContainer input::placeholder {
        color: #94a3b8 !important;
        font-weight: 400 !important;
    }
    
    /* Stunning send button */
    .stChatInputContainer button {
        position: absolute !important;
        right: 6px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        width: 44px !important;
        height: 44px !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 14px !important;
        color: white !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stChatInputContainer button:hover {
        transform: translateY(-50%) scale(1.05) rotate(5deg) !important;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.6) !important;
    }
    
    .stChatInputContainer button:active {
        transform: translateY(-50%) scale(0.95) !important;
    }
    
    /* Premium footer */
    .chat-footer {
        text-align: center;
        padding: 1rem 2rem;
        background: linear-gradient(180deg, 
            rgba(248, 250, 252, 0.8) 0%, 
            rgba(241, 245, 249, 0.8) 100%);
        backdrop-filter: blur(10px);
        border-top: 1px solid rgba(226, 232, 240, 0.6);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.625rem;
        font-size: 0.6875rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #64748b;
    }
    
    .sparkles-icon {
        width: 14px;
        height: 14px;
        opacity: 0.6;
        animation: sparkle 2s ease-in-out infinite;
    }
    
    @keyframes sparkle {
        0%, 100% { transform: scale(1) rotate(0deg); opacity: 0.6; }
        50% { transform: scale(1.2) rotate(180deg); opacity: 1; }
    }
    
    /* Stunning sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, 
            rgba(102, 126, 234, 0.98) 0%, 
            rgba(118, 75, 162, 0.98) 50%,
            rgba(240, 147, 251, 0.98) 100%) !important;
        backdrop-filter: blur(20px);
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.15);
    }
    
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.25) !important;
        backdrop-filter: blur(10px) !important;
        border: 2px solid rgba(255, 255, 255, 0.4) !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 14px !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255, 255, 255, 0.35) !important;
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3) !important;
    }
    
    section[data-testid="stSidebar"] .stTextInput input {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #1e293b !important;
        border: 2px solid rgba(255, 255, 255, 0.4) !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    
    section[data-testid="stSidebar"] .stTextInput input:focus {
        background: white !important;
        box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.3) !important;
    }
    
    section[data-testid="stSidebar"] .stTextInput input::placeholder {
        color: #64748b !important;
    }
    
    /* Premium expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(15px) !important;
        border-radius: 14px !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 1rem 1.25rem !important;
        transition: all 0.3s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.3) !important;
        transform: scale(1.02);
    }
    
    /* Alert messages with premium styling */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 16px !important;
        border: none !important;
        padding: 1.25rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stSuccess {
        background: rgba(16, 185, 129, 0.15) !important;
        border-left: 4px solid #10b981 !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.15) !important;
        border-left: 4px solid #ef4444 !important;
    }
    
    .stWarning {
        background: rgba(245, 158, 11, 0.15) !important;
        border-left: 4px solid #f59e0b !important;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-color: rgba(102, 126, 234, 0.2) !important;
        border-top-color: #667eea !important;
        border-width: 3px !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 10px !important;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.5) !important;
    }
    
    /* Welcome screen animations */
    .welcome-icon {
        animation: welcomeBounce 2s ease-in-out infinite;
    }
    
    @keyframes welcomeBounce {
        0%, 100% { transform: translateY(0) scale(1); }
        50% { transform: translateY(-20px) scale(1.05); }
    }
    
    .feature-card {
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .feature-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Smooth transitions */
    * {
        transition: background-color 0.2s ease, 
                    border-color 0.2s ease, 
                    color 0.2s ease;
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
        page_title="✨ AutoBot AI - Ultra Edition",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply stunning CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Add floating particles effect
    st.markdown("""
    <div class="particles">
        <div class="particle" style="width: 4px; height: 4px; left: 10%; top: 20%; animation-delay: 0s;"></div>
        <div class="particle" style="width: 6px; height: 6px; left: 80%; top: 30%; animation-delay: 2s;"></div>
        <div class="particle" style="width: 3px; height: 3px; left: 50%; top: 50%; animation-delay: 4s;"></div>
        <div class="particle" style="width: 5px; height: 5px; left: 20%; top: 70%; animation-delay: 1s;"></div>
        <div class="particle" style="width: 4px; height: 4px; left: 70%; top: 80%; animation-delay: 3s;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    init_session()
    
    # Stunning sidebar
    with st.sidebar:
        st.markdown("### 🏢 Company Management")
        
        with st.expander("➕ Add New Company", expanded=True):
            company_name = st.text_input("Company Name", placeholder="e.g., Acme Corp")
            website_url = st.text_input("Website URL", placeholder="https://example.com")
            
            if st.button("🚀 Create Chatbot", type="primary", use_container_width=True):
                if not company_name or not website_url:
                    st.warning("⚠️ Please fill in all fields")
                else:
                    slug = re.sub(r'[^a-z0-9]+', '-', company_name.lower()).strip('-')
                    
                    with st.spinner(f"✨ Analyzing {company_name}..."):
                        progress = st.progress(0)
                        status = st.empty()
                        
                        def callback(done, total, url):
                            progress.progress(done / max(total, 1))
                            status.text(f"🔍 Scraping {done}/{total} pages...")
                        
                        chatbot = UniversalChatbot(company_name, website_url)
                        success = chatbot.initialize(callback)
                        
                        if success:
                            st.session_state.chatbots[slug] = chatbot
                            st.session_state.current_company = slug
                            st.session_state.chat_history = []
                            st.success("✅ Chatbot created successfully!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ Error: {chatbot.error}")
        
        if st.session_state.chatbots:
            st.markdown("### 💬 Active Chatbots")
            
            for slug, bot in st.session_state.chatbots.items():
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(
                        f"{'🟢' if st.session_state.current_company == slug else '⚪'} {bot.company_name}", 
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
    
    # Main Content
    if st.session_state.current_company:
        chatbot = st.session_state.chatbots[st.session_state.current_company]
        
        # Premium chat container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Stunning chat header
        st.markdown(f"""
        <div class="chat-header">
            <div class="bot-avatar">
                🤖
                <span class="status-indicator"></span>
            </div>
            <div class="header-text">
                <h3>{chatbot.company_name} Assistant</h3>
                <p><span class="online-dot"></span> Online • Replies instantly</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Messages container
        st.markdown('<div class="messages-container">', unsafe_allow_html=True)
        
        # Display chat history with animations
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
                st.markdown(msg["content"])
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Premium input container
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        # Chat input
        user_input = st.chat_input("✨ Type your message here...")
        
        if user_input:
            # Add user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Get bot response with loading animation
            with st.spinner("🤔 Thinking..."):
                response = chatbot.ask(user_input)
            
            # Add assistant response
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })
            
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Premium footer
        st.markdown("""
        <div class="chat-footer">
            <svg class="sparkles-icon" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0L14.5 8.5L23 11L14.5 13.5L12 22L9.5 13.5L1 11L9.5 8.5L12 0Z"/>
            </svg>
            <span>Powered by Advanced AI</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional info panel
        with st.expander("ℹ️ Chatbot Analytics & Controls"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 Pages Analyzed", len(chatbot.pages))
            with col2:
                st.metric("📧 Emails Found", len(chatbot.contact_info['emails']))
            with col3:
                st.metric("📱 Phones Found", len(chatbot.contact_info['phones']))
            
            st.divider()
            
            st.write(f"**🌐 Website:** {chatbot.website_url}")
            st.write(f"**💬 Messages:** {len(st.session_state.chat_history)}")
            
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.success("✅ Chat cleared!")
                time.sleep(0.5)
                st.rerun()
    
    else:
        # Ultra-stunning welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; position: relative; z-index: 1;">
            <div class="welcome-icon" style="font-size: 6rem; margin-bottom: 1.5rem; filter: drop-shadow(0 10px 30px rgba(102, 126, 234, 0.4));">
                🤖
            </div>
            <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%); 
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       font-size: 3.5rem; font-weight: 900; margin-bottom: 1rem;
                       font-family: 'Space Grotesk', sans-serif;
                       text-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);">
                AutoBot AI Ultra
            </h1>
            <p style="font-size: 1.375rem; color: rgba(255, 255, 255, 0.9); margin-bottom: 3rem;
                      text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3); font-weight: 500;">
                Create intelligent chatbots for any company website in seconds ⚡
            </p>
            
            <style>
                .welcome-feature-card {
                    background: rgba(255, 255, 255, 0.15);
                    backdrop-filter: blur(20px);
                    border-radius: 24px;
                    padding: 3rem;
                    max-width: 700px;
                    margin: 0 auto;
                    border: 2px solid rgba(255, 255, 255, 0.3);
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                }
                .welcome-feature-title {
                    color: white;
                    margin-bottom: 2rem;
                    font-size: 1.75rem;
                    font-weight: 700;
                    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
                }
                .welcome-feature-grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 1.5rem;
                    text-align: left;
                }
                .welcome-feature-item {
                    background: rgba(255, 255, 255, 0.1);
                    padding: 1.25rem;
                    border-radius: 16px;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    backdrop-filter: blur(10px);
                }
                .welcome-feature-icon {
                    font-size: 2rem;
                    margin-bottom: 0.5rem;
                }
                .welcome-feature-name {
                    color: white;
                    font-weight: 600;
                    font-size: 0.95rem;
                }
                .welcome-feature-desc {
                    color: rgba(255, 255, 255, 0.8);
                    font-size: 0.85rem;
                    margin-top: 0.25rem;
                }
                .welcome-divider-container {
                    margin-top: 3rem;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 0.75rem;
                }
                .welcome-divider-line {
                    width: 40px;
                    height: 2px;
                    background: rgba(255, 255, 255, 0.3);
                }
                .welcome-divider-text {
                    color: rgba(255, 255, 255, 0.9);
                    font-size: 1rem;
                    font-weight: 600;
                    margin: 0;
                }
            </style>
            
            <div class="welcome-feature-card">
                <h3 class="welcome-feature-title">✨ Premium Features</h3>
                <div class="welcome-feature-grid">
                    <div class="welcome-feature-item">
                        <div class="welcome-feature-icon">🚀</div>
                        <div class="welcome-feature-name">Instant Creation</div>
                        <div class="welcome-feature-desc">Launch in seconds</div>
                    </div>
                    <div class="welcome-feature-item">
                        <div class="welcome-feature-icon">🔍</div>
                        <div class="welcome-feature-name">Smart Analysis</div>
                        <div class="welcome-feature-desc">Auto content extraction</div>
                    </div>
                    <div class="welcome-feature-item">
                        <div class="welcome-feature-icon">📞</div>
                        <div class="welcome-feature-name">Contact Detection</div>
                        <div class="welcome-feature-desc">Find emails and phones</div>
                    </div>
                    <div class="welcome-feature-item">
                        <div class="welcome-feature-icon">⚡</div>
                        <div class="welcome-feature-name">Lightning Fast</div>
                        <div class="welcome-feature-desc">Instant responses</div>
                    </div>
                </div>
            </div>
            
            <div class="welcome-divider-container">
                <div class="welcome-divider-line"></div>
                <p class="welcome-divider-text">👈 Create your first chatbot in the sidebar</p>
                <div class="welcome-divider-line"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
