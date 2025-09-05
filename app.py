import os, json, asyncio, requests
import streamlit as st
from bs4 import BeautifulSoup
from pydantic import BaseModel
from typing import List
import nest_asyncio
from groq import Groq

# ===============================
# Config
# ===============================
class AppConfig(BaseModel):
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.4
    max_tokens: int = 700

DEFAULT_CONFIG = AppConfig()

# ===============================
# LLM helper
# ===============================
class LLMMessage(BaseModel):
    role: str
    content: str

def llm_chat(messages: List[LLMMessage], model, temperature, max_tokens, api_key) -> str:
    client = Groq(api_key=api_key)   # 👈 Buyer se API key lega
    resp = client.chat.completions.create(
        model=model,
        messages=[m.dict() for m in messages],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

# ===============================
# Agents
# ===============================
class Agent:
    def __init__(self, name, system_prompt, config=DEFAULT_CONFIG):
        self.name = name
        self.system_prompt = system_prompt
        self.config = config

    async def run(self, user_query: str, api_key: str) -> str:
        messages = [
            LLMMessage(role="system", content=self.system_prompt),
            LLMMessage(role="user", content=user_query),
        ]
        loop = asyncio.get_event_loop()
        reply = await loop.run_in_executor(
            None, llm_chat, messages,
            self.config.model, self.config.temperature, self.config.max_tokens, api_key
        )
        return reply

# Specialist prompts
agents = {
    "Web": Agent("Web", "Summarize latest factual information clearly."),
    "Motivation": Agent("Motivation", "Give uplifting motivational advice."),
    "Finance": Agent("Finance", "Provide budgeting & saving tips."),
    "Health": Agent("Health", "Give general wellness and lifestyle advice (no medical diagnosis)."),
    "Business": Agent("Business", "Give strategy and growth ideas."),
    "Marketing": Agent("Marketing", "Provide creative marketing and campaign strategies."),
}

Manager = Agent("Manager", "Classify query into: Web, Motivation, Finance, Health, Business, Marketing, or Multi.")

# ===============================
# Web Fetch Tool
# ===============================
def fetch_url(url: str, max_chars=2000) -> str:
    try:
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script","style","noscript"]): tag.decompose()
        text = " ".join(soup.get_text(" ").split())
        return text[:max_chars]
    except Exception as e:
        return f"[Error fetching {url}: {e}]"

# ===============================
# Orchestrator
# ===============================
async def orchestrator(user_query, mode, urls, api_key):
    web_text = ""
    if urls:
        for u in urls.split():
            web_text += f"\n[From {u}]: {fetch_url(u)}"

    query_full = user_query + web_text

    if mode == "Auto (Manager decides)":
        route = await Manager.run(query_full, api_key)
        chosen = None
        for key in agents.keys():
            if key.lower() in route.lower():
                chosen = key
                break
        if not chosen: chosen = "Motivation"
        reply = await agents[chosen].run(query_full, api_key)
        return f"**Routed to {chosen} Agent**\n\n{reply}"
    else:
        reply = await agents[mode].run(query_full, api_key)
        return f"**{mode} Agent Reply**\n\n{reply}"

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Premium Multi-Agent AI (Groq)", page_icon="🤖", layout="wide")
st.title("🤖 Premium Multi-Agent AI System (Groq)")

api_key = st.text_input("🔑 Enter your Groq API Key", type="password")

user_query = st.text_area("Your Query", placeholder="Ask me anything...")
urls = st.text_input("Web URLs (optional, space-separated)")

mode = st.radio("Mode", ["Auto (Manager decides)"] + list(agents.keys()))

if st.button("Run"):
    if not api_key:
        st.error("❌ Please enter your Groq API Key first.")
    elif not user_query.strip():
        st.warning("⚠️ Please enter a query first.")
    else:
        with st.spinner("🤔 Thinking..."):
            nest_asyncio.apply()
            result = asyncio.run(orchestrator(user_query, mode, urls, api_key))
            st.markdown(result)
