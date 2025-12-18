import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from datetime import datetime
import json

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# DARK GREEN THEME CSS (Like Your Image)
# ============================================================
st.markdown("""
    <style>
    /* Main app background - Dark green */
    .stApp {
        background-color: #0a2e1f;
    }
    
    /* Sidebar styling - Darker green */
    [data-testid="stSidebar"] {
        background-color: #081a11;
        border-right: 1px solid #1a3d2e;
    }
    
    /* Sidebar header */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #4ade80;
        font-weight: 600;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #86efac;
    }
    
    /* Main content text */
    .stMarkdown, p, label, span {
        color: #d1fae5;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: #0f3123;
        border: 1px solid #1a4d35;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    
    /* User message - lighter green */
    [data-testid="stChatMessageContent"]:has(+ [data-testid="chatAvatarIcon-user"]) {
        background-color: #16543b;
        border-left: 3px solid #4ade80;
    }
    
    /* Assistant message - darker */
    [data-testid="stChatMessageContent"]:has(+ [data-testid="chatAvatarIcon-assistant"]) {
        background-color: #0a2617;
        border-left: 3px solid #22c55e;
    }
    
    /* Buttons - Green gradient */
    .stButton>button {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.3);
        transform: translateY(-2px);
    }
    
    /* Category buttons */
    .category-btn {
        background-color: #0f3123;
        border: 1px solid #22c55e;
        border-radius: 8px;
        padding: 12px;
        margin: 6px 0;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .category-btn:hover {
        background-color: #16543b;
        border-color: #4ade80;
    }
    
    /* Chat history items */
    .chat-history-item {
        background-color: #0f3123;
        border: 1px solid #1a4d35;
        border-radius: 8px;
        padding: 12px;
        margin: 6px 0;
        cursor: pointer;
        transition: all 0.2s;
        color: #86efac;
    }
    
    .chat-history-item:hover {
        background-color: #16543b;
        border-color: #22c55e;
    }
    
    /* Input box */
    .stChatInput {
        background-color: #0f3123;
        border: 1px solid #22c55e;
        border-radius: 12px;
        color: #d1fae5;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #0f3123;
        color: #4ade80;
        border-radius: 8px;
        border: 1px solid #1a4d35;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #4ade80;
        font-size: 20px;
    }
    
    [data-testid="stMetricLabel"] {
        color: #86efac;
    }
    
    /* Success/Info boxes */
    .stSuccess {
        background-color: #16543b;
        border: 1px solid #22c55e;
        color: #d1fae5;
    }
    
    .stInfo {
        background-color: #0f3123;
        border: 1px solid #22c55e;
        color: #d1fae5;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #081a11;
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #0f3123;
        border: 1px solid #1a4d35;
        color: #86efac;
        border-radius: 8px 8px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #16543b;
        border-color: #22c55e;
        color: #4ade80;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #4ade80;
    }
    
    /* Divider */
    hr {
        border-color: #1a4d35;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD ENVIRONMENT
# ============================================================
load_dotenv()

# ============================================================
# CATEGORIZED PROMPTS
# ============================================================
PROMPT_CATEGORIES = {
    "📦 Orders & Tracking": [
        "How can I track my order?",
        "Where is my package?",
        "What's my order status?",
        "How do I check shipping updates?",
        "When will my order arrive?"
    ],
    "💳 Payments & Refunds": [
        "What payment methods do you accept?",
        "How do I get a refund?",
        "When will my refund be processed?",
        "Can I pay with PayPal?",
        "Is my payment information secure?"
    ],
    "🔄 Returns & Exchanges": [
        "What is your return policy?",
        "How do I return an item?",
        "Can I exchange for a different size?",
        "Is return shipping free?",
        "What items cannot be returned?"
    ],
    "🚚 Shipping & Delivery": [
        "How long does shipping take?",
        "Do you ship internationally?",
        "What are the shipping costs?",
        "Do you offer express shipping?",
        "Can I change my delivery address?"
    ],
    "💰 Discounts & Promotions": [
        "Are there any discounts available?",
        "Do you have student discounts?",
        "How do I use a promo code?",
        "Is there a loyalty program?",
        "When is your next sale?"
    ],
    "📞 Support & Contact": [
        "How do I contact customer support?",
        "What are your business hours?",
        "Do you have live chat?",
        "Can I speak to a human agent?",
        "Where is your support email?"
    ]
}

# ============================================================
# INITIALIZE SESSION STATE
# ============================================================
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
    
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

if "rag_loaded" not in st.session_state:
    st.session_state.rag_loaded = False

# ============================================================
# FUNCTIONS
# ============================================================
def create_new_chat():
    """Create a new chat session"""
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.chat_sessions[session_id] = {
        "title": "New Chat",
        "messages": [],
        "created": datetime.now().strftime("%I:%M %p")
    }
    st.session_state.current_session_id = session_id
    return session_id

def get_current_messages():
    """Get messages for current session"""
    if st.session_state.current_session_id:
        return st.session_state.chat_sessions[st.session_state.current_session_id]["messages"]
    return []

def add_message(role, content):
    """Add message to current session"""
    if st.session_state.current_session_id:
        messages = st.session_state.chat_sessions[st.session_state.current_session_id]["messages"]
        messages.append({"role": role, "content": content})
        
        # Auto-generate title from first user message
        if role == "user" and len(messages) == 1:
            title = content[:40] + "..." if len(content) > 40 else content
            st.session_state.chat_sessions[st.session_state.current_session_id]["title"] = title

def get_answer(question):
    """Get answer using RAG"""
    try:
        retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])
        
        prompt = f"""You are a professional customer support AI assistant.
Provide a clear, concise, and helpful answer based on the context below.
Use a friendly tone and keep responses under 4 sentences.

Context:
{context}

Customer Question: {question}

Answer:"""
        
        return st.session_state.llm.invoke(prompt).content
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

# ============================================================
# LOAD RAG SYSTEM
# ============================================================
if not st.session_state.rag_loaded:
    with st.spinner("🚀 Loading AI Assistant..."):
        try:
            loader = CSVLoader("data/knowledge_base.csv", encoding="utf-8")
            docs = loader.load()
            
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            st.session_state.vector_db = FAISS.from_documents(docs, embeddings)
            
            st.session_state.llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
            
            st.session_state.rag_loaded = True
            st.session_state.doc_count = len(docs)
            
            # Create first session
            if not st.session_state.chat_sessions:
                create_new_chat()
            
        except Exception as e:
            st.error(f"❌ Error loading system: {e}")
            st.stop()

# ============================================================
# SIDEBAR - CHAT HISTORY & CATEGORIES
# ============================================================
with st.sidebar:
    st.markdown("# 🤖 AI Assistant")
    
    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True, key="new_chat_btn"):
        create_new_chat()
        st.rerun()
    
    st.markdown("---")
    
    # Chat History
    st.markdown("### 💬 Chat History")
    
    if st.session_state.chat_sessions:
        for session_id in reversed(list(st.session_state.chat_sessions.keys())):
            session = st.session_state.chat_sessions[session_id]
            is_current = session_id == st.session_state.current_session_id
            
            button_label = f"{'🟢' if is_current else '⚪'} {session['title']}"
            
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(button_label, key=f"session_{session_id}", use_container_width=True):
                    st.session_state.current_session_id = session_id
                    st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"delete_{session_id}"):
                    del st.session_state.chat_sessions[session_id]
                    if session_id == st.session_state.current_session_id:
                        if st.session_state.chat_sessions:
                            st.session_state.current_session_id = list(st.session_state.chat_sessions.keys())[-1]
                        else:
                            create_new_chat()
                    st.rerun()
    
    st.markdown("---")
    
    # Prompt Categories
    st.markdown("### 📂 Quick Prompts")
    
    tabs = st.tabs([cat.split(" ")[0] for cat in PROMPT_CATEGORIES.keys()])
    
    for idx, (category, prompts) in enumerate(PROMPT_CATEGORIES.items()):
        with tabs[idx]:
            st.markdown(f"**{category}**")
            for prompt in prompts:
                if st.button(f"• {prompt}", key=f"prompt_{category}_{prompt}", use_container_width=True):
                    if not st.session_state.current_session_id:
                        create_new_chat()
                    
                    add_message("user", prompt)
                    with st.spinner("Thinking..."):
                        answer = get_answer(prompt)
                        add_message("assistant", answer)
                    st.rerun()
    
    st.markdown("---")
    
    # System Status
    st.markdown("### ⚡ System")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📚 Docs", f"{st.session_state.doc_count:,}")
    with col2:
        total_chats = len(st.session_state.chat_sessions)
        st.metric("💬 Chats", total_chats)

# ============================================================
# MAIN CHAT INTERFACE
# ============================================================
# Header
st.markdown("## 🤖 AI Customer Support Assistant")

if st.session_state.current_session_id:
    current_session = st.session_state.chat_sessions[st.session_state.current_session_id]
    st.caption(f"Chat started at {current_session['created']}")
else:
    st.info("👈 Click 'New Chat' to start a conversation")
    st.stop()

st.markdown("---")

# Display messages
messages = get_current_messages()

if not messages:
    st.markdown("""
    ### 👋 Welcome! How can I help you today?
    
    **I can assist with:**
    - 📦 Order tracking and status
    - 💳 Payment and refund questions
    - 🔄 Returns and exchanges
    - 🚚 Shipping information
    - 💰 Discounts and promotions
    - 📞 General support
    
    **Choose a quick prompt from the sidebar or type your question below!**
    """)

for msg in messages:
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("💬 Ask me anything...", key="chat_input"):
    add_message("user", prompt)
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            answer = get_answer(prompt)
            st.markdown(answer)
            add_message("assistant", answer)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #86efac; padding: 20px;'>
        <p>🎓 Built for GUVI Project | ⚡ Powered by Groq LLaMA 3.3 & LangChain</p>
    </div>
""", unsafe_allow_html=True)
