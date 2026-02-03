import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from datetime import datetime

# ============================================================
# PAGE CONFIGURATION - MUST BE FIRST
# ============================================================
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================
load_dotenv()

# ============================================================
# INITIALIZE SESSION STATE FIRST
# ============================================================
def init_session_state():
    """Initialize all session state variables"""
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}
    
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    
    if "rag_loaded" not in st.session_state:
        st.session_state.rag_loaded = False
    
    if "doc_count" not in st.session_state:
        st.session_state.doc_count = 0
    
    if "processing" not in st.session_state:
        st.session_state.processing = False

# Initialize immediately
init_session_state()

# ============================================================
# DARK GREEN THEME CSS
# ============================================================
st.markdown("""
    <style>
    .stApp {
        background-color: #0a2e1f;
    }
    
    [data-testid="stSidebar"] {
        background-color: #081a11;
        border-right: 1px solid #1a3d2e;
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #4ade80;
        font-weight: 600;
    }
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #86efac;
    }
    
    .stMarkdown, p, label, span {
        color: #d1fae5;
    }
    
    .stChatMessage {
        background-color: #0f3123;
        border: 1px solid #1a4d35;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    
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
    
    .stChatInput {
        background-color: #0f3123;
        border: 1px solid #22c55e;
        border-radius: 12px;
        color: #d1fae5;
    }
    
    [data-testid="stMetricValue"] {
        color: #4ade80;
        font-size: 20px;
    }
    
    [data-testid="stMetricLabel"] {
        color: #86efac;
    }
    
    .stSuccess {
        background-color: #16543b;
        border: 1px solid #22c55e;
        color: #d1fae5;
    }
    
    .stInfo, .stWarning {
        background-color: #0f3123;
        border: 1px solid #22c55e;
        color: #d1fae5;
    }
    
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
    
    h1, h2, h3 {
        color: #4ade80;
    }
    
    hr {
        border-color: #1a4d35;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# CATEGORIZED PROMPTS
# ============================================================
PROMPT_CATEGORIES = {
    "📦 Orders": [
        "How can I track my order?",
        "Where is my package?",
        "What's my order status?",
        "When will my order arrive?"
    ],
    "💳 Payments": [
        "What payment methods do you accept?",
        "How do I get a refund?",
        "When will my refund be processed?",
        "Is my payment secure?"
    ],
    "🔄 Returns": [
        "What is your return policy?",
        "How do I return an item?",
        "Can I exchange sizes?",
        "Is return shipping free?"
    ],
    "🚚 Shipping": [
        "How long does shipping take?",
        "Do you ship internationally?",
        "What are shipping costs?",
        "Do you offer express shipping?"
    ],
    "💰 Discounts": [
        "Any discounts available?",
        "Do you have student discounts?",
        "How do I use a promo code?",
        "When is your next sale?"
    ],
    "📞 Support": [
        "How do I contact support?",
        "What are your business hours?",
        "Do you have live chat?",
        "Where is your support email?"
    ]
}

# ============================================================
# HELPER FUNCTIONS
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
    if st.session_state.current_session_id and st.session_state.current_session_id in st.session_state.chat_sessions:
        return st.session_state.chat_sessions[st.session_state.current_session_id]["messages"]
    return []

def add_message(role, content):
    """Add message to current session"""
    if st.session_state.current_session_id and st.session_state.current_session_id in st.session_state.chat_sessions:
        messages = st.session_state.chat_sessions[st.session_state.current_session_id]["messages"]
        messages.append({"role": role, "content": content})
        
        # Auto-generate title from first user message
        if role == "user" and len(messages) == 1:
            title = content[:40] + "..." if len(content) > 40 else content
            st.session_state.chat_sessions[st.session_state.current_session_id]["title"] = title

@st.cache_resource(show_spinner=False)
def load_embeddings():
    """Load embeddings model with caching - MUCH FASTER"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_vector_db(_embeddings, csv_path):
    """Load and cache vector database"""
    try:
        loader = CSVLoader(csv_path, encoding="utf-8")
        docs = loader.load()
        vector_db = FAISS.from_documents(docs, _embeddings)
        return vector_db, len(docs)
    except Exception as e:
        st.error(f"Error loading knowledge base: {e}")
        return None, 0

def get_answer(question):
    """Get answer using RAG"""
    try:
        if not hasattr(st.session_state, 'vector_db') or st.session_state.vector_db is None:
            return "❌ System not initialized properly. Please refresh the page."
        
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
        
        response = st.session_state.llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"❌ Sorry, I encountered an error: {str(e)}"

# ============================================================
# LOAD RAG SYSTEM - WITH PROPER ERROR HANDLING
# ============================================================
def initialize_rag():
    """Initialize RAG system with proper error handling"""
    
    # Check for GROQ API key
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        st.error("❌ **GROQ_API_KEY not found!**")
        st.info("Please create a `.env` file with: `GROQ_API_KEY=your_key_here`")
        st.stop()
    
    # Check for CSV file
    csv_path = "data/knowledge_base.csv"
    if not os.path.exists(csv_path):
        st.error(f"❌ **Knowledge base not found!**")
        st.info(f"Please create the file: `{csv_path}`")
        st.stop()
    
    progress_bar = st.progress(0, text="🚀 Initializing AI Assistant...")
    
    try:
        # Step 1: Load embeddings (20%)
        progress_bar.progress(20, text="📥 Loading embeddings model...")
        embeddings = load_embeddings()
        if embeddings is None:
            st.stop()
        
        # Step 2: Load vector database (60%)
        progress_bar.progress(60, text="📚 Loading knowledge base...")
        vector_db, doc_count = load_vector_db(embeddings, csv_path)
        if vector_db is None:
            st.stop()
        
        # Step 3: Initialize LLM (80%)
        progress_bar.progress(80, text="🤖 Connecting to AI model...")
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            groq_api_key=groq_key
        )
        
        # Step 4: Save to session state (100%)
        progress_bar.progress(100, text="✅ Ready!")
        st.session_state.vector_db = vector_db
        st.session_state.llm = llm
        st.session_state.doc_count = doc_count
        st.session_state.rag_loaded = True
        
        # Create first session if none exists
        if not st.session_state.chat_sessions:
            create_new_chat()
        
        progress_bar.empty()
        
    except Exception as e:
        progress_bar.empty()
        st.error(f"❌ **Initialization Error:** {str(e)}")
        st.info("Common fixes:\n- Check your GROQ_API_KEY\n- Ensure knowledge_base.csv exists\n- Check internet connection")
        st.stop()

# Initialize RAG if not loaded
if not st.session_state.rag_loaded:
    initialize_rag()

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
    else:
        st.info("No chat history yet")
    
    st.markdown("---")
    
    # Prompt Categories - SIMPLIFIED (no tabs to avoid complexity)
    st.markdown("### 📂 Quick Prompts")
    
    selected_category = st.selectbox(
        "Choose category:",
        list(PROMPT_CATEGORIES.keys()),
        label_visibility="collapsed"
    )
    
    if selected_category:
        for prompt in PROMPT_CATEGORIES[selected_category]:
            if st.button(prompt, key=f"prompt_{prompt[:20]}", use_container_width=True):
                if not st.session_state.current_session_id:
                    create_new_chat()
                
                # Add user message
                add_message("user", prompt)
                
                # Get answer
                with st.spinner("🤔 Thinking..."):
                    answer = get_answer(prompt)
                    add_message("assistant", answer)
                
                st.rerun()
    
    st.markdown("---")
    
    # System Status
    st.markdown("### ⚡ System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📚 Docs", f"{st.session_state.doc_count:,}")
    with col2:
        total_chats = len(st.session_state.chat_sessions)
        st.metric("💬 Chats", total_chats)

# ============================================================
# MAIN CHAT INTERFACE
# ============================================================
st.markdown("## 🤖 AI Customer Support Assistant")

if st.session_state.current_session_id:
    current_session = st.session_state.chat_sessions[st.session_state.current_session_id]
    st.caption(f"💬 Chat started at {current_session['created']}")
else:
    st.info("👈 Click 'New Chat' to start")
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
else:
    for msg in messages:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

# Chat input - FIXED TO PREVENT CRASHES
if not st.session_state.processing:
    user_input = st.chat_input("💬 Ask me anything...", key="chat_input")
    
    if user_input:
        st.session_state.processing = True
        
        # Add user message
        add_message("user", user_input)
        
        # Display user message immediately
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        
        # Get and display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤔 Thinking..."):
                answer = get_answer(user_input)
                st.markdown(answer)
                add_message("assistant", answer)
        
        st.session_state.processing = False
        st.rerun()

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #86efac; padding: 20px;'>
        <p>🎓 Built for GUVI Project | ⚡ Powered by Groq LLaMA 3.3 & LangChain</p>
    </div>
""", unsafe_allow_html=True)
