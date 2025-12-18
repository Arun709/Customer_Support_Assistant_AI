import os
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file!")

print("✅ API Key loaded successfully")

# ============================================================
# 2. LOAD DATA FROM CSV
# ============================================================
print("\n📂 Loading knowledge base from CSV...")
try:
    loader = CSVLoader(
        file_path="data/knowledge_base.csv", 
        encoding="utf-8"
    )
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} documents")
except FileNotFoundError:
    print("❌ Error: data/knowledge_base.csv not found!")
    print("Please create the data/ folder and add knowledge_base.csv")
    exit(1)

# ============================================================
# 3. CREATE EMBEDDINGS MODEL
# ============================================================
print("\n🧠 Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("✅ Embedding model loaded (HuggingFace)")

# ============================================================
# 4. CREATE VECTOR DATABASE
# ============================================================
print("\n💾 Building FAISS vector database...")
vector_db = FAISS.from_documents(documents, embeddings)
print("✅ Vector database created successfully")

# ============================================================
# 5. SETUP RETRIEVER
# ============================================================
retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Retrieve top 3 most relevant docs
)
print("✅ Retriever configured (Top-3 similarity search)")

# ============================================================
# 6. INITIALIZE GEMINI LLM
# ============================================================
print("\n🤖 Initializing Google Gemini LLM...")
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,  # Lower = more focused, Higher = more creative
    google_api_key=google_api_key
)
print(f"✅ Using model: gemini-1.5-flash")

# ============================================================
# 7. CREATE PROMPT TEMPLATE
# ============================================================
prompt_template = """You are a helpful and professional customer support assistant for an e-commerce company.

Use the following context to answer the customer's question accurately.
If the answer is not in the context, politely say you don't have that information.
Keep your answers concise and friendly.

Context:
{context}

Customer Question: {question}

Your Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)
print("✅ Prompt template configured")

# ============================================================
# 8. CREATE RETRIEVAL QA CHAIN
# ============================================================
print("\n🔗 Building RAG chain...")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff" = put all retrieved docs into prompt
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True  # Returns which docs were used
)
print("✅ RAG pipeline ready!")

# ============================================================
# 9. TEST THE SYSTEM
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 RAG CHATBOT TEST - RUNNING SAMPLE QUERIES")
    print("="*70)
    
    # Sample test questions
    test_questions = [
        "What is your return policy?",
        "How can I track my order?",
        "Do you offer discounts?"
    ]
    
    for idx, question in enumerate(test_questions, 1):
        print(f"\n[Test {idx}/{len(test_questions)}]")
        print(f"❓ Question: {question}")
        print("-" * 70)
        
        try:
            # Invoke the RAG chain
            result = qa_chain.invoke({"query": question})
            answer = result['result']
            sources = result.get('source_documents', [])
            
            # Display answer
            print(f"💬 Answer:\n{answer}")
            
            # Display which documents were retrieved (optional debug info)
            if sources:
                print(f"\n📄 Retrieved {len(sources)} relevant documents")
            
            print("-" * 70)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("-" * 70)
    
    print("\n✅ All tests completed!")
    print("\nℹ️  Next step: Run 'streamlit run app.py' to launch the UI")
