# ============================================================================
# IMPORTS
# ============================================================================
import streamlit as st
import os
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from vector_store import VectorStore
from chat_engine import ChatEngine


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="StudyAI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================================
# ULTRA MODERN CSS
# ============================================================================
st.markdown("""
<style>
    /* Import Modern Font */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');
    
    /* Reset & Base */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* Dark Theme Background */
    .stApp {
        background: #0a0a0a;
        color: #ffffff;
    }
    
    .main {
        background: #0a0a0a;
        padding: 0;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Header */
    .custom-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        position: sticky;
        top: 0;
        z-index: 1000;
    }
    
    .logo {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Main Container */
    .container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 3rem 2rem;
    }
    
    /* Hero Section */
    .hero {
        text-align: center;
        padding: 4rem 0;
        margin-bottom: 3rem;
    }
    
    .hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 4.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        line-height: 1.2;
    }
    
    .hero p {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        color: #a0aec0;
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* Card System */
    .card {
        background: #1a1a1a;
        border: 1px solid #2d2d2d;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        border-color: #667eea;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.2);
        transform: translateY(-2px);
    }
    
    .card-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: white;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Upload Area */
    .upload-area {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 2px dashed #667eea;
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .upload-area:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border-color: #764ba2;
    }
    
    /* Input Fields */
    .stTextInput input {
        background: #1a1a1a !important;
        border: 1px solid #2d2d2d !important;
        border-radius: 12px !important;
        color: white !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    .stTextInput input::placeholder {
        color: #4a5568 !important;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 1rem 2.5rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.5) !important;
    }
    
    /* Answer Box */
    .answer-container {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d1b3d 100%);
        border: 1px solid #667eea;
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.2);
    }
    
    .answer-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
        font-size: 1.2rem;
        font-weight: 600;
        color: #667eea;
    }
    
    .answer-text {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        line-height: 1.8;
        color: #e2e8f0;
    }
    
    /* Source Cards */
    .source-item {
        background: #1a1a1a;
        border-left: 3px solid #667eea;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.2s ease;
    }
    
    .source-item:hover {
        background: #252525;
        transform: translateX(5px);
    }
    
    /* Messages */
    .stSuccess, .stInfo, .stWarning {
        background: #1a1a1a !important;
        border: 1px solid #2d2d2d !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        color: white !important;
    }
    
    /* Chat Message */
    .chat-bubble {
        background: #1a1a1a;
        border: 1px solid #2d2d2d;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.2s ease;
    }
    
    .chat-bubble:hover {
        border-color: #667eea;
        background: #1f1f1f;
    }
    
    .question-text {
        color: #667eea;
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    .answer-preview {
        color: #a0aec0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Stats */
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: white;
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }
    
    /* File Uploader Custom */
    [data-testid="stFileUploader"] {
        background: transparent;
    }
    
    [data-testid="stFileUploader"] section {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 2px dashed #667eea;
        border-radius: 16px;
        padding: 2rem;
    }
    
    [data-testid="stFileUploader"] section:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #1a1a1a !important;
        border: 1px solid #2d2d2d !important;
        border-radius: 12px !important;
        color: white !important;
        font-weight: 500 !important;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #667eea !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        background: rgba(102, 126, 234, 0.2);
        color: #667eea;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================================
load_dotenv()


# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================
if 'processed_chunks' not in st.session_state:
    st.session_state.processed_chunks = None
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Ultra modern UI"""
    
    # Custom Header
    st.markdown("""
        <div class='custom-header'>
            <div class='logo'>✨ StudyAI</div>
            <div style='color: rgba(255,255,255,0.7); font-size: 0.9rem;'>
                Your intelligent study companion
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
        <div class='hero'>
            <h1>Learn Smarter,<br>Not Harder</h1>
            <p>Upload any document and get instant, intelligent answers powered by AI</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Main Container
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        # Upload Section
        st.markdown("<div class='card-title'>📄 Upload Your Document</div>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload",
            type=["pdf"],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            st.success(f"✓ {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
            
            if st.session_state.current_file != uploaded_file.name:
                st.session_state.processed_chunks = None
                st.session_state.vectorstore = None
                st.session_state.current_file = uploaded_file.name
                st.session_state.chat_history = []
            
            if st.button("⚡ Process Document", use_container_width=True):
                with st.spinner("Processing..."):
                    try:
                        processor = DocumentProcessor(uploaded_file, chunk_size=1000, chunk_overlap=200)
                        chunks = processor.process()
                        st.session_state.processed_chunks = chunks
                        
                        # Get API key from sidebar if exists
                        api_key = st.session_state.get('api_key', None)
                        use_openai = st.session_state.get('use_openai', False)
                        
                        vs = VectorStore(api_key=api_key, use_openai=use_openai)
                        vs.create_vectorstore(chunks)
                        st.session_state.vectorstore = vs
                        
                        st.success(f"✨ Ready! {len(chunks)} chunks created")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Stats if processed
        if st.session_state.processed_chunks:
            st.markdown("---")
            cols = st.columns(2)
            with cols[0]:
                st.markdown(f"""
                    <div class='stat-box'>
                        <div class='stat-value'>{len(st.session_state.processed_chunks)}</div>
                        <div class='stat-label'>Chunks</div>
                    </div>
                """, unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f"""
                    <div class='stat-box'>
                        <div class='stat-value'>{len(st.session_state.chat_history)}</div>
                        <div class='stat-label'>Questions</div>
                    </div>
                """, unsafe_allow_html=True)
    
    with col2:
        # Question Section
        st.markdown("<div class='card-title'>💬 Ask Anything</div>", unsafe_allow_html=True)
        
        question = st.text_input(
            "Question",
            placeholder="What would you like to know?",
            label_visibility="collapsed"
        )
        
        if st.button("🔍 Get Answer", use_container_width=True, type="primary"):
            if not st.session_state.vectorstore:
                st.warning("⚠️ Please upload and process a document first")
            elif not question:
                st.warning("⚠️ Please enter a question")
            else:
                with st.spinner("Thinking..."):
                    try:
                        api_key = st.session_state.get('api_key', None)
                        use_openai = st.session_state.get('use_openai', False)
                        
                        chat_engine = ChatEngine(
                            vectorstore=st.session_state.vectorstore,
                            api_key=api_key,
                            use_openai=use_openai and bool(api_key)
                        )
                        
                        result = chat_engine.get_answer(question, k=3)
                        
                        # Display answer
                        st.markdown("""
                            <div class='answer-container'>
                                <div class='answer-header'>✨ Answer</div>
                                <div class='answer-text'>
                        """ + result['answer'].replace('\n', '<br>') + """
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Sources
                        if result["sources"]:
                            with st.expander("📚 View Sources"):
                                for i, doc in enumerate(result["sources"], 1):
                                    st.markdown(f"""
                                        <div class='source-item'>
                                            <strong>Source {i}</strong><br>
                                            {doc.page_content[:200]}...
                                        </div>
                                    """, unsafe_allow_html=True)
                        
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": result["answer"]
                        })
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    # Chat History
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("<div class='card-title'>💭 Recent Questions</div>", unsafe_allow_html=True)
        
        for chat in reversed(st.session_state.chat_history[-3:]):
            st.markdown(f"""
                <div class='chat-bubble'>
                    <div class='question-text'>Q: {chat['question']}</div>
                    <div class='answer-preview'>{chat['answer'][:150]}...</div>
                </div>
            """, unsafe_allow_html=True)
    
    # Sidebar for settings
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        api_key = st.text_input("OpenAI API Key", type="password", key="api_key")
        use_openai = st.checkbox("Use GPT Mode", key="use_openai")
        
        if use_openai and api_key:
            st.success("🤖 AI Mode Active")
        else:
            st.info("🆓 Free Mode")


if __name__ == "__main__":
    main()