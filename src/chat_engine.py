"""
Chat Engine Module

Handles question-answering using retrieved context and LLM.
"""

# ============================================================================
# IMPORTS
# ============================================================================
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from config import LLM_MODEL, LLM_TEMPERATURE, TOP_K


# ============================================================================
# CHAT ENGINE CLASS
# ============================================================================
class ChatEngine:
    """
    Generates answers to questions using retrieved context and LLM.
    
    Attributes:
        vectorstore: VectorStore instance with document embeddings
        api_key: OpenAI API key (optional, for GPT models)
        use_openai: Whether to use OpenAI LLM
    """
    
    def __init__(self, vectorstore, api_key=None, use_openai=False):
        """
        Initialize the chat engine.
        
        Args:
            vectorstore: VectorStore instance
            api_key: OpenAI API key (only if use_openai=True)
            use_openai: If True, use OpenAI GPT. If False, use template-based
        """
        self.vectorstore = vectorstore
        self.api_key = api_key
        self.use_openai = use_openai
        
        # Custom prompt template for better answers
        self.prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Answer:"""
        
        self.PROMPT = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        
        if use_openai and api_key:
            # Use OpenAI GPT (paid, high quality)
            self.llm = ChatOpenAI(
                openai_api_key=api_key,
                model_name=LLM_MODEL,
                temperature=LLM_TEMPERATURE
            )
        else:
            self.llm = None
    
    
    def _extract_key_sentences(self, text, question, max_sentences=3):
        """
        Extract the most relevant sentences from text based on question.
        Simple keyword-based extraction for free mode.
        
        Args:
            text: The text to extract from
            question: User's question
            max_sentences: Maximum sentences to return
            
        Returns:
            String with most relevant sentences
        """
        # Split into sentences
        sentences = text.replace('?', '.').replace('!', '.').split('.')
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # Extract keywords from question (simple approach)
        question_words = set(question.lower().split())
        # Remove common words
        stop_words = {'what', 'is', 'the', 'a', 'an', 'how', 'when', 'where', 'why', 'who', 'about'}
        keywords = question_words - stop_words
        
        # Score sentences based on keyword matches
        scored_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for keyword in keywords if keyword in sentence_lower)
            if score > 0:
                scored_sentences.append((score, sentence))
        
        # Sort by score and take top sentences
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        top_sentences = [s[1] for s in scored_sentences[:max_sentences]]
        
        return top_sentences
    
    
    def get_answer(self, question, k=TOP_K):
        """
        Get answer to a question using retrieved context.
        
        Args:
            question: User's question
            k: Number of relevant chunks to retrieve
            
        Returns:
            Dictionary with answer and source documents
        """
        try:
            # Get relevant chunks from vector store
            relevant_docs = self.vectorstore.similarity_search(question, k=k)
            
            if not relevant_docs:
                return {
                    "answer": "I couldn't find relevant information in the document to answer your question.",
                    "sources": []
                }
            
            if self.use_openai and self.llm:
                # Use OpenAI to generate answer
                qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.vectorstore.get_vectorstore().as_retriever(search_kwargs={"k": k}),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": self.PROMPT}
                )
                
                result = qa_chain({"query": question})
                
                return {
                    "answer": result["result"],
                    "sources": result["source_documents"]
                }
            else:
                # IMPROVED FREE MODE: Extract key sentences
                all_key_sentences = []
                for doc in relevant_docs:
                    key_sentences = self._extract_key_sentences(doc.page_content, question, max_sentences=2)
                    all_key_sentences.extend(key_sentences)
                
                if all_key_sentences:
                    # Format as a clean answer
                    answer = "**Based on your document:**\n\n"
                    for i, sentence in enumerate(all_key_sentences[:4], 1):  # Top 4 sentences
                        answer += f"{i}. {sentence}\n\n"
                    
                    answer += "\n---\n💡 *Using free mode with keyword extraction. For AI-generated natural answers, add an OpenAI API key in the sidebar.*"
                else:
                    # Fallback to showing chunks
                    context = "\n\n---\n\n".join([doc.page_content[:300] + "..." for doc in relevant_docs[:2]])
                    answer = f"**Relevant sections from your document:**\n\n{context}\n\n---\n💡 *Using free mode. For AI-generated answers, add an OpenAI API key.*"
                
                return {
                    "answer": answer,
                    "sources": relevant_docs
                }
        
        except Exception as e:  # noqa: BLE001 - temporary, chat engine is rewritten in Step 2
            return {
                "answer": f"❌ Error generating answer: {e!s}",
                "sources": []
            }