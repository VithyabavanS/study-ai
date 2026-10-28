from langchain_core.documents import Document

from hybrid import BM25Retriever, reciprocal_rank_fusion, tokenize


def doc(text):
    return Document(page_content=text)


def test_tokenize_keeps_tamil_words_whole():
    assert tokenize("வணக்கம், உலகம்!") == ["வணக்கம்", "உலகம்"]


def test_bm25_ranks_exact_keyword_first():
    chunks = [doc("Training data and batching"), doc("We used the Adam optimizer"), doc("Model results")]
    results = BM25Retriever(chunks).similarity_search("Which optimizer was used?", k=1)
    assert results[0].page_content == "We used the Adam optimizer"


def test_rrf_prefers_chunks_ranked_high_in_both_lists():
    a, b, c = doc("a"), doc("b"), doc("c")
    fused = reciprocal_rank_fusion([[a, b, c], [b, c, a]])
    assert fused[0].page_content == "b"


def test_rrf_removes_duplicates():
    a, b = doc("a"), doc("b")
    fused = reciprocal_rank_fusion([[a, b], [a, b]])
    assert len(fused) == 2