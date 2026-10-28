import numpy as np

from embeddings import PrefixedEmbeddings


class FakeModel:
    """Records what it was asked to encode instead of running a real model."""

    def __init__(self):
        self.seen = []

    def encode(self, inputs, normalize_embeddings=True):
        self.seen.append(inputs)
        if isinstance(inputs, list):
            return np.zeros((len(inputs), 2))
        return np.zeros(2)


def test_prefixes_are_added_for_documents_and_queries():
    fake = FakeModel()
    emb = PrefixedEmbeddings("unused", query_prefix="query: ", passage_prefix="passage: ", model=fake)

    emb.embed_documents(["a chunk"])
    emb.embed_query("a question")

    assert fake.seen == [["passage: a chunk"], "query: a question"]