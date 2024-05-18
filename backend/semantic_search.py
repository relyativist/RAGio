import os
from typing import List

import gradio as gr
import lancedb
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer
from torch.nn import Sigmoid


db = lancedb.connect(".lancedb")

DEVICE = os.getenv("DEVICE", "cpu")
VECTOR_COLUMN = os.getenv("VECTOR_COLUMN", "vector")
TEXT_COLUMN = os.getenv("TEXT_COLUMN", "text")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))

retriever = SentenceTransformer(os.getenv("EMB_MODEL"), device=DEVICE)
reranker = CrossEncoder(os.getenv("RERANK_MODEL"), device=DEVICE)


def search(vs_name, query, k = 25):
    query_vec = retriever.encode(query, show_progress_bar=False)
    try:
        table = db.open_table(vs_name)
        search_results = table.search(query_vec, vector_column_name=VECTOR_COLUMN).limit(k)
    except Exception as e:
        raise gr.Error(str(e))

    return search_results


def retrieve(vs_name, query, k : int = 25, rerank : bool = False, top_k : int = 5) -> List[str]:
    """
    Retrieves search results based on the query with optional reranking.
    
    Args:
        vs_name (str): The name of the vector store to use.
        query (str): The input query for search.
        k (int): The number of search results to retrieve (default is 25).
        rerank (bool): Flag indicating whether to perform reranking (default is False).
        top_k (int): The number of top results to return after reranking (default is 5).
        
    Returns:
        List[str]: A list of retrieved text based on the query.
    """
    try:
        semantic_search = search(vs_name, query, k = k)
        if rerank:
            chunks = semantic_search.to_pandas().reset_index(drop=True)
            query_chunks_comb = [[query, chunk] for chunk in chunks["text"]]
            chunks["_distance_reranked"] = reranker.predict(query_chunks_comb, activation_fct=Sigmoid())
            chunks = chunks.sort_values("_distance_reranked", ascending=False).head(top_k)
            retrievals = list(chunks.text)
        else:
            retrievals = [doc[TEXT_COLUMN] for doc in semantic_search.to_list()]
    except Exception as e:
        raise ValueError(str(e))
    
    return retrievals
