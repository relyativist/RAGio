import hashlib
import re, os
from typing import Dict
from typing import List

import fitz
import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa
from sentence_transformers import SentenceTransformer
from spacy.lang.en import English
from tqdm.auto import tqdm


DEVICE = os.getenv("DEVICE", "cpu")

def rem_dub_spaces(chunk):
    return "".join(chunk).replace("  ", " ").strip()


def add_space_dot(chunk):
    return re.sub(r"\.([A-Z])", r". \1", chunk)


def text_processor(text: str) -> str:

    proc_text = text.replace("\n", " ")

    return proc_text


def textsplitter(text, num_sents: int = 4) -> List[List[str]]:
    """
    Group sentences in chunks
    args:
      text : str
      num_sents : int  - number of sentences in one chunk, e.g. [25] -> [int, int ,mod]
    return : List
    """

    nlp = English()
    nlp.add_pipe("sentencizer")
    dot_sep_list = list(nlp(text).sents)
    dot_sep_str = [str(x) for x in dot_sep_list]

    sentence_chunks = [
        dot_sep_str[i : i + num_sents] for i in range(0, len(dot_sep_str), num_sents)
    ]

    text_chunks = []

    for chunk in sentence_chunks:
        chunkd = {}
        joined_chunk = "".join(chunk).replace("  ", " ").strip()
        joined_chunk = re.sub(
            r"\.([A-Z])", r". \1", joined_chunk  # add space to sentence ending dot, e.g. .A => . A
        )  
        chunkd["sentence_chunk"] = joined_chunk
        chunkd["chunk_char_count"] = len(joined_chunk)
        chunkd["chunk_word_count"] = len([word for word in joined_chunk.split(" ")])
        chunkd["chunk_sentence_count"] = len(joined_chunk.split("."))
        chunkd["chunk_token_count"] = len(joined_chunk) / 4
        text_chunks.append(chunkd)

    return text_chunks


def chunking(text: str, chunk_size: int = 4, method: str = "sentence") -> List[str]:

    if method == "sentence":
        chunks = textsplitter(text, num_sents=chunk_size)

    return chunks


def pdf_parser(pdf_path: str) -> List[Dict]:
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in tqdm(enumerate(doc)):
        text = page.get_text()
        pages.append(
            {
                "page": i + 1,
                "page_chars": len(text),
                "page_words": len(text.split(" ")),
                "page_sentences_naive": len(text.split(". ")),
                "page_tokens": len(text)/ 4,  # ~ 1 token is 4 chars https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
                "text": text,
            }
        )
    return pages


def embedder(doc):

    SENTS_EMBEDDER_MODEL = os.getenv("EMB_MODEL")  # sentence embedder model

    sents_embedder = SentenceTransformer(SENTS_EMBEDDER_MODEL, device=DEVICE)
    sents_embedder.eval()

    
    BATCH_SIZE = 32  #  batches for embedder, we don't want to embed all sentences at once)
    LANCE_DB_LOC = "./.lancedb"  #  location for .lancedb on host
    NUM_SUB_VEC = sents_embedder.max_seq_length  #  max token length of embedder model
    EMBEDDING_DIM_MODEL = 768  #  Transformer embedding dimension
    NUM_PARTITIONS_VEC = 128
    NUM_SUB_VEC = 96
    VEC_COLUMN = "vector"  #  vector table embeddings column name
    TEXT_COLUMN = "text"

    assert (
        EMBEDDING_DIM_MODEL % NUM_SUB_VEC == 0
    ), "Embedding size must be divisible by the num of sub vectors"

    db = lancedb.connect(LANCE_DB_LOC)

    schema = pa.schema(
        [
            pa.field(VEC_COLUMN, pa.list_(pa.float32(), EMBEDDING_DIM_MODEL)),
            pa.field(TEXT_COLUMN, pa.string()),
        ]
    )  #  vector table text column name

    vs_hash = hashlib.sha256(f"{SENTS_EMBEDDER_MODEL}_{NUM_SUB_VEC}".encode("utf-8")).hexdigest()
    vs_name = f"vs_{vs_hash}"
    tbl = db.create_table(vs_name, schema=schema, mode="overwrite")

    pdf_page_texts = pdf_parser(doc)
    sentences = []
    for page in tqdm(pdf_page_texts):
        page["sentence_chunks"] = chunking(page["text"], method="sentence")
        page["page_sentences_splitter"] = len(page["sentence_chunks"])
        for chunk in page["sentence_chunks"]:
            sentences.append(chunk["sentence_chunk"])

    for i in tqdm(range(0, int(np.ceil(len(sentences) / BATCH_SIZE)))):
        try:
            batch = [sent for sent in sentences[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] if len(sent) > 0
            ]
            encoded = sents_embedder.encode(
                batch, normalize_embeddings=True
            )
            encoded = [list(vec) for vec in encoded]

            df = pd.DataFrame({VEC_COLUMN: encoded, TEXT_COLUMN: batch})
            tbl.add(df)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Error on batch #{i}: {e}")

    tbl.create_index(
        num_partitions=NUM_PARTITIONS_VEC,
        num_sub_vectors=NUM_SUB_VEC,
        vector_column_name=VEC_COLUMN,
        accelerator="cuda" if DEVICE == "gpu" else None
    )
    return vs_name
