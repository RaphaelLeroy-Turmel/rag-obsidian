from sentence_transformers import SentenceTransformer
from chunking import Chunk, MODEL_NAME
import numpy as np


_model = None

def get_model():
    '''return the model SentenceTransformer(MODEL_NAME) loaded once'''
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed_passages(chunks : list[Chunk]):
    '''embeds the clean chunks (already in format "passage : [...]") return a numpy array '''
    texts = [chunk.text_for_embedding for chunk in chunks]
    return get_model().encode(texts, batch_size=32, normalize_embeddings=True)

def embed_query(query :str):
    ''' parse and embed the query'''
    parsed_query = "query: " + query
    vector = get_model().encode(parsed_query, normalize_embeddings=True) 
    return vector
    


def get_best_embedding(vectors:np.ndarray, n:int, ref_vector):
    ''' return the indice of the n best corresponding vectors, by decreasing order (most similar first)'''
    scores = vectors @ ref_vector            # (n_chunks,): one cosine similarity per chunk
    return np.argsort(scores)[::-1][:n]      # argsort sorts ascending, [::-1] reverses it


if __name__ == "__main__":
    from cleaning import clean_files, load_env_file,env_path
    from chunking import chunk_files
    load_env_file()                                   # charge le .env (chemins)
    model = get_model()
    all_chunks = chunk_files(clean_files())  
    embeddings = embed_passages(all_chunks)         
    query = "What is the structure of the encoder layer in a Transformer" 
    answer ="The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network."
    embeded_query = embed_query(query)
    embeded_answer = _model.encode(answer, normalize_embeddings=True)
    index_best_embeddings = get_best_embedding(embeddings, n=10 ,ref_vector=embeded_query) 
    # an embeding and its original chunk have the same index
    best_indices = get_best_embedding(embeddings, n=10, ref_vector=embeded_query)
    for rank, i in enumerate(best_indices, start=1):
        chunk = all_chunks[i]
        print(f"\n#{rank} | score {embeddings[i] @ embeded_query:.3f} | {chunk.source} | {chunk.headings}")
        print(chunk.text[:300], "...")

