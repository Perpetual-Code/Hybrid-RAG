import bm25s
from pathlib import Path
import pandas as pd 


DATA_DIR=Path(__file__).parent /"data"/"fiqa"
INDEX_DIR=Path(__file__).parent/"indexes"/"bm25"

# Fiqa docs are forum posts , we need to index them to be able to work with bm25 


corpus=pd.read_parquet(DATA_DIR/"corpus.parquet")
id_col='_id' if '_id' in corpus.columns else 'id'
doc_ids=corpus[id_col].tolist()
doc_texts=corpus['text'].tolist()

print(f"Indexing {len(doc_texts)} documents with BM25")

#Step 2
# mb25s.tokenize lowercase,srips punctuations and removes English stopwords
# The result is toeknized object that can be passed straight to bm25 index

tokens=bm25s.tokenize(doc_texts,stopwords="en")

retriever=bm25s.BM25()
retriever.index(tokens)


# Step 3 
# Save the index and the doc_ids to disk in matching order so we can map back later
INDEX_DIR.mkdir(parents=True,exist_ok=True)
retriever.save(str(INDEX_DIR),corpus=doc_ids)
file=(INDEX_DIR/"doc_ids.txt")
file.write_text("\n".join(doc_ids))

def bm25_search(query:str,k:int=10)->list[tuple[int,float]]:
    query_tokens=bm25s.tokenize([query],stopwords="en")
    indices,scores=retriever.retrieve(query_tokens=query_tokens,k=k)
    

    return [
        (doc_ids[i],float(scores[0][j])) for j,i in enumerate(indices[0].tolist())
    ]


if __name__=="__main__":
    query="where should I park my rainy-day fund?"

    print(f"\nQuery : {query}\n")
    for i,(doc_id,score) in enumerate(bm25_search(query,k=5),1):
        text=corpus.loc[corpus["_id"]==doc_id,'text'].iloc[0]
        print(f"{i}, [{score:6,.2f}] {doc_id} {text[:80]}")
