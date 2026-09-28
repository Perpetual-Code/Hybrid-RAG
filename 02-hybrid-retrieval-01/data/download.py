from datasets import load_dataset
from pathlib import Path
import pandas as pd

ROOT_DIR=Path(__file__).resolve().parent
DATA_DIR=ROOT_DIR/"data"/"fiqa"

def download_fiqa_data():
    download_paths={
        "corpus":DATA_DIR/"corpus.parquet",
        "queries":DATA_DIR/"queries.parquet",
        "qrels":DATA_DIR/"qrels.parquet"
    }

    for path in download_paths.values():
        if path.exists():
            print(f"File already saved")
        else:
            corpus_data=load_dataset("BeIR/fiqa","corpus",split="corpus")
            queries_data=load_dataset("BeIR/fiqa","queries",split="queries")
            qrels_data=load_dataset("BeIR/fiqa-qrels",split="test")

            corpus=corpus_data.to_parquet(DATA_DIR/"corpus.parquet")
            queries=queries_data.to_parquet(DATA_DIR/"queries.parquet")
            qrels=qrels_data.to_parquet(DATA_DIR/"qrels.parquet")

if __name__=="__main__":
        print("downloading data")
        download_fiqa_data()




