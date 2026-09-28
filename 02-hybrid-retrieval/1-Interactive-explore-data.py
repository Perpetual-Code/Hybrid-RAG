from pathlib import Path
import pandas as pd

DATA_DIR=Path(__file__).parent/"data"/"fiqa"

corpus=pd.read_parquet(DATA_DIR/"corpus.parquet")
queries=pd.read_parquet(DATA_DIR/"queries.parquet")
qrels=pd.read_parquet(DATA_DIR/"qrels.parquet")

