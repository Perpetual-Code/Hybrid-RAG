from pathlib import Path

import pandas as pd
from datasets import load_dataset

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "fiqa"
print(DATA_DIR)

download_paths = {
    "corpus": DATA_DIR / "corpus.parquet",
    "queries": DATA_DIR / "queries.parquet",
    "qrels": DATA_DIR / "qrels.parquet",
}


def download_data():
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(exist_ok=True, parents=True)
    corpus_data = load_dataset("BeIR/fiqa", "corpus", split="corpus")
    queries_data = load_dataset("BeIR/fiqa", "queries", split="queries")
    qrels_data = load_dataset("BeIR/fiqa-qrels", split="test")

    corpus_data.to_parquet(DATA_DIR / "corpus.parquet")
    queries_data.to_parquet(DATA_DIR / "queries.parquet")
    qrels_data.to_parquet(DATA_DIR / "qrels.parquet")


def main():
    for path in download_paths.values():
        if path.exists():
            print("file saved")
        else:
            download_data()

    corpus_file = DATA_DIR / "corpus.parquet"
    print(pd.read_parquet(corpus_file).columns)


if __name__ == "__main__":
    main()
