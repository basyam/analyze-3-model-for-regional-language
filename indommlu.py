"""Mengambil dan menampilkan dataset IndoMMLU dari Hugging Face."""

import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download


# Repo asli IndoMMLU berisi satu file CSV dan satu split: test.
# Memakai loader CSV menghindari loader SEACrowd dan dependency tambahan.
csv_path = hf_hub_download(
    repo_id="indolem/IndoMMLU",
    filename="IndoMMLU.csv",
    repo_type="dataset",
)
df = pd.read_csv(csv_path)
dataset = DatasetDict(
    {"test": Dataset.from_pandas(df, preserve_index=False)}
)

print(dataset)
print("Kolom:", dataset["test"].column_names)
print("Jumlah data:", len(dataset["test"]))
print("Contoh pertama:")
print(dataset["test"][0])
