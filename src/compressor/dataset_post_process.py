from datasets import Dataset
from pathlib import Path
from copy import deepcopy
import json
import tqdm


stg3_dataset_path = Path("dataset/stage2/")
print(stg3_dataset_path)

def save_pads_number(row):
    pads_number = len(row['token_ids']) - len(json.loads(row['braced_token_ids'])) + len(json.loads(row['compress_ids']))
    assert pads_number >= 0, "The pads number should be greater than or equal to 0."
    row['pads_number'] = pads_number
    return row

for data_file in tqdm.tqdm(list(stg3_dataset_path.glob("*.parquet_test10000"))):
    ds = Dataset.from_parquet(str(data_file))
    ds = ds.map(save_pads_number)
    print(ds.column_names)
    ds.to_parquet(str(data_file))
    
r1 = ds[0]
print("1")

