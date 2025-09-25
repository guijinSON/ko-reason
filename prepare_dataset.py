from datasets import load_dataset
from tqdm.auto import tqdm
import pandas as pd

df = load_dataset(
    "XXXX/kor1-YiSang", split="train"
).to_pandas()

msg = {"messages": []}
for _,row in tqdm(df.iterrows(), total=len(df)):
    msg["messages"].append([
        {
            "role": "user",
            "content": row.instruction
        },
        {
            "role": "assistant",
            "content": row.response
        }
    ])

ds = pd.DataFrame(msg)
ds.to_json("train_YS.jsonl", lines=True, orient="records")
