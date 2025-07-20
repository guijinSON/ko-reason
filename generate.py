import json
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
from vllm import LLM, SamplingParams
from datasets import load_dataset
import torch
import pandas as pd
from more_itertools import batched

def safe_parse(output,i):
    try:
        return output.outputs[i].text
    except:
        return None
        
model_name = "Qwen/Qwen3-32B"
model = LLM(model=model_name, tensor_parallel_size=torch.cuda.device_count())
tokenizer = model.get_tokenizer()


df = load_dataset(
    'amphora/korean-stem',
    token='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',split='train'
                 ).to_pandas()


N=4
params = SamplingParams(
        temperature=0.6,top_p=0.95,
        top_k=20, min_p=0, max_tokens=16384, n=8
    )

idx=0
for batch in batched(df.instruction.values,500):
    qrys = []
    for query in batch:
        qrys.append(
                tokenizer.apply_chat_template([
                    {"role": "user", "content": 'Think carefully, after you finish thinking, state your answer in fluent and coherent Korean.'},
                    {"role": "user", "content": query}
                ], tokenize=False, add_generation_prompt=True, enable_thinking=True)
            )
    outputs = model.generate(qrys, params)
    dfs = []
    for i in range(N):
        responses = [safe_parse(out,i) for out in outputs]
        dfs.append(
            pd.DataFrame({'query':batch,'responses':responses})
        )
    save = pd.concat(dfs)
    save.to_csv(f'ko-stem-{idx}.csv',index=False)
    idx+=1