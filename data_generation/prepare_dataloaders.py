

import torch

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)

model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = T5Tokenizer.from_pretrained(model_name)