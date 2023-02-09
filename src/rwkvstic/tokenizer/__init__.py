from transformers import PreTrainedTokenizerFast
import os
path = "20B_tokenizer.json"
path = os.path.join(os.path.dirname(__file__), path)


def tokenizer(x=None): return PreTrainedTokenizerFast(
    tokenizer_file=x if x is not None else path)
