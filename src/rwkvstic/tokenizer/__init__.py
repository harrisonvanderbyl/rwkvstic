from transformers import PreTrainedTokenizerFast
import os
path = "20B_tokenizer.json"
path = os.path.join(os.path.dirname(__file__), path)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=path)
