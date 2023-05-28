from pathlib import Path
import os
from transformers import RobertaTokenizerFast
from tokenizers.decoders import ByteLevel
from tokenizers import ByteLevelBPETokenizer

def train_tokenizer():
    paths = [str(x) for x in Path('./').glob('*.txt')]
    paths = paths[0:50]
    print("Starting tokenizer training")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=paths,
        vocab_size=30_522,
        min_frequency=2, show_progress=True,
        special_tokens=[
            '<s>', '<pad>', '</s>', '<unk>', '<mask>'
        ]
    )
    print("Finished tokenizer training")
    print("Saving tokenizer")
    os.mkdir('./srberta_tokenizer')
    tokenizer.save_model('srberta_tokenizer')
    srberta_tokenizer = RobertaTokenizerFast.from_pretrained("srberta_tokenizer")
    print("Tokenizer saved in folder: srberta_tokenizer")
