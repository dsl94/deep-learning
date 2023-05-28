from transformers import pipeline, RobertaTokenizerFast
from transformers import RobertaForMaskedLM

tokenizer = RobertaTokenizerFast.from_pretrained("srberta_tokenizer")
model = RobertaForMaskedLM.from_pretrained("./srberta_law_model")
model.to('cpu')


def guess(sentence):
    fill = pipeline('fill-mask', model=model, tokenizer=tokenizer)
    fill(sentence)

    return fill