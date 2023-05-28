import torch
from pathlib import Path
from transformers import RobertaTokenizerFast
from tqdm.auto import tqdm


def mlm(tensor):
    rand = torch.rand(tensor.shape) #[0,1]
    mask_arr = (rand < 0.15)* (tensor!=0)* (tensor!=1)* (tensor!=2)
    for i in range(tensor.shape[0]):
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        tensor[i, selection] = 4

    return tensor

def create_input_tensors():
    print("Reading all data files")
    paths = [str(x) for x in Path('./law-data').glob('*.txt')]

    print("Loading tokenizer")
    tokenizer_srberta = RobertaTokenizerFast.from_pretrained("srberta_tokenizer")

    input_ids = []
    mask = []  # attention mask
    labels = []

    print("Iterating through files and creating tensors")
    for path in tqdm(paths):
        print("File: " + path)
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        sample = tokenizer_srberta(lines, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        labels.append(sample.input_ids)
        mask.append(sample.attention_mask)
        input_ids.append(mlm(sample.input_ids.detach().clone()))

    input_ids = torch.cat(input_ids)
    mask = torch.cat(mask)
    labels = torch.cat(labels)
    print("Saving tensors to disk")
    torch.save(input_ids, './law-data/input_ids.pt')
    torch.save(mask, './law-data/mask.pt')
    torch.save(labels, './law-data/labels.pt')