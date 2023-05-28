import torch


class Dataset(torch.utils.data.Dataset):

    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, i):
        return {key: tensor[i] for key, tensor in self.encodings.items()}


def load_data(batch_size=8):
    print("Loading tensors from disk")
    input_ids = torch.load("./law-data/input_ids.pt")
    mask = torch.load("./law-data/mask.pt")
    labels = torch.load("./law-data/labels.pt")

    encodings = {
        'input_ids': input_ids,
        'attention_mask': mask,
        'labels': labels
    }
    print("Creating dataloader")
    dataset = Dataset(encodings)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("Dataloader created")
    return dataloader
