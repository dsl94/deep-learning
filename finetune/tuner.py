from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
import torch
from transformers import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


def save(model, optimizer):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, 'output_model.pt')


def train_model(dataloader, epochs=2):

    model = RobertaForMaskedLM("./srberta-model")  # randomly initialized weights

    torch.cuda.empty_cache()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    print("Device for training: " + str(device))

    model.to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=3e-5)
    writer = SummaryWriter("./runs_v2")

    print("Starting training in " + str(epochs) + " epochs")
    for epoch in range(epochs):
        step = 0
        # setup loop with TQDM and dataloader
        loop = tqdm(dataloader, leave=True)

        for batch in loop:

            optim.zero_grad()

            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optim.step()

            loop.set_description(f'Epoch: {epoch}')
            loop.set_postfix(loss=loss.item())

            writer.add_scalar("Loss/train", loss, step)
            writer.flush()
            step += 1

    print("Training completed, saving model to folder: srberta_model")
    torch.save({
        'optimizer_law_state_dict': optim.state_dict()
    }, 'optimizer_law.pt')

    model.save_pretrained("./srberta_law_model")
    save(model, optim)

