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
    print("Loading tokenizer")
    tokenizer_srberta = RobertaTokenizerFast.from_pretrained("srberta_tokenizer")
    config = RobertaConfig(
        vocab_size=tokenizer_srberta.vocab_size,
        max_position_embeddings=514,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )
    model = RobertaForMaskedLM(config)  # randomly initialized weights

    torch.cuda.empty_cache()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    print("Device for training: " + str(device))

    model.to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=1e-4)
    writer = SummaryWriter("./runs_v2")

    print("Starting training in " + str(epochs) + " epochs")
    for epoch in range(epochs):
        step = 0
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
        'optimizer_state_dict': optim.state_dict()
    }, 'optimizer.pt')

    model.save_pretrained("./srberta_model")
    save(model, optim)

