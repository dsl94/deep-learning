from pretrain import data
from pretrain import tokenizer
from pretrain import input_pipeline
from pretrain import data_loader
from pretrain import trainer

if __name__ == '__main__':
    # data.get_oscar_dataset()
    # tokenizer.train_tokenizer()
    # input_pipeline.create_input_tensors()
    dataloader = data_loader.load_data()
    trainer.train_model(dataloader)