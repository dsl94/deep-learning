from datasets import load_dataset
from tqdm.auto import tqdm

hugging_face_token = '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'

def get_oscar_dataset():
    print("Downloading oscar-corpus for Serbian language")
    dataset = load_dataset("oscar-corpus/OSCAR-2301",
                           cache_dir="dataset_cache",
                           use_auth_token=hugging_face_token,
                           language="sr",
                           streaming=False)
    print("Downloading oscar-corpus for Serbian language")
    print("Writing dataset to files")
    text_data = []
    file_count = 0
    for sample in tqdm(dataset['train']):

        sample = sample['text'].replace('\n', '')
        text_data.append(sample)

        if len(text_data) == 10_000:
            with open(f'sr_{file_count}.txt', 'w', encoding='utf-8') as fp:
                fp.write('\n'.join(text_data))
            text_data = []
            file_count += 1

    # after saving in 10K chunks, we have to add leftovers
    with open(f'sr_{file_count}.txt', 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(text_data))
    print("Finished writing dataset to files")