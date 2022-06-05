from transformers import BigBirdModel, AutoTokenizer, AutoConfig, AutoModel
import tqdm
import numpy as np

MODEL_NAME = "SajjadAyoubi/distil-bigbird-fa-zwnj"

# TODO: check for fine-tuning
model = BigBirdModel.from_pretrained(MODEL_NAME, attention_type="original_full")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def batch_series(iterable, n=2000):
    length = len(iterable)
    for ndx in range(0, length, n): yield iterable[ndx:min(ndx + n, length)]


def get_transformer_embedding(documents):
    series = [doc for doc in documents]
    result = None

    for batch in tqdm.tqdm(batch_series(series)):
        output = model(**tokenizer(batch, return_tensors='pt', padding=True))
        output = np.mean(output.last_hidden_state.detach().numpy(), axis=1)

        if result is None:
            result = output
        else:
            result = np.concatenate((result, output))

    return result


poems_embeddings = get_transformer_embedding(df.poems.tolist())

with open('../models/embeddings.npy', 'w') as file:
    np.save(file, poems_embeddings)