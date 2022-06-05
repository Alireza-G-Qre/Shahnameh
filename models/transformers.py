from transformers import BigBirdModel, AutoTokenizer, AutoConfig, AutoModel
import tqdm
import numpy as np


def batch_series(iterable, n=2000):
    length = len(iterable)
    for ndx in range(0, length, n): yield iterable[ndx:min(ndx + n, length)]


def get_transformer_embedding(documents, model, tokenizer):
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


class TransformerEmbedding:

    def __init__(self, df, model_name="SajjadAyoubi/distil-bigbird-fa-zwnj", output_addr="../models/embeddings.npy"):
        self.model_name = model_name
        self.output_addr = output_addr
        self.df = df

    def run(self):
        model = BigBirdModel.from_pretrained(self.model_name, attention_type="original_full")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        poems_embeddings = get_transformer_embedding(self.df.poems.tolist(), model, tokenizer)

        with open(self.output_addr, 'wb') as file:
            np.save(file, poems_embeddings)
