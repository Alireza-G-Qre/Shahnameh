from transformers import BigBirdModel, AutoTokenizer, AutoConfig, AutoModel
import tqdm
import numpy as np


def batch_series(iterable, n=2000):
    length = len(iterable)
    for ndx in range(0, length, n): yield iterable[ndx:min(ndx + n, length)]


class TransformerEmbedding:

    def __init__(self, df):
        self.df = df
        self.model_names = {
            'bigbird': 'SajjadAyoubi/distil-bigbird-fa-zwnj',
            'parsbert': 'HooshvareLab/bert-base-parsbert-uncased'
        }
        self.tokenizers = {
            'bigbird': AutoTokenizer.from_pretrained(self.model_names['bigbird']),
            'parsbert': AutoTokenizer.from_pretrained(self.model_names['parsbert'])
        }
        self.models = {
            'bigbird': BigBirdModel.from_pretrained(self.model_names['bigbird'], attention_type='original_full'),
            'parsbert': AutoModel.from_pretrained(
                self.model_names['parsbert'], config=AutoConfig.from_pretrained(self.model_names['parsbert']))
        }

    def get_transformer_embedding(self, documents, model_name):
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
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

    def run(self):
        poems_embeddings = self.get_transformer_embedding(self.df.poems.tolist(), 'bigbird')
        with open('../models/embeddings-bigbird.npy', 'wb') as file:
            np.save(file, poems_embeddings)

        poems_embeddings = self.get_transformer_embedding(self.df.poems.tolist(), 'parsbert')
        with open('../models/embeddings-parsbert.npy', 'wb') as file:
            np.save(file, poems_embeddings)
