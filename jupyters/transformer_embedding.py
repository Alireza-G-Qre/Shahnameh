from transformers import BigBirdModel, AutoTokenizer, AutoConfig, AutoModel
import tqdm
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz


def batch_series(iterable, n=2_000):
    length = len(iterable)
    for ndx in range(0, length, n): yield iterable[ndx:min(ndx + n, length)]


class TransformerEmbedding:

    def __init__(self, df, batch_size=2_000):
        
        self.batch_size, self.df = df = batch_size, df
        
        self.model_names = {
            'bigbird': 'SajjadAyoubi/distil-bigbird-fa-zwnj',
            'parsbert': 'HooshvareLab/bert-base-parsbert-uncased'
        }
        
        self.tokenizers = {
            'bigbird': AutoTokenizer.from_pretrained(self.model_names['bigbird']),
            'parsbert': AutoTokenizer.from_pretrained(self.model_names['parsbert'])
        }
        
        self.models = {
            'bigbird': BigBirdModel.from_pretrained(
                self.model_names['bigbird'], attention_type='original_full'),
            
            'parsbert': AutoModel.from_pretrained(
                self.model_names['parsbert'], 
                config=AutoConfig.from_pretrained(self.model_names['parsbert']))
        }

    def get_transformer_embedding(self, documents, model_name):
        model, tokenizer = \
            self.models[model_name], self.tokenizers[model_name]
        
        series = [doc for doc in documents]
        result = None

        for batch in tqdm.tqdm(batch_series(series, self.batch_size)):
            output = model(**tokenizer(batch, return_tensors='pt', padding=True))
            output = np.mean(output.last_hidden_state.detach().numpy(), axis=1)

            if result is None:
                result = output
            else:
                result = np.concatenate((result, output))

        return result
    
    def load_embeddings(self, file_address): return load_npz(file_address).toarray()

    def run_and_dump(self):
        
        poems_embeddings = self.get_transformer_embedding(self.df.poems.tolist(), 'bigbird')
        csr_embeddings = csr_matrix(poems_embeddings)
        save_npz('../models/embeddings-bigbird.npz', csr_embeddings)

        poems_embeddings = self.get_transformer_embedding(self.df.poems.tolist(), 'parsbert')
        csr_embeddings = csr_matrix(poems_embeddings)
        save_npz('../models/embeddings-parsbert.npz', csr_embeddings)
