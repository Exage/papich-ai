import os

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')


def tokenizer(config):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(config.embedding_model, trust_remote_code=False)


def token_counter(config):
    tok = tokenizer(config)
    if config.chunk_max_tokens > tok.model_max_length:
        raise ValueError('chunk_max_tokens exceeds tokenizer context length')
    return lambda text: len(tok.encode(config.passage_prefix + text, add_special_tokens=True))


class Embedder:
    def __init__(self, config):
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        self.config = config
        self.model = SentenceTransformer(config.embedding_model,
                                         device=config.embedding_device,
                                         trust_remote_code=False)
        dimension = getattr(self.model, 'get_embedding_dimension', None)
        self.dimension = (dimension() if dimension else self.model.get_sentence_embedding_dimension())

    def encode(self, texts, query=False):
        prefix = self.config.query_prefix if query else self.config.passage_prefix
        inputs = [prefix + text for text in texts]
        for text in inputs:
            tokens = self.model.tokenizer.encode(text, add_special_tokens=True)
            if len(tokens) > self.model.max_seq_length:
                raise ValueError(f'Text exceeds embedding context ({self.model.max_seq_length} tokens)')
        return self.model.encode(inputs, batch_size=self.config.embedding_batch_size,
                                 normalize_embeddings=True, show_progress_bar=False).tolist()
