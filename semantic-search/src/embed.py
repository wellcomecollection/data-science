import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Literal
import openai


class TextEmbedder:
    def __init__(
        self,
        model: Literal[
            "text-embedding-ada-002",
            "all-distilroberta-v1",
            "msmarco-bert-base-dot-v5",
            "all-mpnet-base-v2",
            "sentence-t5-xxl",
            "sentence-t5-xl",
        ],
        cache_dir: str | Path = None,
    ):
        self.model = model
        if self.model == "text-embedding-ada-002":
            self.embed = self._embed_with_openai
        else:
            self.encoder = SentenceTransformer(
                model_name_or_path=self.model,
                cache_folder="/data/models",
            )
            self.embed = self._embed_with_sentencetransformers

        if cache_dir is not None:
            self.cache_dir = Path(cache_dir) / self.model
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def use_cache(func):
        """Use a cache to save and load pre-computed embeddingss"""

        def wrapper(self, *args, **kwargs):
            if not self.cache_dir:
                return func(self, *args, **kwargs)
            if self.cache_dir:
                string_to_embed = args[0]
                cache_file = self.cache_dir / str(hash(string_to_embed))
                if cache_file.exists():
                    embedding = np.load(cache_file)
                    return embedding
                else:
                    embedding = func(self, *args, **kwargs)
                    np.save(cache_file, embedding)
                    return embedding

        return wrapper

    @use_cache
    def _embed_with_openai(self, string_to_embed) -> np.ndarray:
        """
        Embed a string using OpenAI's text-embedding-ada-002 model
        https://openai.com/blog/new-and-improved-embedding-model

        Requires a valid OpenAI API key to be set in the OPENAI_API_KEY
        environment variable
        """
        response = openai.Embedding.create(
            model=self.model, input=string_to_embed
        )
        embedding = response["data"][0]["embedding"]
        return np.array(embedding)

    @use_cache
    def _embed_with_sentencetransformers(self, string_to_embed) -> np.ndarray:
        """
        Embed a string using an open SentenceTransformers model
        https://www.sbert.net/docs/pretrained_models.html
        """
        return self.encoder.encode(string_to_embed)
