import os
import torch
import clip

from src.elasticsearch import get_prototype_elastic_client


es = get_prototype_elastic_client()


index_name = f"images-clip-{os.environ.get('INDEX_DATE')}"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocessor = clip.load(
    "ViT-B/32", download_root="data/models", device=device
)


search_terms = input("Enter search terms: ")

with torch.no_grad():
    tokens = clip.tokenize([search_terms]).to(device)
    text_embedding = model.encode_text(tokens).squeeze(0)

text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

response = es.search(
    index=index_name,
    knn={
        "field": "embedding",
        "query_vector": text_embedding.tolist(),
        "k": 10,
        "num_candidates": 1000,
    },
    size=10,
)

for result in response["hits"]["hits"]:
    print(result["_source"]["title"])
    print(result["_source"]["thumbnail_url"])
    print(result["_score"])
    print()
