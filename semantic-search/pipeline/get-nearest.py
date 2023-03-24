from rich.prompt import Prompt
from sentence_transformers import SentenceTransformer
from src.elasticsearch import get_elastic_client
import rich
import os

target_es = get_elastic_client()
index = f"prismic-{os.environ['MODEL_NAME']}"


def get_nearest(query_vector: str, k: int = 10) -> list[str]:
    """run a knn query against the target index"""
    response = target_es.search(
        index=index,
        knn={
            "field": "embedding",
            "query_vector": query_vector,
            "k": k,
            "num_candidates": 100,
        },
    )
    return response["hits"]["hits"]


model = SentenceTransformer(
    model_name_or_path="paraphrase-distilroberta-base-v1",
    cache_folder="/data/models",
)

query = Prompt.ask("Enter a query")
print()

query_embedding = model.encode(query)
nearest_neighbours = get_nearest(query_embedding)

for i, hit in enumerate(nearest_neighbours):
    rich.print(
        f"[bold blue]{i+1}.[/bold blue]\t"
        f"https://wellcomecollection.org/articles/{hit['_source']['id']}\t"
        f"[green]{hit['_score']}[/green]\n"
        f"{hit['_source']['text']}\n"
    )
