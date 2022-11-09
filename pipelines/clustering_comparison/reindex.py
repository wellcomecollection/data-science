import json
import time
from tqdm import tqdm
from src.elasticsearch import get_local_elastic_client, get_rank_elastic_client

local_es = get_local_elastic_client()
rank_es = get_rank_elastic_client()

index_name = "images-knn-256"

# reindex from local to rank
reindex_response = rank_es.reindex(
    source={"index": index_name},
    dest={
        "remote": {
            "host": "http://elasticsearch:9200",
            "username": "elastic",
            "password": "changeme",
        },
        "index": index_name,
    },
    wait_for_completion=False,
)

print(json.dumps(reindex_response, indent=2))

# monitor the task, and run a tqdm progress bar
status = local_es.tasks.get(task_id=reindex_response["task"])
print(json.dumps(status, indent=2))
progress_bar = tqdm(total=status["task"]["status"]["total"])
while status["completed"] == False:
    time.sleep(1)
    status = local_es.tasks.get(task_id=reindex_response["task"])
    progress_bar.update(status["task"]["status"]["updated"])
