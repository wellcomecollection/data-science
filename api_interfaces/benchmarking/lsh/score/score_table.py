import os

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch, helpers

from elo import Elo

elo = Elo(5000)

es = Elasticsearch(
    hosts=[{
        'host': os.environ["ES_HOST"],
        'port': os.environ["ES_PORT"]
    }],
    http_auth=(
        os.environ["ES_USER"],
        os.environ["ES_PASS"]
    )
)

data = pd.DataFrame([thing["_source"] for thing in list(
    helpers.scan(
        es,
        index="assessment",
        query={"query": {"match_all": {}}}
    ))
])

for candidate in np.unique(data[['candidate_a', 'candidate_b']]):
    elo.addPlayer(candidate)

for candidate_a, candidate_b, winner in data.values:
    elo.recordMatch(candidate_a, candidate_b, winner=winner)

n_classifiers_options = [32, 64, 128, 256, 512]
n_clusters_options = [8, 16, 32, 64, 128, 256]
results = pd.DataFrame({
    str(n_classifiers): {
        str(n_clusters): None
        for n_clusters in n_clusters_options}
    for n_classifiers in n_classifiers_options
})

for index_name, score in dict(elo.getRatingList()).items():
    n_classifiers, n_clusters = index_name.split("-")
    results[n_classifiers][n_clusters] = score

print(results)
