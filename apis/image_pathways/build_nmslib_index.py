import nmslib
import numpy as np

feature_vectors = np.load('./data/feature_vectors.npy')

index = nmslib.init(method='hnsw', space='cosinesimil')
index.addDataPointBatch(feature_vectors)
index.createIndex({'post': 2}, print_progress=True)

index.saveIndex('./data/feature_vectors.hnsw', save_data=True)
