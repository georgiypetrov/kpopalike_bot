import pickle

import numpy as np
import pandas as pd

from insightface.embedder import InsightfaceEmbedder

model_path = "models/model-y1-test2/model"
model = InsightfaceEmbedder(model_path=model_path, epoch_num='0000', image_size=(112, 112), no_face_raise=False)

idols_data = pd.read_csv('idols.csv')

with open('embeddings.pickle', 'rb') as f:
    embeddings_with_path = pickle.load(f)

embeddings = embeddings_with_path[range(embeddings_with_path.shape[0]), 2]
embeddings = np.stack(embeddings.tolist(), axis=0).reshape(-1, 128)


def get_features_from_image(image):
    features = model.embed_image(image)
    return features


def get_face_neighbours(image, n=3):
    features = get_features_from_image(image)
    dists = np.linalg.norm(features - embeddings, axis=1)
    closest_ids = np.argsort(dists)
    return (
        (idols_data.iloc[int(embeddings_with_path[closest_ids[i], 0])]['name'], embeddings_with_path[closest_ids[i], 1])
        for i in range(n))
