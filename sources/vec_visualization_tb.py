import torch
from tensorboardX import SummaryWriter
from gensim.models.doc2vec import Doc2Vec

vec_path = "./doc2vec.model"
writer = SummaryWriter()
model = Doc2Vec.load(vec_path)
weights =[]
labels =[]

for i in range(0,len(model.docvecs)):
    weights.append(model.docvecs[i].tolist())
    labels.append(model.docvecs.index_to_doctag(i))

# Visualize vectors up to 1,000.
# - Embedding projector only loads first 100,000 vectors.
# - Refer: https://github.com/tensorflow/tensorboard/issues/773
N = 1000
weights = weights[:N]
labels = labels[:N]
writer.add_embedding(torch.FloatTensor(weights), metadata=labels)
vec2tfb.py (END)
