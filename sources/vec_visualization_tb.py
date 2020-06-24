import torch
from tensorboardX import SummaryWriter
from gensim.models.doc2vec import Doc2Vec
import json
import re
import sys

re_hiragana = re.compile(r'^[あ-ん]+$')
re_katakana = re.compile(r'[\u30A1-\u30F4]+')
re_roman = re.compile(r'^[a-zA-Z]+$')
re_kanji = re.compile(r'^[\u4E00-\u9FD0]+$')
def gomi_eraser(t):
    tt = ""
    for c in t:
        if re_hiragana.fullmatch(c) or re_katakana.fullmatch(c) or re_roman.fullmatch(c) or re_kanji.fullmatch(c):
            tt += c
        else:
            pass
    return tt

with open("./tv_program.json", "r") as f:
    tv_program_dict = json.load(f)

with open("./files.txt", "r") as f:
    files_list = [ff.split()[-1:][0].replace(".ass", "") for ff in f.readlines()]

vec_path = "./doc2vec.model"
writer = SummaryWriter()
model = Doc2Vec.load(vec_path)
weights =[]
labels =[]

for i in files_list:
    title = tv_program_dict[i][1].replace(" ", "").replace("　", "").replace("[終]", "").replace("[続]", "").replace("[新]", "").replace("[再]", "")
    title = title[:title.find("[")]
    labels.append(title)

    # https://github.com/tensorflow/tensorflow/issues/9330
    if len(title) == 0:
        print("There is a blank meta.", i)
        print(tv_program_dict[i])
        sys.exit()

for i in range(0,len(model.docvecs)):
    weights.append(model.docvecs[i].tolist())
    #labels.append(model.docvecs.index_to_doctag(i))

# Visualize vectors up to 1,000.
# - Embedding projector only loads first 100,000 vectors.
# - Refer: https://github.com/tensorflow/tensorboard/issues/773
N = 1000
weights = weights[:N]
labels = labels[:N]
print(labels)
print(len(labels))
writer.add_embedding(torch.FloatTensor(weights), metadata=labels)
