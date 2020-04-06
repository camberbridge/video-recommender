This is a VIDEO Recommender
## About this
- A Flask Web App named "Jagaimo".
## About sources/
- Main algorithms.
## Requirements
- Dataset: texts/1.txt   (N.txt is a document)
- Labelset: texts/tv_program.json   ({"1": ["00:00", "00:30", "20200101", "VIDEO TITLE", "21", "1.1"], "2": [...], ...})

***

## How to run the Jagaimo(GUI)

```$python3 jagaimo.py```

- Display struct elems.

```$python tv_elem.py N(TV ID)```

- Display a similar TV.

```$python tv_analysis M(types) N(TV ID)```

- Gen a word vector from text set.

```$python3 freq_based_vectorize.py document_set.txt```

- Gen a co-occurrence network.

```$python3 cooccurrence_wordnet.py```


## How to learn models

- Gen a text set (document_set.txt).

```$python3 doc_gen.py```

- Prepro to a text set.

```$less document_set.txt | tr -d "（" | tr -d "）" | tr -d "『" | tr -d "』" | tr -d "→" | tr -d "☆" | tr -d "〈" | tr -d "〉" | tr -d "・" > document_set.txt```

- Gen an index relation file.

```$ls -la text/ > files.txt```

```$less files.txt | xargs grep ".txt" > files.txt

- Create a feature data (.json).

```$python3 freq_based_vectorize.py document_set.txt```

- Create a distributed representation from text set (.model) with ipa_dic.

```$python3 program2vec.py document_set.txt``` 

- Create a semantic elems from text set with neologd_dic by LDA and HDP-LDA.

```$python3 documents_vectorize.py document_set.txt```
