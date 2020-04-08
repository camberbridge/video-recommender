#coding: utf-8

from __future__ import print_function
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import json
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import os
import sys
from sources import cooccurrence_wordnet as cw
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)


with open("/Users/ruffy/Desktop/video-recommender/sources/texts/tv_program.json", "r") as f:
    program_list = json.load(f)

with open("/Users/ruffy/Desktop/video-recommender/sources/data/files.txt", "r") as f:
    file_list = f.readlines()
    tvlist_dic = program_list
    tvlist_list = [tvlist_dic[str(i+1)][3].replace(" ","").replace(u"　","")+"("+tvlist_dic[str(i+1)][2]+u"放送)_"+str(i+1) for i in range(len(tvlist_dic))]
    result_list = tvlist_list

def tv_elem(program_id):

    def interpolation(l):
        if len(l) != 10:
            n = 10 - len(l)
            for i in range(n):
                l.append("None")
        return l

    # 上位概念≒概略 (LDAが上位の理由は, トピック数が20の事前分布なので)
    with open("/Users/ruffy/Desktop/video-recommender/sources/models/lda20_2_30per.json", "r") as f:
        lda_result = json.load(f)
        
    # 下位概念≒詳細 ( HDP_LDAが下位の理由は, トピック数がMAX150の事前分布なので詳細であるとして)
    with open("/Users/ruffy/Desktop/video-recommender/sources/models/hdplda_2_30per.json", "r") as f:
        hdp_result = json.load(f)

    if program_id in lda_result:
        lda_result = lda_result[program_id]
        lda_result = interpolation(lda_result)
    else:
        lda_result = ["None" for i in range(10)]

    if program_id in hdp_result:
        hdp_result = hdp_result[program_id]
        hdp_result = interpolation(hdp_result)
    else:
        hdp_result = ["None" for i in range(10)]

    return lda_result, hdp_result

def program2vec():
    m = Doc2Vec.load("/Users/ruffy/Desktop/video-recommender/sources/models/doc2vec.model")
    return m

def similarities_inference(m, doc_id, _topn):
    return m.docvecs.most_similar(doc_id, topn=_topn)

def any_similarities_inference(m, doc_id1, doc_id2):
    return m.docvecs.similarity(doc_id1, doc_id2)

def addition(m, positive_list):
    return m.docvecs.most_similar(positive = positive_list)

def subtraction(m, positive_list, negative_list):
    return m.docvecs.most_similar(positive = positive_list, negative = negative_list)

def formatting(l):
    pivot = l.index("_")+1
    return l[pivot:]

def tv_analysis(types, posi_num=[], nega_num=[]):
    model = program2vec()
    topn = 10

    # Similarity
    if types == 0:
        param = posi_num[0]

        sim_list = []
        dis_list = []
        for c in similarities_inference(model, [param], topn):
            txt_label = c[0]
            prob = c[1]
            txt_n = int(file_list[txt_label].split()[-1:][0].replace(".txt", ""))
            #print(prob, txt_label, result_list[txt_n-1])
            sim_list.append([txt_label, result_list[txt_n-1]])

        dissim_list = similarities_inference(model, [param], _topn=len(result_list))
        dissim_list.reverse()
        for c in dissim_list[:topn]:
            txt_label = c[0]
            prob = c[1]
            txt_n = int(file_list[txt_label].split()[-1:][0].replace(".txt", ""))
            #print(prob, txt_label, result_list[txt_n-1])
            dis_list.append([txt_label, result_list[txt_n-1]])

        return sim_list, dis_list
    # Similar degree
    elif types == 1:
        params1 = posi_num[0]
        params2 = posi_num[1]
        #print(any_similarities_inference(model, params1, params2))
        return any_similarities_inference(model, params1, params2)
    # Addtion
    elif types == 2:
        positive_list = posi_num
        add_list = []
        for c in addition(model, positive_list):
            txt_label = c[0]
            prob = c[1]
            txt_n = int(file_list[txt_label].split()[-1:][0].replace(".txt", ""))
            #print(prob, txt_label, result_list[txt_n-1])
            add_list.append([txt_label, result_list[txt_n-1]])

        return add_list
    # Arithmetic
    elif types == 3:
        positive_list = posi_num
        negative_list = nega_num
        arith_list = []
        for c in subtraction(model, positive_list, negative_list):
            txt_label = c[0]
            prob = c[1]
            txt_n = int(file_list[txt_label].split()[-1:][0].replace(".txt", ""))
            #print(prob, txt_label, result_list[txt_n-1])
            arith_list.append([txt_label, result_list[txt_n-1]])

        return arith_list


@app.route("/", methods=['GET', 'POST'])
def index():
    title = u"R&D"

    if request.method == 'POST':
        #tv_program = request.form["tv_add"]  # for 1 name
        _tv_add = request.form.getlist("tv_add")  # for multi name
        _tv_remove = request.form.getlist("tv_remove")

        posi_tv = _tv_add
        nega_tv = _tv_remove

        tv_add = [i for i in _tv_add if len(i) > 1]
        tv_remove = [i for i in _tv_remove if len(i) > 1]
        N = 1
        if len(tv_remove) > 0:
            N = 2

        tv_program = []
        tv_program.append(tv_add)
        tv_program.append(tv_remove)

        file_n_list = []  # [[add],[remove]]
        message_list = []  # [[add],[remove]]

        try:
            for i in range(N):
                _file_n_list = []
                _message_list = []
                for j in tv_program[i]:
                    pivot = j.index("_")+1
                    file_name = " "+str(j[pivot:])+".txt"
                    message = j[:pivot-1]
                    _file_n_list.append(file_name)
                    _message_list.append(message)
                file_n_list.append(_file_n_list)
                message_list.append(_message_list)

            _elem_list = []
            filenumber_list = []
            for j in file_n_list:
                for k in j:
                    file_n = [i for i in range(len(file_list)) if k in file_list[i]][0]
                    _elem_list.append(tv_elem(str(file_n)))
                    filenumber_list.append(file_n)

            elem_high_list = [i[0] for i in _elem_list]
            elem_low_list = [i[1] for i in _elem_list]

            # Calculate a sim/dissim.
            if len(file_n_list[0]) == 1 and len(file_n_list) == 1:
                sim_list, dis_list = tv_analysis(0, [i for i in range(len(file_list)) if file_n_list[0][0] in file_list[i]], [])

                sim_elem_list = []
                for sim in sim_list:
                    sim_elem_list.append(tv_elem(str(sim[0])))
                dis_elem_list = []
                for dis in dis_list:
                    dis_elem_list.append(tv_elem(str(dis[0])))

                return render_template("index.html", title=title, message=message_list,
                                main_text=elem_low_list,
                                sim_list=sim_list, dis_list=dis_list, tv_list=tvlist_list,
                                high_semantic=elem_high_list, low_semantic=elem_low_list,
                                message_1dim=[i for j in message_list for i in j], 
                                posi_tv=posi_tv, nega_tv=nega_tv,
                                sim_high_semantic=[i[0] for i in sim_elem_list], 
                                sim_low_semantic=[i[1] for i in sim_elem_list],
                                dis_high_semantic=[i[0] for i in dis_elem_list], 
                                dis_low_semantic=[i[1] for i in dis_elem_list],
                                filenumber_list=filenumber_list,
                                sim_num_list=[i[0] for i in sim_list],
                                dis_num_list=[i[0] for i in dis_list])

            # Addition.
            if len(file_n_list[0]) > 1 and len(file_n_list) == 1:
                sets = []
                for j in file_n_list[0]:
                    sets.append([i for i in range(len(file_list)) if j in file_list[i]][0])
                add_result = tv_analysis(2, sets, [])

                sim_elem_list = []
                for sim in add_result:
                    sim_elem_list.append(tv_elem(str(sim[0])))

                # Similar digree.
                if len(file_n_list[0]) == 2:
                    pair = [[i for i in range(len(file_list)) if file_n_list[0][0] in file_list[i]][0],
                            [i for i in range(len(file_list)) if file_n_list[0][1] in file_list[i]][0]]
                    similarity = round(tv_analysis(1, pair, []), 2)

                    return render_template("index.html", title=title, message=message_list,
                                    main_text=elem_low_list, add_result=add_result,
                                    similarity=similarity, tv_list=tvlist_list,
                                    high_semantic=elem_high_list, low_semantic=elem_low_list,
                                    message_1dim=[i for j in message_list for i in j], 
                                    posi_tv=posi_tv, nega_tv=nega_tv,
                                    sim_high_semantic=[i[0] for i in sim_elem_list],
                                    sim_low_semantic=[i[1] for i in sim_elem_list],
                                    filenumber_list=filenumber_list,
                                    add_result_num_list=[i[0] for i in add_result])

                return render_template("index.html", title=title, message=message_list,
                                main_text=elem_low_list,
                                add_result=add_result, tv_list=tvlist_list,
                                high_semantic=elem_high_list, low_semantic=elem_low_list,
                                message_1dim=[i for j in message_list for i in j], 
                                posi_tv=posi_tv, nega_tv=nega_tv,
                                sim_high_semantic=[i[0] for i in sim_elem_list],
                                sim_low_semantic=[i[1] for i in sim_elem_list],
                                filenumber_list=filenumber_list,
                                add_result_num_list=[i[0] for i in add_result])

            # Add/Sub.
            if len(file_n_list[0]) > 0 and len(file_n_list) == 2:
                posi_list = []
                nega_list = []
                for j in file_n_list[0]:
                    posi_list.append([i for i in range(len(file_list)) if j in file_list[i]][0])
                for j in file_n_list[1]:
                    nega_list.append([i for i in range(len(file_list)) if j in file_list[i]][0])
                add_sub_result = tv_analysis(3, posi_list, nega_list)

                sim_elem_list = []
                for sim in add_sub_result:
                    sim_elem_list.append(tv_elem(str(sim[0])))

                return render_template("index.html", title=title, message=message_list,
                                main_text=elem_low_list,
                                add_sub_result=add_sub_result, tv_list=tvlist_list,
                                high_semantic=elem_high_list, low_semantic=elem_low_list,
                                message_1dim=[i for j in message_list for i in j], 
                                posi_tv=posi_tv, nega_tv=nega_tv,
                                sim_high_semantic=[i[0] for i in sim_elem_list],
                                sim_low_semantic=[i[1] for i in sim_elem_list],
                                filenumber_list=filenumber_list,
                                add_sub_result_num_list=[i[0] for i in add_sub_result])

            message = u""
            return render_template("index.html", message = message, title = title, tv_list = tvlist_list)
            """
            return render_template("index.html", title=title, message=message_list,
                                main_text=elem_low_list, tv_list=tvlist_list, 
                                high_semantic=elem_high_list, low_semantic=elem_low_list,
                                message_1dim=[i for j in message_list for i in j])
            """ 
                
        except:
            message = u""
            return render_template("index.html", message = message, title = title, tv_list = tvlist_list)
    else:
        message = u""
        return render_template("index.html", message = message, title = title, tv_list = tvlist_list)


def plot_network(data, edge_threshold=0., fig_size=(8, 8), file_name=None, dir_path=None, w_score_dict=None):
    nodes = list(set(data['node1'].tolist()+data['node2'].tolist()))
    G = nx.Graph()

    #  頂点の追加
    G.add_nodes_from(nodes)
    #  辺の追加
    for i in range(len(data)):
        row_data = data.iloc[i]
        if row_data['value'] > edge_threshold:
            G.add_edge(row_data['node1'], row_data['node2'], weight=row_data['value'])

    # 孤立したnodeを削除
    isolated = [n for n in G.nodes if len([i for i in nx.all_neighbors(G, n)]) == 0]
    for n in isolated:
        G.remove_node(n)

    plt.figure(figsize=fig_size)
    pos = nx.spring_layout(G, k=0.3)  # k = node間反発係数
    pr = nx.pagerank(G)

    # nodeの大きさ
    if w_score_dict != None:
        for w, s in pr.items():
            try:
                pr[w] = w_score_dict[w]
            except Exception as e:
                print(e)
                print(w)
    size_weight = 5000
    nx.draw_networkx_nodes(G, pos, node_color=list(pr.values()),
                           cmap=plt.cm.Reds,
                           alpha=0.7,
                           node_size=[v*size_weight for v in pr.values()])

    # 日本語ラベル
    nx.draw_networkx_labels(G, pos, fontsize=12, font_family='YuGothic', font_weight="bold")

    # エッジの太さ調節
    thickness_weight = 1
    edge_width = [d["weight"] * thickness_weight for (u, v, d) in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color="darkgrey", width=edge_width)
    plt.axis('off')

    # Dynamic plots to display
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    #plt.show()

    return '<img src="data:image/png;base64,{}">'.format(plot_url)

@app.route("/plot/<doc_num>")
def plot_graph(doc_num='0'):
    line_list, N = cw.create_doc_dict(doc_num)

    # Feature value:
    f_type = 0  # 0: TF-IDF, 1: BM25
    feature_dict = cw.word_rank(int(doc_num), f_type)

    # Calculate P(X, Y)
    p_xy_dict = cw.cal_p_xy(line_list)

    # Plot net
    word_association = cw.create_df(p_xy_dict)
    edge_threshold = 1  # For pruning. If TF: 1<, else: N<
    plot_img_path = plot_network(data=word_association, edge_threshold=edge_threshold, w_score_dict=feature_dict)

    print("Plotted!!!!!")
    return plot_img_path

if __name__ == "__main__":
    print('Flask is activated!!!')
    app.debug = True
    app.run(host="127.0.0.1", port=5000)
