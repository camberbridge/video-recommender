# coding: utf-8

import re
import MeCab


_mecab = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/")
#_mecab = MeCab.Tagger()
# 品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用形,活用型,原形,読み,発音
_mecab_feat_labels = 'pos cat1 cat2 cat3 conj conj_t orig read pron'.split(' ')


def _mecab_node2seq(node, decode_surface=True, feat_dict=True,):

    def _mecab_parse_feat(feat):
        return dict(zip(_mecab_feat_labels, feat.split(',')))

    while node:
        if feat_dict:  # Save POS info to dict.
            node.feat_dict = _mecab_parse_feat(
               node.feature
            )
        yield node
        node = node.next


def not_stopword(n):

    def is_stopword(n): 
        if len(n.surface) == 0:
            return True
        elif re.search(r'^[\s!-@\[-`\{-~　、-〜！-＠［-｀]+$', n.surface):
            return True
        elif re.search(r'^(接尾|非自立)', n.feat_dict['cat1']):
            return True
        elif u'サ変・スル' == n.feat_dict['conj'] or u'ある' == n.feat_dict['orig']:
            return True
        elif re.search(r'^(名詞|動詞|形容詞|副詞)', n.feat_dict['pos']):
            return False
        else:
            return True

    return not is_stopword(n)


def node2word(n): 
    return n.surface


def node2norm_word(n): 
    if n.feat_dict['orig'] != '*':
        return n.feat_dict['orig']
    else:
        return n.surface


def word_segmenter_ja(sent, node_filter=not_stopword,
                      node2word=node2norm_word, mecab_encoding='utf-8'):
    if type(sent) == bytes:  # python2: unicode, python3: bytes
        sent = str(sent.encode(mecab_encoding))

    _mecab.parse("")
    nodes = list(
        _mecab_node2seq(_mecab.parseToNode(sent))
    )
    if node_filter:
        nodes = [n for n in nodes if node_filter(n)]
    words = [node2word(n) for n in nodes]

    return words


if __name__ == '__main__':
    text = u'今日はいい天気ですね。ところで君の名はは観た？'
    print('|'.join(word_segmenter_ja(text)))
