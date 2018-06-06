import codecs
import os
import re
from collections import OrderedDict

from utils.data_utils import create_dico, create_mapping, zero_digits
from utils.data_utils import iob2, iob_iobes, get_seg_features


def load_sentences(path, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    读取文件，一行一个字和对应tag，返回句子的列表，句子间以空行分隔
    """
    sentences = []
    sentence = []
    num = 0
    for line in codecs.open(path, 'r', 'utf8'):
        num += 1
        # 是否替换所有数字为0
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()

        # 若是空行（是句子分隔符），代表读取完了一个sentence
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    # sentences = [[(words11, feature11, ..., tag11), ...], [(word21, feature21, ..., tag21), ...], ...]
                    sentences.append(sentence)
                sentence = []
        else:
            # 将首字符的空格替换为$，再用空格切分
            if line[0] == " ":
                line = "$" + line[1:]
                word = line.split()
                # word[0] = " "
            else:
                word = line.split()
            assert len(word) >= 2, print([word[0]])
            # sentence = [(word1, feature1, ..., tag1), (word2, feature2, ..., tag2), ...]
            sentence.append(word)
    # 判断最后一个句子
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def char_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    创建的字表并没有按频次来截取，使用训练集中所有字作为字表
    """
    chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(chars)
    dico["<PAD>"] = 10000001
    dico['<UNK>'] = 10000000
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in chars)
    ))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[char[-1] for char in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def feature_mapping(sentences, features):
    """
    创建特征和index映射关系
    :param sentences: list of list of tuple, [[(words11, features11, ..., tag11), ...], [(word21, feature21, ..., tag21), ...], ...]
    :param features: string, 特征的列index, 以逗号分隔
    :return: 
        feature_to_id: dict
        id_to_feature: dict
    """
    dico = OrderedDict()
    feature_to_id = OrderedDict()
    id_to_feature = OrderedDict()

    features_list = features.split(",")
    for feature_i in features_list:
        if feature_i == "0":
            continue
        cur_feature = [[t[int(feature_i)] for t in s] for s in sentences]
        cur_dico = create_dico(cur_feature)
        print("%sth feature found %i unique features" % (feature_i, len(cur_dico)))
        cur_dico["<UNK>"] = 10000000
        cur_feature_to_id, cur_id_to_feature = create_mapping(cur_dico)

        dico[feature_i] = cur_dico
        feature_to_id[feature_i] = cur_feature_to_id
        id_to_feature[feature_i] = cur_id_to_feature

    return dico, feature_to_id, id_to_feature


def prepare_dataset(sentences, char_to_id, tag_to_id, feature_to_id, lower=False, train=True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - word seg indexes
        - feature indexes
        - tag indexes
    :param char_to_id: key为字，value为index的词典
    :param tag_to_id: key为字，value为index的词典
    :return 
        data: 每个句子的 字序列，字index序列，分词特征序列，其他特征序列字典，tag index序列
    """

    none_index = tag_to_id["O"]

    # 小写转换函数
    def f(x):
        return x.lower() if lower else x

    data = []
    for s in sentences:
        cur_data = []

        # 只取字，s = [(word1, tag1), (word2, tag2), ...]
        # string = [word1, word2, ...]
        string = [w[0] for w in s]
        cur_data.append(string)

        # 将vocab外的字都替换为<UNK>
        chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>']
                 for w in string]
        cur_data.append(chars)

        # 使用jieba对string进行分词得到分词特征，分词特征就是字是在词的第几个位置
        # 我 爱 打游戏 =》 [0, 0, 1, 2, 3]
        segs = get_seg_features("".join(string))
        cur_data.append(segs)

        features = OrderedDict()
        for feature_i, feature_i_to_id in feature_to_id.items():
            if feature_i == "0":
                continue
            cur_features = [feature_i_to_id[t[int(feature_i)]]if t[int(feature_i)] in feature_i_to_id else "<UNK>"
                            for t in s]
            features[feature_i] = cur_features
        cur_data.append(features)

        # 若是训练阶段，则需要tag转化为index
        if train:
            tags = [tag_to_id[w[-1]] for w in s]
        # 若不是训练阶段，所有字的tag全部初始化为O
        else:
            tags = [none_index for _ in chars]
        cur_data.append(tags)

        data.append(cur_data)

    return data


def augment_with_pretrained(dictionary, ext_emb_path, chars):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `chars` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `chars` (typically the words in the development and test sets.)
    
    增大训练集字的字典，若chars为None则将pre-trained的所有字都加入，否则只加入pre-trained和chars中共同的字
    
    :param dictionary: 训练集字的字典
    :param ext_emb_path: pre-trained file path
    :param chars: 测试集的字的set
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    # 得到pre-trained中所有字的set
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    # 若测试集字为None
    if chars is None:
        # 将pre-trained的所有字加入训练集字的字典
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0
    else:
        # 将测试集中存在与pre-trained的字加入训练集字的字典
        for char in chars:
            if any(x in pretrained for x in [
                char,
                char.lower(),
                re.sub('\d', '0', char.lower())
            ]) and char not in dictionary:
                dictionary[char] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


def save_maps(save_path, *params):
    """
    Save mappings and invert mappings
    """
    pass
    # with codecs.open(save_path, "w", encoding="utf8") as f:
    #     pickle.dump(params, f)


def load_maps(save_path):
    """
    Load mappings from the file
    """
    pass
    # with codecs.open(save_path, "r", encoding="utf8") as f:
    #     pickle.load(save_path, f)
