# encoding=utf8
import itertools
import os
import pickle

import tensorflow as tf

from data_loaders.loader import feature_mapping, tag_mapping, char_mapping, augment_with_pretrained, \
    update_tag_scheme, load_sentences
from models.bilstm_model import BiLSTMModel, create_model
from trainers.bilstm_trainer import train
from utils.config_utils import load_config, create_config_model
from utils.data_utils import input_from_line, load_word2vec
from utils.utils import clean, make_path, get_logger, print_config

flags = tf.app.flags
flags.DEFINE_boolean("clean", True, "clean train folder")
flags.DEFINE_boolean("train", True, "Wither train the model")

# configurations for the model
flags.DEFINE_integer("char_dim", 100, "Embedding size for characters")
flags.DEFINE_integer("seg_dim", 20, "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("lstm_dim", 100, "Num of hidden units in LSTM")
flags.DEFINE_string("tag_schema", "iobes", "tagging schema iobes or iob")
flags.DEFINE_bool("use_other_features", False, "除字序列外是否使用其他特征序列")
flags.DEFINE_string("features", "1", "使用的其他特征列，多列以逗号分隔")
flags.DEFINE_string("feature_dim", "32", "使用的其他特征的embedding维度，多列以逗号分隔")

# configurations for training
flags.DEFINE_float("clip", 5, "Gradient clip")
flags.DEFINE_float("dropout", 0.5, "Dropout rate")
flags.DEFINE_float("batch_size", 20, "batch size")
flags.DEFINE_float("lr", 0.001, "Initial learning rate")
flags.DEFINE_string("optimizer", "adam", "Optimizer for training")
flags.DEFINE_boolean("pre_emb", True, "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros", True, "Wither replace digits with zero")
flags.DEFINE_boolean("lower", True, "Wither lower case")

flags.DEFINE_integer("max_epoch", 100, "maximum training epochs")
flags.DEFINE_integer("steps_check", 100, "steps per checkpoint")
flags.DEFINE_string("ckpt_path", "ckpt", "Path to save model")
flags.DEFINE_string("summary_path", "summary", "Path to store summaries")
flags.DEFINE_string("log_file", "log/train.log", "File for log")
flags.DEFINE_string("map_file", "resources/maps.pkl", "File for maps")
flags.DEFINE_string("vocab_file", "resources/vocab.json", "File for vocab")
flags.DEFINE_string("config_file", "resources/config_file", "File for config")
flags.DEFINE_string("script", "conlleval", "evaluation script")
flags.DEFINE_string("result_path", "result", "Path for results")
flags.DEFINE_string("emb_file", "resources/wiki_100.utf8", "Path for pre_trained embedding")
flags.DEFINE_string("train_file", os.path.join("data", "example.train"), "Path for train data")
flags.DEFINE_string("dev_file", os.path.join("data", "example.dev"), "Path for dev data")
flags.DEFINE_string("test_file", os.path.join("data", "example.test"), "Path for test data")

FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]


def main_train():
    # load data sets
    # sentences = [[(words11, tag11), ...], [(word21, tag21), ...], ...]
    train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)
    dev_sentences = load_sentences(FLAGS.dev_file, FLAGS.lower, FLAGS.zeros)
    test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)

    # Use selected tagging scheme (IOB / IOBES)
    # 更新在train_sentences和test_sentences中
    update_tag_scheme(train_sentences, FLAGS.tag_schema)
    update_tag_scheme(test_sentences, FLAGS.tag_schema)

    # create maps if not exist
    # 创建或加载字符、词、特征、target的映射字典
    if not os.path.isfile(FLAGS.map_file):
        # create dictionary for word
        # 若存在pre-trained embedding file，则同时使用pre-trained和训练集构建字典
        if FLAGS.pre_emb:
            # 统计train中字，返回字典
            dico_chars_train = char_mapping(train_sentences, FLAGS.lower)[0]
            # 使用pre-trained的字增大训练集字的字典
            dico_chars, char_to_id, id_to_char = augment_with_pretrained(
                dico_chars_train.copy(),
                FLAGS.emb_file,
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in test_sentences])
                )
            )
        # 否则只是用训练集构建字典
        else:
            _c, char_to_id, id_to_char = char_mapping(train_sentences, FLAGS.lower)

        # Create a dictionary and a mapping for tags
        _t, target_to_id, id_to_target = tag_mapping(train_sentences)

        # 创建其他特征的映射字典，返回的三个都为dict
        _f, feature_to_id, id_to_feature = feature_mapping(train_sentences, FLAGS.features)

        # 存储字、target、feature的映射关系
        with open(FLAGS.map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, target_to_id, id_to_target, feature_to_id, id_to_feature], f)
    else:
        with open(FLAGS.map_file, "rb") as f:
            char_to_id, id_to_char, target_to_id, id_to_target, feature_to_id, id_to_feature = pickle.load(f)

    # make path for store log and model config if not exist
    make_path(FLAGS)
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = create_config_model(FLAGS, char_to_id, target_to_id, feature_to_id)
    logger = get_logger(FLAGS.log_file)
    print_config(config, logger)

    train(config, train_sentences, dev_sentences, test_sentences, char_to_id, feature_to_id, target_to_id, id_to_char,
          id_to_target, logger)


def evaluate_one():
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, target_to_id, id_to_target, feature_to_id, id_to_feature = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, BiLSTMModel, config, load_word2vec, id_to_char, logger)
        while True:
            try:
                line = input("请输入测试句子: ")
                if line == "exit":
                    exit(0)
                features = dict()
                if config["use_other_features"]:
                    for feature_i in config["features"]:
                        if feature_i == "0":
                            continue
                        features[feature_i] = input("请输入 feature_" + feature_i + " : ").split()
                result = model.evaluate_one(sess, input_from_line(line, features, char_to_id, feature_to_id),
                                            id_to_target)
                print(result)
            except Exception as e:
                print(e)
                logger.info(e)


def main(_):
    if FLAGS.train:
        if FLAGS.clean:
            clean(FLAGS)
        main_train()
    else:
        evaluate_one()


if __name__ == "__main__":
    tf.app.run(main)
