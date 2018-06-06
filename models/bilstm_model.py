# encoding = utf8
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers

from models import rnncell
from utils.data_utils import iobes_iob
from utils.utils import result_to_json


class BiLSTMModel(object):
    def __init__(self, config):
        self.config = config

        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()

        # add placeholders for the model
        # 字序列、词序列、target序列
        self.char_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name="ChatInputs")
        self.seg_inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name="SegInputs")
        # dict，存储多个其他特征的placeholder
        self.features_inputs = parse_features(self.config.features)

        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None],
                                      name="Targets")
        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32,
                                      name="Dropout")

        used = tf.sign(tf.abs(self.char_inputs))
        # 序列长度
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]

        self.bilstm()

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def bilstm(self):
        # embeddings for chinese character and segmentation representation
        embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, self.features_inputs)

        # apply dropout before feed to lstm layer
        lstm_inputs = tf.nn.dropout(embedding, self.dropout)

        # bi-directional lstm layer
        lstm_outputs = self.biLSTM_layer(lstm_inputs)

        # logits for tags
        self.logits = self.project_layer(lstm_outputs)

        # loss of the model from crf
        self.loss = self.loss_layer(self.logits)

        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.config.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.config.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.config.lr)
            else:
                raise KeyError

            # apply grad clip to avoid gradient explosion
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in grads_vars]
            # 计算梯度时需要使用global_step
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

    def embedding_layer(self, char_inputs, seg_inputs, features_inputs):
        """
        :param char_inputs: one-hot encoding of sentence
        :param seg_inputs: segmentation feature
        :param features_inputs: 其他特征
        :return: [1, num_steps, embedding size], 
        """

        embedding = []

        with tf.variable_scope("char_embedding"), tf.device('/cpu:0'):
            # 基于字的embedding
            self.char_lookup = tf.get_variable(
                name="char_embedding",
                shape=[self.config.num_chars, self.config.char_dim],
                initializer=self.initializer)
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))

            # 基于词的embedding
            if self.config.seg_dim > 0:
                with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        shape=[self.config.num_segs, self.config.seg_dim],
                        initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))

            # 基于其他特征的embedding
            if self.config.use_other_features:
                with tf.variable_scope("feature_embedding"), tf.device("/cpu:0"):
                    self.feature_lookup = dict()
                    for feature_i, feature_i_inputs in features_inputs.items():
                        if feature_i == 0:
                            continue
                        self.feature_lookup[feature_i] = tf.get_variable(
                            name="feature_" + feature_i,
                            shape=[self.config.num_features[int(feature_i) - 1],
                                   self.config.feature_dim[int(feature_i) - 1]],
                            initializer=self.initializer)
                        embedding.append(
                            tf.nn.embedding_lookup(self.feature_lookup[feature_i], features_inputs[feature_i]))
            embed = tf.concat(embedding, axis=-1)
        return embed

    def biLSTM_layer(self, lstm_inputs, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, 2*lstm_dim] 
        """
        with tf.variable_scope("char_BiLSTM" if not name else name):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnncell.CoupledInputForgetGateLSTMCell(
                        self.config.lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                lstm_inputs,
                dtype=tf.float32,
                sequence_length=self.lengths)
        return tf.concat(outputs, axis=2)

    def project_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.config.lstm_dim * 2, self.config.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)
                b = tf.get_variable("b", shape=[self.config.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.config.lstm_dim * 2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.config.lstm_dim, self.config.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)
                b = tf.get_variable("b", shape=[self.config.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.config.num_tags])

    def loss_layer(self, project_logits, name=None):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss，负对数似然损失
        """
        with tf.variable_scope("crf_loss" if not name else name):
            small = -1000.0
            # pad logits for crf loss
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.config.num_tags]),
                 tf.zeros(shape=[self.batch_size, 1, 1])],
                axis=-1)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.config.num_tags * tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.config.num_tags + 1, self.config.num_tags + 1],
                initializer=self.initializer)
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=self.lengths + 1)
            return tf.reduce_mean(-log_likelihood)

    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data 
        :return: structured data to feed
        """
        # 忽略了字序列
        _, chars, segs, features, tags = batch
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.seg_inputs: np.asarray(segs),
            self.dropout: 1.0,
        }

        features_transformed = OrderedDict()
        for cur_feature in features:
            for cur_feature_i, cur_feature_i_inputs in cur_feature.items():
                if cur_feature_i not in features_transformed:
                    features_transformed[cur_feature_i] = []
                features_transformed[cur_feature_i].append(cur_feature_i_inputs)
        for feature_i, feature_i_inputs in features_transformed.items():
            if feature_i == "0":
                continue
            feed_dict[self.features_inputs[feature_i]] = np.asarray(feature_i_inputs)

        # 训练阶段才需要设置target序列和使用dropout
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits

    def decode(self, logits, lengths, matrix):
        """
        Inference final labels use Viterbi Algorithm
        使用维特比算法解码得到最终tag序列
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """

        paths = []
        small = -1000.0
        start = np.asarray([[small] * self.config.num_tags + [0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)

            paths.append(path[1:])
        return paths

    def evaluate(self, sess, data_generator, id_to_target):
        """
        :param sess: session  to run the model 
        :param data_generator: list of data
        :param id_to_target: index to tag name
        :return: evaluate result，格式为 字 真值 预测值
        """
        results = []
        # crf layer的转移概率矩阵
        trans = self.trans.eval()
        for batch in data_generator.iter_batch():
            results.append(self.evaluate_batch(sess, batch, id_to_target, trans))
        return results

    def evaluate_batch(self, sess, batch, id_to_target, trans):
        strings = batch[0]
        tags = batch[-1]
        lengths, scores = self.run_step(sess, False, batch)

        # 使用维特比算法对序列进行解码得到tag序列
        batch_paths = self.decode(scores, lengths, trans)

        result = []
        for i in range(len(strings)):
            string = strings[i][:lengths[i]]
            gold = iobes_iob([id_to_target[int(x)] for x in tags[i][:lengths[i]]])
            pred = iobes_iob([id_to_target[int(x)] for x in batch_paths[i][:lengths[i]]])
            for char, gold, pred in zip(string, gold, pred):
                result.append(" ".join([char, gold, pred]))

        return result

    def evaluate_one(self, sess, inputs, id_to_target):
        trans = self.trans.eval(session=sess)
        lengths, scores = self.run_step(sess, False, inputs)
        batch_paths = self.decode(scores, lengths, trans)
        tags = [id_to_target[idx] for idx in batch_paths[0]]
        return result_to_json(inputs[0][0], tags)


def parse_features(features):
    """
    根据传入的features创建相应的placeholder
    :param features: 
    :return: dict，对应features的placeholder
    """
    result = OrderedDict()
    for feature_i in features:
        if feature_i == "0":
            continue
        result[feature_i] = tf.placeholder(dtype=tf.int32,
                                           shape=[None, None],
                                           name="feature_" + feature_i)
    return result


def create_model(session, model_class, config, load_vec, id_to_char, logger):
    """
    创建/加载模型，读取pre-trained的embedding文件来初始化embedding layer权重
    :param session: 
    :param model_class: 模型类
    :param load_vec: 加载pre-trained embedding文件的函数方法
    :param config: 模型配置
    :param id_to_char: key为index，value为字的词典
    :param logger: 
    :return: 模型
    """
    # create model, reuse parameters if exists
    model = model_class(config)

    ckpt = tf.train.get_checkpoint_state(config.ckpt_path)
    # 若存在checkpoint则加载
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logger.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        if config.pre_emb:
            # 读取随机初始化的embedding矩阵
            emb_weights = session.run(model.char_lookup.read_value())
            # 读取pre-trained文件来更新embedding矩阵
            emb_weights = load_vec(config.emb_file, id_to_char, config.char_dim, emb_weights)
            # 重新对char_lookup这个variable进行赋值
            session.run(model.char_lookup.assign(emb_weights))
            logger.info("Load pre-trained embedding.")
    return model
