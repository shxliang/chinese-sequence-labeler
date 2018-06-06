import numpy as np
import tensorflow as tf

from data_loaders.data_generator import BatchGenerator
from data_loaders.loader import prepare_dataset
from models.bilstm_model import BiLSTMModel, create_model
from utils.data_utils import load_word2vec
from utils.utils import save_model, test_ner


def train(config, train_sentences, dev_sentences, test_sentences, char_to_id, feature_to_id, target_to_id, id_to_char,
          id_to_target, logger):
    # prepare data, get a collection of list containing index
    train_data = prepare_dataset(
        train_sentences, char_to_id, target_to_id, feature_to_id, config.lower)
    dev_data = prepare_dataset(
        dev_sentences, char_to_id, target_to_id, feature_to_id, config.lower)
    test_data = prepare_dataset(
        test_sentences, char_to_id, target_to_id, feature_to_id, config.lower)
    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), len(dev_sentences), len(test_data)))

    train_batch_generator = BatchGenerator(train_data, config.batch_size)
    dev_batch_generator = BatchGenerator(dev_data, 100)
    test_batch_generator = BatchGenerator(test_data, 100)

    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # 一个epoch的batch数
    steps_per_epoch = train_batch_generator.len_data
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, BiLSTMModel, config, load_word2vec, id_to_char, logger)
        logger.info("start training")
        loss = []
        for i in range(model.config.max_epoch):
            for batch in train_batch_generator.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)

                if step % model.config.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("epoch iteration: {}, step: {}/{}, "
                                "NER loss: {:>9.6f}".format(iteration, step % steps_per_epoch, steps_per_epoch,
                                                            np.mean(loss)))
                    loss = []

            # 评估在验证集（dev）上F1值是否有提升，返回值为布尔型
            best = evaluate(sess, model, "dev", dev_batch_generator, id_to_target, logger)
            if best:
                save_model(sess, model, model.config.ckpt_path, logger)

            # 跑完一个epoch就用测试集（test）对模型进行评估
            evaluate(sess, model, "test", test_batch_generator, id_to_target, logger)

        print("final best dev f1 score: " + str(model.best_dev_f1.eval()))
        print("final best test f1 score: " + str(model.best_test_f1.eval()))


def evaluate(sess, model, name, data, id_to_tag, logger):
    logger.info("evaluate: {}".format(name))
    # 预测得到序列标注结果
    ner_results = model.evaluate(sess, data, id_to_tag)
    # 计算评估指标
    eval_lines = test_ner(ner_results, model.config.result_path)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score: {:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score: {:>.3f}".format(f1))
        return f1 > best_test_f1
