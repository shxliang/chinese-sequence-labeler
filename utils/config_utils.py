from collections import OrderedDict
import json
from bunch import Bunch


def create_config_model(params, char_to_id, tag_to_id, feature_to_id):
    config_dict = OrderedDict()

    config_dict["num_chars"] = len(char_to_id)
    config_dict["char_dim"] = params.char_dim
    config_dict["num_tags"] = len(tag_to_id)
    config_dict["seg_dim"] = params.seg_dim
    config_dict["lstm_dim"] = params.lstm_dim
    config_dict["batch_size"] = params.batch_size
    config_dict["num_features"] = [len(v) for k, v in feature_to_id.items()]
    config_dict["use_other_features"] = params.use_other_features
    config_dict["features"] = params.features.split(",")
    config_dict["feature_dim"] = params.feature_dim.split(",")

    config_dict["emb_file"] = params.emb_file
    config_dict["clip"] = params.clip
    # tf中的参数其实是保留率
    config_dict["dropout_keep"] = 1.0 - params.dropout
    config_dict["optimizer"] = params.optimizer
    config_dict["lr"] = params.lr
    config_dict["tag_schema"] = params.tag_schema
    config_dict["pre_emb"] = params.pre_emb
    config_dict["zeros"] = params.zeros
    config_dict["lower"] = params.lower
    # 分词特征的取值个数，也就是一个词最多只有4个字
    config_dict["num_segs"] = 4

    config_dict["ckpt_path"] = params.ckpt_path
    config_dict["max_epoch"] = params.max_epoch
    config_dict["steps_check"] = params.steps_check
    config_dict["result_path"] = params.result_path

    save_config(config_dict, params.config_file)
    config = Bunch(config_dict)
    return config


def save_config(config, config_file):
    """
    Save configuration of the model
    parameters are stored in json format
    """
    with open(config_file, "w", encoding="utf8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


def load_config(config_file):
    """
    Load configuration of the model
    parameters are stored in json format
    """
    with open(config_file, encoding="utf8") as f:
        config_dict = json.load(f)
    config = Bunch(config_dict)
    return config
