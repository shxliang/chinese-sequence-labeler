import json
import re
from sklearn.model_selection import train_test_split


def parse_tag(file_path, feature_tag_list, target_tag_list, feature_exclude_nest=True, target_exclude_nest=True,
              only_contain_feature=False):
    sentence_sep_re = re.compile("[。？?！!]")
    tag_start_re = re.compile("<(\w{1,3}?)>")
    tag_end_re = re.compile("</(\w{1,3}?)>")

    result = []
    feature_tag_stack = []
    target_tag_stack = []

    contain_feature_tag = True if (feature_tag_list is not None and len(feature_tag_list) > 0) else False

    with open(file_path, "r", encoding="utf8") as f:
        for line in f.readlines():
            json_dict = json.loads(line)
            id = json_dict["id"]
            if not id.startswith("T"):
                continue
            content = json_dict["text"]
            sentences = sentence_sep_re.split(content)

            # 遍历每个句子
            for sentence in sentences:
                if len(sentence) < 1:
                    continue
                # print(sentence)

                target_tag_stack.clear()
                feature_tag_stack.clear()

                cur_target_type = "O"
                cur_feature_type = "O"
                cur_target_loc = ""
                cur_feature_loc = ""
                target_start = 0
                feature_start = 0
                sentence_result = []
                cur_index = 0

                is_contain_feature = False

                # 遍历句子中每个字符
                while cur_index < len(sentence):
                    cur_char = sentence[cur_index]
                    # 当前字符是否为tag开始符
                    is_tag = False

                    if cur_char == "<":
                        # 匹配是否为tag开始标记
                        tag_start_matcher = tag_start_re.search(sentence[cur_index: cur_index + 3])
                        if tag_start_matcher is not None:
                            is_tag = True
                            cur_index += 3
                            tag = tag_start_matcher.group(1)

                            if contain_feature_tag:
                                if tag in feature_tag_list:
                                    feature_start = 1
                                    feature_tag_stack.append(tag)
                                else:
                                    if feature_exclude_nest:
                                        feature_tag_stack.append(tag)
                            if tag in target_tag_list:
                                target_start = 1
                                target_tag_stack.append(tag)
                            else:
                                if target_exclude_nest:
                                    target_tag_stack.append(tag)

                        # 匹配是否为tag结束标记
                        tag_end_matcher = tag_end_re.search(sentence[cur_index: cur_index + 4])
                        if tag_end_matcher is not None:
                            is_tag = True
                            cur_index += 4
                            tag = tag_end_matcher.group(1)

                            if contain_feature_tag:
                                if tag in feature_tag_list:
                                    feature_tag_stack.pop()
                                else:
                                    if feature_exclude_nest:
                                        feature_tag_stack.pop()
                            if tag in target_tag_list:
                                target_tag_stack.pop()
                            else:
                                if target_exclude_nest:
                                    target_tag_stack.pop()

                    if is_tag:
                        continue

                    # 计算当前target tag
                    if len(target_tag_stack) > 0 and target_tag_stack[-1] in target_tag_list:
                        cur_target_type = target_tag_stack[-1]
                        if target_start == 1:
                            cur_target_loc = "B-"
                        else:
                            cur_target_loc = "I-"
                        target_start += 1
                    else:
                        cur_target_type = "O"
                        cur_target_loc = ""
                    cur_target = cur_target_loc + cur_target_type

                    # 若存在feature，则计算当前feature tag
                    if contain_feature_tag:
                        if len(feature_tag_stack) > 0 and feature_tag_stack[-1] in feature_tag_list:
                            is_contain_feature = True
                            cur_feature_type = feature_tag_stack[-1]
                            if feature_start == 1:
                                cur_feature_loc = "B-"
                            else:
                                cur_feature_loc = "I-"
                            feature_start += 1
                        else:
                            cur_feature_type = "O"
                            cur_feature_loc = ""
                        cur_feature = cur_feature_loc + cur_feature_type

                        sentence_result.append((cur_char, cur_feature, cur_target))
                    else:
                        sentence_result.append((cur_char, cur_target))

                    cur_index += 1

                if only_contain_feature and not is_contain_feature:
                    continue

                if len(sentence_result) > 0:
                    result.append(sentence_result)
                    print(sentence_result)

    return result


if __name__ == "__main__":
    feature_tag_list = ["T"]
    target_tag_list = ["P"]

    result = parse_tag("data/1-133_1.txt", feature_tag_list=feature_tag_list, target_tag_list=target_tag_list,
                       feature_exclude_nest=True, only_contain_feature=True)

    train_set, test_set = train_test_split(result, test_size=0.2)

    with open("data/1-133_1_train7.txt", "w", encoding="utf8") as f:
        for sentence in train_set:
            for word in sentence:
                f.writelines(" ".join(word) + "\n")
            f.writelines("\n")

    with open("data/1-133_1_test7.txt", "w", encoding="utf8") as f:
        for sentence in test_set:
            for word in sentence:
                f.writelines(" ".join(word) + "\n")
            f.writelines("\n")
