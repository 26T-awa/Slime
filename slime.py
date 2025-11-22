import json
import os
import random
import tkinter as tk
from PIL import Image, ImageTk
import datetime
import math
import numpy as np


class Slime:
    # 用法·检查文件完整性
    def Check_integrity(self):
        # 文件列表
        required_files = {
            "attributes": ["data.json", "vocabularies.json"],
            "gui": ["gui.json", "slime.png"],
        }
        missings = []

        # 分析
        for folders, files in required_files.items():
            if not os.path.exists(folders):
                missings.append(f"{folders}/ (整个文件夹)")
            else:
                for files in files:
                    if not os.path.exists(f"{folders}/{files}"):
                        missings.append(f"{folders}/{files}")

        # 输出
        if missings:
            print(
                f">- ……[史莱姆的灵魂好像有一点不完整qwq……]\n   检查{missings}文件夹是否存在！"
            )
            return False
        else:
            print(f">- 姆姆~[史莱姆的灵魂正在完整地跳跃awa~]")
            return True

    # 用法·开启对话
    def Start_chat(self):

        print("=" * 30)

        conversation_count = 0

        while True:
            user_text = input("> ")
            conversation_count += 1

            if not user_text.strip():
                print("咕？")
                continue
            self.analyse_user_input(user_text)

    # 用法·打招呼
    def Meet(self, user_text=""):
        # 文件读取
        with open("attributes/vocabularies.json", "r", encoding="utf-8") as file:
            _Vocab = json.load(file)
        with open("attributes/data.json", "r", encoding="utf-8") as file:
            _Data = json.load(file)

        # 调用文件参数
        _MeetOutput = _Vocab["output"]["Meet"]
        morning_texts = _MeetOutput["morning_texts"]
        noon_texts = _MeetOutput["noon_texts"]
        afternoon_texts = _MeetOutput["afternoon_texts"]
        evening_texts = _MeetOutput["evening_texts"]
        midnight_texts = _MeetOutput["midnight_texts"]
        wrong_texts = _MeetOutput["wrong_texts"]
        weights = _MeetOutput["weights"]

        # 分析
        current_hour = datetime.datetime.now().hour

        _MeetMatrix = _Data["Methods_effects"]["to_Meet"]
        _MeetMatrix_ = self.matrix_operation(_MeetMatrix)
        _Input = _Vocab["input"]
        meet_keyword_groups = _Input["key_words"]["Meet"]

        is_meet_command = "/meet" in user_text

        time_keywords = {
            "morning": (meet_keyword_groups[0] if len(meet_keyword_groups) > 0 else []),
            "noon": (meet_keyword_groups[1] if len(meet_keyword_groups) > 1 else []),
            "afternoon": (
                meet_keyword_groups[2] if len(meet_keyword_groups) > 2 else []
            ),
            "evening": (meet_keyword_groups[3] if len(meet_keyword_groups) > 3 else []),
        }

        current_time_segment = None
        if 5 <= current_hour <= 11:
            current_time_segment = "morning"
        elif 12 <= current_hour <= 13:
            current_time_segment = "noon"
        elif 14 <= current_hour <= 18:
            current_time_segment = "afternoon"
        elif 19 <= current_hour <= 22:
            current_time_segment = "evening"
        else:
            current_time_segment = "midnight"

        if not is_meet_command:
            time_correct = True
            if current_time_segment != "midnight":
                has_correct_keyword = False
                for keyword in time_keywords[current_time_segment]:
                    if keyword in user_text:
                        has_correct_keyword = True
                        break

                has_wrong_keyword = False
                for segment, keywords in time_keywords.items():
                    if segment != current_time_segment and segment != "midnight":
                        for keyword in keywords:
                            if keyword in user_text:
                                has_wrong_keyword = True
                                break
                        if has_wrong_keyword:
                            break

                if has_wrong_keyword or not has_correct_keyword:
                    time_correct = False

            if not time_correct:
                text_keys = list(wrong_texts.keys())
                selected_key = random.choices(text_keys, weights=weights[5])[0]
                output = wrong_texts[selected_key]
                selected_index = text_keys.index(selected_key)

                for i in range(len(wrong_texts)):
                    if i == selected_index:
                        weights[5][i] = 0
                    else:
                        weights[5][i] += random.randint(6, 12)

                # 更新权重文件
                _MeetOutput["weights"] = weights
                _Vocab["output"]["Meet"] = _MeetOutput
                with open(
                    "attributes/vocabularies.json", "w", encoding="utf-8"
                ) as file:
                    json.dump(_Vocab, file, ensure_ascii=False, indent=4)

                return output

        if current_time_segment == "morning":
            text_keys = list(morning_texts.keys())
            selected_key = random.choices(text_keys, weights=weights[0])[0]
            output = morning_texts[selected_key]
            selected_index = text_keys.index(selected_key)

            for i in range(len(morning_texts)):
                if i == selected_index:
                    weights[0][i] = 0
                else:
                    weights[0][i] += random.randint(6, 12)

        elif current_time_segment == "noon":
            text_keys = list(noon_texts.keys())
            selected_key = random.choices(text_keys, weights=weights[1])[0]
            output = noon_texts[selected_key]
            selected_index = text_keys.index(selected_key)

            for i in range(len(noon_texts)):
                if i == selected_index:
                    weights[1][i] = 0
                else:
                    weights[1][i] += random.randint(6, 12)

        elif current_time_segment == "afternoon":
            text_keys = list(afternoon_texts.keys())
            selected_key = random.choices(text_keys, weights=weights[2])[0]
            output = afternoon_texts[selected_key]
            selected_index = text_keys.index(selected_key)

            for i in range(len(afternoon_texts)):
                if i == selected_index:
                    weights[2][i] = 0
                else:
                    weights[2][i] += random.randint(6, 12)

        elif current_time_segment == "evening":
            text_keys = list(evening_texts.keys())
            selected_key = random.choices(text_keys, weights=weights[3])[0]
            output = evening_texts[selected_key]
            selected_index = text_keys.index(selected_key)

            for i in range(len(evening_texts)):
                if i == selected_index:
                    weights[3][i] = 0
                else:
                    weights[3][i] += random.randint(6, 12)

        else:
            text_keys = list(midnight_texts.keys())
            selected_key = random.choices(text_keys, weights=weights[4])[0]
            output = midnight_texts[selected_key]
            selected_index = text_keys.index(selected_key)

            for i in range(len(midnight_texts)):
                if i == selected_index:
                    weights[4][i] = 0
                else:
                    weights[4][i] += random.randint(6, 12)

        # 更新权重文件
        _MeetOutput["weights"] = weights
        _Vocab["output"]["Meet"] = _MeetOutput
        with open("attributes/vocabularies.json", "w", encoding="utf-8") as file:
            json.dump(_Vocab, file, ensure_ascii=False, indent=4)
        return output

    # 用法·戳戳
    def Click(self):
        # 文件读取
        with open("attributes/vocabularies.json", "r", encoding="utf-8") as file:
            _Vocab = json.load(file)
        with open("attributes/data.json", "r", encoding="utf-8") as file:
            _Data = json.load(file)

        # 调用文件参数
        _ClickOutput = _Vocab["output"]["Click"]
        normal_texts = _ClickOutput["normal_texts"]
        special_texts = _ClickOutput["special_texts"]
        weights = _ClickOutput["weights"]
        standard_chance = _ClickOutput["standard_chance"]
        intelligence = _Data["intelligence"]

        # 分析
        _ClickMatrix = _Data["Methods_effects"]["to_Click"]
        _ClickMatrix_ = self.matrix_operation(_ClickMatrix)

        if random.random() <= standard_chance:
            text_keys = list(normal_texts.keys())
            selected_key = random.choices(text_keys, weights=weights[0])[0]
            output = normal_texts[selected_key]
            selected_index = text_keys.index(selected_key)

            for i in range(len(normal_texts)):
                if i == selected_index:
                    weights[0][i] = 0
                else:
                    weights[0][i] += random.randint(6, 12)

            standard_chance -= 0.01 + intelligence * 0.0002
        else:
            text_keys = list(special_texts.keys())
            selected_key = random.choices(text_keys, weights=weights[1])[0]
            output = special_texts[selected_key]
            selected_index = text_keys.index(selected_key)

            for i in range(len(special_texts)):
                if i == selected_index:
                    weights[1][i] = 0
                else:
                    weights[1][i] += random.randint(6, 12)

            standard_chance += 0.07 - intelligence * 0.0002
            standard_chance = round(standard_chance, 4)

        if standard_chance >= 1.00:
            standard_chance = 0.99

        weights[0][0] -= _Data["will"] * 2

        if weights[0][0] < 0:
            weights[0][0] = 0

        # 文件覆写
        _ClickOutput["weights"] = weights
        _ClickOutput["standard_chance"] = standard_chance
        _Vocab["output"]["Click"] = _ClickOutput

        with open("attributes/vocabularies.json", "w", encoding="utf-8") as file:
            json.dump(_Vocab, file, ensure_ascii=False, indent=4)

        return output

    # 用法·退出对话
    def Quit(self):
        # 文件读取
        with open("attributes/vocabularies.json", "r", encoding="utf-8") as file:
            _Vocab = json.load(file)

        # 调用文件参数
        _QuitOutput = _Vocab["output"]["Quit"]
        quit_texts = _QuitOutput["texts"]

        weights = _QuitOutput["weights"]

        text_keys = list(quit_texts.keys())
        selected_key = random.choices(text_keys, weights=weights)[0]
        output = quit_texts[selected_key]
        selected_index = text_keys.index(selected_key)

        for i in range(len(quit_texts)):
            if i == selected_index:
                weights[i] = 0
            else:
                weights[i] += random.randint(6, 12)

        # 文件覆写
        _QuitOutput["weights"] = weights
        _Vocab["output"]["Quit"] = _QuitOutput

        with open("attributes/vocabularies.json", "w", encoding="utf-8") as file:
            json.dump(_Vocab, file, ensure_ascii=False, indent=4)

        return output

    def analyse_emotion(self):
        # 读取文件
        with open("attributes/vocabularies.json", "r", encoding="utf-8") as file:
            _Vocab = json.load(file)
        with open("attributes/data.json", "r", encoding="utf-8") as file:
            _Data = json.load(file)
        _Input = _Vocab["input"]
        emotion_possibilities = _Data["emotions"]["data"]
        # 分析
        count = 0
        while count < 20:
            count += 1
            # 开心 伤心 好奇 愤怒 害怕 惊讶
            if count != 1:
                for i in range(0,5):
                    emotion_possibilities[i] *= 0.6
            user_text = input()
            emotion_types = [
                "happiness",
                "sadness",
                "curiosity",
                "anger",
                "fear",
                "surprise",
            ]
            emotion_typecount = 0

            for emotion_typecount in range(0, 5):
                c = 0
                m = 1
                words = _Input[emotion_types[emotion_typecount]]["words"]
                coiiu = 0
                for word in words:
                    coiiu += 1
                    if word in user_text:
                        if emotion_typecount == 1 or 4:
                            emotion_possibilities[0] *= 0.8

                        if emotion_typecount == 0:
                            emotion_possibilities[1] *= 0.85
                            emotion_possibilities[4] *= 0.85

                        c += _Input[emotion_types[emotion_typecount]][word] ** max(
                            1 / math.sqrt(coiiu), 0.3
                        )

                words = _Input["intensifiers"]["words"]
                for word in words:
                    if word in user_text:
                        m *= _Input["intensifiers"][word]

                emotion_possibilities[emotion_typecount] += c * m
                emotion_typecount += 1

            for i in range(0, 5):
                if emotion_possibilities[i] > 1:
                    for j in range(0, 5):
                        if i != j:
                            emotion_possibilities[j] /= emotion_possibilities[i]
                    emotion_possibilities[i] = 1
            for i in range(0, 5):
                emotion_possibilities[i] = round(emotion_possibilities[i], 3)
            print(f"{emotion_possibilities}")

    # 用法·分析对话
    def analyse_user_input(self, user_text):
        # 读取文件
        with open("attributes/vocabularies.json", "r", encoding="utf-8") as file:
            _Vocab = json.load(file)
        _Input = _Vocab["input"]

        # 分析
        slime_response = ""
        triggered_actions = set()
        should_quit = False

        for action_type, keyword_groups in _Input["key_words"].items():
            all_keywords = []
            for group in keyword_groups:
                if isinstance(group, list):
                    all_keywords.extend(group)
                else:
                    all_keywords.append(group)

            for keyword in all_keywords:
                if keyword in user_text:
                    triggered_actions.add(action_type)
                    break

        if triggered_actions:
            action_priority = {"Meet": 1, "Click": 2, "Quit": 3}
            sorted_actions = sorted(
                triggered_actions, key=lambda x: action_priority.get(x, 999)
            )

            responses = []

            for action_type in sorted_actions:
                if hasattr(self, action_type):
                    if action_type == "Meet":
                        method = getattr(self, action_type)
                        response = method(user_text)
                    else:
                        method = getattr(self, action_type)
                        response = method()

                    if response:
                        print(f">- {response}")
                        responses.append(response)

                    if action_type == "Quit":
                        should_quit = True
                else:
                    print(f">- 警告: 未找到方法 {action_type}")

            if responses:
                slime_response = " ".join(responses)

        self.save_conversation(user_text, slime_response)

        # 文件覆写
        with open("attributes/data.json", "r", encoding="utf-8") as file:
            _Data = json.load(file)

        will_change = -1

        _Data["will"] += will_change
        if _Data["will"] < -50:
            _Data["will"] = -50
        if _Data["will"] > 100:
            _Data["will"] = 100

        with open("attributes/data.json", "w", encoding="utf-8") as file:
            json.dump(_Data, file, ensure_ascii=False, indent=4)

        if should_quit:
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            log_file = f"memories/{today}.log"
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] $Quit()\n"
            with open(log_file, "a", encoding="utf-8") as file:
                file.write(log_entry)

            exit()

        return list(user_text)

    # 用法·矩阵运算
    def matrix_operation(self, M, intensity=0.1):
        with open("attributes/data.json", "r", encoding="utf-8") as file:
            _Data = json.load(file)

        matrix_M = np.array(_Data["emotions"]["data"]).reshape(6, 1)
        matrix_X = np.array(M)

        change = np.dot(matrix_X, matrix_M) - matrix_M
        matrix_M_ = matrix_M + intensity * change

        matrix_M_ = np.clip(matrix_M_, 0, 1)
        _Data["emotions"]["data"] = matrix_M_.flatten().tolist()

        with open("attributes/data.json", "w", encoding="utf-8") as file:
            json.dump(_Data, file, ensure_ascii=False, indent=4)

        return matrix_M_

    # 用法·保存对话
    def save_conversation(self, user_text, slime_response):
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        log_file = f"memories/{today}.log"

        if not os.path.exists("memories"):
            os.makedirs("memories")

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] >: {user_text}\n[{timestamp}] {slime_response}\n"

        with open(log_file, "a", encoding="utf-8") as file:
            file.write(log_entry)


pet = Slime()
"""
print(">- 史莱姆正在检查文件完整性. . .")
if pet.Check_integrity():
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    log_file = f"memories/{today}.log"
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_entry = f"\n[{timestamp}] $Start_chat()\n"
    with open(log_file, "a", encoding="utf-8") as file:
        file.write(log_entry)
    with open("attributes/data.json", "r", encoding="utf-8") as file:
        _Data = json.load(file)
        _Data["will"] = 50
        _Data["emotions"]["data"] = _Data["emotions"]["default"]
    with open("attributes/data.json", "w", encoding="utf-8") as file:
        json.dump(_Data, file, ensure_ascii=False, indent=4)

    pet.Start_chat()
"""
pet.analyse_emotion()


"""
"我恨死这个人了，他毁了我的一切！"
"天啊！这简直太不可思议了！"
"救命啊！有怪物在追我！"
"虽然分手很伤心，但也为彼此感到解脱"
"这个谜题既让我困惑又让我着迷"
"听到这个消息，我又惊又喜又有点担心"
"雨声让人感到莫名的安宁"
"独自在深夜，有些孤独但也很自在"
"看着夕阳，心里有种说不出的惆怅"
"我该高兴还是难过？这个选择太艰难了"
"既期待又害怕明天的到来"
"对他又爱又恨，心情很复杂"
"迷路在陌生的城市，既害怕又觉得刺激"
"收到意外的礼物，惊喜中带着困惑"
"比赛输了，失望但为对手感到高兴"
"稍微有点不开心"
"我简直气疯了！"
"隐隐约约感到不安"
"这种意境让人心生向往"
"看到这一幕，不禁感慨万千"
"""
