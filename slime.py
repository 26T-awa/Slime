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

            # 现在 analyse_user_input 返回两个值
            user_chars, emotion_matrix = self.analyse_user_input(user_text)

            # 可选：显示情绪矩阵（调试用）
            if any(value > 0 for value in emotion_matrix.values()):
                print(f"   情绪变化: {emotion_matrix}")

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

    # 用法·分析对话
    def analyse_user_input(self, user_text):
        # 读取文件
        with open("attributes/vocabularies.json", "r", encoding="utf-8") as file:
            _Vocab = json.load(file)
        _Input = _Vocab["input"]

        # 分析情绪
        emotion_scores = self.analyze_emotion_simple(user_text)

        # 更新史莱姆情绪数据
        self.update_slime_emotions(emotion_scores)

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
        else:
            # 没有触发动作时，根据情绪生成回应
            slime_response = self.generate_emotion_response(emotion_scores)
            print(f">- {slime_response}")

        self.save_conversation(user_text, slime_response)

        # 文件覆写 - 意志力受情绪影响
        with open("attributes/data.json", "r", encoding="utf-8") as file:
            _Data = json.load(file)

        # 意志力变化：正面情绪减少意志力消耗，负面情绪增加消耗
        will_change = -1
        if emotion_scores["开心"] > 0.5:
            will_change += 1  # 开心时减少消耗
        if emotion_scores["生气"] > 0.5 or emotion_scores["伤心"] > 0.5:
            will_change -= 1  # 负面情绪增加消耗

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

        return list(user_text), emotion_scores  # 返回情绪值

    def analyze_emotion_simple(self, user_text):
        """简单的情绪分析"""
        with open("attributes/vocabularies.json", "r", encoding="utf-8") as file:
            _Vocab = json.load(file)

        if "emotion_triggers" not in _Vocab["input"]:
            # 如果没有配置情绪触发器，返回默认值
            return {"开心": 0, "伤心": 0, "好奇": 0, "生气": 0, "害怕": 0, "惊讶": 0}

        emotion_triggers = _Vocab["input"]["emotion_triggers"]
        intensifiers_config = _Vocab["input"].get(
            "intensifiers", {"words": [], "multiplier": 1.5}
        )

        emotion_scores = {
            "开心": 0,
            "伤心": 0,
            "好奇": 0,
            "生气": 0,
            "害怕": 0,
            "惊讶": 0,
        }

        # 检查强度修饰
        has_intensifier = any(
            word in user_text for word in intensifiers_config["words"]
        )
        intensity_multiplier = (
            intensifiers_config["multiplier"] if has_intensifier else 1.0
        )

        # 分析每个情绪类别
        for emotion, config in emotion_triggers.items():
            base_intensity = config["base_intensity"]
            for word in config["words"]:
                if word in user_text:
                    emotion_scores[emotion] += base_intensity * intensity_multiplier

        # 限制范围
        for emotion in emotion_scores:
            emotion_scores[emotion] = min(1.0, emotion_scores[emotion])

        return emotion_scores

    def update_slime_emotions(self, emotion_scores):
        """更新史莱姆情绪数据"""
        with open("attributes/data.json", "r", encoding="utf-8") as file:
            _Data = json.load(file)

        current_emotions = _Data["emotions"]["data"]
        emotion_names = _Data["emotions"]["name"]

        # 确保情绪名称匹配
        emotion_mapping = {
            "开心": "开心",
            "伤心": "伤心",
            "好奇": "好奇",
            "生气": "生气",
            "害怕": "害怕",
            "惊讶": "惊讶",
        }

        # 更新情绪值（新影响 + 旧情绪衰减）
        for i, emotion_name in enumerate(emotion_names):
            # 使用映射确保名称一致
            mapped_name = emotion_mapping.get(emotion_name, emotion_name)
            current_value = current_emotions[i]
            impact = emotion_scores.get(mapped_name, 0)

            # 新情绪影响 + 旧情绪自然衰减
            new_value = current_value * 0.9 + impact * 0.3
            current_emotions[i] = max(0, min(1.0, new_value))

        # 保存更新
        _Data["emotions"]["data"] = current_emotions
        with open("attributes/data.json", "w", encoding="utf-8") as file:
            json.dump(_Data, file, ensure_ascii=False, indent=4)

    def generate_emotion_response(self, emotion_scores):
        """根据情绪生成回应"""
        # 找出主导情绪
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        dominant_intensity = emotion_scores[dominant_emotion]

        emotion_responses = {
            "开心": [
                "开心地晃动~",
                "噗噜噗噜~",
                "发出快乐的光芒",
                "姆姆！好高兴",
                "变成亮绿色~",
                "咕啾咕啾~",
            ],
            "伤心": [
                "缩成一团...",
                "变成深蓝色",
                "咕…不开心",
                "发出低落的光芒",
                "姆…有点难过",
                "静静地待着",
            ],
            "好奇": [
                "歪着头看着你",
                "发出好奇的光芒",
                "噗噜？",
                "姆~？",
                "轻轻晃动",
                "好奇地靠近",
            ],
            "生气": [
                "变成红色！",
                "气鼓鼓地",
                "发出不满的光芒",
                "转过身去",
                "哼！",
                "不想理你",
            ],
            "害怕": [
                "颤抖着",
                "躲到角落",
                "变成透明色",
                "发出害怕的光芒",
                "缩成一小团",
                "咕...害怕",
            ],
            "惊讶": [
                "突然跳起来",
                "发出闪烁的光芒",
                "哇！",
                "睁大眼睛",
                "噗叽！",
                "被吓到了",
            ],
        }

        if dominant_intensity > 0.3:
            responses = emotion_responses.get(dominant_emotion, ["咕？"])
            return random.choice(responses)
        else:
            # 情绪不明显时的默认回应
            neutral_responses = ["咕？", "轻轻晃动", "发出微弱的光芒", "噗噜~"]
            return random.choice(neutral_responses)

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
