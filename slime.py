import json
import os
import random
import tkinter as tk
from PIL import Image, ImageTk
import datetime
import math
import numpy as np

class Slime:

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
            "morning": (
                meet_keyword_groups[0] if len(meet_keyword_groups) > 0 else []
            ),  # 早上关键词
            "noon": (
                meet_keyword_groups[1] if len(meet_keyword_groups) > 1 else []
            ),  # 中午关键词
            "afternoon": (
                meet_keyword_groups[2] if len(meet_keyword_groups) > 2 else []
            ),  # 下午关键词
            "evening": (
                meet_keyword_groups[3] if len(meet_keyword_groups) > 3 else []
            ),  # 晚上关键词
            # meet_keyword_groups[4] 是 "/meet"，跳过
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
            current_time_segment = "midnight"  # 深夜没有特定关键词

        if not is_meet_command:
            time_correct = True
            if current_time_segment != "midnight":
                # 检查用户输入是否包含当前时段的关键词
                has_correct_keyword = False
                for keyword in time_keywords[current_time_segment]:
                    if keyword in user_text:
                        has_correct_keyword = True
                        break

                # 检查用户输入是否包含其他时段的错误关键词
                has_wrong_keyword = False
                for segment, keywords in time_keywords.items():
                    if segment != current_time_segment and segment != "midnight":
                        for keyword in keywords:
                            if keyword in user_text:
                                has_wrong_keyword = True
                                break
                        if has_wrong_keyword:
                            break

                # 如果有错误关键词或者没有正确关键词，时间不正确
                if has_wrong_keyword or not has_correct_keyword:
                    time_correct = False

            # 如果时间不符合
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

        else:  # midnight
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

    def analyse_user_input(self, user_text):
        with open("attributes/vocabularies.json", "r", encoding="utf-8") as file:
            _Vocab = json.load(file)
        _Input = _Vocab["input"]

        positive = 0
        negative = 0
        slime_response = ""
        triggered_actions = set()
        should_quit = False

        # 收集触发动作
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

        # 按优先级执行所有动作（分多行输出）
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
                        # 每个动作单独输出一行
                        print(f">- {response}")
                        responses.append(response)

                    # 设置退出标志
                    if action_type == "Quit":
                        should_quit = True
                else:
                    print(f">- 警告: 未找到方法 {action_type}")

            # 保存时合并所有回应
            if responses:
                slime_response = " ".join(responses)

        # 情感分析
        for char in user_text:
            if char in _Input["positive"]:
                positive += 1
            elif char in _Input["negative"]:
                negative += 1

        # 如果没有触发动作，给出情感反应
        if not triggered_actions:
            if positive > negative:
                positive_responses = [
                    "+",
                    "开心地晃动~",
                    "噗噜噗噜~",
                    "发出快乐的光芒",
                    "姆姆！好高兴",
                    "变成亮绿色~",
                ]
                slime_response = random.choice(positive_responses)
            elif negative > positive:
                negative_responses = [
                    "-",
                    "缩成一团...",
                    "变成深蓝色",
                    "咕…不开心",
                    "发出低落的光芒",
                    "姆…有点难过",
                ]
                slime_response = random.choice(negative_responses)
            else:
                neutral_responses = [
                    "咕？",
                    "歪着头看着你",
                    "发出好奇的光芒",
                    "噗噜？",
                    "姆~？",
                    "轻轻晃动",
                ]
                slime_response = random.choice(neutral_responses)
            print(f">- {slime_response}")

        # 保存对话到日志
        self.save_conversation(user_text, slime_response)

        # 更新意志力
        with open("attributes/data.json", "r", encoding="utf-8") as file:
            _Data = json.load(file)
        _Data["will"] -= 1 - positive + negative
        if _Data["will"] < -50:
            _Data["will"] = -50
        if _Data["will"] > 100:
            _Data["will"] = 100

        with open("attributes/data.json", "w", encoding="utf-8") as file:
            json.dump(_Data, file, ensure_ascii=False, indent=4)

        # 如果是退出命令，在这里退出
        if should_quit:
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            log_file = f"memories/{today}.log"
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] $Quit()\n"
            with open(log_file, "a", encoding="utf-8") as file:
                file.write(log_entry)

            exit()

        return list(user_text)

    def matrix_operation(self, M, intensity=0.1):
        with open("attributes/data.json", "r", encoding="utf-8") as file:
            _Data = json.load(file)

        matrix_M = np.array(_Data["attributes"]).reshape(6, 1)
        matrix_X = np.array(M)

        change = np.dot(matrix_X, matrix_M) - matrix_M
        matrix_M_ = matrix_M + intensity * change
        
        matrix_M_ = np.clip(matrix_M_, 0, 1)
        _Data["attributes"] = matrix_M_.flatten().tolist()

        with open("attributes/data.json", "w", encoding="utf-8") as file:
            json.dump(_Data, file, ensure_ascii=False, indent=4)

        return matrix_M_

    def save_conversation(self, user_text, slime_response):
        """保存对话到日志文件"""
        # 获取当前日期
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        log_file = f"memories/{today}.log"

        # 确保memories文件夹存在
        if not os.path.exists("memories"):
            os.makedirs("memories")

        # 写入对话
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

    pet.Start_chat()



class SlimeMatrixLab:
    def __init__(self):
        # 预定义史莱姆状态向量 (8维)
        self.slime_state = np.array([0.8, 0.6, 0.3, 0.9, 0.2, 0.7, 0.4, 0.5])
        self.state_names = [
            "能量",
            "心情",
            "饥饿",
            "亲密度",
            "好奇心",
            "活跃度",
            "学习力",
            "社交需求",
        ]

    def show_slime_state(self):
        """显示当前史莱姆状态"""
        print("\n=== 当前史莱姆状态 ===")
        for i, (name, value) in enumerate(zip(self.state_names, self.slime_state)):
            print(f"{i+1}. {name}: {value:.2f}")

    def create_behavior_matrix(self):
        """创建行为影响矩阵"""
        print("\n=== 创建行为影响矩阵 ===")
        print(
            "矩阵将影响: [能量, 心情, 饥饿, 亲密度, 好奇心, 活跃度, 学习力, 社交需求]"
        )

        matrix = []
        for i, target_attr in enumerate(self.state_names):
            row = []
            print(f"\n设置对 [{target_attr}] 的影响:")
            for j, source_attr in enumerate(self.state_names):
                try:
                    effect = float(input(f"  {source_attr} 的影响系数: "))
                    row.append(effect)
                except ValueError:
                    print("  使用默认值 0.0")
                    row.append(0.0)
            matrix.append(row)

        return np.array(matrix)

    def apply_behavior(self):
        """应用行为矩阵"""
        print("\n=== 应用行为影响 ===")
        matrix = self.create_behavior_matrix()

        print(f"\n行为矩阵:")
        print(matrix)

        # 计算新状态
        new_state = np.dot(matrix, self.slime_state)
        new_state = np.clip(new_state, 0, 1)  # 限制在0-1范围

        print(f"\n应用前状态: {self.slime_state}")
        print(f"应用后状态: {new_state}")

        # 显示变化
        print("\n状态变化:")
        for i, (name, old, new) in enumerate(
            zip(self.state_names, self.slime_state, new_state)
        ):
            change = new - old
            print(f"{name}: {old:.2f} → {new:.2f} ({change:+.2f})")

        self.slime_state = new_state

    def test_feeding_behavior(self):
        """测试喂食行为"""
        print("\n=== 测试喂食行为 ===")
        # 喂食行为的影响矩阵
        feeding_matrix = np.array(
            [
                [1.2, 0.1, -0.5, 0.1, 0.0, 0.1, 0.0, 0.1],
                [0.1, 1.1, -0.3, 0.2, 0.1, 0.2, 0.0, 0.1],
                [0.0, 0.2, 0.8, 0.1, 0.1, 0.0, 0.1, 0.1],
                [0.1, 0.1, 0.1, 1.1, 0.0, 0.1, 0.0, 0.2],
                [0.0, 0.1, 0.0, 0.0, 1.0, 0.1, 0.1, 0.0],
                [0.1, 0.2, 0.0, 0.1, 0.1, 1.1, 0.0, 0.1],
                [0.0, 0.0, 0.1, 0.0, 0.2, 0.0, 1.0, 0.0],
                [0.1, 0.1, 0.0, 0.2, 0.0, 0.1, 0.0, 1.1],
            ]
        )

        old_state = self.slime_state.copy()
        new_state = np.dot(feeding_matrix, old_state)
        new_state = np.clip(new_state, 0, 1)

        print("喂食后的状态变化:")
        for i, (name, old, new) in enumerate(
            zip(self.state_names, old_state, new_state)
        ):
            change = new - old
            print(f"{name}: {old:.2f} → {new:.2f} ({change:+.2f})")

        self.slime_state = new_state

    def run(self):
        """运行实验室"""
        print("🧪 史莱姆矩阵实验室")
        print("=" * 30)

        while True:
            self.show_slime_state()
            print("\n选择操作:")
            print("1. 创建行为矩阵")
            print("2. 应用行为影响")
            print("3. 测试喂食行为")
            print("4. 重置史莱姆状态")
            print("0. 退出")

            choice = input("请输入选择: ").strip()

            if choice == "1":
                matrix = self.create_behavior_matrix()
                print("创建的行为矩阵:")
                print(matrix)
            elif choice == "2":
                self.apply_behavior()
            elif choice == "3":
                self.test_feeding_behavior()
            elif choice == "4":
                self.slime_state = np.array([0.8, 0.6, 0.3, 0.9, 0.2, 0.7, 0.4, 0.5])
                print("状态已重置")
            elif choice == "0":
                print("实验室关闭！")
                break
            else:
                print("无效选择")


# 运行实验室
if __name__ == "__main__":
    lab = SlimeMatrixLab()
    # lab.run()


"""
    def __init__(self):
        self.gui = tk.Tk()
        self.gui.overrideredirect(True)
        self.gui.attributes("-alpha", 0.8)
        self.gui.attributes("-topmost", True)
        self.gui.attributes("-transparentcolor", "white")
        try:
            with open("gui/gui.json", "r", encoding="utf-8") as file:
                size_of_gui = json.load(file)
            x = size_of_gui["initial_pos"][0]
            y = size_of_gui["initial_pos"][1]
            self.size = size_of_gui["initial_size"]
            self.gui.geometry(f"{self.size}x{self.size}+{x}+{y}")
        except FileNotFoundError:
            print("gui.json文件不存在！")
            return 0
        except Exception as e:
            print(f"读取配置文件出错: {e}")
            return 0

        try:
            from PIL import Image, ImageTk

            image = Image.open("gui/slime.png")
            image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image)
            self.original_image = image

            self.label = tk.Label(self.gui, image=self.photo, bg="white")
            self.label.pack()

        except Exception as e:
            print(f"加载图片失败: {e}")

        self.load_animation_settings()  # 加载动画设置
        self.start_idle_animation()  # 开始动画

        self.gui.bind("<Escape>", self.quit)

    def load_animation_settings(self):
        #加载动画设置
        try:
            with open("gui/gui.json", "r", encoding="utf-8") as file:
                animation_data = json.load(file)

            animation_config = animation_data["animation"]
            self.swing_range = animation_config["swing_range"]  # 修正键名
            self.animation_speed = animation_config["speed"]
            self.animation_phase = animation_config["phase"]  # 初始化相位

        except Exception as e:
            print(f"加载动画设置失败: {e}")
            # 默认值
            self.swing_range = 10
            self.animation_speed = 0.2
            self.animation_phase = 0

    def start_idle_animation(self):
        self.animate_swing()

    def animate_swing(self):
        #print(f"动画执行中... 相位: {self.animation_phase}, 幅度: {self.swing_range}")  # 调试

        if hasattr(self, "original_image"):
            swing_offset = int(math.sin(self.animation_phase) * self.swing_range)

            new_x = self.gui.winfo_x() + swing_offset
            current_y = self.gui.winfo_y()

            self.gui.geometry(f"{self.size}x{self.size}+{new_x}+{current_y}")

            self.animation_phase += self.animation_speed

            self.gui.after(50, self.animate_swing)
    def quit(self, event=None):
        self.gui.quit()

    def run(self):
            self.gui.mainloop()
"""
