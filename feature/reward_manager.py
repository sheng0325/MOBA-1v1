#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import math
from ppo.config import GameConfig

# Used to record various reward information
# 用于记录各个奖励信息
class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight
        self.min_value = -1
        self.is_first_arrive_center = True


# Used to initialize various reward information
# 用于初始化各个奖励信息
def init_calc_frame_map():
    calc_frame_map = {}
    for key, weight in GameConfig.REWARD_WEIGHT_DICT.items():
        calc_frame_map[key] = RewardStruct(weight)
    return calc_frame_map


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.main_hero_hp = -1
        self.main_hero_organ_hp = -1
        self.m_reward_value = {}
        self.m_last_frame_no = -1
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.m_init_calc_frame_map = {}
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        self.m_main_hero_config_id = -1
        self.m_each_level_max_exp = {}
        self.main_hero_hp_last = None
        self.hp_decrease_last_frame = 0  # 记录上一帧HP减少量

    # Used to initialize the maximum experience value for each agent level
    # 用于初始化智能体各个等级的最大经验值
    def init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp.clear()
        self.m_each_level_max_exp[1] = 160
        self.m_each_level_max_exp[2] = 298
        self.m_each_level_max_exp[3] = 446
        self.m_each_level_max_exp[4] = 524
        self.m_each_level_max_exp[5] = 613
        self.m_each_level_max_exp[6] = 713
        self.m_each_level_max_exp[7] = 825
        self.m_each_level_max_exp[8] = 950
        self.m_each_level_max_exp[9] = 1088
        self.m_each_level_max_exp[10] = 1240
        self.m_each_level_max_exp[11] = 1406
        self.m_each_level_max_exp[12] = 1585
        self.m_each_level_max_exp[13] = 1778
        self.m_each_level_max_exp[14] = 1984

    def result(self, frame_data):
        self.init_max_exp_of_each_hero()
        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value)

        frame_no = frame_data["frameNo"]
        if self.time_scale_arg > 0:
            for key in self.m_reward_value:
                self.m_reward_value[key] *= math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)

        return self.m_reward_value

    # Calculate the value of each reward item in each frame
    # 计算每帧的每个奖励子项的信息
    def set_cur_calc_frame_vec(self, cul_calc_frame_map, frame_data, camp):

        # Get both agents
        # 获取双方智能体
        main_hero, enemy_hero = None, None
        hero_list = frame_data["hero_states"]
        for hero in hero_list:
            hero_camp = hero["actor_state"]["camp"]
            if hero_camp == camp:
                main_hero = hero
            else:
                enemy_hero = hero
        main_hero_hp = main_hero["actor_state"]["hp"]
        main_hero_max_hp = main_hero["actor_state"]["max_hp"]
        main_hero_ep = main_hero["actor_state"]["values"]["ep"]
        main_hero_max_ep = main_hero["actor_state"]["values"]["max_ep"]

        # 更新 main_hero_hp_last
        if self.main_hero_hp_last is None:
            self.main_hero_hp_last = main_hero_hp

        # Get both defense towers
        # 获取双方防御塔
        main_tower, main_spring, enemy_tower, enemy_spring = None, None, None, None
        npc_list = frame_data["npc_states"]
        for organ in npc_list:
            organ_camp = organ["camp"]
            organ_subtype = organ["sub_type"]
            if organ_camp == camp:
                if organ_subtype == "ACTOR_SUB_TOWER":  # 21 is ACTOR_SUB_TOWER, normal tower
                    main_tower = organ
                elif organ_subtype == "ACTOR_SUB_CRYSTAL":  # 24 is ACTOR_SUB_CRYSTAL, base crystal
                    main_spring = organ
            else:
                if organ_subtype == "ACTOR_SUB_TOWER":  # 21 is ACTOR_SUB_TOWER, normal tower
                    enemy_tower = organ
                elif organ_subtype == "ACTOR_SUB_CRYSTAL":  # 24 is ACTOR_SUB_CRYSTAL, base crystal
                    enemy_spring = organ
                    

        for reward_name, reward_struct in cul_calc_frame_map.items():
            reward_struct.last_frame_value = reward_struct.cur_frame_value
            # Money
            # 金钱
            if reward_name == "money":
                reward_struct.cur_frame_value = main_hero["moneyCnt"]
            # Health points
            # 生命值
            elif reward_name == "hp_point":
                reward_struct.cur_frame_value = math.sqrt(math.sqrt(1.0 * main_hero_hp / main_hero_max_hp))
            # Energy points
            # 法力值
            elif reward_name == "ep_rate":
                if main_hero_max_ep == 0 or main_hero_hp <= 0:
                    reward_struct.cur_frame_value = 0
                else:
                    reward_struct.cur_frame_value = main_hero_ep / float(main_hero_max_ep)
            # Kills
            # 击杀
            elif reward_name == "kill":
                reward_struct.cur_frame_value = main_hero["killCnt"]
            # Deaths
            # 死亡
            elif reward_name == "death":
                reward_struct.cur_frame_value = main_hero["deadCnt"]
            # Tower health points
            # 塔血量
            elif reward_name == "tower_hp_point":
                reward_struct.cur_frame_value = 1.0 * main_tower["hp"] / main_tower["max_hp"]
            # Last hit
            # 补刀
            elif reward_name == "last_hit":
                reward_struct.cur_frame_value = 0.0
                frame_action = frame_data["frame_action"]
                if "dead_action" in frame_action:
                    dead_actions = frame_action["dead_action"]
                    for dead_action in dead_actions:
                        if (
                            dead_action["killer"]["runtime_id"] == main_hero["actor_state"]["runtime_id"]
                            and dead_action["death"]["sub_type"] == "ACTOR_SUB_SOLDIER"
                        ):
                            reward_struct.cur_frame_value += 1.0
                        elif (
                            dead_action["killer"]["runtime_id"] == enemy_hero["actor_state"]["runtime_id"]
                            and dead_action["death"]["sub_type"] == "ACTOR_SUB_SOLDIER"
                        ):
                            reward_struct.cur_frame_value -= 1.0
                        # **策略1：限制过度激进的追击**
            elif reward_name == "avoid_over_aggressive":
                reward_struct.cur_frame_value = 0.0
                # 检查英雄是否在敌方防御塔范围内
                if self.is_in_enemy_tower_range(main_hero, enemy_tower):
                    # 在敌方防御塔范围内，给予惩罚
                    reward_struct.cur_frame_value = -1.0
                else:
                    # 不在敌方防御塔范围内，给予轻微奖励，鼓励安全行动
                    reward_struct.cur_frame_value = 0.1
                # 检查英雄的生命值
                hero_hp_percentage = main_hero_hp / main_hero_max_hp
                if hero_hp_percentage < 0.3:
                    # 生命值过低，且靠近敌方英雄，给予惩罚
                    if self.is_near_enemy_hero(main_hero, enemy_hero):
                        reward_struct.cur_frame_value += -0.5
                # 如果英雄生命值较低且远离己方防御塔，给予惩罚
                if self.is_far_from_own_tower(main_hero, main_tower) and hero_hp_percentage < 0.3:
                    reward_struct.cur_frame_value += -0.5
            # **策略2：在无法检测小兵的情况下攻击防御塔**
            elif reward_name == "attack_enemy_tower":
                reward_struct.cur_frame_value = 0.0
                # 如果英雄靠近敌方防御塔
                if self.is_in_enemy_tower_range(main_hero, enemy_tower):
                    # 检查英雄的生命值变化
                    hp_decrease = self.main_hero_hp_last - main_hero_hp
                    if hp_decrease > 400:
                        # 生命值正在减少，可能被防御塔攻击，给予惩罚
                        reward_struct.cur_frame_value = -1.0
                    else:
                        # 生命值未减少，可能安全攻击防御塔，给予奖励
                        reward_struct.cur_frame_value = 1.0
            # **新增部分：攻击敌方英雄策略奖励**
            # --------------------------------------------------------
            elif reward_name == "attack_enemy_hero":
                # 计算我方与敌方英雄的生命值百分比差距
                hp_diff_percentage = self.calculate_hp_diff(main_hero, enemy_hero)
                
                if hp_diff_percentage >= 0.1:
                    # 优势范围：主动进攻敌方英雄，奖励 +5
                    reward_struct.cur_frame_value = 5.0
                    print(f"优势范围：主动进攻敌方英雄，奖励+20，敌方英雄ID={enemy_hero['actor_state']['runtime_id']}")
                elif -0.1 <= hp_diff_percentage < 0.1:
                    # 对峙范围：保持对峙，奖励 +2
                    reward_struct.cur_frame_value = 2.0
                    print(f"对峙范围：保持对峙，奖励+5，敌方英雄ID={enemy_hero['actor_state']['runtime_id']}")
                else:
                    # 劣势范围：撤退或防御，奖励 +1
                    reward_struct.cur_frame_value = 1.0
                    print(f"劣势范围：撤退或防御，奖励+3，敌方英雄ID={enemy_hero['actor_state']['runtime_id']}")
            # --------------------------------------------------------
             # --------------------------------------------------------
            elif reward_name == "pick_health_pack":
                reward_struct.cur_frame_value = 0.0  # 初始化当前帧的奖励值

                # 判断我方英雄的生命值是否低于60%
                hero_hp_percentage = main_hero_hp / main_hero_max_hp

                if hero_hp_percentage < 0.6:
                    # 生命值低于60%，考虑拾取血包

                    # 评估战场安全性
                    is_safe = self.evaluate_battlefield_safety(frame_data, camp, main_hero)

                    if is_safe:
                        # 查找最近的血包
                        nearest_health_pack = self.find_nearest_health_pack(frame_data, main_hero)
                        if nearest_health_pack:
                            # 判断是否成功拾取了血包
                            if self.has_picked_health_pack(frame_data, main_hero, nearest_health_pack):
                                # 成功拾取血包，给予奖励
                                reward_struct.cur_frame_value = 1.0  # 可以根据需要调整奖励值
                                print(f"成功拾取血包，奖励+{reward_struct.cur_frame_value}")
                            else:
                                # 尚未拾取血包，引导前往血包位置
                                reward_struct.cur_frame_value = 0.5  # 引导前往血包
                                print(f"生命值低于60%，引导前往最近的血包")
                        else:
                            # 没有可用的血包，可能需要等待
                            reward_struct.cur_frame_value = 0.0
                            print("没有可用的血包")
                    else:
                        # 战场不安全，建议返回防御塔
                        reward_struct.cur_frame_value = -0.5  # 惩罚试图在不安全情况下拾取血包
                        print("战场不安全，建议返回防御塔")
            # --------------------------------------------------------

            # Experience points
            # 经验值
            elif reward_name == "exp":
                reward_struct.cur_frame_value = self.calculate_exp_sum(main_hero)
            # Forward
            # 前进
            elif reward_name == "forward":
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower)
            
        self.main_hero_hp_last = main_hero_hp

    
    # **新增辅助方法**
    def is_in_enemy_tower_range(self, main_hero, enemy_tower):
        """
        判断英雄是否在敌方防御塔的攻击范围内
        """
        hero_pos = (main_hero["actor_state"]["location"]["x"], main_hero["actor_state"]["location"]["z"])
        tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        distance = self.calculate_distance(hero_pos, tower_pos)
        tower_attack_range = 8000  # 根据实际游戏参数调整
        return distance < tower_attack_range

    def is_near_enemy_hero(self, main_hero, enemy_hero):
        """
        判断英雄是否靠近敌方英雄
        """
        hero_pos = (main_hero["actor_state"]["location"]["x"], main_hero["actor_state"]["location"]["z"])
        enemy_pos = (enemy_hero["actor_state"]["location"]["x"], enemy_hero["actor_state"]["location"]["z"])
        distance = self.calculate_distance(hero_pos, enemy_pos)
        proximity_threshold = 8000  # 根据需要调整
        return distance < proximity_threshold

    def is_far_from_own_tower(self, main_hero, main_tower):
        """
        判断英雄是否远离己方防御塔
        """
        hero_pos = (main_hero["actor_state"]["location"]["x"], main_hero["actor_state"]["location"]["z"])
        tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
        distance = self.calculate_distance(hero_pos, tower_pos)
        safe_distance = 15000  # 根据需要调整
        return distance > safe_distance

    def evaluate_battlefield_safety(self, frame_data, camp, main_hero):
        """
        评估战场安全性，判断是否安全去拾取血包
        """
        # 获取敌方英雄列表
        enemy_heroes = []
        for hero in frame_data["hero_states"]:
            if hero["actor_state"]["camp"] != camp:
                enemy_heroes.append(hero)

        hero_position = (
            main_hero["actor_state"]["location"]["x"],
            main_hero["actor_state"]["location"]["z"]
        )

        # 判断附近是否有敌方英雄
        safe_distance_threshold = 8000  # 根据实际情况调整，单位可能是厘米

        for enemy_hero in enemy_heroes:
            enemy_position = (
                enemy_hero["actor_state"]["location"]["x"],
                enemy_hero["actor_state"]["location"]["z"]
            )
            distance = self.calculate_distance(hero_position, enemy_position)
            if distance < safe_distance_threshold:
                # 敌方英雄在附近，认为不安全
                return False

        # 没有敌方英雄在附近，认为安全
        return True

    def find_nearest_health_pack(self, frame_data, main_hero):
        """
        查找距离我方英雄最近的血包
        """
        hero_position = (
            main_hero["actor_state"]["location"]["x"],
            main_hero["actor_state"]["location"]["z"]
        )

        # 从 frame_data 中获取 health_packs
        health_packs = frame_data.get("health_packs", [])

        if not health_packs:
            return None

        min_distance = float('inf')
        nearest_pack = None
        for pack in health_packs:
            # 由于没有 is_available 字段，我们假设所有血包都是可用的
            pack_position = (
                pack["collider"]["location"]["x"],
                pack["collider"]["location"]["z"]
            )
            distance = self.calculate_distance(hero_position, pack_position)
            if distance < min_distance:
                min_distance = distance
                nearest_pack = pack

        return nearest_pack

    def has_picked_health_pack(self, frame_data, main_hero, health_pack):
        """
        判断我方英雄是否成功拾取了血包
        """
        # 检查英雄的生命值是否增加
        hp_increased = main_hero["actor_state"]["hp"] > self.main_hero_hp_last

        # 检查英雄是否接近血包位置
        hero_position = (
            main_hero["actor_state"]["location"]["x"],
            main_hero["actor_state"]["location"]["z"]
        )
        pack_position = (
            health_pack["collider"]["location"]["x"],
            health_pack["collider"]["location"]["z"]
        )
        distance_to_pack = self.calculate_distance(hero_position, pack_position)

        pick_up_distance_threshold = 200  # 根据实际情况调整，表示拾取血包的距离阈值

        if hp_increased and distance_to_pack < pick_up_distance_threshold:
            return True
        else:
            return False

    def calculate_distance(self, pos1, pos2):
        """
        计算两个位置之间的距离
        """
        return math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
    
    def calculate_hp_diff(self, player_hero, enemy_hero):
        """
        计算我方英雄与敌方英雄的生命值百分比差距
        """
        player_hp_percentage = player_hero["actor_state"]["hp"] / player_hero["actor_state"]["max_hp"]
        enemy_hp_percentage = enemy_hero["actor_state"]["hp"] / enemy_hero["actor_state"]["max_hp"]
        return player_hp_percentage - enemy_hp_percentage


    # Calculate the total amount of experience gained using agent level and current experience value
    # 用智能体等级和当前经验值，计算获得经验值的总量
    def calculate_exp_sum(self, this_hero_info):
        exp_sum = 0.0
        for i in range(1, this_hero_info["level"]):
            exp_sum += self.m_each_level_max_exp[i]
        exp_sum += this_hero_info["exp"]
        return exp_sum

    # Calculate the forward reward based on the distance between the agent and both defensive towers
    # 用智能体到双方防御塔的距离，计算前进奖励
    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        main_tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
        enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        hero_pos = (
            main_hero["actor_state"]["location"]["x"],
            main_hero["actor_state"]["location"]["z"],
        )
        forward_value = 0
        dist_hero2emy = math.dist(hero_pos, enemy_tower_pos)
        dist_main2emy = math.dist(main_tower_pos, enemy_tower_pos)
        if main_hero["actor_state"]["hp"] / main_hero["actor_state"]["max_hp"] > 0.99 and dist_hero2emy > dist_main2emy:
            forward_value = (dist_main2emy - dist_hero2emy) / dist_main2emy
        return forward_value

    # Calculate the reward item information for both sides using frame data
    # 用帧数据来计算两边的奖励子项信息
    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1

        for hero in frame_data["hero_states"]:
            if hero["player_id"] == self.main_hero_player_id:
                main_camp = hero["actor_state"]["camp"]
                self.main_hero_camp = main_camp
            else:
                enemy_camp = hero["actor_state"]["camp"]
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

    # Use the values obtained in each frame to calculate the corresponding reward value
    # 用每一帧得到的奖励子项信息来计算对应的奖励值
    def get_reward(self, frame_data, reward_dict):
        reward_dict.clear()
        reward_sum, weight_sum = 0.0, 0.0
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            if reward_name == "hp_point":
                if (
                    self.m_main_calc_frame_map[reward_name].last_frame_value == 0.0
                    and self.m_enemy_calc_frame_map[reward_name].last_frame_value == 0.0
                ):
                    reward_struct.cur_frame_value = 0
                    reward_struct.last_frame_value = 0
                elif self.m_main_calc_frame_map[reward_name].last_frame_value == 0.0:
                    reward_struct.cur_frame_value = 0 - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    reward_struct.last_frame_value = 0 - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                elif self.m_enemy_calc_frame_map[reward_name].last_frame_value == 0.0:
                    reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - 0
                    reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - 0
                else:
                    reward_struct.cur_frame_value = (
                        self.m_main_calc_frame_map[reward_name].cur_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    )
                    reward_struct.last_frame_value = (
                        self.m_main_calc_frame_map[reward_name].last_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                    )
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
            elif reward_name == "ep_rate":
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value
                if reward_struct.last_frame_value > 0:
                    reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
                else:
                    reward_struct.value = 0
            elif reward_name == "exp":
                main_hero = None
                for hero in frame_data["hero_states"]:
                    if hero["player_id"] == self.main_hero_player_id:
                        main_hero = hero
                if main_hero and main_hero["level"] >= 15:
                    reward_struct.value = 0
                else:
                    reward_struct.cur_frame_value = (
                        self.m_main_calc_frame_map[reward_name].cur_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    )
                    reward_struct.last_frame_value = (
                        self.m_main_calc_frame_map[reward_name].last_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                    )
                    reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
            elif reward_name == "forward":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name == "last_hit":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name == "attack_enemy_hero":  # **新增部分**
                reward_struct.value = self.m_cur_calc_frame_map[reward_name].cur_frame_value
            elif reward_name == "pick_health_pack":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name == "avoid_over_aggressive":
                reward_struct.value = self.m_cur_calc_frame_map[reward_name].cur_frame_value
            elif reward_name == "attack_enemy_tower":
                reward_struct.value = self.m_cur_calc_frame_map[reward_name].cur_frame_value
            else:
                # Calculate zero-sum reward
                # 计算零和奖励
                reward_struct.cur_frame_value = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                )
                reward_struct.last_frame_value = (
                    self.m_main_calc_frame_map[reward_name].last_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                )
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            weight_sum += reward_struct.weight
            reward_sum += reward_struct.value * reward_struct.weight
            reward_dict[reward_name] = reward_struct.value
        reward_dict["reward_sum"] = reward_sum

    # 需要在初始化时记录上一次的英雄生命值
    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1

        for hero in frame_data["hero_states"]:
            if hero["player_id"] == self.main_hero_player_id:
                main_camp = hero["actor_state"]["camp"]
                self.main_hero_camp = main_camp
                # 初始化或更新上一次的生命值
                if not hasattr(self, 'main_hero_hp_last'):
                    self.main_hero_hp_last = hero["actor_state"]["hp"]
                else:
                    self.main_hero_hp_last = self.main_hero_hp_last
            else:
                enemy_camp = hero["actor_state"]["camp"]
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)