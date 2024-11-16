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
        # 新增：用于跟踪小兵位置
        self.minion_positions_last = []
        self.minion_positions_current = []

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
        self.update_reward_weights(frame_data["frameNo"])

        frame_no = frame_data["frameNo"]
        self.game_time = frame_no
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
        
        # 新增：获取当前帧的小兵位置
        self.minion_positions_current = []
        for npc in npc_list:
            if npc["sub_type"] == "ACTOR_SUB_SOLDIER" and npc["camp"] == camp:
                location = npc["location"]
                self.minion_positions_current.append((location["x"], location["z"]))

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
            # 修改：集成 last_hit_optimized 的逻辑，传递 frame_data
                reward_struct.cur_frame_value = self.calculate_last_hit(frame_data, main_hero, enemy_hero)
            # **新增部分：与小兵保持协同**
            elif reward_name == "stay_with_minions":  # 新增
                reward_struct.cur_frame_value = self.calculate_hero_minion_proximity(main_hero)
            # **新增部分：避免不必要的伤害**
            elif reward_name == "avoid_unnecessary_damage":
                reward_struct.cur_frame_value = self.calculate_avoid_damage(main_hero, enemy_hero, enemy_tower)
            # **新增部分：与小兵协同攻击**
            elif reward_name == "coordinate_attack":  # 新增
                reward_struct.cur_frame_value = self.calculate_coordinate_attack(main_hero, enemy_tower)

            # **新增部分：优化补刀**（如果需要保留，请取消注释）
            # elif reward_name == "last_hit_optimized":  # 可选新增
            #     reward_struct.cur_frame_value = self.calculate_last_hit_reward(frame_data, main_hero)
            # **新增部分：攻击敌方英雄策略奖励**


                        # **策略1：限制过度激进的追击**
            # elif reward_name == "avoid_over_aggressive":
            #     reward_struct.cur_frame_value = 0.0
            #     # 检查英雄是否在敌方防御塔范围内
            #     if self.is_in_enemy_tower_range(main_hero, enemy_tower):
            #         # 在敌方防御塔范围内，给予惩罚
            #         reward_struct.cur_frame_value = -1.0
            #     else:
            #         # 不在敌方防御塔范围内，给予轻微奖励，鼓励安全行动
            #         reward_struct.cur_frame_value = 0.1
            #     # 检查英雄的生命值
            #     hero_hp_percentage = main_hero_hp / main_hero_max_hp
            #     if hero_hp_percentage < 0.3:
            #         # 生命值过低，且靠近敌方英雄，给予惩罚
            #         if self.is_near_enemy_hero(main_hero, enemy_hero):
            #             reward_struct.cur_frame_value += -0.5
            #     # 如果英雄生命值较低且远离己方防御塔，给予惩罚
            #     if self.is_far_from_own_tower(main_hero, main_tower) and hero_hp_percentage < 0.3:
            #         reward_struct.cur_frame_value += -0.5
            # **策略2：在无法检测小兵的情况下攻击防御塔**
            elif reward_name == "attack_enemy_tower":
                            reward_struct.cur_frame_value = self.calculate_attack_enemy_tower(main_hero, enemy_tower)
            # **新增部分：攻击敌方英雄策略奖励**
            # --------------------------------------------------------
            # elif reward_name == "attack_enemy_hero":
            #     # 计算我方与敌方英雄的生命值百分比差距
            #     hp_diff_percentage = self.calculate_hp_diff(main_hero, enemy_hero)
                
            #     if hp_diff_percentage >= 0.1:
            #         # 优势范围：主动进攻敌方英雄，奖励 +5
            #         reward_struct.cur_frame_value = 5.0
            #         print(f"优势范围：主动进攻敌方英雄，奖励+20，敌方英雄ID={enemy_hero['actor_state']['runtime_id']}")
            #     elif -0.1 <= hp_diff_percentage < 0.1:
            #         # 对峙范围：保持对峙，奖励 +2
            #         reward_struct.cur_frame_value = 2.0
            #         print(f"对峙范围：保持对峙，奖励+5，敌方英雄ID={enemy_hero['actor_state']['runtime_id']}")
            #     else:
            #         # 劣势范围：撤退或防御，奖励 +1
            #         reward_struct.cur_frame_value = 1.0
            #         print(f"劣势范围：撤退或防御，奖励+3，敌方英雄ID={enemy_hero['actor_state']['runtime_id']}")
            # --------------------------------------------------------
             # --------------------------------------------------------
            elif reward_name == "pick_health_pack":
                            reward_struct.cur_frame_value = self.calculate_pick_health_pack(frame_data, main_hero, main_spring, camp)
            # --------------------------------------------------------
            # **策略1和策略2逻辑融合到攻击敌方英雄奖励中**
            elif reward_name == "attack_enemy_hero":
                reward_struct.cur_frame_value = self.calculate_attack_enemy_hero(main_hero, enemy_hero, enemy_tower, main_tower)
            # Experience points
            # 经验值
            elif reward_name == "exp":
                reward_struct.cur_frame_value = self.calculate_exp_sum(main_hero)
            # Forward
            # 前进
            elif reward_name == "forward":
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower)
            # 技能使用奖励（新增）
            elif reward_name == "skill_usage":
                reward_struct.cur_frame_value = self.calculate_skill_usage(main_hero, enemy_hero)
            # 技能命中奖励（新增）
            elif reward_name == "skill_hit":
                reward_struct.cur_frame_value = self.calculate_skill_hit(main_hero, enemy_hero)
            # 技能连招奖励（新增）
            elif reward_name == "skill_combo":
                reward_struct.cur_frame_value = self.calculate_skill_combo(main_hero, enemy_hero)
            
        self.main_hero_hp_last = main_hero_hp

    def calculate_pick_health_pack(self, frame_data, main_hero, main_spring, camp):
        reward = 0.0
        hero_hp = main_hero["actor_state"]["hp"]
        hero_max_hp = main_hero["actor_state"]["max_hp"]
        hero_hp_percentage = hero_hp / hero_max_hp

        if hero_hp_percentage < 0.6:
            # 生命值低于60%，考虑拾取血包
            is_safe = self.evaluate_battlefield_safety(frame_data, camp, main_hero)
            if is_safe:
                nearest_health_pack = self.find_nearest_health_pack(frame_data, main_hero)
                if nearest_health_pack:
                    if self.has_picked_health_pack(frame_data, main_hero, nearest_health_pack):
                        # 成功拾取血包，给予奖励
                        reward = 1.0
                    else:
                        # 尚未拾取血包，引导前往血包位置
                        reward = 0.5
                else:
                    # 没有可用的血包
                    reward = 0.0
            else:
                # 战场不安全，建议返回防御塔
                reward = -0.5

            if hero_hp_percentage < 0.3:
                # 生命值低于30%，优先回到己方泉水
                if main_spring:
                    own_spring_pos = (
                        main_spring["location"]["x"],
                        main_spring["location"]["z"]
                    )
                    hero_pos = (
                        main_hero["actor_state"]["location"]["x"],
                        main_hero["actor_state"]["location"]["z"]
                    )
                    distance_to_spring = self.calculate_distance(hero_pos, own_spring_pos)
                    if distance_to_spring < 500:
                        reward += 2.0  # 增加奖励值
                        print("英雄靠近己方泉水，给予额外奖励+2.0")
                    else:
                        # 引导英雄朝己方泉水移动，给予引导奖励
                        reward += 1.0  # 引导前往泉水
                        print("生命值低于30%，引导前往己方泉水")
        return reward
    
        # 攻击敌方英雄策略奖励
    def calculate_attack_enemy_hero(self, main_hero, enemy_hero, enemy_tower, main_tower):
        hp_diff_percentage = self.calculate_hp_diff(main_hero, enemy_hero)
        reward = 0.0

        if self.is_in_enemy_tower_range(main_hero, enemy_tower):
            # 在敌方防御塔范围内，不鼓励攻击敌方英雄，给予惩罚
            reward += -1.0
        else:
            if hp_diff_percentage >= 0.1:
                # 优势范围：主动进攻敌方英雄，奖励 +5
                reward += 5.0
            elif -0.1 <= hp_diff_percentage < 0.1:
                # 对峙范围：保持对峙，奖励 +2
                reward += 2.0
            else:
                # 劣势范围：撤退或防御，奖励 +1
                reward += 1.0

        # 检查英雄的生命值
        hero_hp_percentage = main_hero["actor_state"]["hp"] / main_hero["actor_state"]["max_hp"]
        if hero_hp_percentage < 0.3:
            if self.is_near_enemy_hero(main_hero, enemy_hero):
                reward += -0.5
            if self.is_far_from_own_tower(main_hero, main_tower):
                reward += -0.5
        return reward
    
    # 攻击敌方防御塔
    def calculate_attack_enemy_tower(self, main_hero, enemy_tower):
        reward = 0.0
        if self.is_in_enemy_tower_range(main_hero, enemy_tower):
            # 检查英雄的生命值变化
            hp_decrease = self.main_hero_hp_last - main_hero["actor_state"]["hp"]
            if hp_decrease > 400:
                # 生命值正在减少，可能被防御塔攻击，给予惩罚
                reward = -1.0
            else:
                # 生命值未减少，可能安全攻击防御塔，给予奖励
                reward = 1.0
        return reward
    
    # 修改后的 last_hit 计算方法
    def calculate_last_hit(self, frame_data, main_hero, enemy_hero):
        """
        优化后的补刀奖励计算，考虑小兵的可用性和英雄的位置
        """
        reward = 0.0
        frame_action = frame_data.get("frame_action", {})
        if "dead_action" in frame_action:
            dead_actions = frame_action["dead_action"]
            for dead_action in dead_actions:
                if (
                    dead_action["killer"]["runtime_id"] == main_hero["actor_state"]["runtime_id"]
                    and dead_action["death"]["sub_type"] == "ACTOR_SUB_SOLDIER"
                ):
                    # 奖励成功补刀
                    reward += 1.0
                # elif (
                #     dead_action["killer"]["runtime_id"] == enemy_hero["actor_state"]["runtime_id"]
                #     and dead_action["death"]["sub_type"] == "ACTOR_SUB_SOLDIER"
                # ):
                #     # 惩罚对手补刀
                #     reward -= 1.0

        # 优化：如果当前没有小兵在附近，减少或取消补刀奖励
        if not self.minion_positions_current:
            # 奖励降低，因为没有小兵支援
            reward *= 0.5  # 调整比例，根据需要
        return reward
    
    # 新增：计算英雄与小兵的接近度
    def calculate_hero_minion_proximity(self, main_hero):
        hero_pos = (main_hero["actor_state"]["location"]["x"], main_hero["actor_state"]["location"]["z"])
        if not self.minion_positions_current:
            return -1.0  # 如果没有小兵在附近，给予惩罚
        distances = [self.calculate_distance(hero_pos, minion_pos) for minion_pos in self.minion_positions_current]
        min_distance = min(distances)
        proximity_threshold = 5000  # 根据需要调整
        if min_distance <= proximity_threshold:
            return 1.0  # 奖励保持与小兵的接近
        else:
            return -1.0  # 惩罚过于远离小兵

    # 新增：避免不必要的伤害
    def calculate_avoid_damage(self, main_hero, enemy_hero, enemy_tower):
        hp_lost = self.main_hero_hp_last - main_hero["actor_state"]["hp"]
        reward = 0.0
        if hp_lost > 0:
            # 假设任何生命值损失都是不必要的伤害，进行负向奖励
            reward -= hp_lost / main_hero["actor_state"]["max_hp"]
        # 额外负向奖励如果在敌方防御塔范围内
        if self.is_in_enemy_tower_range(main_hero, enemy_tower):
            reward -= 1.0
        return reward

    # 新增：与小兵协同攻击
    def calculate_coordinate_attack(self, main_hero, enemy_tower):
        hero_pos = (main_hero["actor_state"]["location"]["x"], main_hero["actor_state"]["location"]["z"])
        # 检查是否有小兵在敌方防御塔攻击范围内
        enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        minion_in_range = any(
            self.calculate_distance(minion_pos, enemy_tower_pos) <= enemy_tower["attack_range"]
            for minion_pos in self.minion_positions_current
        )
        # 如果有小兵在范围内且英雄也在范围内，给予奖励
        if minion_in_range and self.is_in_enemy_tower_range(main_hero, enemy_tower):
            return 2.0
        else:
            return 0.0  # 如果英雄单独攻击，给予惩罚
    
    # 技能使用奖励
    def calculate_skill_usage(self, main_hero, enemy_hero):
        reward = 0.0
        skill_states = main_hero["skill_state"]["slot_states"]

        # 获取技能槽对应的技能
        skill_mapping = {
            "SLOT_SKILL_0": "normal_attack",
            "SLOT_SKILL_1": "skill_one",
            "SLOT_SKILL_2": "skill_two",
            "SLOT_SKILL_3": "skill_three",
        }

        # 技能优先级
        skill_priority = ["skill_three", "skill_one", "skill_two"]

        # 判断敌方英雄是否在三技能范围内
        enemy_in_skill3_range = False
        if self.is_enemy_in_skill_range(main_hero, enemy_hero, "SLOT_SKILL_3"):
            enemy_in_skill3_range = True

        # 技能可用性和命中情况
        for skill in skill_states:
            slot_type = skill["slot_type"]
            skill_name = skill_mapping.get(slot_type, None)
            if skill_name and skill_name in skill_priority:
                if skill["usedTimes"] > 0:
                    # 检查三技能是否仅对敌方英雄使用
                    if skill_name == "skill_three":
                        # 检查是否命中敌方英雄
                        if skill["hitHeroTimes"] > 0 and enemy_in_skill3_range:
                            # 奖励使用技能
                            reward += 1.0
                            # 奖励技能命中敌方英雄
                            reward += 2.0
                            # 额外奖励
                            reward += 2.0
                        else:
                            # 三技能未命中敌方英雄，给予惩罚
                            reward -= 2.0
                    else:
                        # 对于其他技能，正常奖励
                        reward += 1.0
                        if skill["hitHeroTimes"] > 0:
                            reward += 2.0
                else:
                    # 如果三技能可命中但未使用，给予惩罚
                    if skill_name == "skill_three" and enemy_in_skill3_range:
                        reward -= 1.0
        return reward

    # 技能命中奖励
    def calculate_skill_hit(self, main_hero, enemy_hero):
        reward = 0.0
        skill_states = main_hero["skill_state"]["slot_states"]

        for skill in skill_states:
            if skill["hitHeroTimes"] > 0:
                if skill["slot_type"] == "SLOT_SKILL_3":
                    # 三技能命中敌方英雄，奖励较高
                    if self.is_enemy_in_skill_range(main_hero, enemy_hero, "SLOT_SKILL_3"):
                        reward += 3.0
                elif skill["slot_type"] == "SLOT_SKILL_1":
                    # 一技能命中敌方英雄
                    reward += 2.0
                elif skill["slot_type"] == "SLOT_SKILL_2":
                    # 二技能命中敌方英雄
                    reward += 1.0
        return reward

    # 技能连招奖励
    def calculate_skill_combo(self, main_hero, enemy_hero):
        reward = 0.0
        skill_states = main_hero["skill_state"]["slot_states"]

        # 检查技能使用次数
        skill_usage = {
            "SLOT_SKILL_1": 0,  # 一技能
            "SLOT_SKILL_2": 0,  # 二技能
            "SLOT_SKILL_3": 0,  # 三技能
        }

        for skill in skill_states:
            if skill["slot_type"] in skill_usage:
                skill_usage[skill["slot_type"]] = skill["usedTimes"]

        # 检查连招是否成功
        if skill_usage["SLOT_SKILL_3"] > 0 and skill_usage["SLOT_SKILL_1"] > 0:
            # 检查三技能是否在敌方英雄可命中范围内
            if self.is_enemy_in_skill_range(main_hero, enemy_hero, "SLOT_SKILL_3"):
                # 成功连招，奖励较高
                reward += 3.5
        return reward

    # 判断敌方英雄是否在指定技能的攻击范围内
    def is_enemy_in_skill_range(self, main_hero, enemy_hero, skill_slot):
        """
        判断敌方英雄是否在指定技能的攻击范围内
        """
        skill_ranges = {
            "SLOT_SKILL_3": 8000,  # 三技能范围
            "SLOT_SKILL_1": 8000,  # 一技能范围
            "SLOT_SKILL_2": 8000,  # 二技能范围
        }
        if skill_slot not in skill_ranges:
            return False

        skill_range = skill_ranges[skill_slot]
        hero_pos = (main_hero["actor_state"]["location"]["x"], main_hero["actor_state"]["location"]["z"])
        enemy_pos = (enemy_hero["actor_state"]["location"]["x"], enemy_hero["actor_state"]["location"]["z"])
        distance = self.calculate_distance(hero_pos, enemy_pos)
        return distance <= skill_range

    def calculate_distance(self, pos1, pos2):
        return math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
    
    def is_in_enemy_tower_range(self, main_hero, enemy_tower):
        """
        判断英雄是否在敌方防御塔的攻击范围内
        """
        hero_pos = (main_hero["actor_state"]["location"]["x"], main_hero["actor_state"]["location"]["z"])
        tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        distance = self.calculate_distance(hero_pos, tower_pos)
        tower_attack_range = enemy_tower["attack_range"]  # 使用实际防御塔攻击范围
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
        safe_distance = 8800  # 防御塔的攻击范围:8800
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

        pick_up_distance_threshold = 1000  # 根据实际情况调整，表示拾取血包的距离阈值

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
    
    # 需要在初始化时记录上一次的英雄生命值
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
            elif reward_name in ["attack_enemy_hero", "pick_health_pack", "stay_with_minions",
                                 "avoid_unnecessary_damage", "coordinate_attack", "attack_enemy_tower",
                                 "skill_usage", "skill_hit", "skill_combo"]:
                # 直接使用当前帧的计算值
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
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

    def update_reward_weights(self, frame_no):
        # Define time thresholds for changing focus
        early_game = 1000  # Adjust based on game duration
        mid_game = 2000

        # Clear minions focus in early game
        if frame_no <= early_game:
            self.m_cur_calc_frame_map["last_hit"].weight = 1.0
            self.m_cur_calc_frame_map["attack_enemy_hero"].weight = 0.1
            self.m_cur_calc_frame_map["attack_enemy_tower"].weight = 0.5
        # Shift focus to attacking hero and towers in mid game
        elif early_game < frame_no <= mid_game:
            self.m_cur_calc_frame_map["last_hit"].weight = 0.5
            self.m_cur_calc_frame_map["attack_enemy_hero"].weight = 0.5
            self.m_cur_calc_frame_map["attack_enemy_tower"].weight = 1.0
        # Late game focus on pushing towers and crystals
        else:
            self.m_cur_calc_frame_map["last_hit"].weight = 0.1
            self.m_cur_calc_frame_map["attack_enemy_hero"].weight = 1.0
            self.m_cur_calc_frame_map["attack_enemy_tower"].weight = 2.0