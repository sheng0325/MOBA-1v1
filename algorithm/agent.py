#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import json

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
from ppo.model.model import Model
from ppo.feature.definition import *
import numpy as np
from kaiwu_agent.agent.base_agent import (
    BaseAgent,
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
)

from ppo.config import Config
from kaiwu_agent.utils.common_func import attached
from ppo.feature.reward_manager import GameRewardManager

# 打印帧信息
def save_state_to_file(state_dict, file_name="state_dump.json"):
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接完整的文件路径
    file_path = os.path.join(current_dir, file_name)

    try:
        # 将状态信息保存到文件
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=4, ensure_ascii=False)
        print(f"状态信息已保存到: {file_path}")
    except Exception as e:
        print(f"保存状态信息时出错: {e}")

@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.cur_model_name = ""
        self.device = device
        # Create Model and convert the model to achannel-last memory format to achieve better performance.
        # 创建模型, 将模型转换为通道后内存格式，以获得更好的性能。
        self.model = Model().to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)

        # config info
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.cut_points = [value[0] for value in Config.data_shapes]

        # env info
        self.hero_camp = 0
        self.player_id = 0
        self.game_id = None

        # learning info
        self.train_step = 0
        initial_lr = Config.INIT_LEARNING_RATE_START
        parameters = self.model.parameters()
        self.optimizer = torch.optim.Adam(params=parameters, lr=initial_lr, betas=(0.9, 0.999), eps=1e-8)
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]

        # tools
        self.reward_manager = None
        self.logger = logger
        self.monitor = monitor

        super().__init__(agent_type, device, logger, monitor)

    def _model_inference(self, list_obs_data):
        # 使用网络进行推理
        # Using the network for inference
        feature = [obs_data.feature for obs_data in list_obs_data]
        legal_action = [obs_data.legal_action for obs_data in list_obs_data]
        lstm_cell = [obs_data.lstm_cell for obs_data in list_obs_data]
        lstm_hidden = [obs_data.lstm_hidden for obs_data in list_obs_data]

        input_list = [np.array(feature), np.array(lstm_cell), np.array(lstm_hidden)]
        torch_inputs = [torch.from_numpy(nparr).to(torch.float32) for nparr in input_list]
        for i, data in enumerate(torch_inputs):
            data = data.reshape(-1)
            torch_inputs[i] = data.float()

        feature, lstm_cell, lstm_hidden = torch_inputs
        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = lstm_hidden.reshape(-1, self.lstm_unit_size)
        lstm_cell_state = lstm_cell.reshape(-1, self.lstm_unit_size)

        format_inputs = [feature_vec, lstm_hidden_state, lstm_cell_state]

        self.model.set_eval_mode()
        with torch.no_grad():
            output_list = self.model(format_inputs, inference=True)

        np_output = []
        for output in output_list:
            np_output.append(output.numpy())

        logits, value, _lstm_cell, _lstm_hidden = np_output[:4]

        _lstm_cell = _lstm_cell.squeeze(axis=0)
        _lstm_hidden = _lstm_hidden.squeeze(axis=0)

        list_act_data = list()
        for i in range(len(legal_action)):
            prob, action, d_action = self._sample_masked_action(logits[i], legal_action[i])
            list_act_data.append(
                ActData(
                    action=action,
                    d_action=d_action,
                    prob=prob,
                    value=value,
                    lstm_cell=_lstm_cell[i],
                    lstm_hidden=_lstm_hidden[i],
                )
            )
        return list_act_data

    @predict_wrapper
    def predict(self, list_obs_data):
        return self._model_inference(list_obs_data)

    @exploit_wrapper
    def exploit(self, state_dict):
        # Evaluation task will not call agent.reset in the workflow. Users can use the game_id to determine whether a new environment
        # 评估任务不会在workflow中重置agent，用户可以通过game_id判断是否是新的对局，并根据新对局对agent进行重置
        game_id = state_dict["game_id"]
        if self.game_id != game_id:
            player_id = state_dict["player_id"]
            camp = state_dict["player_camp"]
            self.reset(camp, player_id)
            self.game_id = game_id

        # exploit is automatically called when submitting an evaluation task.
        # The parameter is the state_dict returned by env, and it returns the action used by env.step.
        # exploit在提交评估任务时自动调用，参数为env返回的state_dict, 返回env.step使用的action
        obs_data = self.observation_process(state_dict)
        # Call _model_inference for model inference, executing local model inference
        # 模型推理调用_model_inference, 执行本地模型推理
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        return self.action_process(state_dict, act_data, False)

    def train_predict(self, state_dict):
        obs_data = self.observation_process(state_dict)
        # Call agent.predict for distributed model inference
        # 调用agent.predict，执行分布式模型推理
        act_data = self.predict([obs_data])[0]
        self.update_status(obs_data, act_data)
        return self.action_process(state_dict, act_data, True)

    def eval_predict(self, state_dict):
        obs_data = self.observation_process(state_dict)
        # Call agent.predict for distributed model inference
        # 调用agent.predict，执行分布式模型推理
        act_data = self.predict([obs_data])[0]
        self.update_status(obs_data, act_data)

        # 新增策略逻辑
        player_id = state_dict["player_id"]
        player_hero, enemy_heroes, minions = parse_frame_state(state_dict, player_id)

        # 打印所有小兵数据
        print_all_minions(minions)

        # 决策行动
        target_minion = decide_action(player_hero, enemy_heroes, minions)
        if target_minion:
            # 构建攻击指令
            action = self.build_attack_action(target_minion)
        else:
            # 使用原有的行动
            action = self.action_process(state_dict, act_data, False)

        return action
    
    def build_attack_action(self, target_minion):
        # 根据具体的环境和动作空间，构建攻击指令
        # 这里需要根据您的动作定义来构建
        # 例如，假设动作是一个数组，包含动作类型和目标ID

        minion_id = target_minion["actor_state"]["runtime_id"]
        # 构建攻击小兵的动作，例如 [动作类型, 目标ID]
        attack_action = [1, minion_id]  # 具体取决于动作空间的定义

        return attack_action
    
    def action_process(self, state_dict, act_data, is_stochastic):
        if is_stochastic:
            # Use stochastic sampling action
            # 采用随机采样动作 action
            return act_data.action
        else:
            # Use the action with the highest probability
            # 采用最大概率动作 d_action
            return act_data.d_action

    def observation_process(self, state_dict):
        feature_vec, legal_action = (
            state_dict["observation"],
            state_dict["legal_action"],
        )

        # 打印
        # 打印并保存帧状态信息
        print("保存帧状态信息到文件...")
        save_state_to_file(state_dict)  # 调用保存函数
        
        return ObsData(
            feature=feature_vec, legal_action=legal_action, lstm_cell=self.lstm_cell, lstm_hidden=self.lstm_hidden
        )
    
    def parse_frame_state(state_dict, player_id):
        frame_state = state_dict["frame_state"]
        hero_states = frame_state["hero_states"]
        npc_states = frame_state["npc_states"]
        player_hero = None
        enemy_heroes = []
        minions = []

        # 提取英雄信息
        for hero in hero_states:
            if hero["player_id"] == player_id:
                player_hero = hero
            else:
                enemy_heroes.append(hero)

        # 提取小兵信息
        for npc in npc_states:
            if npc["actor_state"]["sub_type"] == "ACTOR_SUB_SOLDIER":  # 小兵类型
                minions.append(npc)

        if not player_hero:
            raise ValueError("未找到玩家英雄状态")

        return player_hero, enemy_heroes, minions
    
    def decide_action(player_hero, enemy_heroes, minions):
        # 检查敌方英雄是否在攻击范围内
        if is_enemy_hero_in_range(player_hero, enemy_heroes):
            print("敌方英雄在攻击范围内，停止补兵，保持安全")
            return None  # 或者返回一个保守的行动，例如后退

        # 找到生命值百分比最低的小兵
        target_minion = get_lowest_hp_minion(minions)
        if target_minion:
            print("攻击目标小兵的全部数据如下：")
            print(json.dumps(target_minion, indent=4, ensure_ascii=False))
            print(f"选择攻击小兵: ID={target_minion['actor_state']['runtime_id']}，剩余HP百分比={target_minion['actor_state']['hp'] / target_minion['actor_state']['max_hp']:.2f}")
            return target_minion

        print("没有可以补刀的小兵")
        return None
    
    def is_enemy_hero_in_range(player_hero, enemy_heroes, safe_distance=10000):
        hero_location = player_hero["actor_state"]["location"]

        for enemy_hero in enemy_heroes:
            enemy_location = enemy_hero["actor_state"]["location"]
            distance = ((hero_location["x"] - enemy_location["x"]) ** 2 +
                        (hero_location["z"] - enemy_location["z"]) ** 2) ** 0.5

            if distance <= safe_distance:
                return True

        return False

    def get_lowest_hp_minion(minions):
        lowest_hp_minion = None
        lowest_hp_percentage = float('inf')

        for minion in minions:
            minion_state = minion["actor_state"]
            if minion_state["hp"] > 0:
                hp_percentage = minion_state["hp"] / minion_state["max_hp"]
                if hp_percentage < lowest_hp_percentage:
                    lowest_hp_percentage = hp_percentage
                    lowest_hp_minion = minion

        return lowest_hp_minion
    
    def print_all_minions(minions):
        print("当前帧所有小兵的数据如下：")
        for minion in minions:
            print(json.dumps(minion, indent=4, ensure_ascii=False))

    @learn_wrapper
    def learn(self, list_sample_data):
        list_npdata = [sample_data.npdata for sample_data in list_sample_data]
        _input_datas = np.stack(list_npdata, axis=0)
        _input_datas = torch.from_numpy(_input_datas).to(self.device)
        results = {}

        data_list = list(_input_datas.split(self.cut_points, dim=1))
        for i, data in enumerate(data_list):
            data = data.reshape(-1)
            data_list[i] = data.float()

        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        feature, legal_action = seri_vec.split(
            [
                np.prod(self.seri_vec_split_shape[0]),
                np.prod(self.seri_vec_split_shape[1]),
            ],
            dim=1,
        )
        init_lstm_cell = data_list[-2]
        init_lstm_hidden = data_list[-1]

        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = init_lstm_hidden.reshape(-1, self.lstm_unit_size)
        lstm_cell_state = init_lstm_cell.reshape(-1, self.lstm_unit_size)

        format_inputs = [feature_vec, lstm_hidden_state, lstm_cell_state]

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        rst_list = self.model(format_inputs)
        total_loss, info_list = self.model.compute_loss(data_list, rst_list)
        results["total_loss"] = total_loss.item()

        total_loss.backward()

        # grad clip
        if Config.USE_GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)

        self.optimizer.step()
        self.train_step += 1

        _info_list = []
        for info in info_list:
            if isinstance(info, list):
                _info = [i.item() for i in info]
            else:
                _info = info.item()
            _info_list.append(_info)
        if self.monitor:
            _, (value_loss, policy_loss, entropy_loss) = _info_list
            results["value_loss"] = round(value_loss, 2)
            results["policy_loss"] = round(policy_loss, 2)
            results["entropy_loss"] = round(entropy_loss, 2)
            self.monitor.put_data({os.getpid(): results})

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files, and it is important to ensure that
        #  each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files, and it is important to ensure that
        # each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if self.cur_model_name == model_file_path:
            self.logger.info(f"current model is {model_file_path}, so skip load model")
        else:
            self.model.load_state_dict(
                torch.load(
                    model_file_path,
                    map_location=self.device,
                )
            )
            self.cur_model_name = model_file_path
            self.logger.info(f"load model {model_file_path} successfully")

    def reset(self, hero_camp, player_id):
        self.hero_camp = hero_camp
        self.player_id = player_id
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.reward_manager = GameRewardManager(player_id)

    def update_status(self, obs_data, act_data):
        self.obs_data = obs_data
        self.act_data = act_data
        self.lstm_cell = act_data.lstm_cell
        self.lstm_hidden = act_data.lstm_hidden

    # get final executable actions
    def _sample_masked_action(self, logits, legal_action):
        """
        Sample actions from predicted logits and legal actions
        return: probability, stochastic and deterministic actions with additional []
        """
        """
        从预测的logits和合法动作中采样动作
        返回：以列表形式概率、随机和确定性动作
        """

        prob_list = []
        action_list = []
        d_action_list = []
        label_split_size = [sum(self.label_size_list[: index + 1]) for index in range(len(self.label_size_list))]
        legal_actions = np.split(legal_action, label_split_size[:-1])
        logits_split = np.split(logits, label_split_size[:-1])
        for index in range(0, len(self.label_size_list) - 1):
            probs = self._legal_soft_max(logits_split[index], legal_actions[index])
            prob_list += list(probs)
            sample_action = self._legal_sample(probs, use_max=False)
            action_list.append(sample_action)
            d_action = self._legal_sample(probs, use_max=True)
            d_action_list.append(d_action)

        # deals with the last prediction, target
        # 处理最后的预测，目标
        index = len(self.label_size_list) - 1
        target_legal_action_o = np.reshape(
            legal_actions[index],  # [12, 8]
            [
                self.legal_action_size[0],
                self.legal_action_size[-1] // self.legal_action_size[0],
            ],
        )
        one_hot_actions = np.eye(self.label_size_list[0])[action_list[0]]  # [12]
        one_hot_actions = np.reshape(one_hot_actions, [self.label_size_list[0], 1])  # [12, 1]
        target_legal_action = np.sum(target_legal_action_o * one_hot_actions, axis=0)

        legal_actions[index] = target_legal_action  # [12]
        probs = self._legal_soft_max(logits_split[-1], target_legal_action)
        prob_list += list(probs)
        sample_action = self._legal_sample(probs, use_max=False)
        action_list.append(sample_action)

        # target_legal_action = tf.gather(target_legal_action, action_idx, axis=1)
        one_hot_actions = np.eye(self.label_size_list[0])[d_action_list[0]]
        one_hot_actions = np.reshape(one_hot_actions, [self.label_size_list[0], 1])
        target_legal_action_d = np.sum(target_legal_action_o * one_hot_actions, axis=0)

        # legal_actions[index] = target_legal_action
        probs = self._legal_soft_max(logits_split[-1], target_legal_action_d)

        # prob_list.append(probs)
        d_action = self._legal_sample(probs, use_max=True)
        d_action_list.append(d_action)

        return [prob_list], action_list, d_action_list

    def _legal_soft_max(self, input_hidden, legal_action):
        _lsm_const_w, _lsm_const_e = 1e20, 1e-5
        _lsm_const_e = 0.00001

        tmp = input_hidden - _lsm_const_w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        # Not necessary max clip 1
        tmp = np.clip(tmp - tmp_max, -_lsm_const_w, 1)
        # tmp = tf.exp(tmp - tmp_max)* legal_action + _lsm_const_e
        tmp = (np.exp(tmp) + _lsm_const_e) * legal_action
        # tmp_sum = tf.reduce_sum(tmp, axis=1, keepdims=True)
        probs = tmp / np.sum(tmp, keepdims=True)
        return probs

    def _legal_sample(self, probs, legal_action=None, use_max=False):
        # Sample with probability, input probs should be 1D array
        # 根据概率采样，输入的probs应该是一维数组
        if use_max:
            return np.argmax(probs)

        return np.argmax(np.random.multinomial(1, probs, size=1))
