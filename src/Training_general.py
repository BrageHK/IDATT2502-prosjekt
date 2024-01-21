import torch.multiprocessing as mp
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import os
import random
import copy

from Connect_four_env import ConnectFour
from TicTacToe import TicTacToe
import gym
import gym_chess

from MCTS.MCTS import MCTS
from NeuralNet import AlphaPredictorNerualNet
from NeuralNetThreshold import NeuralNetThreshold
from NeuralNetThreshold_lightweight import NeuralNetThresholdLightweight
from NeuralNetChess import AlphaPredictorNerualNetChess
from Node.NodeType import NodeType
from tqdm import tqdm


def reward_tuple(reward):
    return (1, 0, 0) if reward == 1 else (0, 1, 0) if reward == 0 else (0, 0, 1)


@torch.no_grad()
def play_game_alpha_zero(env, mcts, match_id):
    print(f"Starting match {match_id}")

    memory = []
    player = 1
    done = False
    state = env.get_initial_state()

    while True:
        neutral_state = env.change_perspective(state, player)
        mcts_prob, action = mcts.search(neutral_state, training=True)

        memory.append((neutral_state, mcts_prob, player))

        mcts_prob = np.power(mcts_prob, 1 / 1.25)  # Temperature
        mcts_prob = mcts_prob / np.sum(mcts_prob)  # Normalize probabilities
        action = np.random.choice(
            env.action_space, p=mcts_prob
        )  # Choose action according to probability distribution

        state, reward, done = env.step(state, action=action, player=player)
        if done:
            return_memory = []
            for historical_state, historical_mcts_prob, historical_player in memory:
                historical_outcome = (
                    reward
                    if historical_player == player
                    else env.get_opponent_value(reward)
                )
                return_memory.append(
                    (
                        env.get_encoded_state(historical_state),
                        historical_mcts_prob,
                        historical_outcome,
                    )
                )
            return return_memory

        player = env.get_opponent(player)


@torch.no_grad()
def play_game_alpha_zero_chess(env, mcts, match_id):
    print(f"Starting chess match {match_id}")

    memory = []
    player = 1
    done = False
    while True:
        mcts_prob, action = mcts.search(copy.deepcopy(env), training=True)

        memory.append(
            (env.get_observation(), mcts_prob, player)
        )  # First element in tuple is the neutral state

        mcts_prob = np.power(mcts_prob, 1 / 1.25)  # Temperature
        mcts_prob = mcts_prob / np.sum(mcts_prob)  # Normalize probabilities
        action = np.random.choice(
            len(mcts_prob), p=mcts_prob
        )  # Choose action according to probability distribution

        state_info = env.step(action)
        reward = state_info[1]
        done = state_info[2]

        if done:
            return_memory = []
            for historical_state, historical_mcts_prob, historical_player in memory:
                historical_outcome = reward if historical_player == player else -reward
                return_memory.append(
                    (historical_state, historical_mcts_prob, historical_outcome)
                )
            return return_memory

        player = -player


@torch.no_grad()
def play_game_threshold(env, mcts, match_id):
    print(f"Starting match {match_id}")

    memory = []
    player = 1
    done = False
    state = env.get_initial_state()
    turn = 0

    while True:
        neutral_state = env.change_perspective(state, player)
        action = mcts.search(neutral_state)

        memory.append((neutral_state, player))

        state, reward, done = env.step(state, action=action, player=player)

        if done:
            return_memory = []
            for historical_state, historical_player in memory:
                historical_outcome = (
                    reward
                    if historical_player == player
                    else env.get_opponent_value(reward)
                )
                historical_outcome = reward_tuple(historical_outcome)
                return_memory.append(
                    (env.get_encoded_state(historical_state), historical_outcome)
                )
            return return_memory

        turn += 1
        player = env.get_opponent(player)


@torch.no_grad()
def play_game_threshold_lightweight(env, mcts, match_id):
    print(f"Starting match {match_id}")

    memory = []
    player = 1
    done = False
    state = env.get_initial_state()
    turn = 0

    while True:
        neutral_state = env.change_perspective(state, player)
        action = mcts.search(neutral_state)

        memory.append((neutral_state, player))

        state, reward, done = env.step(state, action=action, player=player)

        if done:
            return_memory = []
            for historical_state, historical_player in memory:
                historical_outcome = (
                    reward
                    if historical_player == player
                    else env.get_opponent_value(reward)
                )
                historical_outcome = reward_tuple(historical_outcome)
                historical_state = np.array(historical_state)
                return_memory.append((historical_state.flatten(), historical_outcome))
            return return_memory

        turn += 1
        player = env.get_opponent(player)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.policy_loss_history = []
        self.value_loss_history = []
        self.game_length_history = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.get_model_from_node()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.02)
        self.mcts = MCTS(
            env=config["env"],
            num_iterations=config["num_iterations"],
            NODE_TYPE=config["node_type"],
            model=self.model,
        )

        print(
            "Starting training using: ", "cuda" if torch.cuda.is_available() else "cpu"
        )
        print("Cores used for training: ", mp.cpu_count())

    def get_model_from_node(self):
        if config["node_type"] == NodeType.NODE_NN:
            return AlphaPredictorNerualNet(
                num_resBlocks=config["num_res_blocks"],
                env=config["env"],
                device=self.device,
            )
        elif config["node_type"] == NodeType.NODE_NN_CHESS:
            return AlphaPredictorNerualNetChess(
                num_resBlocks=config["num_res_blocks"], device=self.device
            )
        elif config["node_type"] == NodeType.NODE_THRESHOLD:
            return NeuralNetThreshold(env=config["env"], device=self.device)
        elif config["node_type"] == NodeType.NODE_THRESHOLD_LIGHTWEIGHT:
            return NeuralNetThresholdLightweight(env=config["env"], device=self.device)
        else:
            raise ValueError("Unknown node type")

    def is_threshold(self):
        return (
            self.config["node_type"] == NodeType.NODE_THRESHOLD
            or self.config["node_type"] == NodeType.NODE_THRESHOLD_LIGHTWEIGHT
        )

    def save_data(self, iteration=None):
        if iteration is not None:
            subpath = (
                lambda file: self.config["save_folder"]
                + "/"
                + self.config["node_type"].name
                + "/"
                + self.config["env"].__repr__()
                + "/"
                + file
                + "/"
                + file
                + f"_{iteration}.pt"
            )
        else:
            subpath = (
                lambda file: self.config["save_folder"]
                + "/"
                + self.config["node_type"].name
                + "/"
                + self.config["env"].__repr__()
                + "/"
                + file
                + "/"
                + file
                + ".pt"
            )
        with open(subpath("loss_values"), "wb") as file:
            if self.is_threshold():
                pickle.dump(self.value_loss_history, file)
            else:
                pickle.dump((self.policy_loss_history, self.value_loss_history), file)
        with open(subpath("game_lengths"), "wb") as file:
            pickle.dump(self.game_length_history, file)
        torch.save(self.model.state_dict(), subpath("model"))

    def load_data(self, iteration=None):
        def load_loss_history(filename):
            with open(filename, "rb") as file:
                return pickle.load(file)

        if iteration is not None:
            path = (
                lambda file: self.config["save_folder"]
                + "/"
                + self.config["node_type"].name
                + "/"
                + self.config["env"].__repr__()
                + "/"
                + file
                + "/"
                + file
                + f"_{iteration}.pt"
            )
        else:
            path = (
                lambda file: self.config["save_folder"]
                + "/"
                + self.config["node_type"].name
                + "/"
                + self.config["env"].__repr__()
                + "/"
                + file
                + "/"
                + file
                + ".pt"
            )

        err_print = lambda file: print("No ", file, " found from file: ", path(file))
        try:
            self.model.load_state_dict(torch.load(path("model")))
            print("loaded model")
        except FileNotFoundError:
            err_print("model")
        try:
            if self.is_threshold():
                self.value_loss_history = load_loss_history(path("loss_values"))
            else:
                self.policy_loss_history, self.value_loss_history = load_loss_history(
                    path("loss_values")
                )
            print("loaded loss values")
        except FileNotFoundError:
            err_print("loss_values")
        try:
            with open(path("game_lengths"), "rb") as file:
                self.game_length_history = pickle.load(file)
            print("loaded game lengths")
        except FileNotFoundError:
            err_print("game_lengths")

    def self_play(self):
        self.model.eval()
        match_id = 1
        args_list = []
        memory = []
        for _ in range(self.config["num_games"]):
            args_list.append((copy.deepcopy(self.config["env"]), self.mcts, match_id))
            match_id += 1

        if self.config["node_type"] == NodeType.NODE_NN_CHESS:
            game_func = play_game_alpha_zero_chess
        else:
            game_func = (
                play_game_threshold_lightweight
                if self.config["node_type"] == NodeType.NODE_THRESHOLD_LIGHTWEIGHT
                else play_game_threshold
                if self.config["node_type"] == NodeType.NODE_THRESHOLD
                else play_game_alpha_zero
            )

        print("Starting self play")
        mp.set_start_method("spawn", force=True)
        with mp.Pool(mp.cpu_count()) as pool:
            result_list = list(
                tqdm(pool.starmap(game_func, args_list), total=len(args_list))
            )

        for i in range(len(result_list)):
            memory += result_list[i]
            self.game_length_history.append(len(result_list[i]))

        return memory

    def loss_alpha_zero(
        self, policy_logits, value_logits, policy_target, value_target
    ):  #
        policy_loss = F.cross_entropy(policy_logits, policy_target)
        value_loss = F.mse_loss(value_logits, value_target)
        self.value_loss_history.append(value_loss.item())
        self.policy_loss_history.append(policy_loss.item())
        return policy_loss + value_loss

    def loss_threshold(self, value_logits, value_target):
        value_loss = F.mse_loss(value_logits, value_target)
        self.value_loss_history.append(value_loss.item())
        return value_loss

    def optimize(self, memory):
        self.model.train()  # Sets training mode
        for i in range(self.config["num_epochs"]):
            random.shuffle(memory)

            print("Starting epoch: ", i + 1)
            for batch_index in range(0, len(memory), self.config["batch_size"]):
                sample = memory[
                    batch_index : min(
                        len(memory) - 1, batch_index + self.config["batch_size"]
                    )
                ]

                if self.is_threshold():
                    states, value_targets = zip(*sample)
                else:
                    states, policy_targets, value_targets = zip(*sample)

                states = np.array(states).transpose(0, 3, 1, 2)

                if not self.is_threshold():
                    policy_targets = np.array(policy_targets)
                value_targets = np.array(
                    [np.array(item).reshape(-1, 1) for item in value_targets]
                )

                states = torch.tensor(
                    states, dtype=torch.float32, device=self.model.device
                )
                if not self.is_threshold():
                    policy_targets = torch.tensor(
                        policy_targets, dtype=torch.float32, device=self.model.device
                    )
                value_targets = torch.tensor(
                    value_targets, dtype=torch.float32, device=self.model.device
                )

                if self.is_threshold():
                    value_output = self.model(states)
                else:
                    policy_output, value_output = self.model(states)

                if not self.is_threshold():
                    loss = self.loss_alpha_zero(
                        policy_output,
                        value_output,
                        policy_targets,
                        value_targets.squeeze(-1),
                    )
                else:
                    loss = self.loss_threshold(value_output, value_targets.squeeze(-1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def create_folder(self, path):
        subfolders = ["model", "loss_values", "game_lengths"]
        subpath = (
            path
            + "/"
            + self.config["node_type"].name
            + "/"
            + self.config["env"].__repr__()
        )
        for subfolder in subfolders:
            if not os.path.exists(subpath + "/" + subfolder):
                os.makedirs(subpath + "/" + subfolder)

    def learn(self):
        self.create_folder(self.config["save_folder"])
        self.create_folder(self.config["load_folder"])

        if self.config["load_data"]:
            trainer.load_data(self.config["starting_iteration"])
        iteration = (
            self.config["starting_iteration"] + 1
            if self.config["starting_iteration"] is not None
            else 1
        )

        while True:
            print("Starting iteration: ", iteration)
            positions = trainer.self_play()

            # Optimize
            trainer.optimize(positions)

            # Save model and loss history
            trainer.save_data(iteration=iteration)
            trainer.save_data()

            iteration += 1


if __name__ == "__main__":
    # Here are the arguments for the current training session.
    # Set them to the correct value for your training.
    config = {
        "num_iterations": 200,  # Bare for testing, set til 600 etterpå
        "num_games": 100,  # Bare for testing, sett til 400 etterpå
        "num_epochs": 4,
        "batch_size": 128,
        "node_type": NodeType.NODE_NN_CHESS,
        "env": gym.make(
            "ChessAlphaZero-v0"
        ),  # ConnectFour(), TicTacToe() or gym.make("ChessAlphaZero-v0")
        "save_folder": "data",
        "load_folder": "data",
        "load_data": False,
        "num_res_blocks": 20,  # Use 9 for connect4 and 4 for tictactoe. Use more for chess. Not relevant for threshold
        "starting_iteration": None,  # Set this to None if you want the newest one
    }
    config["env"].reset()

    trainer = Trainer(config)

    trainer.learn()
