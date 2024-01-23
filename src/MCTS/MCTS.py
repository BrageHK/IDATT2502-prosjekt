import numpy as np
import time
import torch
import gym
import gym_chess

from Node.Node import Node
from Node.Node_normalized import NodeNormalized
from Node.Node_NN import NodeNN
from Node.NodeType import NodeType
from Node.Node_threshold import NodeThreshold
from Node.Node_threshold_lightweight import NodeThresholdLightweight
from Node.Node_NN_chess import NodeNNChess


class MCTS:
    def __init__(
        self,
        env,
        num_iterations=69,
        NODE_TYPE=NodeType.NODE_NORMALIZED,
        model=None,
        turn_time=None,
    ):
        self.env = env
        self.num_iterations = num_iterations
        self.NODE_TYPE = NODE_TYPE
        self.model = model
        self.turn_time = turn_time

    def create_node(self, state):
        if self.NODE_TYPE == NodeType.NODE:
            return Node(self.env, state)
        elif self.NODE_TYPE == NodeType.NODE_NORMALIZED:
            return NodeNormalized(self.env, state)
        elif self.NODE_TYPE == NodeType.NODE_NN:
            return NodeNN(self.env, state)
        elif self.NODE_TYPE == NodeType.NODE_NN_CHESS:
            return NodeNNChess(state)
        elif self.NODE_TYPE == NodeType.NODE_THRESHOLD:
            return NodeThreshold(env=self.env, state=state, model=self.model)
        elif self.NODE_TYPE == NodeType.NODE_THRESHOLD_LIGHTWEIGHT:
            return NodeThresholdLightweight(env=self.env, state=state, model=self.model)
        else:
            raise Exception("Invalid node type")


    @torch.no_grad()
    def get_neural_network_predictions(self, state, env=None):
        tensor_state = torch.tensor(state, device=self.model.device, dtype=torch.float)
        tensor_state = tensor_state.permute(2, 0, 1).unsqueeze(0)
        policy, value = self.model.forward(tensor_state)

        policy = torch.softmax(policy, dim=1).squeeze(0).detach()

        legal_moves = self.env.legal_actions if env is None else env.legal_actions
        mask = torch.zeros_like(policy)
        mask[legal_moves] = 1

        policy *= mask

        sum_policy = torch.sum(policy)
        if sum_policy > 0:
            policy /= sum_policy
        else:
            policy = torch.zeros_like(policy)  # Avoid division by zero

        return policy, value.item()


    @torch.no_grad()
    def mcts_AlphaZero(self, root):
        # Select
        node = root.select()

        result, done = self.env.check_game_over(node.state, node.action_taken)
        result = self.env.get_opponent_value(result)

        if not done:
            # Predicts probabilities for each move and winner probability
            policy, result = self.get_neural_network_predictions(node.state)

            # Expand node
            node.expand(policy)

        # Backpropagate with simulation result
        node.backpropagate(result)

    @torch.no_grad()
    def mcts_AlphaZero_chess(self, root):
        # Select
        node = root.select()
        done = node.env.done
        result = node.env.reward  # TODO: does this work

        if not done:
            # Predicts probabilities for each move and winner probability
            policy, result = self.get_neural_network_predictions(
                node.env.get_observation(), node.env
            )

            # Expand node
            node.expand(policy)

        # Backpropagate with simulation result
        node.backpropagate(result)

    @torch.no_grad()
    def mcts(self, root):
        # Select
        node = root.select()

        result, done = self.env.check_game_over(node.state, node.action_taken)
        result = self.env.get_opponent_value(result)

        if not done:
            # Expand node
            node.expand()
            # Simulate root node
            result, _ = node.simulate()

        # Backpropagate with simulation result
        node.backpropagate(result)

    @torch.no_grad()
    def search(self, state, training=False):
        root = self.create_node(state)

        mcts_method = (
            self.mcts
            if self.NODE_TYPE
            in [
                NodeType.NODE,
                NodeType.NODE_NORMALIZED,
                NodeType.NODE_THRESHOLD,
                NodeType.NODE_THRESHOLD_LIGHTWEIGHT,
            ]
            else self.mcts_AlphaZero
            if self.NODE_TYPE == NodeType.NODE_NN
            else self.mcts_AlphaZero_chess
        )
        if mcts_method == self.mcts_AlphaZero_chess:
            self.env = state

        if mcts_method is None:
            raise Exception("Invalid node type")

        if self.turn_time is None:
            for _ in range(self.num_iterations):
                mcts_method(root)
        else:
            start = time.time()
            while time.time() - start < self.turn_time:
                mcts_method(root)

        visits = torch.zeros(self.env.action_space.n, device=self.model.device)
        for child in root.children:
            visits[child.action_taken] = child.visits

        # print("Probabilties: ", probabilties)
        best_action = torch.argmax(visits)
        if training:
            probabilties = visits / torch.sum(visits)
            return probabilties, best_action
        return best_action
