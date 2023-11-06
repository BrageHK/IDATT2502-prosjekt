import numpy as np
from Node.NodeType import NodeType
from Node.Node_single import NodeSingle
from Node.Node_double import NodeDouble
from Node.Node_NN import NodeNN

class MCTS():
    def __init__(self, num_simulations=1000, NODE_TYPE=NodeType.NODE_DOUBLE, model=None):   
        self.NODE_TYPE = NODE_TYPE     
        self.num_simulations = num_simulations
        self.model = model

    def create_node(self, env):
        if self.NODE_TYPE == NodeType.NODE_SINGLE:
            return NodeSingle(env=env)
        elif self.NODE_TYPE == NodeType.NODE_DOUBLE:
            return NodeDouble(env=env)
        elif self.NODE_TYPE == NodeType.NODE_NN:
            return NodeNN(env=env)
        
    def mcts_AlphaZero(self, root):
        # Select
        node = root.select()
        
        # Predicts probabilities for each move and winner probability
        policy, value = node.get_neural_network_predictions()
        
        # Expand node
        node.expand(policy)

        # Backpropagate with simulation result
        node.backpropagate(value)
    
    def mcts(self, root):
        # Select
        node = root.select()
        
        # Expand node
        node.expand()
        
        print("Hei")
        # Simulate root node
        result, _ = node.simulate()

        # Backpropagate with simulation result
        node.backpropagate(result)
    
    def get_action(self, env, training=False):
        env = env.deepcopy()
        # Invert board if player is -1
        #print("turn: ", env.turn)
        env.invert_board_for_second_player()
    
        root = self.create_node(env=env)
        
        if self.NODE_TYPE == NodeType.NODE_SINGLE or self.NODE_TYPE == NodeType.NODE_DOUBLE:
            for _ in range(self.num_simulations):
                self.mcts(root)
        elif self.NODE_TYPE == NodeType.NODE_NN:
            for _ in range(self.num_simulations):
                self.mcts_AlphaZero(root)
        else:
            raise Exception("Invalid node type")

            
        # If player was -1 invert board back
        env.reset_invert()
               
        visits = np.array([child.visits for child in root.children])
        print("Visits: ", visits)
        best_action = max(root.children, key=lambda child: child.visits, default=None).action
        
        worst_action = min(root.children, key=lambda child: child.visits, default=None).action

        if training:
            visits = np.zeros(env.COLUMN_COUNT)
            for child in root.children:
                visits[child.action] = child.visits
            probabilties = visits / np.sum(visits)
            return probabilties, best_action
        
        return best_action