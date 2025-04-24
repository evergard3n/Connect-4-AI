import pyspiel
import numpy as np

game = pyspiel.load_game("connect_four")
state = game.new_initial_state()


class ConnectFourEvaluator(pyspiel.Evaluator):
    """Final working implementation of a Connect-4 evaluator"""
    
    def __init__(self, n_rollouts=10, rollout_depth=20, seed=None):
        # No super().__init__() needed - special OpenSpiel base class
        self.n_rollouts = n_rollouts
        self.rollout_depth = rollout_depth
        self.rng = np.random.RandomState(seed if seed else 42)
        
    def evaluate(self, state):
        """Main evaluation function required by the interface"""
        if state.is_terminal():
            return state.returns()
            
        player = state.current_player()
        total = [0.0, 0.0]
        
        # Perform multiple rollouts for more stable evaluation
        for _ in range(self.n_rollouts):
            result = self._single_rollout(state.clone(), player)
            total[0] += result[0]
            total[1] += result[1]
            
        return [total[0]/self.n_rollouts, total[1]/self.n_rollouts]
    
    def _single_rollout(self, state, player):
        """Perform one rollout with Connect-4 specific knowledge"""
        for _ in range(self.rollout_depth):
            if state.is_terminal():
                return state.returns()
                
            # Smart action selection during rollout
            action = self._select_rollout_action(state)
            state.apply_action(action)
            
        return self._heuristic_evaluation(state, player)
    
    def _select_rollout_action(self, state):
        """Choose actions strategically during rollout"""
        legal_actions = state.legal_actions()
        
        # 1. First check for immediate wins
        for action in legal_actions:
            child = state.clone()
            child.apply_action(action)
            if child.is_terminal():
                return action
                
        # 2. Then check for opponent threats to block
        opponent = 1 - state.current_player()
        for action in legal_actions:
            child = state.clone()
            child.apply_action(action)
            for opp_action in child.legal_actions():
                grandchild = child.clone()
                grandchild.apply_action(opp_action)
                if grandchild.is_terminal() and grandchild.returns()[opponent] == 1:
                    return action
                    
        # 3. Fallback to weighted random selection (prefer center)
        center_weights = [1, 2, 3, 4, 3, 2, 1]  # Column weights
        weights = [center_weights[a] for a in legal_actions]
        total = sum(weights)
        probs = [w/total for w in weights]
        return self.rng.choice(legal_actions, p=probs)
    
    def _heuristic_evaluation(self, state, player):
        """Position evaluation when rollout doesn't reach terminal state"""
        # Implement your Connect-4 specific heuristics here
        return [0.0, 0.0]  # Neutral evaluation as fallback

bot = pyspiel.MCTSBot(
    game,
    evaluator=pyspiel.RandomRolloutEvaluator(20, 62), # Shorter rollouts (endgame doesn't need long ones)
    uct_c=0.1,  # Lower exploration for endgame precision
    max_simulations=50000,  # Increased simulations for endgame accuracy
    max_memory_mb=500,  # Slightly increased but still constrained
    solve=True,  # Keep solving enabled for endgame
    seed=82,
    verbose=False
)

while not state.is_terminal():
    print("\n=== Turn of player", state.current_player(), "===")
    print(state)

    if state.current_player() == 0:
        action = bot.step(state)
        print("Bot chọn cột:", action)
    else:
        legal = state.legal_actions()
        print("Các cột hợp lệ:", legal)
        action = int(input("Chọn cột để đánh (0-6): "))
        if action not in legal:
            print("Cột không hợp lệ. Thử lại.")
            continue

    state.apply_action(action)

print("\nGame Over!")
print("Kết quả:", state.returns())