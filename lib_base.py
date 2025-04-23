import pyspiel

game = pyspiel.load_game("connect_four")
state = game.new_initial_state()

bot = pyspiel.MCTSBot(
    game,
    evaluator=pyspiel.RandomRolloutEvaluator(20, 42),  # Increased rollout depth
    uct_c=0.5,  # Lowered exploration constant for more exploitation
    max_simulations=10000,  # Increased number of simulations
    max_memory_mb=1000,  # Increased memory limit
    solve=True,
    seed=42,
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