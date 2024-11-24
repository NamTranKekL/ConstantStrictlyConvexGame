import random
from itertools import combinations

def generate_convex_game(n, varsigma):
    # Initialize the game values dictionary
    game_values = {}

    # Step 1: Initialize singleton values
    for i in range(n):
        game_values[frozenset([i])] = varsigma + random.uniform(0, 1)

    # Step 2 and 3: Define values for larger sets and generate values for all subsets
    for size in range(2, n+1):
        for subset in combinations(range(n), size):
            subset = frozenset(subset)
            for i in subset:
                smaller_subset = frozenset(subset - {i})
                omega = random.uniform(0, 1)
                new_value = game_values[smaller_subset] + (len(smaller_subset) + 1) * varsigma + omega
                if subset in game_values:
                    game_values[subset] = max(game_values[subset], new_value)
                else:
                    game_values[subset] = new_value

    return game_values

# Example usage
n = 10  # Number of players
varsigma = 1  # Strictness parameter
convex_game = generate_convex_game(n, varsigma)
print(convex_game)  # Display the generated game values


