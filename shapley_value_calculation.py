from itertools import chain, combinations
from math import factorial
from convex_game import generate_convex_game, compute_marginal_contributions
import numpy as np
def powerset(iterable):
    # "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2,) (1,3,) (2,3,) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def shapley_value(game_values, n, epoch=100):
    # Calculate Shapley values for each player
    shapley_values = {i: 0 for i in range(n)}
    player_set = frozenset(range(n))
    total_players = len(player_set)
    for n in range(epoch):
        for i in player_set:
            for S in powerset(player_set - {i}):
                S = frozenset(S)
                S_with_i = S | {i}
                # Calculate the marginal contribution of player i
                if S:
                    marginal_contribution = game_values[S_with_i] - game_values[S]
                else:
                    marginal_contribution = game_values[S_with_i]
                # Update the Shapley value of player i
                shapley_values[i] += (marginal_contribution *
                                    factorial(len(S)) *
                                    factorial(total_players - len(S) - 1) /
                                    factorial(total_players))
        # Normalize the Shapley values (as the factorial grows very fast)
        for i in player_set:
            shapley_values[i] /= total_players
    shapley_values_array = np.array(list(shapley_values.values()))
    return shapley_values_array

# Example usage with a game generated by the generate_convex_game function
n = 10
varsigma = 1
convex_game_values = generate_convex_game(n, varsigma)
shapley_values = shapley_value(convex_game_values, n)
ground_truth = compute_marginal_contributions(convex_game_values,n)
# print("Shapley Values:", shapley_values)
# print("Ground Truth Value:", ground_truth)