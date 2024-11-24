import random
from itertools import chain, combinations, permutations
from math import factorial
import numpy as np

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
            # omega = random.uniform(0, 1)
            for i in subset:
                smaller_subset = frozenset(subset - {i})
                omega = random.uniform(0, 1)
                #omega = np.random.binomial(1,0.5)
                new_value = game_values[smaller_subset] + (len(smaller_subset) + 1) * varsigma + omega
                if subset in game_values:
                    game_values[subset] = max(game_values[subset], new_value)
                else:
                    game_values[subset] = new_value

    return game_values


def generate_convex_game_add(n, varsigma):
    # Initialize the game values dictionary
    game_values = {}
    c = np.random.uniform(0.1, 1, n)
    # Step 1: Initialize singleton values
    for i in range(n):
        game_values[frozenset([i])] = c[i]

    # Step 2 and 3: Define values for larger sets and generate values for all subsets
    for size in range(2, n+1):
        for subset in combinations(range(n), size):
            subset = frozenset(subset)
            # omega = random.uniform(0, 1)
            for i in subset:
                smaller_subset = frozenset(subset - {i})
                omega = random.uniform(0, 1)
                new_value = game_values[smaller_subset] + c[i]
                if subset in game_values:
                    game_values[subset] = max(game_values[subset], new_value)
                else:
                    game_values[subset] = new_value

    return game_values

# def generate_convex_game(n, varsigma):
#     # Initialize the game values dictionary
#     game_values = {}

#     # Generate all permutations of length 'length'
#     all_permutations = list(permutations(range(n), n))

#     # Define values for the specified length
#     for permutation in all_permutations:
#         subset = tuple(permutation)
#         for i in subset:
#             smaller_subset = tuple(x for x in subset if x != i)
#             omega = random.uniform(0, 1)
#             new_value = game_values.get(smaller_subset, varsigma) + (len(smaller_subset) + 1) * varsigma + omega
#             game_values[subset] = max(game_values.get(subset, 0), new_value)

#     return game_values, all_permutations


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2,) (1,3,) (2,3,) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def compute_marginal_contributions(game_values, n):
    # Initialize the marginal contributions dictionary
    marginal_contributions = {i: 0 for i in range(n)}

    # Generate all subsets of players
    all_subsets = list(powerset(range(n)))
    
    # Calculate the marginal contribution for each player
    for i in range(n):
        for subset in all_subsets:
            if i not in subset:
                # Convert the subset to a frozenset for indexing
                subset = frozenset(subset)
                # Calculate the value of the subset with and without the current player
                value_with_i = game_values.get(subset | {i}, 0)
                value_without_i = game_values.get(subset, 0)
                # The marginal contribution is the difference in value
                marginal_contributions[i] += value_with_i - value_without_i

    # Normalize the marginal contributions by the number of subsets considered
    for i in range(n):
        marginal_contributions[i] /= factorial(n)

    return marginal_contributions


n = 3  # For a quick example, use a smaller number of players
varsigma = 1  # Strictness parameter for the convex game
game_values= generate_convex_game(n, varsigma)
# print(game_values)
# print(compute_marginal_contributions(game_values, n))


def generate_all_permutations(n):
    if isinstance(n, int):
        lst = [i for i in range(n)]
    elif isinstance(n, list):
        lst = n
    all_perms = []
    for length in range(1, len(lst) + 1):
        perms = permutations(lst, length)
        all_perms.extend(perms)
    return all_perms


def sample_permutation(permutation_set):
    random_permutation_sample = random.choice(list(permutation_set))
    return random_permutation_sample

def create_marginal_vector(n,all_permutations):
    total_dict = {}
    n = len(all_permutations)
    for i in all_permutations:
        dict_sample = {}
        for j in range(n):
            dict_sample[j] = 0
        total_dict[tuple(i)] = dict_sample
    return total_dict


def cyclic_permutation(n):
    if n <= 0:
        return []
    
    # The initial order
    original_order = list(range(n))
    # A list to hold the cyclic permutations
    cyclic_permutations = [original_order]

    for i in range(1, n):
        # Generate the next permutation by moving the first element to the end
        new_order = cyclic_permutations[-1][1:] + [cyclic_permutations[-1][0]]
        cyclic_permutations.append(new_order)
    
    return cyclic_permutations








# all_permutations = generate_all_permutations(3)
# samples = sample_permutation(all_permutations)
# print(all_permutations)
# print(samples)
# n = np.random.choice(list(samples))
# print(n)
# samples_i = [x for x in all_permutations if n in list(x)]
# dict_sample = {}
# for element in samples_i:
#     dict_sample[element] = None 
# print(dict_sample)
# print(samples_i)
# total_dict = create_marginal_vector(3)
# print(total_dict)