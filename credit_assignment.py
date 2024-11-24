import numpy as np
from convex_game import generate_convex_game, generate_convex_game_add, sample_permutation, create_marginal_vector, generate_all_permutations, cyclic_permutation, compute_marginal_contributions
from shapley_value_calculation import shapley_value
import math
import itertools
import copy
import time
import pandas as pd
from tqdm import tqdm

# Define the scoring function (Placeholder)


def r_s(coalition, i=None):
    # # reward value for coalitions
    if isinstance(coalition, int):
        coalition = [coalition]
    return S_CONVEX_GAME[frozenset(coalition)]


# Save the dictionary for coalitions corresponding to cyclic permutations only. For quicker search
def r_cyclic_s(coalition, i=None):
    # # reward value for coalitions
    return S_CONVEX_GAME_CYCLIC[frozenset(coalition)]

# Compute phi_hat^j_i(k) for a given permutation and epoch
# i stands for the index of element we want to calculate
# k stands for the epoch number
# permutation_j stands for the permutation we query in current time step
# NOTE: We use the same value for each epoch, since we predefined the game value. 
# TODO: Add reward value calculation process in the epoch iteration
def compute_phi_hat_j_i(k, permutation, permutation_without_i):
    phi_hat_j_i_sum = 0
    for s in range(1, k+1):
        # Compute the score with and without the additional element i
        score_with_i = r_s(permutation)
        if permutation_without_i == []:
            phi_hat_j_i_sum += score_with_i
        else:
            score_without_i = r_s(permutation_without_i)
            phi_hat_j_i_sum += score_with_i - score_without_i
    return phi_hat_j_i_sum / k





def compute_x(Q,p):
    X = []
  #  for i in range(N):
   #     if i == p:
    for j in range(N):
        X.append(Q[tuple(CYCLIC_PERMUTATION_LIST[p])][j])
    return np.array(X)

def compute_vp_value(Q, e_ep, I):
    X = {}
 #   for p in Q.keys():
  #      X[p] = []

    for p in range(N):
        X[tuple(CYCLIC_PERMUTATION_LIST[p])] = []
        for j in range(N):
            if p == I:
                X[tuple(CYCLIC_PERMUTATION_LIST[p])].append(0)
            else:
                X[tuple(CYCLIC_PERMUTATION_LIST[p])].append(Q[tuple(CYCLIC_PERMUTATION_LIST[p])][j]) #CHANGE: Now each element in X is a list representing the marginal value for each agent

    v_p_1,c_p_1 = compute_v_p_based_c_p(X, I)
    if np.dot(v_p_1, X[tuple(CYCLIC_PERMUTATION_LIST[I])]) < c_p_1:
        return v_p_1, c_p_1 - e_ep
    else:
        return -v_p_1, -c_p_1 - e_ep # need to check if it is -e_ep or +e_ep


def compute_v_p_based_c_p(X, p):
    c_p = 1
    v_p = np.zeros((N,N))
    Vec_c_p = np.ones(N)*c_p
    Vec_c_p[p] = 0
    X_matrix = np.zeros((N,N))
    for i in range(N):
        if i == p:
            X_matrix[i] = np.ones(N)
        else:
            X_matrix[i] = np.array(X[tuple(CYCLIC_PERMUTATION_LIST[i])])

    v_p = np.matmul(np.linalg.inv(X_matrix), Vec_c_p )
#    tmp_sum = np.zeros((N,N))
    # for i in range(N):
    #     tmp_sum[i] = v_p[i] ** 2 # NOTE: Using the 2-norm is not reasonable since the value of v_p would always be negative
   # for i in range(N):
   #     tmp_sum[i] = v_p[i] # NOTE: In this case, we will always choose c_p = -1, don't think this is the result you want
   # v_p[p] = np.sqrt(1 - tmp_sum.sum(0))

    return v_p/np.linalg.norm(v_p), c_p/np.linalg.norm(v_p)
# Compute the stopping condition for the algorithm
def stopping_condition(Q_set, b_ep, ep, max_iters, min_iters):
    if ep < min_iters:
        return False
    e_ep = 2 * np.sqrt(len(Q_set)) * b_ep
#    print(e_ep)
    c_p = 1
    Q = copy.deepcopy(Q_set)
    if not Q_set:
        return False
    if ep >= max_iters:
        print("False")
        return True
    for p in range(N):
        # Assuming Hp(Q) computation is done elsewhere and v_p, c_p are returned
        v_p, c_p = compute_vp_value(Q_set,e_ep,p)  # Placeholder function
        x = compute_x(Q,p)
        h_p = c_p - v_p @ x
      #  print("h_p=", h_p)
       # print("N * e_ep= ", N * e_ep)
        if  h_p < N * e_ep - 0.9: # pow(N-2,0)*1 - 1: # 2:-1 3:pow(N-2,1)*1 -0.2; 4-10:pow(N-2,0.8)*1;
            return False
        else:
            return True
    return True

# Main function for the Common Points Picking Algorithm
def common_points_picking(n, max_iters, min_iter):
    t = 0
    ep = 0
    b_ep = 0
    Q_set = {}
    P = copy.deepcopy(CYCLIC_PERMUTATION_LIST)
    marginal_contribution_dict = copy.deepcopy(MARGINAL_VECTOR_DICT)
    while not stopping_condition(Q_set, b_ep, ep, max_iters, min_iter):
     #   print("ep% = ",ep*100/max_iters)
        ep += 1
        phi_hat_sigma_ep = []
        Q_set = {}
        for p in marginal_contribution_dict.keys():
            for i in range(N): # i for index of permutation order, p(i) is stands for the player
                P_ip = list(p)[:i+1] # Set of player that contain p(i)
                #Reward_P_ip = r_s(P_ip) # deterministic: reward of the smallest set contain player p(i)

                Reward_P_ip = np.random.binomial(1, r_cyclic_s(P_ip)) # stochastic: reward of the smallest set contain player p(i)
                if len(P_ip) == 1:
                    Reward_P_ip_without_i = 0 # deterministic
                else:
                    #Reward_P_ip_without_i = r_s(P_ip[:i]) # deterministic: reward of the largest set does not containt player p(i)
                    Reward_P_ip_without_i = np.random.binomial(1, r_cyclic_s(P_ip[:i])) # stochastic: reward of the largest set does not containt player p(i)
            #    p_without_i = [x for x in P_ip if x != P_ip[i]]
                marginal_contribution_i = (Reward_P_ip - Reward_P_ip_without_i)
 #               print("marginal_contribution_i=", marginal_contribution_i)
                marginal_contribution_dict[p][list(p)[i]] = (marginal_contribution_dict[p][list(p)[i]]*(ep-1) + marginal_contribution_i)/ep
            # update the marginal value vector
#              marginal_contribution_dict[p][list(p)[i]] += marginal_contribution_i
            if abs(sum(marginal_contribution_dict[p].values())) > 0.00005:
                norm_marginal_vec = sum(marginal_contribution_dict[p].values())
                for j in range(N): # need to take care of the case when all point is 0.
                    marginal_contribution_dict[p][j] = marginal_contribution_dict[p][j]/norm_marginal_vec
                    # print(marginal_contribution_dict)
#            print(marginal_contribution_dict[p])
        Q_set = copy.deepcopy(marginal_contribution_dict)
  #      print('Q_set = ',  Q_set)
  #      print("Q_set = ",Q_set)
        
        # Compute b_ep 
        b_ep = math.sqrt((2 * math.log(1/DELTA)) / ep)
    #    print("b_ep=",  b_ep)
    print('ep = ',ep)
    # Compute x_star as the average of all phi_hat_sigma(ep)
    x_star = mean_of_nested_dict(Q_set)
 #   for key in x_star.keys():
  #      x_star[key] = x_star[key] / len(x_star.keys())
    return x_star, ep

def mean_of_nested_dict(nested_dict):
    mean_values = np.zeros(N)
    for key, inner_dict in nested_dict.items():
        for i in range(N):
            mean_values[i] += inner_dict[i]

    return mean_values/N



def is_in_ecore(x, N, tolerance=0.0005):
    # check if all the vector in x equals to r(N)
    if not (1 - tolerance <= sum(x) <= 1 + tolerance):
        return False

    for C in itertools.chain.from_iterable(itertools.combinations(np.arange(N), r) for r in range(1, N+1)):
        subset_sum = sum(x[i - 1] for i in C)
        if not (r_s(C) - tolerance <= subset_sum):
            return False
            
    # if all the conditions are satisfied, then it belongs to a E-core
    return True

###########################|  Main |####################################################################################

for N in range(2, 11):
    #N=3
    print(N)
    VARSIGMA = 1
    DELTA = 0.05

    CYCLIC_PERMUTATION_LIST = cyclic_permutation(N)
    MARGINAL_VECTOR_DICT = create_marginal_vector(N, CYCLIC_PERMUTATION_LIST)


    Simulation_iter = 100
    Data_save = []
    with tqdm(total=Simulation_iter) as pbar:

        for iter in range(Simulation_iter):
            # Initialise environment
            S_CONVEX_GAME = generate_convex_game(N, VARSIGMA)
            for i in S_CONVEX_GAME.keys():
                #    S_CONVEX_GAME[i] /= sum(S_CONVEX_GAME.values())
                S_CONVEX_GAME[i] /= max(S_CONVEX_GAME.values())

            # S_CONVEX_GAME_CYCLIC = {key:S_CONVEX_GAME[key] for key in CYCLIC_PERMUTATION_LIST}
            S_CONVEX_GAME_CYCLIC = {}
            for cyclic_p in CYCLIC_PERMUTATION_LIST:
                for i in range(len(cyclic_p)):
                    S_CONVEX_GAME_CYCLIC[frozenset(cyclic_p[:i + 1])] = S_CONVEX_GAME[frozenset(cyclic_p[:i + 1])]


            # Run algorithm
            initial_time = time.time()
            max_iter = 5000000
            min_iter = 100
            x, ep = common_points_picking(N, max_iter, min_iter)
            print("Time: ", time.time() - initial_time)
    #        print(x)
    #        print("Commont point in Core?", is_in_ecore(x,N))
            new_data = {"Epochs": ep, "Status": is_in_ecore(x,N) }
            Data_save.append(new_data)
            pbar.update(1)
    #print(Data_save)
    df = pd.DataFrame(Data_save)
    excel_file_path = 'data' + str(N) + '.xlsx'

    df.to_excel(excel_file_path, index=False)

    print(f"Data saved to {excel_file_path}")



#print("Average episode time: ", (time.time() - initial_time)/)
#ground_truth = compute_marginal_contributions(S_CONVEX_GAME,N)
#shapley_value = shapley_value(S_CONVEX_GAME,N)
#print("Shapley approximation in Core?", is_in_ecore(shapley_value,N))