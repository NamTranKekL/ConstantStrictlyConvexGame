import random
import math
from random import sample
class Poly_Shap():
    def __init__(self,env,N,reward_list,w,m):
        self.env = env
        self.N = N
        self.reward_list = reward_list
        self.weight = w
        self.epoch = m
        self.sh = [0] * self.N
    def run(self):
        for i in range(self.epoch):
            P = self.sample()
            reward = [0] * self.N
            if self.env == 'symmetric_vote':
                reward[P[501]]+=1
            elif self.env == 'non_symmetric_vote':
                reward_lim = sum(self.weight)/2
                for number in P:
                    index = P.index(number)
                    P.remove(number)
                    P_remove = P.copy()
                    P.insert(index,number)
                    if sum([self.weight[i] for i in P]) > reward_lim:
                        if sum([self.weight[i] for i in P_remove]) < reward_lim:
                            reward[index]+=1
            self.sh = [self.sh[i] + reward[i] for i in range(self.N)]
        self.sh = [x/self.epoch for x in self.sh]
        return self.sh
    def sample(self):
        permutation_set = [ x for x in range(self.N)]
        sample_set = random.sample(permutation_set,self.N)
        return sample_set
