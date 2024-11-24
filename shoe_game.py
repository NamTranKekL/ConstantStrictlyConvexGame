import argparse
from poly_shap import Poly_Shap
from config import get_experiment_config
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='symmetric_vote', help='NAME OF ENV')
parser.add_argument('--N', type=int, default=1000, help='TOTAL NUMBER OF AGENTS')
parser.add_argument('--reward_list', type=list, default=[0]*500+[1]*500, help='REWARD LIST')
parser.add_argument('--weight', type=list, default=[1]*1000, help='WEIGHT FOR AGENTS')
parser.add_argument('--m', type=int, default=1000, help='NUMBER OF SAMPLES')
args = parser.parse_args()
args = get_experiment_config(args)
agent = Poly_Shap(args.env, args.N, args.reward_list, args.weight, args.m)
epoch = 1
print(args.env)
shaply_value = [0]*args.N
for i in range(epoch):
    print(i)
    value = agent.run()
    shaply_value = [value[i] + shaply_value[i] for i in range(len(shaply_value))]
shaply_value = [i/epoch for i in shaply_value]
print(sum(shaply_value)/len(shaply_value))
