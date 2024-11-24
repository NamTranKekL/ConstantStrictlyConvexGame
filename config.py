def get_experiment_config(args):
    if args.env == 'non_symmetric_vote':
        args.weight = [45,41,27,26,26,25,21,17,17,14,13,13,12,12,12,11,10,10,10,10,9,9,9,9,8,8,7,7,7,7,6,6,6,6,5,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3]
        args.N = 51
        args.m = 10000000
    elif args.env == 'airport':
        args.weight = [1]*8 + [2]*12 + [3]*6 + [4]*14 + [5]*8 + [6]*9 + [7]*13 + [8]*10 + [9]*10 + [10]*10
        args.N = 100
        args.m = 100000000
    return args

