import argparse
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import sys
import torch

import preprocess as data

miss_type_map = {
    'MCAR': data.MCAR,
    'MAR': data.MAR,
    'label': data.label_dependent_missingness_no_noise,
    'labelnoise': data.label_dependent_missingness,
    'logit': data.logit_missingness,
    'MNARsum': data.MNARsum,
    'MNAR1var': data.MNAR1var,
    'MNAR1varMCAR': data.MNAR1varMCAR,
    'MNAR2var': data.MNAR2var,
}


def parse_args():
    """Parses the arguments for the model.
    
    TODO: check back later which of these arguments are actually used in code
    TODO incomplete. further arguments will need to be passed (e.g. see __init__function).
    """
    parser = argparse.ArgumentParser(description='Deep heterogenous latent factor modelling')

    parser.add_argument('--data-file', nargs='?', default='data/breast', help='input data directory')
    parser.add_argument('--header', nargs='?', default=None, help='does data set have header')
    parser.add_argument('--image-dim-0', type=int, default=28, help='image dim 0 (default: 29)')
    parser.add_argument('--train-pct', type=int, default=0.8, metavar='N', help='Percentage of train data set (default: 80)')
    parser.add_argument('--val-pct', type=int, default=0.1, metavar='N', help='Percentage of train data set (default: 80)')
    parser.add_argument('--miss-type', nargs='?', default='MNAR1var', choices=miss_type_map.keys(), help='missingness type to be induced')
    parser.add_argument('--miss-ratio', type=float, default=0.8, help='missingness ratio')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 1)')

    args = parser.parse_args()
    args.uniform = True

    return(args)


def main(args):
    """Pipeline for preprocessing data.""" 

    # check if file exists
    miss_file_name = args.miss_type + "_" + (1-args.uniform)*"not" + "uniform_frac_" + str(int(args.miss_ratio*100)) + "_seed_" + str(args.seed)  
    Path(os.path.join(args.data_file, "miss_data")).mkdir(parents=True, exist_ok=True)
    print("File " + args.data_file + "/miss_data/" + miss_file_name + " will now be generated.")

    # load data
    data = pd.read_csv(os.path.join(args.data_file, "data.csv"), engine="python", header=args.header)
    try: 
        targets = pd.read_csv(os.path.join(args.data_file, "targets.csv"), header=None)
    except FileNotFoundError:
        pass

    # induce missingness
    np.random.seed(args.seed)
    induce_missingness = miss_type_map[args.miss_type]
    if 'label' in args.miss_type:
        labels = pd.read_csv(os.path.join(args.data_file, "targets.csv"), engine="python", header=None)
        M, patternsets = induce_missingness(data, labels, missingness_ratio=args.miss_ratio, seed=args.seed)
    else:
        M, patternsets = induce_missingness(data, missingness_ratio=args.miss_ratio, seed=args.seed)

    # train test split
    train_idx = int(args.train_pct*len(data))
    val_idx = int((args.train_pct+args.val_pct)*len(data))
    random_permute = np.random.RandomState(seed=args.seed).permutation(len(data))
    M_train, M_val, M_test = M.iloc[random_permute[:train_idx]], M.iloc[random_permute[train_idx:val_idx]], M.iloc[random_permute[val_idx:]]  
    data_train, data_val, data_test = data.iloc[random_permute[:train_idx]], data.iloc[random_permute[train_idx:val_idx]], data.iloc[random_permute[val_idx:]]  

    # save induced missing data
    M_train.to_csv(os.path.join(args.data_file, "miss_data", miss_file_name + ".train"), header=False, index=False)
    M_val.to_csv(os.path.join(args.data_file, "miss_data", miss_file_name + ".val"), header=False, index=False)
    M_test.to_csv(os.path.join(args.data_file, "miss_data", miss_file_name + ".test"), header=False, index=False)
    data_train.to_csv(os.path.join(args.data_file, f"data_{args.seed}.train"), header=False, index=False)
    data_val.to_csv(os.path.join(args.data_file, f"data_{args.seed}.val"), header=False, index=False)
    data_test.to_csv(os.path.join(args.data_file, f"data_{args.seed}.test"), header=False, index=False)
    
    if patternsets != None:
        patternsets_train, patternsets_val, patternsets_test = patternsets.iloc[random_permute[:train_idx]], patternsets.iloc[random_permute[train_idx:val_idx]], patternsets.iloc[random_permute[val_idx:]]  
        patternsets_train.to_csv(os.path.join(args.data_file, "miss_data", f"{miss_file_name}_patternsets.train"), header=False, index=False)
        patternsets_val.to_csv(os.path.join(args.data_file, "miss_data", f"{miss_file_name}_patternsets.val"), header=False, index=False)
        patternsets_test.to_csv(os.path.join(args.data_file, "miss_data", f"{miss_file_name}_patternsets.test"), header=False, index=False)
    
    try:
        targets_train, targets_val, targets_test = targets.iloc[random_permute[:train_idx]], targets.iloc[random_permute[train_idx:val_idx]], targets.iloc[random_permute[val_idx:]]  
        targets_train.to_csv(os.path.join(args.data_file, f"targets_{args.seed}.train"), header=False, index=False)
        targets_val.to_csv(os.path.join(args.data_file, f"targets_{args.seed}.val"), header=False, index=False)
        targets_test.to_csv(os.path.join(args.data_file, f"targets_{args.seed}.test"), header=False, index=False)
    except UnboundLocalError:
        pass


if __name__ == "__main__":
    args = parse_args() # sys.argv[0] is file name
    main(args)