import argparse
import random
import numpy as np
import torch

def control_randomness(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    print(f'control_seed: {seed}')    


def set_arguments():
    parser = argparse.ArgumentParser(description="Run Causal Structure Model.")

    parser.add_argument('-seed', 
                        default=1234,
                        type=int,
                        help='Fix random seed. Default seed is 1234.')   

    parser.add_argument('-initialize', 
                        type=str, 
                        default='xavier',
                        help='Learning rate') 
    parser.add_argument('-dropout', 
                        type=float, 
                        default=0.5) 

    parser.add_argument('-lr', 
                        type=float, 
                        default=0.05) 

    parser.add_argument("-epochs",
                        type=int,
                        default=50)

    parser.add_argument('-batch', 
                        type=int,
                        default=1024)

    parser.add_argument('-gpu', 
                        default=0, 
                        type=str)  

    return parser.parse_args()