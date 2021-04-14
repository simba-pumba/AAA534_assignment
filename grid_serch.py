
import os

if __name__=="__main__":
    exp = 0
    for init in ['xavier', 'he']:
        for drop_prop in [0.3, 0.4, 0.5]:
            for lr in [0.0005, 0.001, 0.005, 0.01, 0.05]:
                exp = exp + 1
                print(f"{exp} experiment . . .")
                print(f'python train.py -initialize {init} -dropout {drop_prop} -lr {lr}')
                os.system(f'python train.py -initialize {init} -dropout {drop_prop} -lr {lr}')

