## https://github.com/GaParmar/clean-fid/blob/main/cleanfid/fid.py

from cleanfid import fid
import torch
'''adam_score = fid.compute_fid('/Users/allisonlettiere/Downloads/cs231n-project/data/full_covers/resized_images/', '/Users/allisonlettiere/Downloads/cs231n-project/results/adam-results-combined-epoch-100/', device=torch.device("cpu"), num_workers=0)
print("Adam FID:", adam_score)'''

lbfgs_score = fid.compute_fid('/Users/allisonlettiere/Downloads/cs231n-project/data/full_covers/resized_images/', '/Users/allisonlettiere/Downloads/cs231n-project/results/adam-longtraining/', device=torch.device("cpu"), num_workers=0)
print("L-BFGS FID:", lbfgs_score)

'''rmsprop_score = fid.compute_fid('/Users/allisonlettiere/Downloads/cs231n-project/data/full_covers/resized_images/', '/Users/allisonlettiere/Downloads/cs231n-project/results/rmsprop-results-combined-epoch-100/', device=torch.device("cpu"), num_workers=0)
print("RMSProp FID:", rmsprop_score)'''