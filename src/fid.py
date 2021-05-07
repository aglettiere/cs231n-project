## https://github.com/GaParmar/clean-fid/blob/main/cleanfid/fid.py

from cleanfid import fid
import torch
score = fid.compute_fid('/Users/allisonlettiere/Downloads/cs231n-project/data/images/', '/Users/allisonlettiere/Downloads/cs231n-project/results/100epochsResults/', device=torch.device("cpu"), num_workers=0)
print("FID:", score)

