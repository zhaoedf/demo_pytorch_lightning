
import argparse
from pytorch_lightning import Trainer
import json

parser = argparse.ArgumentParser()
parser = Trainer.add_argparse_args(parser)

args_temp = parser.parse_args()

with open('./trainer.json') as f:
    trainer_params = json.load(f)

args_dict = vars(args_temp)

#print(args_dict)
for k,v in trainer_params.items():
    args_dict[k] = v

#print("*"*100)

args = argparse.Namespace(**args_dict)

#print(args)

