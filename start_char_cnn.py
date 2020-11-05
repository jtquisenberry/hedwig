# python -m models.char_cnn --dataset Reuters --batch-size 128 --lr 0.001 --seed 3435

# from .models import char_cnn
#import models.char_cnn

import sys
# sys.argv.append('abcdef')
#sys.argv.extend(['--dataset', 'Reuters', '--batch-size', '128', '--lr', '0.001', '--seed', '3435'])
#sys.argv.append('3')



from models.char_cnn.runner import Runner
from models.char_cnn.runner import CustomArgs
args = CustomArgs()

args.cuda = False
args.dataset='Reuters'
args.batch_size=128
args.lr=0.001
args.seed=3435
args.data_dir = r'E:\Development\corpora\hedwig-data\datasets'
# args.epochs = 1

print(args)



runner = Runner(args)
runner.start()





