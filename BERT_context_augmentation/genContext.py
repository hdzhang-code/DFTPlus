import torch
import argparse
import time
from transformers import AutoTokenizer

from utils.models import IntentBERT
from utils.IntentDataset import IntentDataset
from utils.Trainer import FewShotTuneTrainer
from utils.commonVar import *
from utils.printHelper import *
from utils.tools import *
from utils.Logger import logger
from utils.DataAugmenter import DataAugmenter
from torch.utils.data import DataLoader
import pdb
import copy as cp
import os

def parseArgument():
    parser = argparse.ArgumentParser(description='Evaluate few-shot performance')

    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--tokenizer', default='bert-base-uncased',
                        help="Name of tokenizer")
    parser.add_argument('--LMName', default='bert-base-uncased',
                        help='Name for models and path to saved model')
    parser.add_argument('--augNum', default=1, type=int)

    parser.add_argument('--testDataset', help="Dataset names included in this experiment. For example:'OOS'.")
    parser.add_argument('--testDomain', help='Test domain names and separated by comma.')

    
    parser.add_argument('--shot', type=int, default=2)
    parser.add_argument('--disableCuda', action="store_true")
    
    parser.add_argument('--loggingLevel', default='INFO',
                        help="python logging level")

    args = parser.parse_args()

    return args

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = parseArgument()
    print(args)

    if args.loggingLevel == LOGGING_LEVEL_INFO:
        loggingLevel = logging.INFO
    elif args.loggingLevel == LOGGING_LEVEL_DEBUG:
        loggingLevel = logging.DEBUG
    else:
        raise NotImplementedError("Not supported logging level %s", args.loggingLevel)
    logger.setLevel(loggingLevel)

    if args.seed >= 0:
        set_seed(args.seed)
        logger.info("The random seed is set %d"%(args.seed))
    else:
        logger.info("The random seed is not set any value.")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    logger.info("----- Testing Data -----")
    logger.info(f"Loading data from {args.testDataset}")
    dataset = IntentDataset(args.testDataset, mode=DATASET_MODE_PARTITION_MERGE_INTO_TEST)
    dataset = dataset.selectByDomains(splitName(args.testDomain))
    dataset.tokenize(tok)

    logger.info("----- IntentBERT initializing -----")
    modelConfig = {}
    modelConfig['device'] = torch.device('cuda:0' if not args.disableCuda else 'cpu')
    modelConfig['clsNumber'] = 90
    modelConfig['LMName'] = args.LMName
    model = IntentBERT(modelConfig)

    set_seed(args.seed)   # reset seed so that the seed can control the following random data sample process.

    tensorDataset, fewShotUttList, labList = dataset.trainPart.randomSliceTorchDataset(args.shot, tok)
    logger.info(f"Sliced {args.shot} data from dataset. K-shot data size is {len(tensorDataset)}. Signature: {getMD5(fewShotUttList)}.")

    dataAugmenterParam={"cachePath": "./dataAugCache", \
            "cacheFile": f"DS{args.testDataset}_SE{args.seed}_ST{args.shot}_DN{args.augNum}.pk"}
    dataAugmenter = DataAugmenter(dataAugmenterParam)
    unlabTensorData, newUttList, newLabList = dataAugmenter.dataAugContextGPT3(fewShotUttList, labList, augNum = args.augNum, device=model.device, returnTensorDataset = True, mergeOriData = True, oriTensorDataset = tensorDataset, dataset=dataset, tokenizerOri = tok)

    logger.info(args)
    logger.info(time.asctime())

if __name__ == "__main__":
    main()
    exit(0)
