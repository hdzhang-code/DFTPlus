import torch
import argparse
import time
from transformers import AutoTokenizer

from utils.models import IntentBERT, LinearClsfier
from utils.IntentDataset import IntentDataset
from utils.Evaluator import FewShotEvaluator, TestPartitionEvaluator
from utils.commonVar import *
from utils.printHelper import *
from utils.tools import *
from utils.Logger import logger
import pdb
import os

def parseArgument():
    parser = argparse.ArgumentParser(description='Evaluate few-shot performance')

    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--tokenizer', default='bert-base-uncased',
                        help="Name of tokenizer")
    parser.add_argument('--LMName', default='bert-base-uncased',
                        help='Name for models and path to saved model')
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--testDataset', help="Dataset names included in this experiment. For example:'OOS'.")
    parser.add_argument('--testDomain', help='Test domain names and separated by comma.')
    
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
    modelConfig['device']    = torch.device('cuda:0' if not args.disableCuda else 'cpu')
    modelConfig['clsNumber'] = 90
    modelConfig['LMName']    = args.LMName
    model = IntentBERT(modelConfig)
    lcConfig = {}
    lcConfig['device']          = torch.device('cuda:0' if not args.disableCuda else 'cpu')
    lcConfig['clsNumber']       = dataset.getLabNum()
    lcConfig['initializeValue'] = None
    lc = LinearClsfier(lcConfig)
    lc.loadFromDisk(args.LMName)

    testParam = {'batch_size': args.batch_size}

    tester = TestPartitionEvaluator(testParam, dataset)
    logger.info("Evaluating model ...")

    set_seed(args.seed)   # reset seed so that the seed can control the following random data sample process.
    acc, pre, rec, fsc = tester.evaluate(model, lc, tok, logLevel='INFO')

    itemList = ["acc", "pre", "rec", "fsc"]
    logger.info("Evaluate statistics: ")
    printMean([acc, pre, rec, fsc], itemList, debugLevel=logging.INFO)

    logger.info(args)
    logger.info(time.asctime())

if __name__ == "__main__":
    main()
    exit(0)
