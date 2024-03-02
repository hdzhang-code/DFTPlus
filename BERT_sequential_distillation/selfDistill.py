import torch
import argparse
import time
from transformers import AutoTokenizer

from utils.models import IntentBERT, LinearClsfier
from utils.IntentDataset import IntentDataset
from utils.Trainer import SelfDistillationTrainer
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
    parser.add_argument('--MLMWeight', type=float, default=1.0)
    parser.add_argument('--augNum', default=1, type=int)
    parser.add_argument('--KDIter', default=1, type=int)

    parser.add_argument('--testDataset', help="Dataset names included in this experiment. For example:'OOS'.")
    parser.add_argument('--testDomain', help='Test domain names and separated by comma.')
    
    parser.add_argument('--shot', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--inTaskEpoch', type=int, default=2)
    parser.add_argument('--disableCuda', action="store_true")
    parser.add_argument('--lrBackbone', type=float, default=2e-5)
    parser.add_argument('--lrClsfier', type=float, default=2e-5)
    parser.add_argument('--weightDecay', type=float, default=0)
    parser.add_argument('--sdWeight', default=0.0, type=float)
    parser.add_argument('--KDTemp', default=10.0, type=float)
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--epochMonitorWindow', type=int, default=1)
    
    # ==== other things ====
    parser.add_argument('--loggingLevel', default='INFO',
                        help="python logging level")
    parser.add_argument('--wandb', help='use wandb or not', action="store_true")
    parser.add_argument('--wandbProj', help='wandb project name')
    parser.add_argument('--saveModel', action='store_true')
    parser.add_argument('--saveName', default='none',
                        help="Specify a unique name to save your model"
                        "If none, then there will be a specific name controlled by how the model is trained")
    parser.add_argument('--monitorTestPerform', action='store_true')
    parser.add_argument('--loadModelPath', default=SAVE_PATH, help='where to load the model')


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
    modelConfig['loadModelPath'] = args.loadModelPath
    model = IntentBERT(modelConfig)
    lcConfig = {}
    lcConfig['device']          = torch.device('cuda:0' if not args.disableCuda else 'cpu')
    lcConfig['clsNumber']       = dataset.getLabNum()
    lcConfig['initializeValue'] = None
    lcConfig['loadModelPath'] = args.loadModelPath
    lc = LinearClsfier(lcConfig)
    lc.loadFromDisk(args.LMName)

    trainingParam = {"shot":args.shot,  \
            'batch_size': args.batch_size,  \
            'lrBackbone': args.lrBackbone, \
            'lrClsfier': args.lrClsfier, \
            'weight_decay': args.weightDecay,  \
            'seed':args.seed,  \
            'inTaskEpoch':args.inTaskEpoch,  \
            'sdWeight': args.sdWeight,
            "wandb": args.wandb, \
            "wandbProj":args.wandbProj, \
            "wandbRunName": "KD-al%.2f-T%.2f-AU%d-lrBackbone%.5f-lrCls%.5f-Te%s-%s-seed%d-st%s"%(args.alpha, args.KDTemp, args.augNum, args.lrBackbone, args.lrClsfier, args.testDataset, args.LMName, args.seed, args.shot), \
            "wandbConfig": {},  \
            "epochMonitorWindow" : args.epochMonitorWindow, \
            "monitorTestPerform": args.monitorTestPerform,  \
            "MLMWeight": args.MLMWeight, \
            "KDTemp": args.KDTemp, \
            "KDIter": args.KDIter, \
            "alpha": args.alpha
    }
    trainer = SelfDistillationTrainer(trainingParam, dataset)

    set_seed(args.seed)   # reset seed so that the seed can control the following random data sample process.

    tensorDataset, fewShotUttList, labList = dataset.trainPart.randomSliceTorchDataset(args.shot, tok)
    dataloader = DataLoader(tensorDataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    logger.info(f"Sliced {args.shot} data from dataset. K-shot data size is {len(tensorDataset)}.")

    mode = 2
    if mode == 1:
        dataAugmenterParam={"cachePath": "./dataAugCache", \
                "cacheFile": f"DS{args.testDataset}_SE{args.seed}_ST{args.shot}_DN{args.augNum}.pk"}
        dataAugmenter = DataAugmenter(dataAugmenterParam)
        unlabTensorData, newUttList, newLabList = dataAugmenter.dataAugContextGPT3(fewShotUttList, labList, augNum = args.augNum, device=model.device, returnTensorDataset = True, mergeOriData = True, oriTensorDataset = tensorDataset, dataset=dataset, tokenizerOri = tok)
    elif mode ==2:   # exploit few-shot data
        unlabTensorData = cp.deepcopy(tensorDataset)
    else:
        unlabTensorData = dataset.testPart.generateTorchDataset(tok)

    model, lc = trainer.seqSelfDistill(model, lc, tok, dataset, dataloader, unlabTensorData, logLevel='INFO')

    if args.saveModel:
        save_path = os.path.join(SAVE_PATH, args.saveName)
        logger.info("Saving model.pth into folder: %s", save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.save(save_path)
        lc.save(save_path)

    logger.info(args)
    logger.info(time.asctime())

if __name__ == "__main__":
    main()
    exit(0)
