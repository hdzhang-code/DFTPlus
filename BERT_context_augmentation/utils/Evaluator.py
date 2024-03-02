from utils.IntentDataset import IntentDataset
from utils.tools import *
from utils.printHelper import *
from utils.Logger import logger
from utils.commonVar import *
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import logging
import torch
import numpy as np
import pdb
import random
from tqdm import tqdm
import copy as cp
import wandb

from utils.models import IntentBERT, LinearClsfier

class EvaluatorBase():
    def __init__(self):
        self.roundN = 4
        pass

    def round(self, floatNum):
        return round(floatNum, self.roundN)

    def evaluate(self):
        raise NotImplementedError("train() is not implemented.")

class FewShotEvaluator(EvaluatorBase):
    def __init__(self, evalParam, taskParam, dataset: IntentDataset):
        super(FewShotEvaluator, self).__init__()
        self.way   = taskParam['way']
        self.shot  = taskParam['shot']
        self.query = taskParam['query']

        self.dataset = dataset

        self.multi_label = evalParam['multi_label']
        self.clsFierName = evalParam['clsFierName']
        self.evalTaskNum = evalParam['evalTaskNum']
        logger.info("In evaluator classifier %s is used.", self.clsFierName)

        self.beforeBatchNorm = evalParam['beforeBatchNorm']
        logger.info("In evaluator, beforeBatchNorm %s"%(self.beforeBatchNorm))

        if self.multi_label:
            self.taskSampler = MultiLabTaskSampler(self.dataset, self.shot, self.query)
        else:
            self.taskSampler = UniformTaskSampler(self.dataset, self.way, self.shot, self.query)

    def evaluate(self, model, tokenizer, mode='multi-class', logLevel='DEBUG'):
        model.eval()

        performList = []   # acc, pre, rec, fsc
        with torch.no_grad():
            for task in range(self.evalTaskNum):
                # sample a task
                task = self.taskSampler.sampleOneTask()

                # collect data
                supportX = task[META_TASK_SHOT_TOKEN]
                queryX = task[META_TASK_QUERY_TOKEN]
                if mode == 'multi-class':
                    supportY = task[META_TASK_SHOT_LOC_LABID]
                    queryY = task[META_TASK_QUERY_LOC_LABID]
                elif mode == 'entailment-utt':
                    supportY = task[META_TASK_SHOT_LOC_LABID]
                    queryY = task[META_TASK_QUERY_LOC_LABID]
                elif mode == 'entailment-lab':
                    supportY = task[META_TASK_SHOT_LAB]
                    queryY = task[META_TASK_QUERY_LAB]

                # padding
                supportX, supportY, queryX, queryY =\
                    makeEvalExamples(supportX, supportY, queryX, queryY, tokenizer, mode=mode)

                # forward
                queryPrediction = model.fewShotPredict(supportX.to(model.device),
                                                       supportY,
                                                       queryX.to(model.device),
                                                       self.clsFierName,
                                                       mode=mode, 
                                                       beforeBatchNorm=self.beforeBatchNorm)

                if mode == 'entailment-utt':
                    queryPrediction = np.stack([p.reshape(-1, self.shot).sum(-1) for p in queryPrediction.reshape(-1, self.way*self.shot)]).argmax(-1)
                elif mode == 'entailment-lab':
                    queryPrediction = queryPrediction.reshape(-1, self.way).argmax(-1)
                
                # calculate acc
                acc = accuracy_score(queryY, queryPrediction)   # acc
                if self.multi_label:
                    performDetail = precision_recall_fscore_support(queryY, queryPrediction, average='micro', warn_for=tuple())
                else:
                    performDetail = precision_recall_fscore_support(queryY, queryPrediction, average='macro', warn_for=tuple())

                performList.append([acc, performDetail[0], performDetail[1], performDetail[2]])
        
        # performance mean and std
        performMean = np.mean(np.stack(performList, 0), 0)
        performStd  = np.std(np.stack(performList, 0), 0)

        if logLevel == 'DEBUG':
            itemList = ["acc", "pre", "rec", "fsc"]
            logger.debug("Evaluate statistics: ")
            printMeanStd(performMean, performStd, itemList, debugLevel=logging.DEBUG)
        else:
            itemList = ["acc", "pre", "rec", "fsc"]
            logger.info("Evaluate statistics: ")
            printMeanStd(performMean, performStd, itemList, debugLevel=logging.INFO)

        # acc, pre, rec, F1
        return performMean[0], performMean[1], performMean[2], performMean[3]


class KShotEvaluator(EvaluatorBase):
    def __init__(self, evalParam, dataset: IntentDataset):
        super(KShotEvaluator, self).__init__()
        self.shot  = evalParam['shot']
        self.dataset = dataset
        self.batch_size = evalParam['batch_size']
        self.repetition = evalParam['repetition']
        self.seed = evalParam['seed']
        self.clf = None

    def evaluate(self, model, tokenizer, logLevel='DEBUG'):
        model.eval()

        performList = []
        set_seed(self.seed)   # reset seed so that the seed can control the following random data sample process.
        for repeatID in tqdm(range(self.repetition)):
            with torch.no_grad():
                # sample K-shot in the training partition
                tensorDataset = self.dataset.trainPart.randomSliceTorchDataset(self.shot, tokenizer)
                dataloader = DataLoader(tensorDataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

                # logger.info("Generating embeddings for shots ...")
                Y, embeddings = model.forwardEmbeddingDataLoader(dataloader)
                Y = Y.cpu()
                embeddings = embeddings.cpu()

                # fit the model
                if self.clf == None:
                    self.clf = LogisticRegression(penalty='l2',
                            random_state=0,
                            C=1.0,
                            solver='lbfgs',
                            max_iter=1000,
                            multi_class='multinomial')
                    # fit and predict
                self.clf.fit(embeddings, Y)

                # evaluate on test partition
                tensorDatasetTest = self.dataset.testPart.generateTorchDataset(tokenizer)
                dataloaderTest = DataLoader(tensorDatasetTest, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
                # logger.info("Generating embeddings for queries ...")
                queryY, embeddingsTest = model.forwardEmbeddingDataLoader(dataloaderTest)
                queryY = queryY.cpu()
                embeddingsTest = embeddingsTest.cpu()
                # sample K-shot in the training partition
                query_pred = self.clf.predict(embeddingsTest)

                # calculate acc
                acc = accuracy_score(queryY, query_pred)   # acc
                performDetail = precision_recall_fscore_support(queryY, query_pred, average='macro', warn_for=tuple())
                performList.append([acc, performDetail[0], performDetail[1], performDetail[2]])

        # performance mean and std
        performMean = np.mean(np.stack(performList, 0), 0)
        performStd  = np.std(np.stack(performList, 0), 0)

        if logLevel == 'DEBUG':
            itemList = ["acc", "pre", "rec", "fsc"]
            logger.debug("Evaluate statistics: ")
            printMeanStd(performMean, performStd, itemList, debugLevel=logging.DEBUG)
        else:
            itemList = ["acc", "pre", "rec", "fsc"]
            logger.info("Evaluate statistics: ")
            printMeanStd(performMean, performStd, itemList, debugLevel=logging.INFO)

        # acc, pre, rec, F1
        return performMean[0], performMean[1], performMean[2], performMean[3]

##
# @brief MetaEvaluator used to do meta evaluation. Tasks are sampled and the model is evaluated task by task.
class FineTuneEvaluator(EvaluatorBase):
    def __init__(self, evalParam, taskParam, optimizer, dataset: IntentDataset):
        super(FineTuneEvaluator, self).__init__()
        self.way   = taskParam['way']
        self.shot  = taskParam['shot']
        self.query = taskParam['query']

        self.dataset   = dataset
        self.optimizer = optimizer

        self.finetuneSteps = evalParam['finetuneSteps']
        self.evalTaskNum   = evalParam['evalTaskNum']

        self.taskSampler = UniformTaskSampler(self.dataset, self.way, self.shot, self.query)

    def evaluate(self, model, tokenizer, mode='multi-class', logLevel='DEBUG'):
        performList = []   # acc, pre, rec, fsc
        initial_model = model.state_dict().copy()
        initial_optim = self.optimizer.state_dict().copy()

        for task in tqdm(range(self.evalTaskNum)):
            # sample a task
            task = self.taskSampler.sampleOneTask()

            # collect data
            supportX = task[META_TASK_SHOT_TOKEN]
            queryX = task[META_TASK_QUERY_TOKEN]
            if mode == 'multi-class':
                supportY = task[META_TASK_SHOT_LOC_LABID]
                queryY = task[META_TASK_QUERY_LOC_LABID]
            elif mode == 'entailment-utt':
                supportY = task[META_TASK_SHOT_LOC_LABID]
                queryY = task[META_TASK_QUERY_LOC_LABID]
            elif mode == 'entailment-lab':
                supportY = task[META_TASK_SHOT_LAB]
                queryY = task[META_TASK_QUERY_LAB]

            # padding
            supportX, supportY, queryX, queryY =\
                makeEvalExamples(supportX, supportY, queryX, queryY, tokenizer, mode=mode)

            # finetune
            model.train()
            for _ in range(self.finetuneSteps):
                logits = model(supportX.to(model.device))
                loss = model.loss_ce(logits, torch.tensor(supportY).to(model.device))
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.optimizer.step()

            model.eval()
            with torch.no_grad():
                if mode == 'multi-class':
                    queryPrediction = model(queryX.to(model.device)).argmax(-1)
                elif mode == 'entailment-utt':
                    queryPrediction = model(queryX.to(model.device))
                    queryPrediction = torch.stack([p.reshape(-1, self.shot).sum(-1) for p in queryPrediction.reshape(-1, self.way*self.shot)]).argmax(-1)
                elif mode == 'entailment-lab':
                    queryPrediction = model(queryX.to(model.device))
                    queryPrediction = queryPrediction.reshape(-1, self.way).argmax(-1)
                
                queryPrediction = queryPrediction.cpu().numpy()

                # calculate acc
                acc = accuracy_score(queryY, queryPrediction)   # acc
                performDetail = precision_recall_fscore_support(queryY, queryPrediction, average='macro', warn_for=tuple())

                performList.append([acc, performDetail[0], performDetail[1], performDetail[2]])
            
            model.load_state_dict(initial_model)
            self.optimizer.load_state_dict(initial_optim)
        
        model.load_state_dict(initial_model)
        # performance mean and std
        performMean = np.mean(np.stack(performList, 0), 0)
        performStd  = np.std(np.stack(performList, 0), 0)

        if logLevel == 'DEBUG':
            itemList = ["acc", "pre", "rec", "fsc"]
            logger.debug("Evaluate statistics: ")
            printMeanStd(performMean, performStd, itemList, debugLevel=logging.DEBUG)
        else:
            itemList = ["acc", "pre", "rec", "fsc"]
            logger.info("Evaluate statistics: ")
            printMeanStd(performMean, performStd, itemList, debugLevel=logging.INFO)

        # acc, pre, rec, F1
        return performMean[0], performMean[1], performMean[2], performMean[3]


##
# @brief InTaskFineTuneEvaluator used to do evaluation. FineTuneEvaluator is for meta-evaluation, there is a concept of task. InTaskFineTuneEvaluator imitates supervised learning, it is supposed to be used on a standard supervised scenario, we have training data and testing data. We need to finetune the model on a small portion of training data and then conduct evaluation on test data.
class InTaskFineTuneEvaluator(EvaluatorBase):
    def __init__(self, evalParam, dataset: IntentDataset):
        super(InTaskFineTuneEvaluator, self).__init__()
        self.shot  = evalParam['shot']
        self.batch_size = evalParam['batch_size']
        self.seed = evalParam['seed']
        self.repetition = evalParam['repetition']
        self.inTaskEpoch = evalParam['inTaskEpoch']

        self.dataset      = dataset
        self.lr           = evalParam['lr']
        self.weight_decay = evalParam['weight_decay']
        self.sdWeight = evalParam['sdWeight']

        self.wandb        = evalParam['wandb']
        self.wandbProj    = evalParam['wandbProj']
        self.wandbRunName = evalParam['wandbRunName']
        self.wandbConfig  = evalParam['wandbConfig']

    def evaluate(self, model, tokenizer, logLevel='DEBUG'):
        performList = []   # acc, pre, rec, fsc
        initial_model = cp.deepcopy(model.state_dict())

        set_seed(self.seed)   # reset seed so that the seed can control the following random data sample process.
        for repeatID in tqdm(range(self.repetition)):
            # sample K-shot in the training partition
            tensorDataset = self.dataset.trainPart.randomSliceTorchDataset(self.shot, tokenizer)
            dataloader = DataLoader(tensorDataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

            logger.info("Initializing linear classifier ...")
            model.eval()
            Y, embeddings = model.forwardEmbeddingDataLoader(dataloader)

            # calculate proto for classifier initialization
            Y = Y.detach().cpu()
            embeddings = embeddings.detach().cpu()
            Y2EmbeddingList = {}
            for y, embeddings in zip(Y, embeddings):
                yValue = y.item()
                if yValue not in Y2EmbeddingList:
                    Y2EmbeddingList[yValue] = []
                Y2EmbeddingList[yValue].append(embeddings)
            protoList = []
            for y in sorted(Y2EmbeddingList):
                proto = torch.stack(Y2EmbeddingList[y]).mean(0)
                protoList.append(proto)
            protoTensor = torch.stack(protoList)

            # finetune
            if self.wandb:
                run = wandb.init(project=self.wandbProj, reinit=True)
                wandb.config.update(self.wandbConfig)
                wandbRunName = self.wandbRunName + f'-rp{repeatID}'
                wandb.run.name=(wandbRunName)

            model.train()
            # initialize linear classifier
            lcConfig = {'device': model.device, 'clsNumber': self.dataset.getLabNum(), 'initializeValue': protoTensor}
            lc = LinearClsfier(lcConfig)
            # initialize model
            optimizer = optim.AdamW(list(model.parameters()) + list(lc.parameters()), lr=self.lr, weight_decay=self.weight_decay)
            patience = 5
            patience = 200
            lowLossEpoch = 0
            for epoch in range(self.inTaskEpoch):
                batchLossList = []
                batchCEList = []
                batchSDList = []
                trainAccList = []
                for batchID, batch in enumerate(dataloader):
                    # batch data
                    Y, ids, types, masks = batch
                    X = {'input_ids':ids.to(model.device),
                            'token_type_ids':types.to(model.device),
                            'attention_mask':masks.to(model.device)}
                    batchEmbedding = model.forwardEmbedding(X)
                    logits = lc(batchEmbedding)
                    # loss = model.loss_ce(logits, Y.to(model.device))
                    loss, crossEntropy, sd = model.loss_ce_SD(logits, Y.to(model.device), self.sdWeight)
                    batchLossList.append(loss.item())
                    batchCEList.append(crossEntropy.item())
                    batchSDList.append(sd.item())
                    optimizer.zero_grad()
                    loss.backward()
                    # logger.info(f"Fine-tuning inside task: loss = {loss.item()}")
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(lc.parameters(), 1.0)
                    optimizer.step()

                    # calculate train acc
                    YTensor = Y.cpu()
                    logits = logits.detach().clone()
                    if torch.cuda.is_available():
                        logits = logits.cpu()
                    logits = logits.numpy()
                    predictResult = np.argmax(logits, 1)
                    trainAcc = accuracy_score(YTensor, predictResult)
                    trainAccList.append(trainAcc)

                # this epoch is done
                avrgLoss = sum(batchLossList) / len(batchLossList)
                avrgLossCE = sum(batchCEList) / len(batchCEList)
                avrgLossSD = sum(batchSDList) / len(batchSDList)
                avrgTrainAcc = sum(trainAccList) / len(trainAccList)
                if avrgLoss < 0.01:
                    lowLossEpoch = lowLossEpoch + 1
                else:
                    lowLossEpoch = 0

                # validation on test partition
                logger.info("Monitoring performance on test partition ...")
                acc, pre, rec, fsc = self.evaluateOnTestPartition(model, lc, tokenizer)
                model.train()
                logger.info(f"In-task fine-tuning epoch {epoch}, averLoss = {avrgLoss}, avrgLossCE={avrgLossCE}, avrgLossSD={avrgLossSD}, lowLossE = {lowLossEpoch}/{patience}, tranAcc={avrgTrainAcc}, testPartAcc={acc}.")

                # log in wandb
                if self.wandb:
                    wandb.log({'avrgLoss': avrgLoss, \
                            'avrgLossCE': avrgLossCE, \
                            'avrgLossSD': avrgLossSD, \
                            'avrgTrainAcc': avrgTrainAcc, \
                            'epoch': epoch, \
                            'testPartAcc': acc, \
                            })
                if lowLossEpoch >= patience:
                    break

            if self.wandb:
                run.finish()

            # evaluate model on test partition
            logger.info("Fine-tuning is done, evaluating on test partition ...")
            acc, pre, rec, fsc = self.evaluateOnTestPartition(model, lc, tokenizer)
            performList.append([acc, pre, rec, fsc])

            # recover optimizer and model parameters
            model.load_state_dict(initial_model)
        
        # performance mean and std
        model.load_state_dict(initial_model)
        performMean = np.mean(np.stack(performList, 0), 0)
        performStd  = np.std(np.stack(performList, 0), 0)

        if logLevel == 'DEBUG':
            itemList = ["acc", "pre", "rec", "fsc"]
            logger.debug("Evaluate statistics: ")
            printMeanStd(performMean, performStd, itemList, debugLevel=logging.DEBUG)
        else:
            itemList = ["acc", "pre", "rec", "fsc"]
            logger.info("Evaluate statistics: ")
            printMeanStd(performMean, performStd, itemList, debugLevel=logging.INFO)

        # acc, pre, rec, F1
        return performMean[0], performMean[1], performMean[2], performMean[3]


    ##
    # @brief 
    #
    # @param model
    # @param lc linear classifier
    # @param tokenizer
    #
    # @return 
    def evaluateOnTestPartition(self, model, lc, tokenizer):
        # evaluate on test partition
        model.eval()
        with torch.no_grad():
            # loop test partition to predict the label
            tensorDatasetTest = self.dataset.testPart.generateTorchDataset(tokenizer)
            dataloaderTest = DataLoader(tensorDatasetTest, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            YTruthList = []
            predResultList = []
            for batchID, batch in tqdm(enumerate(dataloaderTest)):
                # batch data
                Y, ids, types, masks = batch
                X = {'input_ids':ids.to(model.device),
                        'token_type_ids':types.to(model.device),
                        'attention_mask':masks.to(model.device)}
                batchEmbedding = model.forwardEmbedding(X)
                logits = lc(batchEmbedding)

                YTensor = Y.cpu()
                logits = logits.detach().clone()
                if torch.cuda.is_available():
                    logits = logits.cpu()
                logits = logits.numpy()
                predResult = np.argmax(logits, 1)

                YTruthList.extend(Y.tolist())
                predResultList.extend(predResult.tolist())
        # calculate acc
        acc = accuracy_score(YTruthList, predResultList)   # acc
        performDetail = precision_recall_fscore_support(YTruthList, predResultList, average='macro', warn_for=tuple())

        return acc, performDetail[0], performDetail[1], performDetail[2]


##
# @brief TestPartitionEvaluator used to do evaluation on the test partition. This class is supposed not to train the model, in contrast, InTaskFineTuneEvaluator do model updating.
class TestPartitionEvaluator(EvaluatorBase):
    def __init__(self, evalParam, dataset: IntentDataset):
        super(TestPartitionEvaluator, self).__init__()
        self.dataset      = dataset
        self.batch_size = evalParam['batch_size']

    def evaluate(self, model, lc, tokenizer, logLevel='DEBUG'):
        # evaluate model on test partition
        acc, pre, rec, fsc = self.evaluateOnTestPartition(model, lc, tokenizer)

        return acc, pre, rec, fsc

    ##
    # @brief 
    #
    # @param model
    # @param lc linear classifier
    # @param tokenizer
    #
    # @return 
    def evaluateOnTestPartition(self, model, lc, tokenizer):
        # evaluate on test partition
        model.eval()
        with torch.no_grad():
            # loop test partition to predict the label
            tensorDatasetTest = self.dataset.testPart.generateTorchDataset(tokenizer)
            dataloaderTest = DataLoader(tensorDatasetTest, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            YTruthList = []
            predResultList = []
            for batchID, batch in tqdm(enumerate(dataloaderTest)):
                # batch data
                Y, ids, types, masks = batch
                X = {'input_ids':ids.to(model.device),
                        'token_type_ids':types.to(model.device),
                        'attention_mask':masks.to(model.device)}
                batchEmbedding = model.forwardEmbedding(X)
                logits = lc(batchEmbedding)

                YTensor = Y.cpu()
                logits = logits.detach().clone()
                if torch.cuda.is_available():
                    logits = logits.cpu()
                logits = logits.numpy()
                predResult = np.argmax(logits, 1)

                YTruthList.extend(Y.tolist())
                predResultList.extend(predResult.tolist())
        # calculate acc
        acc = accuracy_score(YTruthList, predResultList)   # acc
        performDetail = precision_recall_fscore_support(YTruthList, predResultList, average='macro', warn_for=tuple())

        return acc, performDetail[0], performDetail[1], performDetail[2]
