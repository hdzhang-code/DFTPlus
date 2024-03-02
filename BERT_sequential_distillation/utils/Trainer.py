import transformers
from utils.IntentDataset import IntentDataset
from utils.Evaluator import EvaluatorBase
from utils.Logger import logger
from utils.commonVar import *
from utils.tools import mask_tokens, makeTrainExamples
from utils.models import IntentBERT, LinearClsfier
import time
import torch
import numpy as np
import copy
from sklearn.metrics import accuracy_score, r2_score, precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import copy as cp
import wandb
import pdb

class TrainerBase():
    def __init__(self, wandb, wandbProj, wandbConfig, wandbRunName):
        self.finished=False
        self.bestModelStateDict = None
        self.roundN = 4

        self.wandb = wandb
        self.wandbProjName = wandbProj
        self.wandbConfig = wandbConfig
        self.runName = wandbRunName
        pass

    def round(self, floatNum):
        return round(floatNum, self.roundN)

    def train(self):
        raise NotImplementedError("train() is not implemented.")

    def getBestModelStateDict(self):
        return self.bestModelStateDict

class SelfDistillationTrainer(TrainerBase):
    def __init__(self, trainingParam, dataset: IntentDataset):
        super(SelfDistillationTrainer, self).__init__(trainingParam["wandb"], trainingParam["wandbProj"], trainingParam["wandbConfig"], trainingParam["wandbRunName"])
        self.shot  = trainingParam['shot']
        self.batch_size = trainingParam['batch_size']
        self.seed = trainingParam['seed']
        self.inTaskEpoch = trainingParam['inTaskEpoch']
        self.monitorTestPerform = trainingParam['monitorTestPerform']


        self.dataset      = dataset
        self.lrBackbone          = trainingParam['lrBackbone']
        self.lrClsfier           = trainingParam['lrClsfier']
        self.weight_decay = trainingParam['weight_decay']
        self.sdWeight     = trainingParam['sdWeight']
        self.MLMWeight    = trainingParam['MLMWeight']
        self.mlm = True

        self.KDTemp     = trainingParam['KDTemp']
        self.kd_criterion = torch.nn.KLDivLoss(reduction="batchmean")

        self.KDIter = trainingParam['KDIter']
        self.alpha = trainingParam['alpha']

        self.epochMonitorWindow = trainingParam['epochMonitorWindow']

    def initLinearClassifierRandom(self, model, dataset):
        lcConfig = {'device': model.device, 'clsNumber': dataset.getLabNum(), 'initializeValue': None}
        lc = LinearClsfier(lcConfig)

        return lc

    def initLinearClassifier(self, model, dataset, dataloader):
        model.eval()
        Y, embeddings = model.forwardEmbeddingDataLoader(dataloader)

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

        lcConfig = {'device': model.device, 'clsNumber': dataset.getLabNum(), 'initializeValue': protoTensor}
        lc = LinearClsfier(lcConfig)

        return lc

    def makeTrainExamples(data:list, tokenizer, label=None, mode='unlabel'):
        """
        unlabel: simply pad data and then convert into tensor
        multi-class: pad data and compose tensor dataset with labels
        entailment-utt: compose entailment examples with other utterances
        entailment-lab: compose entailment examples with labels
        """
        examples = tokenizer.pad(data, padding='longest', return_tensors='pt')
        examples = TensorDataset(examples['input_ids'],
                                 examples['token_type_ids'],
                                 examples['attention_mask'])
        return examples

    def evaluateOnTestPartition(self, model, lc, dataset, tokenizer):
        model.eval()
        lc.eval()
        with torch.no_grad():
            tensorDatasetTest = dataset.testPart.generateTorchDataset(tokenizer)
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

    def seqSelfDistill(self, model, lc, tokenizer, dataset, dataloader, unlabeledDataset, logLevel='DEBUG'):
        for kdIterID in range(self.KDIter):
            model, lc = self.selfDistill(model, lc, tokenizer, dataset, dataloader, unlabeledDataset, logLevel=logLevel, iterID = kdIterID)
        return model, lc


    def selfDistill(self, model, lc, tokenizer, dataset, dataloader, unlabeledDataset, logLevel='DEBUG', iterID = None):
        studentModel = cp.deepcopy(model)
        studentLc = cp.deepcopy(lc)
        if self.wandb:
            run = wandb.init(project=self.wandbProjName, reinit=True)
            wandb.config.update(self.wandbConfig)
            wandbRunName = f"KD{iterID}-{self.runName}"
            wandb.run.name=(wandbRunName)

        paramList = [{'params': studentModel.parameters(), 'lr': self.lrBackbone}, \
                {'params': studentLc.parameters(), 'lr': self.lrClsfier}]
        optimizer = optim.AdamW(paramList, weight_decay=self.weight_decay)
        t_total = len(dataloader) * self.inTaskEpoch
        warmup_steps = round(t_total/20)
        logger.info(f"Learning rate scheduler: warmup_steps={warmup_steps}, t_total={t_total}.")
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

        logger.info(f"Generating teacher soft target ...")
        dataloaderUnlabeled = DataLoader(unlabeledDataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        targetSoftLabel = self.getSoftTarget(model, lc, dataloader)
        tensorDatasetWithLogits = TensorDataset(torch.stack(targetSoftLabel['Y']),   \
                torch.stack(targetSoftLabel['input_ids']), \
                torch.stack(targetSoftLabel['token_type_ids']), \
                torch.stack(targetSoftLabel['attention_mask']), \
                torch.stack(targetSoftLabel['logits']), \
                )
        tensorDataloaderWithLogits = DataLoader(tensorDatasetWithLogits, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        for epoch in range(self.inTaskEpoch):
            studentModel.train()
            studentLc.train()

            batchLossList = []
            batchCEList = []
            batchSDList = []
            trainAccList = []
            batchMLMList = []
            for batchID, batch in enumerate(tensorDataloaderWithLogits):
                Y, ids, types, masks, logitsTeacher = batch
                X = {'input_ids':ids.to(studentModel.device),
                        'token_type_ids':types.to(studentModel.device),
                        'attention_mask':masks.to(studentModel.device)}
                batchEmbedding = studentModel.forwardEmbedding(X)
                logits = studentLc(batchEmbedding)

                logitsTeacher = logitsTeacher.to(studentModel.device)
                loss_KL = self.kd_criterion(F.log_softmax(logits / self.KDTemp, 1), F.softmax(logitsTeacher / self.KDTemp, 1))

                lossCE = model.loss_ce(logits, Y.to(model.device))

                loss = self.alpha * lossCE + (1 - self.alpha) * loss_KL

                batchLossList.append(loss.item())
                optimizer.zero_grad()
                loss.backward()

                batchLossList.append(loss.item())
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(lc.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                logger.info("Learning rate:")
                logger.info([group['lr'] for group in optimizer.param_groups])

            avrgLoss     = sum(batchLossList) / len(batchLossList)

            logger.info("Monitoring performance on test partition ...")
            if self.monitorTestPerform:
                acc, pre, rec, fsc = self.evaluateOnTestPartition(studentModel, studentLc, dataset, tokenizer)
            else:
                acc = -1
                pre = -1
                rec = -1
                fsc = -1
            logger.info(f"In-task fine-tuning epoch {epoch}, averLoss = {avrgLoss}, testPartAcc={acc}.")

            # log in wandb
            if self.wandb:
                if epoch % self.epochMonitorWindow == 0:
                    wandb.log({'avrgLoss': avrgLoss, \
                            'epoch': epoch, \
                            'testPartAcc': acc, \
                            })

        if self.wandb:
            run.finish()

        return studentModel, studentLc

    def getSoftTarget(self, model, lc, dataloader):
        dataDict = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'Y': [], 'logits': []}
        model.eval()
        with torch.no_grad():
            for batchID, batch in tqdm(enumerate(dataloader)):
                # batch data
                Y, ids, types, masks = batch
                X = {'input_ids':ids.to(model.device),
                        'token_type_ids':types.to(model.device),
                        'attention_mask':masks.to(model.device)}
                batchEmbedding = model.forwardEmbedding(X)
                logits = lc(batchEmbedding)

                dataDict['input_ids'].extend(ids.cpu())
                dataDict['token_type_ids'].extend(types.cpu())
                dataDict['attention_mask'].extend(masks.cpu())
                dataDict['Y'].extend(Y.cpu())
                dataDict['logits'].extend(logits.cpu())

        return dataDict
