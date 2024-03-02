import transformers
from utils.IntentDataset import IntentDataset
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
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
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

class FewShotTuneTrainer(TrainerBase):
    def __init__(self, trainingParam, dataset: IntentDataset):
        super(FewShotTuneTrainer, self).__init__(trainingParam["wandb"], trainingParam["wandbProj"], trainingParam["wandbConfig"], trainingParam["wandbRunName"])
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

        self.epochMonitorWindow = trainingParam['epochMonitorWindow']

    def train(self, model, tokenizer, dataset, dataloader, unlabeledDataset, logLevel='DEBUG'):
        logger.info("Initializing linear classifier ...")
        lc = self.initLinearClassifierRandom(model, dataset)

        logger.info("Fine-tuning the model with K-shot data ...")
        self.fineTuneKshot(model, lc, dataset, dataloader, unlabeledDataset, tokenizer)

        return model, lc

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

    def fineTuneKshot(self, model, lc, dataset, dataloader, unlabTensorData, tokenizer):
        if self.wandb:
            run = wandb.init(project=self.wandbProjName, reinit=True)
            wandb.config.update(self.wandbConfig)
            wandbRunName = self.runName
            wandb.run.name=(wandbRunName)

        paramList = [{'params': model.parameters(), 'lr': self.lrBackbone}, \
                {'params': lc.parameters(), 'lr': self.lrClsfier}]
        optimizer = optim.AdamW(paramList, weight_decay=self.weight_decay)
        t_total = len(dataloader) * self.inTaskEpoch
        warmup_steps = round(t_total/20)
        logger.info(f"Learning rate scheduler: warmup_steps={warmup_steps}, t_total={t_total}.")
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

        if self.mlm:
            unlabeledloader = DataLoader(unlabTensorData, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            unlabelediter = iter(unlabeledloader)

        for epoch in range(self.inTaskEpoch):
            model.train()
            lc.train()

            batchLossList = []
            batchCEList = []
            batchSDList = []
            trainAccList = []
            batchMLMList = []
            for batchID, batch in enumerate(dataloader):
                Y, ids, types, masks = batch
                X = {'input_ids':ids.to(model.device),
                        'token_type_ids':types.to(model.device),
                        'attention_mask':masks.to(model.device)}
                batchEmbedding = model.forwardEmbedding(X)
                logits = lc(batchEmbedding)

                lossCE = model.loss_ce(logits, Y.to(model.device))

                if self.mlm:   # mlm
                    try:
                        _, ids, types, masks = unlabelediter.next()
                    except StopIteration:
                        unlabelediter = iter(unlabeledloader)
                        _, ids, types, masks = unlabelediter.next()
                    X_un = {'input_ids':ids.to(model.device),
                            'token_type_ids':types.to(model.device),
                            'attention_mask':masks.to(model.device)}
                    mask_ids, mask_lb = mask_tokens(X_un['input_ids'].cpu(), tokenizer)
                    X_un = {'input_ids':mask_ids.to(model.device),
                            'token_type_ids':X_un['token_type_ids'],
                            'attention_mask':X_un['attention_mask']}
                    lossMLM = model.mlmForward(X_un, mask_lb.to(model.device))

                loss = lossCE + self.MLMWeight * lossMLM

                batchLossList.append(loss.item())
                batchCEList.append(lossCE.item())
                batchMLMList.append(lossMLM.item())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(lc.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                logger.info("Learning rate:")
                logger.info([group['lr'] for group in optimizer.param_groups])

                # calculate train acc
                YTensor = Y.cpu()
                logits = logits.detach().clone()
                if torch.cuda.is_available():
                    logits = logits.cpu()
                logits = logits.numpy()
                predictResult = np.argmax(logits, 1)
                trainAcc = accuracy_score(YTensor, predictResult)
                trainAccList.append(trainAcc)

            avrgLoss     = sum(batchLossList) / len(batchLossList)
            avrgLossCE   = sum(batchCEList) / len(batchCEList)
            avrgLossMLM = sum(batchMLMList) / len(batchMLMList)
            avrgTrainAcc = sum(trainAccList) / len(trainAccList)

            logger.info("Monitoring performance on test partition ...")
            if self.monitorTestPerform:
                acc, pre, rec, fsc = self.evaluateOnTestPartition(model, lc, dataset, tokenizer)
            else:
                acc = -1
                pre = -1
                rec = -1
                fsc = -1
            logger.info(f"In-task fine-tuning epoch {epoch}, averLoss = {avrgLoss}, avrgLossCE={avrgLossCE}, avrgLossMLM={avrgLossMLM}, tranAcc={avrgTrainAcc}, testPartAcc={acc}.")

            # log in wandb
            if self.wandb:
                if epoch % self.epochMonitorWindow == 0:
                    wandb.log({'avrgLoss': avrgLoss, \
                            'avrgLossCE': avrgLossCE, \
                            'avrgLossMLM': avrgLossMLM, \
                            'avrgTrainAcc': avrgTrainAcc, \
                            'epoch': epoch, \
                            'testPartAcc': acc, \
                            })

        if self.wandb:
            run.finish()

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
        acc = accuracy_score(YTruthList, predResultList)   # acc
        performDetail = precision_recall_fscore_support(YTruthList, predResultList, average='macro', warn_for=tuple())

        return acc, performDetail[0], performDetail[1], performDetail[2]
