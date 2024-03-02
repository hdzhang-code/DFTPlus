#coding=utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from transformers import AutoModelForMaskedLM
# from utils.transformers import AutoModelForMaskedLM
from utils.commonVar import *
from utils.Logger import logger

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
import pdb

class IntentBERT(nn.Module):
    def __init__(self, config):
        super(IntentBERT, self).__init__()
        self.device = config['device']
        self.LMName = config['LMName']
        self.clsNum = config['clsNumber']
        self.featureDim = 768

        self.linearClsfier = nn.Linear(self.featureDim, self.clsNum)
        self.dropout = nn.Dropout(0.1) # follow the default in bert model
        # self.word_embedding = nn.DataParallel(self.word_embedding)

        # load from Huggingface or from disk
        try:
            self.word_embedding = AutoModelForMaskedLM.from_pretrained(self.LMName)
        except:
            if 'loadModelPath' in config:
                modelPath = os.path.join(config['loadModelPath'], self.LMName)
            else:
                modelPath = os.path.join(SAVE_PATH, self.LMName)
            logger.info("Loading model from %s"%(modelPath))
            self.word_embedding = AutoModelForMaskedLM.from_pretrained(modelPath)

        self.word_embedding.to(self.device)
        self.linearClsfier.to(self.device)

    def loss_ce(self, logits, Y):
        loss = nn.CrossEntropyLoss()
        output = loss(logits, Y)
        return output


    def loss_mse(self, logits, Y):
        loss = nn.MSELoss()
        output = loss(torch.sigmoid(logits).squeeze(), Y)
        return output

    def loss_kl(self, logits, label):
        # KL-div loss
        probs = F.log_softmax(logits, dim=1)
        # label_probs = F.log_softmax(label, dim=1)
        loss = F.kl_div(probs, label, reduction='batchmean')
        return loss

    def getUttEmbeddings(self, X):
        # BERT forward
        outputs = self.word_embedding(**X, output_hidden_states=True)

        # extract [CLS] for utterance representation
        CLSEmbedding = outputs.hidden_states[-1][:,0]

        return CLSEmbedding


    def forwardEmbedding(self, X):
        # get utterances embeddings
        CLSEmbedding = self.getUttEmbeddings(X)

        return CLSEmbedding

    def mlmForward(self, X, Y):
        # BERT forward
        outputs = self.word_embedding(**X, labels=Y)

        return outputs.loss

    def forwardEmbeddingDataLoader(self, dataLoader: torch.utils.data.DataLoader, detach=True):
        labelList = []
        embeddingList = []
        for batchID, batch in tqdm(enumerate(dataLoader)):
            Y, ids, types, masks = batch
            X = {'input_ids':ids.to(self.device),
                    'token_type_ids':types.to(self.device),
                    'attention_mask':masks.to(self.device)}

            # forward
            embeddings = self.forwardEmbedding(X)
            if detach:
                embeddings = embeddings.detach()
            labelList.append(Y)
            embeddingList.append(embeddings)

        labelListCat     = torch.cat(labelList)
        embeddingListCat = torch.cat(embeddingList)

        return labelListCat, embeddingListCat
    
    def forward(self, X, returnEmbedding=False):
        # get utterances embeddings
        CLSEmbedding = self.getUttEmbeddings(X)

        # linear classifier
        logits = self.linearClsfier(CLSEmbedding)

        if returnEmbedding:
            return logits, CLSEmbedding
        else:
            return logits
    
    def fewShotPredict(self, supportX, supportY, queryX, clsFierName, mode='multi-class'):
        # calculate word embedding
        supportEmbedding = self.getUttEmbeddings(supportX)
        queryEmbedding   = self.getUttEmbeddings(queryX)

        # select clsfier
        support_features = supportEmbedding.cpu()
        query_features = queryEmbedding.cpu()
        clf = None
        if clsFierName == CLSFIER_LINEAR_REGRESSION:
            clf = LogisticRegression(penalty='l2',
                                     random_state=0,
                                     C=1.0,
                                     solver='lbfgs',
                                     max_iter=1000,
                                     multi_class='multinomial')
            # fit and predict
            clf.fit(support_features, supportY)
        elif clsFierName == CLSFIER_SVM:
            clf = make_pipeline(StandardScaler(), 
                                SVC(gamma='auto',C=1,
                                kernel='linear',
                                decision_function_shape='ovr'))
            # fit and predict
            clf.fit(support_features, supportY)
        elif clsFierName == CLSFIER_MULTI_LABEL:
            clf = MultiOutputClassifier(LogisticRegression(penalty='l2',
                                                           random_state=0,
                                                           C=1.0,
                                                           solver='liblinear',
                                                           max_iter=1000,
                                                           multi_class='ovr',
                                                           class_weight='balanced'))

            clf.fit(support_features, supportY)
        else:
            raise NotImplementedError("Not supported clasfier name %s", clsFierName)
        
        if mode == 'multi-class':
            query_pred = clf.predict(query_features)
        elif mode == 'entailment-utt':
            query_pred = clf.predict_proba(query_features)[:, 1]            
        elif mode == 'entailment-lab':
            query_pred = clf.predict_proba(query_features)[:, 1]

        return query_pred
    
    def reinit_clsfier(self):
        self.linearClsfier.weight.data.normal_(mean=0.0, std=0.02)
        self.linearClsfier.bias.data.zero_()
    
    def set_dropout_layer(self, dropout_rate):
        self.dropout = nn.Dropout(dropout_rate)
    
    def set_linear_layer(self, clsNum):
        self.linearClsfier = nn.Linear(768, clsNum)
    
    def normalize(self, x):
        norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
        out = x.div(norm)
        return out

    def NN(self, support, support_ys, query):
        """nearest classifier"""
        support = np.expand_dims(support.transpose(), 0)
        query = np.expand_dims(query, 2)

        diff = np.multiply(query - support, query - support)
        distance = diff.sum(1)
        min_idx = np.argmin(distance, axis=1)
        pred = [support_ys[idx] for idx in min_idx]
        return pred

    def CosineClsfier(self, support, support_ys, query):
        """Cosine classifier"""
        support_norm = np.linalg.norm(support, axis=1, keepdims=True)
        support = support / support_norm
        query_norm = np.linalg.norm(query, axis=1, keepdims=True)
        query = query / query_norm

        cosine_distance = query @ support.transpose()
        max_idx = np.argmax(cosine_distance, axis=1)
        pred = [support_ys[idx] for idx in max_idx]
        return pred

    def save(self, path):
        # pre-trained LM
        self.word_embedding.save_pretrained(path)


class LinearClsfier(nn.Module):
    def __init__(self, config):
        super(LinearClsfier, self).__init__()
        self.device = config['device']
        self.clsNum = config['clsNumber']
        self.initializeValue = config['initializeValue']
        if 'loadModelPath' in config:
            self.loadModelPath = config['loadModelPath']
        self.featureDim = 768

        self.linearClsfier = nn.Linear(self.featureDim, self.clsNum)
        if self.initializeValue is not None:
            self.linearClsfier.weight = torch.nn.Parameter(self.initializeValue)
            self.linearClsfier.bias   = torch.nn.Parameter(torch.zeros(self.linearClsfier.bias.shape))

        self.linearClsfier.to(self.device)

    def forward(self, embeddings):
        # linear classifier
        logits = self.linearClsfier(embeddings)
        return logits

    def save(self, path):
        lcParamPath = f"{path}/{SAVE_MODEL_LINEAR_CLASSIFER_PARAM_FILE}"
        torch.save(self.linearClsfier, lcParamPath)

    def loadFromDisk(self, path):
        if hasattr(self, "loadModelPath"):
            lcParamPath = os.path.join(self.loadModelPath, path, SAVE_MODEL_LINEAR_CLASSIFER_PARAM_FILE)
        else:
            lcParamPath = os.path.join(SAVE_PATH, path, SAVE_MODEL_LINEAR_CLASSIFER_PARAM_FILE)
        logger.info(f"Loading linear classifier from: {lcParamPath}")
        self.linearClsfier = torch.load(lcParamPath)
        self.linearClsfier.to(self.device)
