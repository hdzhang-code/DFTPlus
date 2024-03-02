#coding=utf-8
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from utils.commonVar import *
import json
from utils.Logger import logger
import os
import copy
import random
import pdb
import copy as cp
import hashlib
import torch
from torch.utils.data import TensorDataset

class IntentDataset():
    def __init__(self, dataSetPath, mode):
        self.datasetPath = dataSetPath
        self.name = dataSetPath.split('/')[-1]
        if mode == DATASET_MODE_ALL:  # read all data into train partition
            dataFilePath = os.path.join(DATA_PATH, dataSetPath, FILE_NAME_DATASET) 
            self.trainPart = self.readDatasetFile(dataFilePath)
            self.valPart  = None
            self.testPart = None
        elif mode in [DATASET_MODE_PARTITION, DATASET_MODE_PARTITION_MERGE_INTO_TEST]: # read different partition separately. All partitions should share the same label names 2 label ind mapping.
            # train
            dataFilePathTrain = os.path.join(DATA_PATH, dataSetPath, FILE_NAME_DATASET_TRAIN) 
            self.trainPart = self.readDatasetFile(dataFilePathTrain)
            labName2LabIDDictTrain = self.trainPart.getLabName2LabIDDict()
            # dev
            dataFilePathDev = os.path.join(DATA_PATH, dataSetPath, FILE_NAME_DATASET_DEV) 
            self.valPart = self.readDatasetFile(dataFilePathDev, labName2LabIDDictTrain)
            # test
            dataFilePathTest = os.path.join(DATA_PATH, dataSetPath, FILE_NAME_DATASET_TEST) 
            self.testPart = self.readDatasetFile(dataFilePathTest, labName2LabIDDictTrain)

            # sanity check: we should have the same labelName-labelID mapping
            labName2LabIDSigTrain = self.trainPart.getLabName2LabIDMapSignature() if self.trainPart else None
            labName2LabIDSigVal   = self.valPart.getLabName2LabIDMapSignature() if self.valPart else None
            labName2LabIDSigTest  = self.testPart.getLabName2LabIDMapSignature() if self.testPart else None
            logger.info(f"LabName2LabID signature, train:      {labName2LabIDSigTrain if labName2LabIDSigTrain else None}.")
            logger.info(f"LabName2LabID signature, validation: {labName2LabIDSigVal if labName2LabIDSigVal else None}.")
            logger.info(f"LabName2LabID signature, test:       {labName2LabIDSigTest if labName2LabIDSigTest else None}.")
            if not len(set([sig for sig in [labName2LabIDSigTrain, labName2LabIDSigVal, labName2LabIDSigTest] if sig is not None])) == 1:
                logger.error(f"Inconsistent label ID for training, validation and testing.")
                exit(1)

            if mode == DATASET_MODE_PARTITION_MERGE_INTO_TEST:  # merge dev and test into test
                partition = Partition()
                partition.absorb(self.valPart)
                partition.absorb(self.testPart)
                valSize  = len(self.valPart.uttList)  if self.valPart is not None else 0
                testSize = len(self.testPart.uttList) if self.testPart is not None else 0
                logger.info(f'Partition absorb: {valSize} + {testSize} -> {len(partition.uttList)}.')
                self.testPart = partition
        else:
            logger.error(f"Invalid mode {mode}.")

    def getLabNum(self):
        return self.trainPart.getLabNum()

    def readDatasetFile(self, filePath, labName2labIDDict=None):
        if os.path.exists(filePath):
            partition = Partition()
            partition.loadDataset(filePath, labName2labIDDict)
            partition.setName(f"{self.name}-{filePath.split('/')[-1]}")
            logger.info(f"Read from {filePath}, data: {len(partition.uttList)}")
        else:
            logger.info(f"File does not exist: {filePath}.")
            partition = None
        return partition


    ##
    # @brief for each partition, select data according to domain name
    #
    # @param list domain names to keep
    #
    # @return 
    def selectByDomains(self, domainNames: list):
        newIntentDataset = cp.deepcopy(self)
        newIntentDataset.trainPart = newIntentDataset.trainPart.splitDomain(domainNames) if not newIntentDataset.trainPart == None else None
        newIntentDataset.valPart   = newIntentDataset.valPart.splitDomain(domainNames) if not newIntentDataset.valPart == None else None
        newIntentDataset.testPart  = newIntentDataset.testPart.splitDomain(domainNames) if not newIntentDataset.testPart == None else None
        return newIntentDataset

    def tokenize(self, tokenizer):
        if self.trainPart is not None:
            self.trainPart.tokenize(tokenizer) 
        if self.valPart is not None:
            self.valPart.tokenize(tokenizer)
        if self.testPart is not None:
            self.testPart.tokenize(tokenizer)

##
# @ brief a partition of a dataset, train, dev, or test.
class Partition():
    def __init__(self,
                 domList=None,
                 labList=None,
                 uttList=None,
                 tokList=None,
                 ):
        self.name = None
        self.domList = None if domList is None else domList
        self.labList = None if labList is None else labList
        self.uttList = None if uttList is None else uttList
        self.tokList = None if tokList is None else tokList

        if (self.labList is not None):
            self.createLabID()
        self.labID2DataInd = None
        self.dataInd2LabID = None

        self.labName2Count = None

        self.labName2LabIDDict = None

        self.tensorDataset = None

    def absorb(self, part):
        if part == None:
            return

        if self.domList == None:
            self.domList = []
        if self.labList == None:
            self.labList = []
        if self.uttList == None:
            self.uttList = []
        self.domList.extend(part.domList)
        self.labList.extend(part.labList)
        self.uttList.extend(part.uttList)
        if part.tokList is not None:
            self.tokList.extend(part.tokList)

        self.labID2DataInd = None
        self.dataInd2LabID = None

        self.labName2Count = None

        self.labName2LabIDDict = None

        self.tensorDataset = None

        if (self.labList is not None):
            self.createLabID()

    ##
    # @brief converte self data to a torch dataset and return
    #
    # @return 
    def generateTorchDataset(self, tokenizer):
        if self.tensorDataset is not None:
            return self.tensorDataset
        paddedTokens = tokenizer.pad(self.tokList, padding='longest', return_tensors='pt')
        label = torch.tensor(self.labIDList)
        tensorDataset = TensorDataset(label,
                                 paddedTokens['input_ids'],
                                 paddedTokens['token_type_ids'],
                                 paddedTokens['attention_mask'])
        self.tensorDataset = tensorDataset
        return self.tensorDataset

    ##
    # @brief randomly sample k data for each class, return a partition
    #
    # @param k
    #
    # @return  a torch TensorDataset to train or evaluate
    def randomSliceTorchDataset(self, k, tokenizer):
        # sample k data for each label
        labID2DataInds = self.getLabID2dataInd()
        selectDataInds = []
        for labID in labID2DataInds:
            dataInds = cp.deepcopy(labID2DataInds[labID])
            random.shuffle(dataInds)
            selectDataInds.extend(dataInds[:k])

        # collect data to return 
        labIDList = [self.labIDList[i] for i in selectDataInds]
        tokList = [self.tokList[i] for i in selectDataInds]
        uttList = [self.uttList[i] for i in selectDataInds]
        labList = [self.labList[i] for i in selectDataInds]

        paddedTokens = tokenizer.pad(tokList, padding='longest', return_tensors='pt')
        label = torch.tensor(labIDList)
        tensorDataset = TensorDataset(label,
                                 paddedTokens['input_ids'],
                                 paddedTokens['token_type_ids'],
                                 paddedTokens['attention_mask'])
        return tensorDataset, uttList, labList

    def getLabName2LabIDMapSignature(self):
        labName2LabIDDict = self.getLabName2LabIDDict()
        strCode = json.dumps(labName2LabIDDict, sort_keys=True).encode('utf-8')
        sign = hashlib.md5(strCode).hexdigest()
        return sign

    def getMD5(self):
        dictionary = {}
        dictionary[KEY_DOM_LIST] = self.domList
        dictionary[KEY_LAB_LIST] = self.labList
        dictionary[KEY_UTT_LIST] = self.uttList

        strCode = json.dumps(dictionary, sort_keys=False).encode('utf-8')
        sign = hashlib.md5(strCode).hexdigest()
        return sign

    def setName(self, name):
        self.name = name

    def getDomList(self):
        return self.domList

    def getLabList(self):
        return self.labList
    
    def getUttList(self):
        return self.uttList
    
    def getTokList(self):
        return self.tokList

    def getAllData(self):
        return self.domList, self.labList, self.uttList, self.tokList
    
    def getLabNum(self):
        labSet = set()
        for lab in self.labList:
            labSet.add(lab)
        return len(labSet)
    
    def getLabID(self):
        return self.labIDList

    def checkData(self, utt: str, label: str):
        if len(label) == 0 or len(utt) == 0:
            logger.warning("Illegal label %s or utterance %s, 0 length", label, utt)
            return 1
        else:
            return 0

    def loadDataset(self, dataFilePath, labName2labIDDict=None):
        self.domList, self.labList, self.uttList = [], [], []
        
        dataList = []
        with open(dataFilePath, 'r') as json_file:
            dataList.append(json.load(json_file))

        delDataNum = 0
        for data in dataList:
            for datasetName in data:
                dataset = data[datasetName]
                for domainName in dataset:
                    domain = dataset[domainName]
                    for dataItem in domain:
                        utt = dataItem[0]
                        labList = dataItem[1]

                        lab = labList[0]
                        
                        if not self.checkData(utt, lab) == 0:
                            logger.warning("Illegal label %s or utterance %s, too short length", lab, utt)
                            delDataNum = delDataNum+1
                        else:
                            self.domList.append(domainName)
                            self.labList.append(lab)
                            self.uttList.append(utt)

        # report deleted data number 
        if (delDataNum>0):
            logger.warning("%d data is deleted from dataset.", delDataNum)

        # sanity check
        countSet = set()
        countSet.add(len(self.domList))
        countSet.add(len(self.labList))
        countSet.add(len(self.uttList))
        if len(countSet) > 1:
            logger.error("Unaligned data list. Length of data list: dataset %d, domain %d, lab %d, utterance %d", len(self.domainList), len(self.labList), len(self.uttList))
            exit(1)
        self.createLabID(labName2labIDDict)
        return 0

    def removeStopWord(self):
        raise NotImplementedError 

        # print info
        logger.info("Removing stop words ...")
        logger.info("Before removing stop words: data count is %d", len(self.uttList))

        # remove stop word
        stopwordsEnglish = stopwords.words('english')
        uttListNew = []
        labListNew = []
        delLabListNew = []
        delUttListNew = []  # Utt for utterance
        maxLen = -1
        for lab, utt in zip(self.labList, self.uttList):
            uttWordListNew = [w for w in utt.split(' ') if not word in stopwordsEnglish]
            uttNew = ' '.join(uttWordListNew)

            uttNewLen = len(uttWordListNew)
            if uttNewLen <= 0:   # too short utterance, delete it from dataset
                delLabListNew.append(lab)
                delUttListNew.append(uttNew)
            else:   # utt with normal length
                if uttNewLen > maxLen:
                    maxLen = uttNewLen
                labListNew.append(lab)
                uttListNew.append(uttNew)
        self.labList = labListNew
        self.uttListNew = uttListNew
        self.delLabList.append(delLabListNew)
        self.delUttList.append(delUttListNew)

        # update data list
        logger.info("After removing stop words: data count is %d", len(self.uttList))
        logger.info("Removing stop words ... done.")
      
        return 0

    def splitDomain(self, domainName: list,  metaTaskInfo = None):
        domList = self.getDomList()

        # collect index
        indList = []
        for ind, domain in enumerate(domList):
            if domain in domainName:
                indList.append(ind)

        # remove label with too few data
        if metaTaskInfo:
            shot = metaTaskInfo['shot']
            query = metaTaskInfo['query']

            deleteIndList = []
            deleteLabSet = set()
            for ind in indList:
                labName = self.mapDataID2Label(ind)
                count = self.getDataCountFromLabName(labName)
                
                if count < (shot + query):
                    deleteLabSet.add(labName)
                    deleteIndList.append(ind)

            if len(deleteLabSet) > 0:
                logger.info("Following labels are deleted due to too few count:")
                logger.info(deleteLabSet)
                logger.info("Data to be removed due to too few count:")
                logger.info(len(deleteIndList))

                for delInd in deleteIndList:
                    indList.remove(delInd)
            else:
                logger.info("No label is deleted due to too few count.")

        # sanity check
        dataCount = len(indList)
        if dataCount<1:
            logger.error("Empty data for domain %s", domainName)
            exit(1)
        
        logger.info(f"In partition {self.name}, for domain {domainName}, {dataCount} data is selected from {len(domList)} data.")
        
        # get all data from dataset
        domList, labList, uttList, tokList = self.getAllData()
        domDomList = [domList[i] for i in indList]
        domLabList = [labList[i] for i in indList]
        domUttList = [uttList[i] for i in indList]
        if self.tokList:
            domTokList = [tokList[i] for i in indList]
        else:
            domTokList = []

        domDataset = Partition(domDomList, domLabList, domUttList, domTokList)

        return domDataset
    
    def tokenize(self, tokenizer):
        self.tokList = []
        for u in self.uttList:
            ut = tokenizer(u)
            if 'token_type_ids' not in ut:
                ut['token_type_ids'] = [0]*len(ut['input_ids'])
            self.tokList.append(ut)
    
    def shuffle_words(self):
        newList = []
        for u in self.uttList:
            replace = copy.deepcopy(u)
            replace = replace.split(' ')
            random.shuffle(replace)
            replace = ' '.join(replace)
            newList.append(replace)
        self.uttList = newList

    def getLabName2LabIDDict(self):
        if self.labName2LabIDDict:
            return self.labName2LabIDDict
        self.labName2LabIDDict = {}
        for lab, labID in zip(self.labList, self.labIDList):
            if lab not in self.labName2LabIDDict:
                self.labName2LabIDDict[lab] = labID
        return self.labName2LabIDDict
    
    # convert label names to label IDs: 0, 1, 2, 3
    def createLabID(self, labName2labIDDict = None):
        if labName2labIDDict is not None:   # if input a map from lab name to labID, then use it
            self.labIDList = []
            for labName in self.labList:
                self.labIDList.append(labName2labIDDict[labName])
        else:   # automatically assign lab ID to label names
            # get unique label
            labSet = set()
            for lab in self.labList:
                labSet.add(lab)
            
            # get number
            self.labNum = len(labSet)
            sortedLabList = list(labSet)
            sortedLabList.sort()    # sort label according to names to remove software uncertainty.

            # fill up dict: lab -> labID
            self.name2LabID = {}
            for ind, lab in enumerate(sortedLabList):
                if not lab in self.name2LabID:
                    self.name2LabID[lab] = ind

            # fill up label ID list
            self.labIDList =[]
            for lab in self.labList:
                self.labIDList.append(self.name2LabID[lab])

        # sanity check
        if not len(self.labIDList) == len(self.uttList):
            logger.error("create labID error. Not consistence labe ID list length and utterance list length.")
            exit(1)

    def mapDataID2Label(self, ind):
        return self.labList[ind]

    def getDataCountFromLabName(self, labName):
        if self.labName2Count == None:
            self.labName2Count = {}
            for lab in self.labList:
                if lab not in self.labName2Count:
                    self.labName2Count[lab] = 0
                
                self.labName2Count[lab] = self.labName2Count[lab] + 1

        return self.labName2Count[labName]
        
    def getLabID2dataInd(self):
        if not self.labID2DataInd == None:
            return self.labID2DataInd
        else:
            self.labID2DataInd = {}
            for dataInd, labID in enumerate(self.labIDList):
                if not labID in self.labID2DataInd:
                    self.labID2DataInd[labID] = []
                self.labID2DataInd[labID].append(dataInd)
            
            # sanity check
            dataCount = 0
            for labID in self.labID2DataInd:
                dataCount = dataCount + len(self.labID2DataInd[labID])
            if not dataCount == len(self.uttList):
                logger.error("Inconsistent data count %d and %d when generating dict, labID2DataInd", dataCount, len(self.uttList))
                exit(1)

            return self.labID2DataInd
    
    def getDataInd2labID(self):
        if not self.dataInd2LabID == None:
            return self.dataInd2LabID
        else:
            self.dataInd2LabID = {}
            for dataInd, labID in enumerate(self.labIDList):
                self.dataInd2LabID[dataInd] = labID
        return self.dataInd2LabID
