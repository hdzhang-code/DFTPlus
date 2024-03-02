from transformers import AutoTokenizer
import gc, os
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
import contractions
import string
from utils.Logger import logger
import pickle
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import pdb
from random import sample

##
# @brief  base class of DataAugmenter
class DataAugmenterBase():
    def __init__(self):
        self.roundN = 4
        pass

    def round(self, floatNum):
        return round(floatNum, self.roundN)

class DataAugmenter(DataAugmenterBase):
    def __init__(self, dataAugmenterParam):
        super(DataAugmenter, self).__init__()
        self.cachePath = dataAugmenterParam['cachePath']
        self.cacheFile = dataAugmenterParam['cacheFile']

    def dataAugGPT3(self, uttList, labList, augNum = 5, device=None):
        # check cache
        cacheFilePath = os.path.join(self.cachePath, self.cacheFile) 
        if os.path.exists(cacheFilePath):   # hit cache
            logger.info(f"Reading from cache: {cacheFilePath}")
            with (open(cacheFilePath, "rb")) as openfile:
                cache = pickle.load(openfile)
            labName2NewUtts = cache
        else:
            logger.info(f"Generating new data with GPT3 ...")
            from transformers import GPTJForCausalLM
            model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True, cache_dir='/data1/.cache/')

            model.to(device)

            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

            labName2Utts = {}
            for lab, utt in zip(labList, uttList):
                if lab not in labName2Utts:
                    labName2Utts[lab] = []
                labName2Utts[lab].append(utt)

            labName2NewUtts = {}
            keys = labName2Utts.keys()
            for lab in tqdm(keys):    # data augmentation for each label
                # compose prompt
                uttList = labName2Utts[lab]
                if True:
                    prompt = "\n".join([f"Example {i+1}: {utt}" for i, utt in enumerate(uttList)])
                    prompt = prompt + f"\nExample {len(uttList) + 1}: "
                    prompt = f"The following sentences belong to the same category '{lab}': \n" + prompt
                    preFixGen = None
                else:
                    prompt = "\n".join([f"{lab}: {utt}" for i, utt in enumerate(uttList)])
                    preFixGen = f"\n{lab}:"
                    prompt = prompt + preFixGen

                newData = self.gptj_complete(prompt, n = augNum, temp=1.0, model=model, tokenizer = tokenizer, top_k=0, top_p=1.0, device = device, preFixGen = preFixGen, oriUttList=uttList, oriLab = lab)
                labName2NewUtts[lab] = newData

            if not os.path.exists(self.cachePath):
                os.makedirs(self.cachePath)
            logger.info(f"Saving cache: {cacheFilePath}")
            with (open(cacheFilePath, "wb")) as openfile:
                pickle.dump(labName2NewUtts, openfile)

        newUttList = []
        newLabList = []
        for lab in labName2NewUtts:
            for utt in labName2NewUtts[lab]:
                newUttList.append(utt)
                newLabList.append(lab)
        return newUttList, newLabList

    def gptj_complete(self, prompt, n, temp, model, tokenizer, top_k, top_p, device, preFixGen=None, oriUttList=None, oriLab=None, queryUtt=None):
        k = len(prompt.splitlines()) - 1
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        stop_token = tokenizer.encode("\n")[0]
        sentences = []
        batchN = 10
        while len(sentences) != n:
            gen_tokens = model.generate(
                    input_ids,
                    do_sample=True,
                    max_length=2 * input_ids.shape[1],
                    temperature=temp,
                    eos_token_id=stop_token,
                    num_return_sequences=min(n, batchN),
                    pad_token_id=stop_token,
                    top_k=0 if not top_k else top_k,
                    top_p=1 if not top_p else top_p,
                    )
            generations = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            del gen_tokens
            for i in range(min(n, batchN)):
                if False:
                    s = generations[i].splitlines()[k:][0][len(f"Example {k}:") :].strip()
                else:
                    s = generations[i].splitlines()[k:][0][len(preFixGen) :].strip()
                s = self.cleanUpSentence(s)
                if s and len(sentences) < n and not self.filterCandidate4(s, oriUttList):
                    sentences.append(s)
                del s
            del generations
        del input_ids
        del model
        gc.collect()
        torch.cuda.empty_cache()
        # return [GPTJChoice(s) for s in sentences]
        return [s for s in sentences]

    def filterCandidate4(self, utt, uttList):
        if len(utt.split()) > 1 and utt not in uttList:
            return False
        else:
            return True

    def filterCandidate3(self, utt, uttList, oriLab, queryUtt=None):
        vocab = set()
        labSplit = queryUtt.split()
        for word in labSplit:
            vocab.add(word)

        vocabUtt = set()
        uttSplit = utt.split()
        for word in uttSplit:
            vocabUtt.add(word)
        rate = len(vocabUtt & vocab) / len(vocab)
        logger.info(rate)
        logger.info(f"utt: {utt}")
        logger.info(f"queryUtt: {queryUtt}")
        if rate > 0.33 and (utt is not queryUtt):
            return False
        else:
            return True

    def filterCandidate2(self, utt, uttList, oriLab):
        stopwordsEnglish = stopwords.words('english')
        if not hasattr(self, 'vocab'):
            vocab = set()
            labSplit = oriLab.split()
            for word in labSplit:
                if word not in stopwordsEnglish:
                    vocab.add(word)
            self.vocab = vocab
        vocabUtt = set()
        uttSplit = utt.split()
        for word in uttSplit:
            if word not in stopwordsEnglish:
                vocabUtt.add(word)
        if len(vocabUtt) == 0:
            for word in uttSplit:
                vocabUtt.add(word)
        if self.vocab in vocabUtt:
            return False
        else:
            return True

    def filterCandidate(self, utt, uttList, oriLab):
        stopwordsEnglish = stopwords.words('english')
        if not hasattr(self, 'vocab'):
            vocab = set()
            for utt in uttList:
                uttSplit = utt.split()
                for word in uttSplit:
                    if word not in stopwordsEnglish:
                        vocab.add(word)
            self.vocab = vocab
        vocabUtt = set()
        uttSplit = utt.split()
        for word in uttSplit:
            if word not in stopwordsEnglish:
                vocabUtt.add(word)
        if len(vocabUtt) == 0:
            for word in uttSplit:
                vocabUtt.add(word)
        if len(vocabUtt) ==0:
            return True
        rate = len(vocabUtt & self.vocab) / len(vocabUtt)
        logger.info(rate)
        # if rate >= 0.15:
        if rate > 0:
            return False
        else:
            return True


    def cleanUpSentence(self, sentence):
        # sentence: a string, like " Hello, do you like apple? I hate it!!  "

        # strip
        sentence = sentence.strip()

        # fix contractions
        sentence = contractions.fix(sentence)

        # lower case
        sentence = sentence.lower()

        # remove '_' and '-'
        sentence = sentence.replace('-',' ')
        sentence = sentence.replace('_',' ')

        # remove all punctuations
        sentence = ''.join(ch for ch in sentence if ch not in string.punctuation)

        return sentence

    def dataAugContextGPT3(self, uttListOri, labListOri, augNum = 5, device=None, returnTensorDataset = True, mergeOriData = True, oriTensorDataset=None, dataset = None, tokenizerOri=None):
        # check cache
        cacheFilePath = os.path.join(self.cachePath, self.cacheFile) 
        if os.path.exists(cacheFilePath):   # hit cache
            logger.info(f"Reading from cache: {cacheFilePath}")
            with (open(cacheFilePath, "rb")) as openfile:
                cache = pickle.load(openfile)
            labName2NewUtts = cache
            labName2NewUttsTemp = {}
            for key in labName2NewUtts:
                if key in labListOri:
                    labName2NewUttsTemp[key] = labName2NewUtts[key]
            labName2NewUtts = labName2NewUttsTemp
        else:
            # prepare GPT model
            logger.info(f"Generating new data with GPT3 ...")
            from transformers import GPTJForCausalLM
            model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True, cache_dir='/data1/.cache/')

            model.to(device)

            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

            labName2Utts = {}
            for lab, utt in zip(labListOri, uttListOri):
                if lab not in labName2Utts:
                    labName2Utts[lab] = []
                labName2Utts[lab].append(utt)

            labName2NewUtts = {}
            keys = labName2Utts.keys()
            for lab in tqdm(keys):    # data augmentation for each label
                uttList = labName2Utts[lab]
                if True:
                    prompt = "\n".join([f"Example {i+1}: {utt}" for i, utt in enumerate(uttList)])
                    prompt = prompt + f"\nExample {len(uttList) + 1}: "
                    prompt = f"The following sentences belong to the same category '{lab}': \n" + prompt
                    preFixGen = f"Example {len(uttList) + 1}:"
                else:
                    prompt = "\n".join([f"{lab}: {utt}" for i, utt in enumerate(uttList)])
                    preFixGen = f"\n{lab}:"
                    prompt = prompt + preFixGen

                newData = self.gptj_complete(prompt, n = augNum, temp=1.0, model=model, tokenizer = tokenizer, top_k=0, top_p=1.0, device = device, preFixGen = preFixGen, oriUttList=uttList, oriLab = lab)
                labName2NewUtts[lab] = newData

            if not os.path.exists(self.cachePath):
                os.makedirs(self.cachePath)
            logger.info(f"Saving cache: {cacheFilePath}")
            with (open(cacheFilePath, "wb")) as openfile:
                pickle.dump(labName2NewUtts, openfile)

        newUttList = []
        newLabList = []
        for lab in labName2NewUtts:
            if lab == 'general quirky':
                continue
            for utt in labName2NewUtts[lab]:
                newUttList.append(utt)
                newLabList.append(lab)

        if returnTensorDataset and mergeOriData:
            labName2LabIDDict = dataset.trainPart.getLabName2LabIDDict()
            newLabIDList = [labName2LabIDDict[labName] for labName in newLabList if labName in labName2LabIDDict]
            label = torch.cat([oriTensorDataset.tensors[0], torch.tensor(newLabIDList)])
            uttListAll = uttListOri + newUttList

            tokListAll = []
            for u in uttListAll:
                ut = tokenizerOri(u)
                if 'token_type_ids' not in ut:
                    ut['token_type_ids'] = [0]*len(ut['input_ids'])
                tokListAll.append(ut)
            paddedTokens = tokenizerOri.pad(tokListAll, padding='longest', return_tensors='pt')
            tensorDataset = TensorDataset(label,
                                     paddedTokens['input_ids'],
                                     paddedTokens['token_type_ids'],
                                     paddedTokens['attention_mask'])
            return tensorDataset, newUttList, newLabList
        else:
            return newUttList, newLabList
