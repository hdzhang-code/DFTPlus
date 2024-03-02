#!/usr/bin/env bash

echo usage: 
echo scriptName.sh : run in normal mode
echo scriptName.sh debug : run in debug mode

# hardware
cudaID=$2

# debug mode
if [[ $# != 0 ]] && [[ $1 == "debug" ]]
then
    debug=true
else
    debug=false
fi

seed=1
seedList=(1 2 3 4 5)

# wandb
wandb=--wandb
wandbProj=BERT_sequential_distillation

# dataset
testDataName=(bank77 mcid HINT3 OOS hwu64_publishedPaper)
testDataName=(OOS)

# fine-tune setting
shot=2
shotList=(5 10)
lrBackboneList=(2e-4)
lrClsfierList=(2e-5)
weightDecay=0.001
inTaskEpoch=25
MLMWeightList=(1.0)
augNumList=(50)
KDTempList=(100.0)
alphaList=(0.0)
KDIterList=(6)

monitorTestPerform=

loadModelPath=../BERT_context_augmentation/saved_models

# model initialization
tokenizer=bert-base-uncased

epochMonitorWindow=2
saveModel=--saveModel

# modify arguments if it's debug mode
RED='\033[0;31m'
GRN='\033[0;32m'
NC='\033[0m' # No Color
if $debug
then
    echo -e "Run in ${RED} debug ${NC} mode."
    epochs=1
    wandb=
else
    echo -e "Run in ${GRN} normal ${NC} mode."
fi

echo "Start Experiment ..."
for seed in ${seedList[@]}
do
    for testData in ${testDataName[@]}
    do
        for lrBackbone in ${lrBackboneList[@]}
        do
            for lrClsfier in ${lrClsfierList[@]}
            do
                for shot in ${shotList[@]}
                do
                    for MLMWeight in ${MLMWeightList[@]}
                    do
                        for augNum in ${augNumList[@]}
                        do
                            for KDTemp in ${KDTempList[@]}
                            do
                                for alpha in ${alphaList[@]}
                                do
                                    for KDIter in ${KDIterList[@]}
                                    do
                                        case ${testData} in
                                            bank77)
                                                testDataset=bank77
                                                testDomain="BANKING"
                                                ;;
                                            mcid)
                                                testDataset=mcid
                                                testDomain="MEDICAL"
                                                ;;
                                            HINT3)
                                                testDataset=HINT3
                                                testDomain='curekart,powerplay11,sofmattress'
                                                ;;
                                            OOS)
                                                testDataset=OOS
                                                testDomain='travel,kitchen_dining'
                                                ;;
                                            hwu64_publishedPaper)
                                                testDataset=hwu64_publishedPaper
                                                testDomain='play,lists,recommendation,iot,general,transport,weather,social,email,music,qa,takeaway,audio,news,datetime,calendar,cooking,alarm'
                                                ;;
                                            *)
                                                echo Invalid testData ${testData}
                                        esac

                                        LMName=FewShotFineTuneModel_seed${seed}_testD${testData}_${shot}shot_LMbert-base-uncased_MW${MLMWeight}_AuN${augNum}
                                        logFolder=./log/
                                        mkdir -p ${logFolder}
                                        logFile=${logFolder}/SelfDistill_${LMName}_T${KDTemp}_alp${alpha}.log
                                        if $debug
                                        then
                                            logFlie=${logFolder}/logDebug.log
                                        fi

                                        saveName=SelfDistill_LM${LMName}_T${KDTemp}_alp${alpha}_it${KDIter}

                                        export CUDA_VISIBLE_DEVICES=${cudaID}
                                        python selfDistill.py  \
                                            --seed ${seed} \
                                            --testDataset ${testDataset} \
                                            --testDomain ${testDomain} \
                                            --tokenizer  ${tokenizer}   \
                                            --shot ${shot}  \
                                            --LMName ${LMName} \
                                            --lrBackbone  ${lrBackbone}  \
                                            --lrClsfier   ${lrClsfier}  \
                                            --weightDecay  ${weightDecay} \
                                            --inTaskEpoch  ${inTaskEpoch}  \
                                            ${wandb}  \
                                            --wandbProj  ${wandbProj}  \
                                            ${saveModel} \
                                            --saveName ${saveName} \
                                            --epochMonitorWindow   ${epochMonitorWindow}  \
                                            ${monitorTestPerform}  \
                                            --MLMWeight   ${MLMWeight}  \
                                            --augNum     ${augNum}  \
                                            --loadModelPath   ${loadModelPath}  \
                                            --KDTemp  ${KDTemp} \
                                            --KDIter  ${KDIter}  \
                                            --alpha   ${alpha}  \
                                            | tee "${logFile}"
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
echo "Experiment finished."
