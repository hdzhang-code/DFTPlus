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
wandbProj=BERT_context_augmentation

# dataset
testDataName=(bank77 mcid HINT3 OOS hwu64_publishedPaper)

# fine-tune setting
shot=2
shotList=(5 10)
lrBackboneList=(2e-4)
lrClsfierList=(2e-5)
weightDecay=0.001
inTaskEpoch=100
MLMWeightList=(1.0)
augNumList=(50)

monitorTestPerform=

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

                            LMName=bert-base-uncased
                            logFolder=./log/
                            mkdir -p ${logFolder}
                            logFile=${logFolder}/fewShotFineTune_seed${seed}_testD${testData}_${shot}shot_${LMName}_MW${MLMWeight}_AuN${augNum}.log
                            if $debug
                            then
                                logFlie=${logFolder}/logDebug.log
                            fi

                            saveName=FewShotFineTuneModel_seed${seed}_testD${testData}_${shot}shot_LM${LMName}_MW${MLMWeight}_AuN${augNum}

                            export CUDA_VISIBLE_DEVICES=${cudaID}
                            python fewShotFineTune.py  \
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
                                | tee "${logFile}"
                            done
                        done
                    done
                done
            done
        done
    done
echo "Experiment finished."
