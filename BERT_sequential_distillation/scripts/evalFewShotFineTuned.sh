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

# dataset
testDataName=(bank77 mcid HINT3 OOS hwu64_publishedPaper)
testDataName=(OOS)

# evaluation setting
shotList=(5 10)
testRepetition=1

# model initialization
tokenizer=bert-base-uncased

lrBackboneListName=(2e-4)
lrClsfierListName=(2e-5)
MLMWeightList=(1.0)
KDTempList=(100.0)
alphaList=(0.0)
KDIterList=(6)
seedModelName=(1 2 3 4 5)
augNumList=(50)

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
for testData in ${testDataName[@]}
do
    for shot in ${shotList[@]}
    do
        for seedName in ${seedModelName[@]}
        do
            for lrBackboneName in ${lrBackboneListName[@]}
            do
                for lrClsfierName in ${lrClsfierListName[@]}
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

                                        LMName=SelfDistill_LMFewShotFineTuneModel_seed${seedName}_testD${testData}_${shot}shot_LMbert-base-uncased_MW${MLMWeight}_AuN${augNum}_T${KDTemp}_alp${alpha}_it${KDIter}
                                        logFolder=./log/
                                        mkdir -p ${logFolder}
                                        logFile=${logFolder}/evalFewShotFT_${LMName}.log
                                        if $debug
                                        then
                                            logFlie=${logFolder}/logDebug.log
                                        fi

                                        export CUDA_VISIBLE_DEVICES=${cudaID}
                                        python evalFewShotFineTuned.py \
                                            --seed ${seed} \
                                            --testDataset ${testDataset} \
                                            --testDomain ${testDomain} \
                                            --tokenizer  ${tokenizer}   \
                                            --LMName ${LMName} \
                                            ${wandb}  \
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
