# Experiments
1. Context augmentation.
    1) cd BERT_context_augmentation
    2) do context augmentation, generate contextual relevant data. This repo already contains generated contextual data in the folder dataAugCache, so the following procedures are optional.
            ./scripts/genContext.sh n 0
    3) Fine-tune PLM with the few data and augmented contextual data.
            ./scripts/fewShotFineTune.sh n 0
2. Sequential knowledge distillation.
    1) cd BERT_sequential_distillation
    2) do sequential knowledge distillation
            ./scripts/selfDistill.sh n 0
    3) evaluate distilled model
            ./scripts/evalFewShotFineTuned.sh n 0

# Environment:
We provide environment in folder environment. There are two .yml file. The following file is used to do context augmentation because GPT-J requires updated version of transformers:
    conda-env_transformer4.20.yml
