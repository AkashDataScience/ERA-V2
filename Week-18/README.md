# Assignment
1. Your assignment is to write the train.py file and optimize training of transformer model.
2. Train the model such that it reaches a loss of 1.8 or less in 18 epochs. 

# Introduction
The goal of this assignment is to implement optimization strategies to speed up transformers training.

## Training log
    Max length of the source sentence before processing: 309
    Max length of the source target before processing: 274
    Removing sentences based on ratio between lenght of en and it sentences
    Spliting long sent sentences
    Sorting by length
    Combining inputs
    Creating random batches
    Max length of the source sentence : 102
    Max length of the source target : 145
    0
    Processing Epoch 00: 100%|██████████| 928/928 [03:00<00:00,  5.14it/s, loss=6.755]
    1
    Processing Epoch 01: 100%|██████████| 928/928 [02:59<00:00,  5.16it/s, loss=5.650]
    2
    Processing Epoch 02: 100%|██████████| 928/928 [03:01<00:00,  5.10it/s, loss=5.025]
    3
    Processing Epoch 03: 100%|██████████| 928/928 [03:01<00:00,  5.10it/s, loss=4.486]
    4
    Processing Epoch 04: 100%|██████████| 928/928 [03:06<00:00,  4.99it/s, loss=3.926]
    5
    Processing Epoch 05: 100%|██████████| 928/928 [03:05<00:00,  5.00it/s, loss=3.387]
    6
    Processing Epoch 06: 100%|██████████| 928/928 [03:01<00:00,  5.10it/s, loss=2.877]
    7
    Processing Epoch 07: 100%|██████████| 928/928 [03:00<00:00,  5.13it/s, loss=2.512]
    8
    Processing Epoch 08: 100%|██████████| 928/928 [02:58<00:00,  5.20it/s, loss=2.262]
    9
    Processing Epoch 09: 100%|██████████| 928/928 [02:59<00:00,  5.16it/s, loss=2.122]
    10
    Processing Epoch 10: 100%|██████████| 928/928 [03:02<00:00,  5.08it/s, loss=2.011]
    11
    Processing Epoch 11: 100%|██████████| 928/928 [03:00<00:00,  5.14it/s, loss=1.908]
    12
    Processing Epoch 12: 100%|██████████| 928/928 [02:58<00:00,  5.19it/s, loss=1.867]
    13
    Processing Epoch 13: 100%|██████████| 928/928 [02:58<00:00,  5.19it/s, loss=1.754]
    14
    Processing Epoch 14: 100%|██████████| 928/928 [02:58<00:00,  5.19it/s, loss=1.693]
    15
    Processing Epoch 15: 100%|██████████| 928/928 [02:58<00:00,  5.20it/s, loss=1.683]
    16
    Processing Epoch 16: 100%|██████████| 928/928 [02:58<00:00,  5.18it/s, loss=1.633]
    17
    Processing Epoch 17: 100%|██████████| 928/928 [02:59<00:00,  5.16it/s, loss=1.632]

## Metrics
Train loss: 1.632

## Acknowledgments
This model is trained using repo listed below
* [language translation optimization](https://github.com/AkashDataScience/language_translation_optimization)