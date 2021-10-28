# Transferability_Black-Box_Attacks

This is the code of the paper: Zhao Ren, Jing Han, Nicholas Cummins, and Björn W. Schuller. "Enhancing Transferability of Black-Box Adversarial Attacks via Lifelong Learning for Speech Emotion Recognition Models." in Proc. INTERSPEECH, Shanghai, China, 2020, pp. 496-500. (http://www.interspeech2020.org/uploadfile/pdf/Mon-2-1-1.pdf)

The implementations of the three frameworks: multitask learning, transfer learning, and lifelong learning, are provided in this github repository.

## Folders 

demos-multitask/demos-cnn: the implementation of the multitask learning

demos-lifelong-transfer/demos-cnn: the implementation of the lifelong learning and transfer learning. 

Note: The difference between lifelong learning and transfer learning is using EWC loss function (Equation (3) in the paper). 

- If transfer learning, please remove the EWC loss function in the file demos-lifelong-transfer/demos-cnn/pytorch/main_pytorch.py 

- If lifelong learning, please keep the EWC loss function in the same file  

## Feature Extraction

For each framework, please run command: python utils/features.py

## Train and Evaluation

For each framework, please run command: sh runme.sh

## Cite

If the code is helpful for you, please feel free to cite it: 

Zhao Ren, Jing Han, Nicholas Cummins, and Björn W. Schuller. "Enhancing Transferability of Black-Box Adversarial Attacks via Lifelong Learning for Speech Emotion Recognition Models." in Proc. INTERSPEECH, Shanghai, China, 2020, pp. 496-500.



Zhao Ren

Chair of Embedded Intelligence for Health Care and Wellbeing

University of Augsburg, Germany

