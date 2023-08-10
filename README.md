# Intermediate Task Fine-Tuning in Cancer Classification

In this repository you will find the necessary information to reproduce the results of article "Intermediate Task Fine-Tuning in Cancer Classification".

The *train.py* file is a Python script that trains a ResNet-18 model with a certain configuration (C1 to C5) starting from an initial model (ImageNet or other).

Initial weights and settings can be easily changed on the first few lines.

To reproduce the experiments it is also necessary to split the datasets in the same way that we did. For PathoNet and DeepHisto we respect the original division that they have in their respective repositories. In the *hiutr_split.zip* file you can see the list of files for the HIUTR dataset.
