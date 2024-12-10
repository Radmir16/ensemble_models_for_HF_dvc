# An ensemble of Hugginface models for the classification of melanoma images ISIC2019 using DVC

![!\[dvc\]()](source/dvc_git.png) 
![alt text](source/image.png)

This repository presents a pipeline for classifying images of skin melanomas using an ensemble of models with HuggingFace

## DVC (Data Version Control) 
[DVC](https://dvc.org/) (Data Version Control) is a dataset versioning system and not only, which is an add-on to git. If you can work with git, congratulations, you can work with DVC. In addition, DVC allows you to log experiments, as well as make Auto-ML.

At this stage, the tree of our project looks like this:
```
├── data
├── dvceval
├── dvclive
├── dvcpreproc
├── dvctrain
├── models
├── scr
    └── eval.py
    ├── markdownwriter.py
    ├── pipeline.ipynb
    ├── preprocess.py
    ├── train.py
    ├── trainmodel.py
├── dvc.yaml
├── params.yaml
├── README.md
```
The dvc.yaml file represents the experiment pipeline.

>!
Before conducting the experiment, it is necessary to upload the ISIC 2019 dataset and .csv file to the data/raw directory
The command to download linux ```wget https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip``` ```wget https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv```

To start the experiment, run the command in the console:
```
dvc exp run -n name_exp
```