# TCMPR: TCM Prescription recommendation based on subnetwork term mapping and deep learning

## 1. Introduction

This repository contains source code and datasets for paper ["TCMPR: TCM Prescription recommendation based on subnetwork 
term mapping and deep learning"](https://ieeexplore.ieee.org/document/9669588/). 

In this study, we proposed a subnetwork-based symptom term mapping method (SSTM), and 
constructed a SSTM-based TCM prescription recommendation method (termed TCMPR). Our SSTM can extract the subnetwork 
structure between symptoms from knowledge network to effectively represent the embedding features of clinical symptom 
terms (especially, the unrecorded terms). 

## 2. Overview
![alt text](img/Fig1.jpg "fig1")
Fig1: An overview of our methods. 
First, we constructed the HSKG and symptom network (a). 
Second, comprehensive embedding of patient symptoms was formed with the SSTM and symptom network (b). 
Finally, the patient’s comprehensive embedding vector was used for TCM prescription recommendation (c), 
the predicted probability of each herb is the output, so as to obtain the recommended prescription.

## 3. Install Python libraries needed
```bash
$ conda create -n tcmpr_env python=3.6
$ conda activate tcmpr_env
$ pip install -r requirements.txt
```

## 4. Basic usage
### (1) dataset
The relevant data required by the model are uniformly placed in the "data" folder. This folder contains the following three data files:
<li><b>Symptom_network.txt</b>: Symptom network constructed based on HSKG. The complete symptom network is not publicly available due to data privacy and other reasons. 
In order to facilitate readers to understand the specific process of this work, we provide 5000 relational data of symptom network as examples to demonstrate the code. 
<li><b>Symptom_Embedding_200.model</b>: The node embedding vector of the symptom network. The formation of this file is based on the above symptom network file and the DeepWalk algorithm is used for network embedding representation. 
The embedding features of all symptom nodes obtained are 200.
<li><b>input_example.xlsx</b>: Examples of experimental data. Due to data privacy and other reasons, the complete clinical case data cannot be made public. 
To give readers an idea of the input format for the model, we provide 100 pieces of data as example for running the program. 
Each data includes three elements: the patient's index number, the patient's symptoms (different symptoms are separated by semicolons), 
and the patient's herbs (different herbs are separated by semicolons).

### (2) run model
The python script file of the model is in the "model" folder, which is the <b>"TCMPR_model.py"</b> file. 
Readers can adjust the relevant parameters in the code according to their needs, and run the code file directly.
```bash
$ python TCMPR_model.py
```
Please see <b>requirements.txt</b> for the environment required for the model.\
If the dependency environment is correct and the parameters are set correctly, the "TCMPR_model.py" file can be run. 
In this model, the experimental data are randomly divided according to the ratio of 8:2 to obtain the training set and the test set. The model is trained using the training set data, and then the test set results are evaluated. 

### (3) result
After running the "TCMPR_model.py" file, the Top@K performance results of the model on the test set can be obtained. 
The result file is placed in the "result" folder, that is, the "Evaluation.xlsx" file. 
The result file contains four columns: 
<li>the "k" column represents the number of k in Top@k (k ranges from 1 to 20 in the results)</li>
<li>the "Precision" column represents the precision@k</li>
<li>the "Recall" column represents the recall@k</li> 
<li>the "F1_score" column represents the F1 score@k</li>

## 5. Citation
If you find TCMPR useful for your research, please consider citing the following paper:
```
@inproceedings{dong2021tcmpr,
  title={TCMPR: TCM Prescription recommendation based on subnetwork term mapping and deep learning},
  author={Dong, Xin and Zheng, Yi and Shu, Zixin and Chang, Kai and Yan, Dengying and Xia, Jianan and Zhu, Qiang and Zhong, Kunyu and Wang, Xinyan and Yang, Kuo and others},
  booktitle={2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={3776--3783},
  year={2021},
  organization={IEEE}
}
```
Dong X, Zheng Y, Shu Z, et al. TCMPR: TCM Prescription recommendation based on subnetwork term mapping and deep learning
[C]//2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). IEEE, 2021: 3776-3783.

<b>If you have better suggestions or questions about our work, please contact us: <a>xindong@bjtu.edu.cn</a></b>