# LPUF-AuthNet: A Lightweight PUF-Based IoT Authentication via Tandem Neural Networks and Split Learning
This code is used in a manuscript submission under the title: "LPUF-AuthNet: A Lightweight PUF-Based IoT Authentication via Tandem Neural Networks & Split Learning".
## Overview
This paper explores the challenges of securing IoT devices, particularly in authentication processes. The widespread adoption of the Internet of Things (IoT) has revolutionized various aspects of daily life through advanced connectivity, automation, and intelligent decision-making. However, traditional cryptographic methods often struggle with the constraints of IoT devices, such as limited computational power and storage.

We investigate physical unclonable functions (PUFs) as robust security solutions, utilizing their inherent physical uniqueness to authenticate devices securely. Our research addresses the limitations of traditional PUF systems, which are vulnerable to machine learning attacks and burdened by large datasets.  We propose a novel lightweight PUF authentication scheme termed  <code style="color : black">**LPUF-AuthNet**</code>, which comprises two ML models: <code style="color : black">**deep neural networks (DNN)**</code>   and <code style="color : black">**tandem neural networks (TNN)**</code> trained using <code style="color : black">**split learning (SL)</code> paradigm. The proposed architecure reduces storage and communication demands, provides mutual authentication, enhances security by resisting different types of attacks, and supports scalable authentication, paving the way for secure integration into future 6G technologies.



<p align="center">
  <div style="display: flex; justify-content: space-around; align-items: center; padding: 0 5%;">
    <div style="text-align: center; margin-right: 10px;">
      <img src="Images/Enrollement_Phase.png" width="100%">
      <p><strong>Figure 1:</strong> Enrollment Phase</p>
    </div>
    <div style="text-align: center;">
      <img src="Images/Authentication_Phase.png" width="100%">
      <p><strong>Figure 2:</strong> Authentication Phase</p>
    </div>
  </div>
</p>



## Repository Contents 
- <code style="color : black">TrainingTNN.py:</code> Main script for training the LPUF-AuthNet models, including the autoencoders, the deep neural networks, and the tandem neural networks.


- <code style="color : black">LPUF-AuthNet-Models.py:</code> A script that contains the definitions of the LPUF-AuthNet models. Specifically, it defines the deep neural networks, autoencoders, and tandem neural networks.


- <code style="color : black">CRP_FPGA_01 - Copy.csv:</code> A subset of the CRPs dataset utilized in our paper, comprising 10% of the total dataset.
 

- <code style="color : black">GeneratorDataset.csv</code>: A dataset containing binary text that indexes each CRP in the file <code style="color : black">CRP_FPGA_01 - Copy.csv</code>. This file is used to train the CRP generator (DNN).


- <code style="color: yellow">DNN.h5</code>: This is the DNN model responsible for generating novel CRPs. The input is a hexadecimal number written in binary format, similar to the dataset, and the model generates the corresponding CRP.

- <code style="color: yellow">**best_model.pth**</code>: This model contains the $Encoder_1$, $Encoder_2$, and $Decoder_2$. It is the result of training as illustrated in the figure below (Phase A).

- <code style="color: yellow">best_model2.pth</code>: This model contains the $Enhanced_Encoder_1$, $Enhanced_Encoder_2$, and $Decoder_1$. It is the result of training as illustrated in the figure below (Phase B).



<p align="center">
  <div style="display: flex; justify-content: center; align-items: center; padding: 0 5%;">
    <div style="text-align: center;">
      <img src="Images/Training Architecture.png" width="40%">
      <p><strong>Figure 2:</strong> Authentication Phase</p>
    </div>
  </div>
</p>

## Requirements
To run the code in this repository, you need the following:

- Python 3.7 or higher
- Torch
- Numpy
- Pandas
- Matplotlib
- Scikit-learn

## Installation 
1. Clone the repository:

```
git clone https://github.com/yourusername/LPUF-AuthNet.git
cd LPUF-AuthNet
```
or donwload the repository from the fellowing link: 

```
https://github.com/BrahiM-Mefgouda/LPUF-AuthNet/archive/refs/heads/main.zip
```

2. Install the required packages using pip:
```
pip install -r requirements.txt
```

## Usage

If you have a dataset of CRPs you should replace it  

1. Train the models by running the script <code style="color : LightSkyBlue">LPUF-AuthNet-Models.py.csv:</code>


## Copyright and license


## Citation

