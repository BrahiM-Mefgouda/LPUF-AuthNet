# LPUF-AuthNet: A novel Lightweight PUF-Based Tandem Neural Networks, AutoEncoders, and DNN for IoT Device Authentication
This code is used in a manuscript submission under the title: "LPUF-AuthNet: A novel Lightweight PUF-Based Tandem Neural Networks, AutoEncoders, and Deep Neural Networks for IoT Device Authentication".
## Overview
This paper explores the challenges of securing IoT devices, particularly in authentication processes. The widespread adoption of the Internet of Things (IoT) has revolutionized various aspects of daily life through advanced connectivity, automation, and intelligent decision-making. However, traditional cryptographic methods often struggle with the constraints of IoT devices, such as limited computational power and storage.

We investigate physical unclonable functions (PUFs) as robust security solutions, utilizing their inherent physical uniqueness to authenticate devices securely. Our research addresses the limitations of traditional PUF systems, which are vulnerable to machine learning attacks and burdened by large datasets. We propose a novel lightweight PUF mechanism, called 
<code style="color : black">**LPUF-AuthNet**</code>, which combines <code style="color : black">**tandem neural networks**</code>, <code style="color : black">**Deep neural networks**</code>, and <code style="color : black">**autoencoders**</code> within a <code style="color : black">**split learning framework**</code>. The proposed architecure reduces storage and communication demands, provides mutual authentication, enhances security by resisting different types of attacks, and supports scalable authentication, paving the way for secure integration into future 6G technologies.



<div style="display: flex; justify-content: center; flex-direction: column; align-items: center;">
  <img src="https://github.com/user-attachments/assets/7fed8979-055b-4589-bb5f-e6dca86458fc" alt="One" style="width: 80%;">
  <figcaption style="text-align: center;"><strong>Figure 1:</strong> Enrollment phase.</figcaption>
</div>



![Two](https://github.com/user-attachments/assets/3c23273f-babb-4bd5-99bd-0f801da83a2a)
**Figure 2:** Authentification phase.



## Repository Contents 
- <code style="color : black">LPUF-AuthNet-Training.py:</code> Main script for training the LPUF-AuthNet models, including the autoencoders, the deep neural networks, and the tandem neural networks.
  
- <code style="color : black">LPUF-AuthNet-Models.py:</code> A script that contains the definitions of the LPUF-AuthNet models. Specifically, it defines the deep neural networks, autoencoders, and tandem neural networks.

- <code style="color : black">CRP_FPGA_01 - Copy.csv:</code> A subset of the CRPs dataset utilized in our paper, comprising 10% of the total dataset.
 

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
1. Train the models by running the script <code style="color : LightSkyBlue">LPUF-AuthNet-Models.py.csv:</code>


## Copyright and license


## Citation
