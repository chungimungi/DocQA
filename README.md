![image](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

# CrossEncoder

A custom cross encoder QA model that gives a diagnosis based on the input of symptoms a user describes in natural language.

# Data

This [Data](https://github.com/chungimungi/CrossEncoder/blob/main/data/s2d.csv) was sourced from [Kaggle](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease)

It contains two columns ```disease``` and ```symptoms```, the current dataset has 24 diseases listed with 50 symptoms for each in natural language.

In [scripts/s2d CrossEncoder](https://github.com/chungimungi/CrossEncoder/blob/main/scripts/s2d%20CrossEncoder.py) comment the line ```model = nn.DataParallel(model)``` if multiple GPUs are unavailable for training.

# Requirements

Install all dependencies using ```pip install -r requirements.txt```

# Model

| Layer (type:depth-idx)                                   | Param #         |
|----------------------------------------------------------|-----------------|
| DataParallel                                            | --              |
| └─CustomCrossEncoder: 1-1                               | --              |
|    └─Embedding: 2-1                                     | 2,568,192       |
|    └─LSTM: 2-2                                          | 6,299,648       |
|    └─MultiheadAttention: 2-3                            | 3,148,800       |
|    │    └─NonDynamicallyQuantizableLinear: 3-1          | 1,049,600       |
|    └─Dropout: 2-4                                       | --              |
|    └─Linear: 2-5                                        | 131,200         |
|    └─Linear: 2-6                                        | 3,096           |

Total params: 13,200,536      
Trainable params: 13,200,536      
Non-trainable params: 0              



# Changes

- Added multihead attention to the model improved accuracy by 7% using the same hyperparamters (80.41% to 87.01%)
    - Model now has 1.3M parameters (from 522K)

- Changed ```embed_dim``` to 1024 and ```hidden_dim``` to 512, while also changing ```batch_size``` to 256.
    - Model trained for 350 epochs this time and improved accuracy by another 7% (87.01% to 94.58%)
    - Model now has 13.2M parameters (from 1.3M)
 



**NOTE: This project is for demo purposes only. For any symptoms/disease, please refer to a Doctor.**
