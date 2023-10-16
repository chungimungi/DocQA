# CrossEncoder

A custom cross encoder used to predict diseases from a user inputted symptoms

# Data

This [Data](https://github.com/chungimungi/CrossEncoder/blob/main/data/s2d.csv) was sourced from [Kaggle](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease)

It contains two columns ```disease``` and ```symptoms```, the current dataset has 24 diseases listed with 50 symptoms for each in natural language.

In [scripts/s2d CrossEncoder.py](https://github.com/chungimungi/CrossEncoder/blob/main/scripts/s2d%20CrossEncoder.py) comment the line ```model = nn.DataParallel(model)``` if multiple GPUs are unavailable for training.

# Requirements

Install all dependencies using ```pip install -r requirements.txt```

# Model

The model is a custom cross encoder created by me to train on the mentioned dataset.


## To Do
- [ ] Increase the number of diseases in the dataset
- [X] Improve model performance

# Changes

- Added multihead attention to the model improved accuracy by 7% using the same hyperparamters (80.41% to 87.01%)
    - Model now has 1.3M parameters (from 522K)

- Changed ```embed_dim``` to 1024 and ```hidden_dim``` to 512, while also changing ```batch_size``` to 256.
    - Model trained for 350 epochs this time and improved accuracy by another 7% (87.01% to 94.58%)
    - Model now has 13.2M parameters (from 1.3M)
