# CrossEncoder

A custom cross encoder used to predict diseases from a user inputted symptoms

# Data


[Data](https://github.com/chungimungi/CrossEncoder/blob/main/data/s2d.csv)

This dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease)

It contains two columns ```disease``` and ```symptoms```, the current dataset has 24 diseases listed with 50 symptoms for each in natural language.

In [scripts](https://github.com/chungimungi/CrossEncoder/tree/main/scripts) there are two ```.py``` files that do that same but one utilizes multiple GPUs where as the other is for single GPU or CPU training.

# Requirements

Install all dependencies using ```pip install -r requirements.txt```

# Model

The model is a custom cross encoder created by me to train on the mentioned dataset.


## To Do
- [ ] Increase the number of diseases in the dataset
- [ ] Improve model performance

# Changes

- Added multihead attention to the model improved accuracy by 7% using the same hyperparamters (80.41 to 87.01)
    - Model now has 1.3M parameters (from 522K)
