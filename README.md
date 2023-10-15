# CrossEncoder

A custom cross encoder used to predict diseases from a user inputted symptoms

# Data


[Data](https://github.com/chungimungi/CrossEncoder/blob/main/data/s2d.csv)

This dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease)

It contains two columns ```disease``` and ```symptoms```, the current dataset has 24 diseases listed with 50 symptoms for each in natural language.

# Requirements

Install all dependencies using ```pip install -r requirements.txt```

*Model trained on two GPUs*

# Model

The model is a custom cross encoder created by me to trainon the mentioned dataset. It has 522K parameters all of them being trainable. The model achieves an accuracy of 80.416% on training for 320 epochs on batch size of 512.
