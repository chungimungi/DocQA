import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np

# Set the device to use two GPUs if available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load and preprocess the data
data = pd.read_csv("s2d.csv")

# Tokenization and creating vocabulary
symptoms = data['symptoms'].tolist()
diseases = data['disease'].tolist()
symptom_vocab = set(" ".join(symptoms).split())
disease_vocab = set(diseases)
symptom2id = {symptom: idx for idx, symptom in enumerate(symptom_vocab)}
disease2id = {disease: idx for idx, disease in enumerate(disease_vocab)}
id2disease = {idx: disease for disease, idx in disease2id.items()}

# Convert text to numerical data
X = [[symptom2id[word] for word in symptom.split()] for symptom in symptoms]
y = [disease2id[disease] for disease in diseases]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a custom dataloader
class CustomDataset(Dataset):
    """Custom PyTorch dataset for handling input sequences and labels."""
    def __init__(self, X, y, max_seq_length):
        self.X = X
        self.y = y
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        input_seq = self.X[index]
        label = self.y[index]

        # Pad input sequences to a fixed length
        padded_input_seq = np.zeros(self.max_seq_length, dtype=np.int64)
        padded_input_seq[:len(input_seq)] = input_seq

        return torch.LongTensor(padded_input_seq), torch.LongTensor([label])

# Define the model architecture
class CustomCrossEncoder(nn.Module):
    """Custom PyTorch model for a cross-encoder with LSTM and attention."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout_prob=0.5):
        super(CustomCrossEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(2 * hidden_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, input):
        embedded = self.embedding(input)
        output, (hidden, _) = self.lstm(embedded)

        # Apply attention
        attended_output = self.attention(output, output, output)

        hidden_concat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        x = self.fc1(hidden_concat)
        x = torch.relu(x)
        x = self.dropout(x)

        output = self.fc2(x)

        return output

# Hyperparameters
vocab_size = len(symptom_vocab)
embed_dim = 1024
hidden_dim = 512
num_classes = len(disease_vocab)
num_epochs = 350
batch_size = 256
learning_rate = 0.001

# Define loss function and optimizer
model = CustomCrossEncoder(vocab_size, embed_dim, hidden_dim, num_classes).to(device)
model = nn.DataParallel(model)  # Comment this line of code if multiple GPUs are not available
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

max_seq_length = 1000

train_dataset = CustomDataset(X_train, y_train, max_seq_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: zip(*batch))

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        inputs = torch.stack(inputs).to(device)
        labels = torch.cat(labels).to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader)}")

# Evaluation
model.eval()
test_dataset = CustomDataset(X_test, y_test, max_seq_length)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels.squeeze()).sum().item()

print(f"Test Accuracy: {100 * correct / total}%")

# Inference function
def predict_disease_from_input():
    """Predict disease from user input."""
    input_text = input("Enter symptoms : ")
    input_text = input_text.split()
    input_ids = [symptom2id[word] for word in input_text]
    input_tensor = torch.LongTensor(input_ids).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_disease = id2disease[predicted.item()]
    print(f"Predicted Disease: {predicted_disease}")

# Usage: predict disease from user input
predict_disease_from_input()

import torchinfo

# Display model summary
torchinfo.summary(model.cuda())
