import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import torch.cuda

torch.cuda.set_device(0)

#Load and preprocess the data
data = pd.read_csv("/kaggle/input/sym2dis/s2d.csv")

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

# Define your custom model architecture
class CustomCrossEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2, dropout_prob=0.5):
        super(CustomCrossEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(2 * hidden_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, input):
        embedded = self.embedding(input)
        output, (hidden, _) = self.lstm(embedded)
        hidden_concat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        x = self.fc1(hidden_concat)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


# Hyperparameters
vocab_size = len(symptom_vocab)
embed_dim = 100
hidden_dim = 128
num_classes = len(disease_vocab)
num_epochs = 300
batch_size = 512
learning_rate = 0.001

# Define loss function and optimizer
model = CustomCrossEncoder(vocab_size, embed_dim, hidden_dim, num_classes)
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
        inputs = torch.stack(inputs)
        labels = torch.cat(labels)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader)}")

# Evaluation
model.eval()
test_dataset = CustomDataset(X_test, y_test,max_seq_length)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_dataloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels.squeeze()).sum().item()

print(f"Test Accuracy: {100 * correct / total}%")
