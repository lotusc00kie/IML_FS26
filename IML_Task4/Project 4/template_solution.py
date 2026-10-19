# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:

import os
# We disable low-level log outputs by default to keep the terminal clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel

# Depending on your approach, you might need to adapt the structure of this template or parts not marked by TODOs.
# It is not necessary to completely follow this template. Feel free to add more code and delete any parts that
# are not required

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32  # TODO: Set the batch size according to both training performance and available memory
NUM_EPOCHS = 10  # TODO: Set the number of epochs
MAX_LEN = 256         
EMBEDDING_CACHE = "embeddings_cache.pt"   
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

print(f"Using device: {DEVICE}")

train_val = pd.read_csv("train.csv")
test_val = pd.read_csv("test_no_score.csv")

def combine_text(df):
    return (df["title"].fillna("") + " [SEP] " + df["sentence"].fillna("")).tolist()
 
train_texts  = combine_text(train_val)
train_labels = train_val["score"].tolist()
test_texts   = combine_text(test_val)
 
# Extract  DistilBERT embeddings 
 
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
 
def extract_embeddings(texts, bert_model, batch_size=64):
    """Return CLS-token embeddings (shape: N × 768) for a list of texts."""
    bert_model.eval()
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        input_ids      = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)
        with torch.no_grad():
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        # outputs.last_hidden_state: (batch, seq_len, 768)
        # CLS token is at position 0
        mask = attention_mask.unsqueeze(-1).float()
        cls_emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        all_embeddings.append(cls_emb.cpu())
    return torch.cat(all_embeddings, dim=0)   # (N, 768)
 
 
if os.path.exists(EMBEDDING_CACHE):
    print("Loading cached embeddings …")
    cache = torch.load(EMBEDDING_CACHE)
    train_embeddings = cache["train"]
    test_embeddings  = cache["test"]
else:
    print("Computing embeddings (this may take a while on CPU) …")
    bert_model = DistilBertModel.from_pretrained(MODEL_NAME).to(DEVICE)
    # Freeze all parameters 
    for param in bert_model.parameters():
        param.requires_grad = False
 
    train_embeddings = extract_embeddings(train_texts, bert_model)
    test_embeddings  = extract_embeddings(test_texts,  bert_model)
 
    torch.save({"train": train_embeddings, "test": test_embeddings}, EMBEDDING_CACHE)
    print(f"Embeddings cached to {EMBEDDING_CACHE}")
 
    # Free GPU memory
    del bert_model
    torch.cuda.empty_cache()
 
print(f"Train embeddings: {train_embeddings.shape}")
print(f"Test  embeddings: {test_embeddings.shape}")

# TODO: Fill out the SentimentDataset
class SentimentDataset(Dataset):
    def __init__(self, embeddings, labels=None):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, index):
        # TODO: Tokenize and return a {input_ids, attention_mask, labels} dictionary, or just {input_ids, attention_mask} for test data.
        item = {"embeddings": self.embeddings[index]}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


train_dataset = SentimentDataset(train_embeddings, train_labels)
test_dataset  = SentimentDataset(test_embeddings)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
			  num_workers=0,         # If you want to utilize multi-processing, set this to the number of your available cores!
			  pin_memory=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
			 num_workers=0,          # If you want to utilize multi-processing, set this to the number of your available cores!
			 pin_memory=True)

# Additional code if needed

# TODO: Fill out SentimentClassifier
class SentimentClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=2, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, embeddings):
        return self.classifier(embeddings)   # (batch, 2)

model = SentimentClassifier().to(DEVICE)

# TODO: Setup loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0
 
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", total=len(train_loader)):
        embeddings = batch["embeddings"].to(DEVICE)
        labels     = batch["labels"].to(DEVICE)
 
        optimizer.zero_grad()
        logits = model(embeddings)           # (batch, 2)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
 
        total_loss += loss.item() * labels.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
 
    scheduler.step()
    avg_loss = total_loss / total
    accuracy = correct / total
    print(f"  Loss: {avg_loss:.4f}  |  Train Acc: {accuracy:.4f}  |  LR: {scheduler.get_last_lr()[0]:.6f}")

model.eval()
results = []
 
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Inference", total=len(test_loader)):
        embeddings = batch["embeddings"].to(DEVICE)
        logits     = model(embeddings)
        preds      = logits.argmax(dim=1).cpu().numpy()
        results.append(preds)
 
predictions = np.concatenate(results)
 
with open("result.txt", "w") as f:
    for val in predictions:
        f.write(f"{val}\n")
 
print(f"\nDone! Wrote {len(predictions)} predictions to result.txt")