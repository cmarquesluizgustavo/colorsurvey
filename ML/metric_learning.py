import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from ML.data_prep import load_and_preprocess_data, ColorDataset
from ML.models.choice_models import BasicChoiceModel as ChoiceModel
from ML.metric_learning import EmbeddingModel
from ML.losses.vanilla_triplet_loss import VanillaTripletLoss


def evaluate(embedding_model, choice_model, dataloader, le):
    embedding_model.eval()
    choice_model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            z = embedding_model(batch_x)
            logits = choice_model(z)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.numpy())
            all_labels.extend(batch_y.numpy())
            
    print("\n" + "="*40)
    print("       EVALUATION REPORT")
    print("="*40)
    
    # Get target names from LabelEncoder, ensure they are strings
    target_names = [str(c) for c in le.classes_]
    
    # Calculate accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"Overall Accuracy: {acc:.4f}")
    
    # Detailed report
    # Note: If there are many classes, this might be long.
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))

def train():
    # Hyperparameterss
    EMBEDDING_DIM = 16
    LR_M = 0.01   # Learning rate for Classifier (M-Step)
    LR_E = 0.001  # Learning rate for Embedding (E-Step)
    LAMBDA = 0.5  # Balance between Metric Loss and CE Loss in E-Step
    CYCLES = 10
    EPOCHS_PER_STEP = 10
    BATCH_SIZE = 64
    TOP_N_COLORS = 329
    CSV_PATH = 'mainsurvey_data.csv'

    # Load Data
    X, y, le = load_and_preprocess_data(CSV_PATH, top_n=TOP_N_COLORS) 
    NUM_CLASSES = len(le.classes_)
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_dataset = ColorDataset(X_train, y_train)
    test_dataset = ColorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Models
    f_theta = EmbeddingModel(embedding_dim=EMBEDDING_DIM)
    g_phi = ChoiceModel(embedding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES)
    triplet_loss = VanillaTripletLoss(margin=0.5)

    # Optimizers (separate to easily freeze/unfreeze)
    opt_theta = optim.Adam(f_theta.parameters(), lr=LR_E)
    opt_phi = optim.Adam(g_phi.parameters(), lr=LR_M)

    print(f"Starting training loop for {CYCLES} cycles...")
    for cycle in range(CYCLES):
        print(f"\n--- Cycle {cycle+1}/{CYCLES} ---")
        
        # Step 1: M-Step (Optimize Classifier g_phi)
        print("M-Step: Training Classifier...")
        
        # Freeze f_theta, Unfreeze g_phi
        for p in f_theta.parameters(): p.requires_grad = False
        for p in g_phi.parameters(): p.requires_grad = True
        
        for epoch in range(EPOCHS_PER_STEP):
            total_loss_m = 0
            for batch_x, batch_y in train_loader:
                opt_phi.zero_grad()
                
                # Forward
                with torch.no_grad():
                    z = f_theta(batch_x) # Embeddings are fixed inputs here
                logits = g_phi(z)
                
                # Loss
                loss_ce = nn.CrossEntropyLoss()(logits, batch_y)
                loss_ce.backward()
                opt_phi.step()
                total_loss_m += loss_ce.item()
            
            print(f"  Epoch {epoch+1}/{EPOCHS_PER_STEP} - Loss: {total_loss_m / len(train_loader):.4f}")

        # Step 2: E-Step (Optimize Embedding f_theta)
        print("E-Step: Training Embedding...")
        
        # Unfreeze f_theta, Freeze g_phi
        for p in f_theta.parameters(): p.requires_grad = True
        for p in g_phi.parameters(): p.requires_grad = False
        
        for epoch in range(EPOCHS_PER_STEP):
            total_loss_e = 0
            metric_loss_sum = 0
            ce_loss_sum = 0
            
            for batch_x, batch_y in train_loader:
                opt_theta.zero_grad()
                
                # Forward
                z = f_theta(batch_x)
                
                # 1. Metric Loss (Triplet)
                loss_metric = triplet_loss(z, batch_y)
                
                # 2. CE Loss (Constraint to keep classifier happy)
                # Note: g_phi is frozen, but gradients flow back through z to f_theta
                logits = g_phi(z)
                loss_ce = nn.CrossEntropyLoss()(logits, batch_y)
                
                # Total Loss
                total_loss = loss_metric + (LAMBDA * loss_ce)
                
                total_loss.backward()
                opt_theta.step()
                
                total_loss_e += total_loss.item()
                metric_loss_sum += loss_metric.item()
                ce_loss_sum += loss_ce.item()
            
            avg_loss = total_loss_e / len(train_loader)
            avg_metric = metric_loss_sum / len(train_loader)
            avg_ce = ce_loss_sum / len(train_loader)
            print(f"  Epoch {epoch+1}/{EPOCHS_PER_STEP} - Total: {avg_loss:.4f} (Metric: {avg_metric:.4f}, CE: {avg_ce:.4f})")

    print("\nTraining Complete.")
    
    # Final Evaluation
    evaluate(f_theta, g_phi, test_loader, le)

if __name__ == "__main__":
    train()
