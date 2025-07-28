import torch
from torch.utils.data import DataLoader
from data import ParityDataset, TOKENS
from model import SimpleTransformerEncoder
import json
from tqdm import tqdm


def train():
    # Load config
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    # Set device (CPU only)
    device = torch.device('cpu')
    torch.manual_seed(config.get('seed', 42))

    # Prepare datasets and dataloaders
    train_dataset = ParityDataset(
        length=config['train_size'],
        seq_len=config['seq_len'],
        pre_generate=config['pre_generate'],
        seed=config.get('seed', 42)
    )
    val_dataset = ParityDataset(
        length=config['val_size'],
        seq_len=config['seq_len'],
        pre_generate=config['pre_generate'],
        seed=config.get('seed', 42)
    )
    train_generator = torch.Generator().manual_seed(config.get('seed', 42))
    test_generator = torch.Generator().manual_seed(config.get('seed', 42))
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, generator=train_generator)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, generator=test_generator)

    # Initialize model, optimizer, loss
    model = SimpleTransformerEncoder(
        num_layers=config['num_layers'],
        d_model=config['hidden_size'],
        num_tokens=len(TOKENS)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    n = 8  # Print eval every n steps
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        step = 0
        for input_ids, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [train]"):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * input_ids.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            step += 1

            if step % n == 0 or step == len(train_loader):
                # Compute running train stats
                running_train_loss = train_loss / train_total if train_total > 0 else 0.0
                running_train_acc = train_correct / train_total if train_total > 0 else 0.0

                # Validation
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for val_input_ids, val_labels in val_loader:
                        val_input_ids = val_input_ids.to(device)
                        val_labels = val_labels.to(device)
                        val_logits = model(val_input_ids)
                        v_loss = criterion(val_logits, val_labels)
                        val_loss += v_loss.item() * val_input_ids.size(0)
                        val_preds = val_logits.argmax(dim=1)
                        val_correct += (val_preds == val_labels).sum().item()
                        val_total += val_labels.size(0)
                val_loss /= val_total
                val_acc = val_correct / val_total

                print(f"Epoch {epoch+1} Step {step}: Train loss={running_train_loss:.4f}, acc={running_train_acc:.4f} | Val loss={val_loss:.4f}, acc={val_acc:.4f}")

                model.train()

if __name__ == "__main__":
    train() 
