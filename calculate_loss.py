import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import json
import os
import time
from transformers import AutoProcessor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.models.qwen_slm import QwenMedVQA

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "data"
EPOCHS_DIR = os.path.join(BASE_DIR, "Epochs")
os.makedirs(EPOCHS_DIR, exist_ok=True)

# SAVE RESULTS TO JSON
def save_loss_to_json(epoch, train_loss, val_loss, train_acc, val_acc):
    loss_data = {
        "epoch": epoch,
        "training_loss": round(float(train_loss), 6),
        "validation_loss": round(float(val_loss), 6),
        "training_accuracy": round(float(train_acc), 4),
        "validation_accuracy": round(float(val_acc), 4),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    file_path = os.path.join(EPOCHS_DIR, f"epoch_{epoch}.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(loss_data, f, indent=4)
    print(f"Saved Epoch {epoch} on: {file_path}")

# LOAD DATASET (STREAMING MODE)
def get_streaming_dataloaders(batch_size=8):
    print("Connecting Streaming with VQA-RAD and PathVQA...")
    
    # Load VQA-RAD (Streaming)
    vqa_rad = load_dataset("flaviagiammarino/vqa-rad", streaming=True)
    # Load PathVQA (Streaming)
    path_vqa = load_dataset("flaviagiammarino/path-vqa", streaming=True)
    
    train_data = path_vqa['train']
    val_data = path_vqa['validation']
    
    train_data = train_data.shuffle(seed=42, buffer_size=1000)
    
    def preprocess_batch(batch):
        """Preprocess batch for VQA training"""
        # Convert images to proper format and extract text labels
        processed_images = []
        processed_questions = []
        processed_labels = []
        
        for i in range(len(batch['image'])):
            # Handle image preprocessing - Qwen2-VL expects specific format
            image = batch['image'][i]
            question = batch['question'][i]
            answer = str(batch['answer'][i]).lower()
            
            processed_images.append(image)
            processed_questions.append(question)
            
            # Convert answer to numerical label (simplified for loss calculation)
            # This is a placeholder - you might want to implement proper answer tokenization
            if answer in ['yes', 'no']:
                label = 1 if answer == 'yes' else 0
            else:
                # For open-ended answers, use hash as placeholder (you should improve this)
                label = hash(answer) % 1000  # Simple placeholder
            
            processed_labels.append(label)
        
        return {
            'image': processed_images,
            'question': processed_questions,
            'label': torch.tensor(processed_labels, dtype=torch.long)
        }
    
    # Custom collate function to handle preprocessing
    def collate_fn(batch):
        # Convert list of dicts to dict of lists
        batch_dict = {}
        for key in batch[0].keys():
            batch_dict[key] = [item[key] for item in batch]
        return preprocess_batch(batch_dict)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_fn)
    
    return train_loader, val_loader

# FOR LOOP CALCULATE LOSS 
def monitor_streaming_loss(model, criterion, optimizer, num_epochs=10):
    train_loader, val_loader = get_streaming_dataloaders(batch_size=2) # Reduce batch_size for memory
    
    # Note: QwenMedVQA handles device placement internally
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n=== Starting Epoch {epoch} ===")
        
        # --- PHASE: TRAINING ---
        model.model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        steps_train = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Preprocess inputs for Qwen2-VL
                images = batch['image']
                questions = batch['question']
                labels = batch['label'].to(DEVICE)
                
                # Format inputs for Qwen2-VL
                texts = []
                for question in questions:
                    texts.append(f"user\n<|vision_start|><|image_pad|><|vision_end|>{question}\nassistant\n")
                
                # Process inputs
                inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model.model(**inputs)
                logits = outputs.logits
                
                # Get the relevant logits for answer prediction
                # This is simplified - you might need to adjust based on your specific task
                batch_size = logits.shape[0]
                seq_len = logits.shape[1]
                vocab_size = logits.shape[2]
                
                # Use the last token's logits for prediction (simplified approach)
                last_token_logits = logits[:, -1, :]
                
                # Calculate loss
                loss = criterion(last_token_logits, labels)
                
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                steps_train += 1
                
                # Calculate accuracy
                _, predicted = torch.max(last_token_logits, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                
                # Print progress every 10 steps
                if batch_idx % 10 == 0:
                    current_acc = 100 * correct_train / total_train if total_train > 0 else 0
                    print(f"  Training Step {batch_idx}: Loss = {loss.item():.4f}, Acc = {current_acc:.2f}%")
                
                # Limit steps for testing (remove for full training)
                if steps_train >= 50: 
                    break
                    
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue

        if steps_train > 0:
            avg_train_loss = total_train_loss / steps_train
            train_accuracy = 100 * correct_train / total_train if total_train > 0 else 0
        else:
            avg_train_loss = 0.0
            train_accuracy = 0.0

        # --- PHASE: VALIDATION ---
        model.model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        steps_val = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    images = batch['image']
                    questions = batch['question']
                    labels = batch['label'].to(DEVICE)
                    
                    # Format inputs for Qwen2-VL
                    texts = []
                    for question in questions:
                        texts.append(f"user\n<|vision_start|><|image_pad|><|vision_end|>{question}\nassistant\n")
                    
                    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                    
                    outputs = model.model(**inputs)
                    logits = outputs.logits
                    last_token_logits = logits[:, -1, :]
                    
                    v_loss = criterion(last_token_logits, labels)
                    
                    total_val_loss += v_loss.item()
                    steps_val += 1
                    
                    # Calculate validation accuracy
                    _, predicted = torch.max(last_token_logits, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
                    
                    # Limit validation steps
                    if steps_val >= 20: 
                        break
                        
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    continue

        if steps_val > 0:
            avg_val_loss = total_val_loss / steps_val
            val_accuracy = 100 * correct_val / total_val if total_val > 0 else 0
        else:
            avg_val_loss = 0.0
            val_accuracy = 0.0
        
        # SAVE RESULT
        save_loss_to_json(epoch, avg_train_loss, avg_val_loss, train_accuracy, val_accuracy)
        print(f"Epoch {epoch} completed | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}%")
        print(f"Training steps: {steps_train} | Validation steps: {steps_val}")

# RUN FILE
if __name__ == "__main__":
    print("=== Initializing Qwen2-VL Model for Loss Calculation ===")
    
    # Initialize model with same configuration as main_federated.py
    model = QwenMedVQA(use_4bit=True)
    model.model = prepare_model_for_kbit_training(model.model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    model.model = get_peft_model(model.model, lora_config)
    
    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-4)
    
    # Start training
    monitor_streaming_loss(model, criterion, optimizer, num_epochs=50)