import torch
from torchinfo import summary
from tqdm import tqdm
from datetime import datetime

from src.model import YAMNet
import src.params as params
from src.feature_extraction import WaveformToMelSpec

def trivial_collate_fn(inputs):
    return inputs

class YAMNetEngine(torch.nn.Module):
    def __init__(self, model: YAMNet, tt_chunk_size, logger):
        super().__init__()

        self.model = model
        self.tt_chunk_size = tt_chunk_size
        self.logger = logger
    
        if torch.cuda.is_available():
            logger.info("Using CUDA")
            self.device = torch.device("cuda")
            logger.info(torch.cuda.get_device_name(torch.cuda.current_device()))
        elif torch.backends.mps.is_available():
            logger.info("Using Metal")
            self.device = torch.device("mps")
        else:
            logger.info("Using CPU")
            self.device = torch.device("cpu")
        self.to(self.device)
        self.model.to(self.device)
        
        self.waveform_transform = WaveformToMelSpec(self.device)
        
    def forward(self, inputs):
        overall_preds = []
        for payload in inputs:
            data, sr = payload["data"], payload["sr"]
            data = data.to(self.device)
            sr = torch.tensor(sr, device=self.device)
            mel_spectro, _ = self.waveform_transform(data, sr)
            chunks = mel_spectro.split(self.tt_chunk_size, dim=0)
            accuracies = []
            for chunk in chunks:
                mask = (chunk != 0).any(dim=(1, 2, 3))
                chunk = chunk[mask]
                
                pred = self.model(chunk)
                accuracies.append(pred.cpu())
            overall_preds.append(accuracies)
        return overall_preds
    
    def train_yamnet(self, train_loader, val_loader, checkpoint_path, num_labels, num_epochs):
        start = datetime.now()
        self.logger.info("Started training")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params.LEARNING_RATE)
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = len(train_loader.sampler.indices)
            
            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
            
            for batch in train_iterator:
                outputs = self.forward(batch)[0][0]
                preds = torch.mean(outputs, dim=0)
                actual_label = batch[0]["label"]
                
                one_hot_label = [0 for _ in range(num_labels)]
                one_hot_label[actual_label] = 1
                loss = criterion(preds, torch.tensor(one_hot_label).float())
                
                _, predicted = torch.max(preds, dim=0)
                correct += (predicted == actual_label)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_iterator.set_postfix(loss=loss.item())
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = correct / total
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            self.model.eval()
            
            val_loss = 0.0
            correct = 0
            total = len(val_loader.sampler.indices)
            num_outputs = 1
            
            val_iterator = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]")
            
            with torch.no_grad():
                for batch in val_iterator:
                    outputs = self.forward(batch)[0][0]
                    preds = torch.mean(outputs, dim=0)
                    actual_label = batch[0]["label"]
                    
                    one_hot_label = [0 for _ in range(num_labels)]
                    one_hot_label[actual_label] = 1
                    loss = criterion(preds, torch.tensor(one_hot_label).float())
                    
                    _, predicted = torch.max(preds, dim=0)
                    correct += (predicted == actual_label)
                    
                    val_loss += loss.item()
                    val_iterator.set_postfix(loss=loss.item())
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = correct / total
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            torch.save(self.model.state_dict(), checkpoint_path)
        
        end = datetime.now()
        duration = end - start
        self.logger.info(f"Runtime: {duration}")
        