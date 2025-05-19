import sys
import torch
from torch.utils import checkpoint
from sklearn.model_selection import train_test_split

from src.engine import YAMNetEngine, trivial_collate_fn
from src.model import YAMNet
from src.logger import YamnetLogger
import src.params as params
from src.data import ESC50Data

if __name__ == "__main__":
    # Usage: python3 example.py /path/to/dataset /path/to/log /path/to/checkpoint
    dataset_path = sys.argv[1]
    log_path = sys.argv[2]
    checkpoint_path = sys.argv[3]
    
    log = YamnetLogger(log_path)
    
    model = YAMNetEngine(
        model=YAMNet(),   
        # To use mbnv3 backbone: model=YAMNet(v3=True)
        tt_chunk_size=params.CHUNK_SIZE,
        logger=log,
    )
    
    dataset = ESC50Data(dataset_path)
    
    train_idxs, val_idxs = train_test_split(range(len(dataset)), test_size=0.2, random_state=9)
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idxs)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idxs)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=train_sampler, num_workers=1, collate_fn=trivial_collate_fn)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=val_sampler, num_workers=1, collate_fn=trivial_collate_fn)
    
    model.train_yamnet(
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=checkpoint_path,
        num_labels=len(dataset.get_labels()),
        num_epochs=50
    )
    