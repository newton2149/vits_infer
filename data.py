from torch.utils.data import Dataset
import torch
from text import text_to_sequence
from text.symbols import symbols
from text import text_to_sequence



class CustomData(Dataset):
    def __init__(self, lines, hps):
        self.lines = lines
        self.hps = hps

    def get_text(self, text, hps):
        text_norm = text_to_sequence(text, hps.data.text_cleaners)
        return torch.LongTensor(text_norm)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip() 
        sid = f"SID_{idx}"  
        stn_tst = self.get_text(line, self.hps)  
        return sid, stn_tst

# # Usage example
# file_path = "path_to_your_text_file.txt"
# hps = {}  # Your hyperparameters

# my_custom_dataset = MyDataset(file_path, hps)

