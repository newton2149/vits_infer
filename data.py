from torch.utils.data import Dataset
import torch
from text import text_to_sequence
from text.symbols import symbols
from text import text_to_sequence
import commons

class CustomData(Dataset):
    def __init__(self, lines, hps):
        self.lines = lines
        self.hps = hps

        
    def get_text_batch(self, text, hps):
        text_norm = text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        max_length = 30  
        if len(text_norm) < max_length:
            text_norm += [0] * (max_length - len(text_norm))
        elif len(text_norm) > max_length:
            text_norm = text_norm[:max_length]

        text_norm = torch.LongTensor(text_norm)
        return text_norm


    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip() 
        sid = f"SID_{idx}"  
        stn_tst = self.get_text_batch(line, self.hps)  
        return sid, stn_tst

