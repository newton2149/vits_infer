from torch.utils.data import Dataset
import torch
from text import text_to_sequence
from text.symbols import symbols
from text import text_to_sequence
import commons

from serverutils import get_text,get_text_fr,get_text_rw,get_text_vctk

switch = {
    "en": get_text,
    "fr": get_text_fr,
    "rw": get_text_rw,
    "vctk": get_text_vctk
}

class CustomData(Dataset):
    def __init__(self, lines, hps,lang):
        self.lines = lines
        self.hps = hps
        self.lang = lang

        
    def get_text_batch(self, text, hps,lang):
        text_norm = switch[lang](text, hps)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        max_length = len(max(self.lines, key=len))+250
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
        stn_tst = self.get_text_batch(line, self.hps,self.lang)  
        return sid, stn_tst

