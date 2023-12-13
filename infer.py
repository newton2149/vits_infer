import os
import torch
import commons
import utils
import argparse
import zipfile
import re

from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io.wavfile import write
import soundfile as sf

'''
Sample Command Line Arguments

python3 inference.py --txt_file "./predict.txt" --device "gpu" --model "./logs/mlb_ezwa/G_*.pth" 
'''

## Inference



def get_lines(input_dir):
    file = open(input_dir, 'r')
    Lines = file.readlines()
    return Lines

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def main_cpu(args, line_number):
    hps = utils.get_hparams_from_file(args.config)

    net_g = SynthesizerTrn(
        58,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)
    _ = net_g.eval()

    _ = utils.load_checkpoint(args.model, net_g, None)

    stn_tst = get_text(args.text, hps)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        sid = torch.LongTensor([102])
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=args.noise_scale, noise_scale_w=args.noise_scale_w, length_scale=args.length_scale)[0][0, 0].data.cpu().float().numpy()
        output_filename = f"{args.output_directory}/generated_{line_number}.wav"
        sf.write(output_filename, audio, hps.data.sampling_rate, 'PCM_24')

def main_gpu(args, line_number):
    hps = utils.get_hparams_from_file(args.config)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()

    _ = utils.load_checkpoint(args.model, net_g, None)

    stn_tst = get_text(args.text, hps)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=args.noise_scale, noise_scale_w=args.noise_scale_w, length_scale=args.length_scale)[0][0, 0].data.float().cpu().numpy()
        output_filename = f"{args.output_directory}/generated_{line_number}.wav"
        sf.write(output_filename, audio, hps.data.sampling_rate, 'PCM_24')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ma-AI-Labs Inference Encoder")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--model", type=str, required=True, help="Model Path")
    parser.add_argument("--config", type=str, default='./configs/mlb_french.json', help="Config File Path")
    parser.add_argument("--txt_file", type=str, default="", help="Input text File")
    parser.add_argument("--speaker_id", type=int, default=0, help="Speaker Id")
    parser.add_argument("--noise_scale", type=float, default=.667, help="Noise Scale")
    parser.add_argument("--noise_scale_w", type=float, default=0.8, help="Noise Scale Weights")
    parser.add_argument("--length_scale", type=float, default=1.2, help="Length Scale")
    parser.add_argument("--text", type=str, default='', help="Input Text")
    parser.add_argument("--output_directory", type=str, default='./output', help="Output Directory")

    args = parser.parse_args()

    parent_dir = "./"
    path = os.path.join(parent_dir, args.output_directory)
    os.makedirs(path, exist_ok=True)

    if len(args.txt_file) != 0:
        lines = get_lines(args.txt_file)
        for line_number, line in enumerate(lines, start=1):
            args.text = line
            if args.device == "cuda":
                main_gpu(args, line_number)
            elif args.device == "cpu":
                main_cpu(args, line_number)

        folder_to_zip = f'./{args.output_directory}'
        output_zip_file = f'./{args.output_directory}.zip'
        #zip_folder(folder_to_zip, output_zip_file)
    else:
        if args.device == "cuda":
            main_gpu(args, 1)
        elif args.device == "cpu":
            main_cpu(args, 1)
