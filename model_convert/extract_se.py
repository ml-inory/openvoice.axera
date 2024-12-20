import os
import torch
import argparse

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from openvoice import se_extractor
from openvoice.api import ToneColorConverter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="Reference speaker audio")
    parser.add_argument("--output", "-o", type=str, required=False, default=None, help="Saved tone color path, *.bin, default is the same name of input")
    return parser.parse_args()


def main():
    args = parse_args()
    reference_speaker = args.input
    save_path = args.output
    if save_path is None:
        save_path = os.path.splitext(os.path.basename(reference_speaker))[0] + ".bin"

    print(f"Reference speaker: {reference_speaker}")
    print(f"Save path: {save_path}")

    ckpt_converter = 'checkpoints_v2/converter'
    device = "cpu"
    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
    
    target_se, _ = se_extractor.get_se(reference_speaker, tone_color_converter, vad=False)
    target_se.numpy().tofile(save_path)
    print("Done")

if __name__ == "__main__":
    main()