import onnxruntime as ort
import numpy as np
import argparse
import os
import soundfile as sf
import librosa
import torch
from mel_processing import spectrogram_torch
from axengine import InferenceSession
import time


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_audio", "-i", type=str, required=True, help="Input audio file(.wav)")
    parser.add_argument("--output_audio", "-o", type=str, required=False, default="./output.wav", help="Seperated wav path")
    parser.add_argument("--encoder", "-e", type=str, required=False, default="./models/encoder.axmodel", help="encoder axmodel")
    parser.add_argument("--decoder", "-d", type=str, required=False, default="./models/decoder.axmodel", help="decoder axmodel")
    parser.add_argument("--g_src", type=str, required=False, default="./models/g_src.bin", help="source speaker feature")
    parser.add_argument("--g_dst", type=str, required=False, default="./models/g_dst.bin", help="target speaker feature")
    return parser.parse_args()


def main():
    args = get_args()
    assert os.path.exists(args.input_audio), f"Input audio {args.input_audio} not exist"
    assert os.path.exists(args.encoder), f"Encoder {args.model} not exist"
    assert os.path.exists(args.decoder), f"Decoder {args.model} not exist"
    assert os.path.exists(args.g_src), f"{args.g_src} not exist"
    assert os.path.exists(args.g_dst), f"{args.g_dst} not exist"

    input_audio = args.input_audio
    output_audio = args.output_audio
    encoder_path = args.encoder
    decoder_path = args.decoder
    g_src = np.fromfile(args.g_src, dtype=np.float32).reshape((1, 256, 1))
    g_dst = np.fromfile(args.g_dst, dtype=np.float32).reshape((1, 256, 1))

    sampling_rate = 22050
    filter_length = 1024
    hop_length = 256
    win_length = 1024
    enc_len = 1024
    dec_len = 128

    print(f"Input audio: {input_audio}")
    print(f"Output audio: {output_audio}")
    print(f"Encoder: {encoder_path}")
    print(f"Decoder: {decoder_path}")

    print("Loading audio...")
    audio, origin_sr = librosa.load(input_audio, sr=sampling_rate)
    # print(f"audio.shape = {audio.shape}")

    print("Loading model...")
    start = time.time()
    sess_enc = InferenceSession.load_from_model(encoder_path)
    sess_dec = InferenceSession.load_from_model(decoder_path)
    print(f"Load model take {(time.time() - start) * 1000}ms")

    print("Preprocessing audio...")
    audio = torch.tensor(audio).float()
    audio = audio.unsqueeze(0)
    start = time.time()
    spec = spectrogram_torch(audio, filter_length,
                            sampling_rate, hop_length, win_length,
                            center=False)
    spec = spec.numpy()
    real_enc_len = spec.shape[-1]
    spec = np.concatenate((spec, np.zeros((*spec.shape[:-1], int(np.ceil(spec.shape[-1] / enc_len)) * enc_len - spec.shape[-1]), dtype=np.float32)), axis=-1)
    # np.save("spec.npy", spec)
    # print(f"spec.size = {spec.shape}")
    print(f"Preprocess take {(time.time() - start) * 1000}ms")

    print("Running model...")
    slice_num = spec.shape[-1] // enc_len
    z = []
    for i in range(slice_num):
        spec_slice = spec[..., i * enc_len : (i + 1) * enc_len]
        start = time.time()
        outputs = sess_enc.run(input_feed={"y": spec_slice})
        z.append(outputs["z"])
        print(f"Run encoder take {(time.time() - start) * 1000}ms")
    z = np.concatenate(z, axis=-1)[..., :real_enc_len]

    # np.save("z.npy", z)
    # np.save("y_mask.npy", y_mask)
    z = np.concatenate((z, np.zeros((*z.shape[:-1], int(np.ceil(z.shape[-1] / dec_len)) * dec_len - z.shape[-1]), dtype=np.float32)), axis=-1)
    slice_num = z.shape[-1] // dec_len
    audio_list = []
    for i in range(slice_num):
        z_slice = z[..., i * dec_len : (i + 1) * dec_len]
        start = time.time()
        audio = sess_dec.run(input_feed={"z": z_slice, "g_src": g_src, "g_dst": g_dst})["audio"]
        
        audio = audio.flatten()
        print(f"Run decoder slice {i + 1}/{slice_num} take {(time.time() - start) * 1000}ms")

        audio_list.append(audio)

    audio = np.concatenate(audio_list, axis=-1)[:real_enc_len * 256]
    # np.save("audio.npy", audio)

    sf.write(output_audio, audio, sampling_rate)
    print(f"Save audio to {output_audio}")


if __name__ == "__main__":
    main()