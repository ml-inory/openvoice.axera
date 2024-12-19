from openvoice import utils
from openvoice.models import SynthesizerTrn
import torch
import onnx
import onnxruntime as ort
from onnxsim import simplify
import librosa
from openvoice.mel_processing import spectrogram_torch
import tarfile
import tqdm
import os
import numpy as np


device = "cpu"
ckpt_converter = 'checkpoints_v2/converter'
config_path = f'{ckpt_converter}/config.json'
ckpt_path = f'{ckpt_converter}/checkpoint.pth'
audio_src_path = "./resources/demo_speaker0.mp3"
dataset_path = "calibration_dataset"

hps = utils.get_hparams_from_file(config_path)

model = SynthesizerTrn(
    len(getattr(hps, 'symbols', [])),
    hps.data.filter_length // 2 + 1,
    n_speakers=hps.data.n_speakers,
    **hps.model,
).to(device)
model.eval()
checkpoint_dict = torch.load(ckpt_path, map_location=torch.device(device))
model.load_state_dict(checkpoint_dict['model'], strict=False)
# Hack decoder
# self.conv_pre.bias.data += self.cond.bias.data
model.dec.conv_pre.bias.data += model.dec.cond.bias.data

# load audio
audio, sample_rate = librosa.load(audio_src_path, sr=hps.data.sampling_rate)
audio = torch.tensor(audio).float()

with torch.no_grad():
    y = torch.FloatTensor(audio).to(device)
    y = y.unsqueeze(0)
    # (1, 513, y_length)
    spec = spectrogram_torch(y, hps.data.filter_length,
                            hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                            center=False).to(device)
    tau = torch.FloatTensor([0.3])                          
    # print(f"spec.size = {spec.size()}")

    # Export encoder
    enc_len = 1024
    model_name = "encoder.onnx"
    model.forward = model.enc_forward
    inputs = (
        torch.rand(1, 513, enc_len),
    )
    input_names = ['y',]
    torch.onnx.export(model,               # model being run
                    inputs,                    # model input (or a tuple for multiple inputs)
                    model_name,              # where to save the model (can be a file or file-like object)
                    opset_version=16,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    dynamic_axes=None,
                    input_names=input_names, # the model's input names
                    output_names=['z'], # the model's output names
                    )
    sim_model,_ = simplify(model_name)
    onnx.save(sim_model, model_name)
    print(f"Export encoder to {model_name}")

    # Export decoder
    model_name = "decoder.onnx"
    model.forward = model.dec_forward
    dec_len = 128
    z = torch.rand(1, 192, dec_len)
    y_mask = torch.ones(1, 1, dec_len)
    g_src = torch.rand(1, 256, 1)
    g_dst = torch.rand(1, 256, 1)
    inputs = (
        z, g_src, g_dst
    )
    input_names = ['z', 'g_src', 'g_dst']
    torch.onnx.export(model,               # model being run
                    inputs,                    # model input (or a tuple for multiple inputs)
                    model_name,              # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=16,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    dynamic_axes=None,
                    input_names = input_names, # the model's input names
                    output_names = ['audio'], # the model's output names
                    )
    sim_model,_ = simplify(model_name)
    onnx.save(sim_model, model_name)
    print(f"Export decoder to {model_name}")

    print("Generating calibration dataset...")
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(f"{dataset_path}/y", exist_ok=True)
    os.makedirs(f"{dataset_path}/z", exist_ok=True)
    os.makedirs(f"{dataset_path}/g_src", exist_ok=True)
    os.makedirs(f"{dataset_path}/g_dst", exist_ok=True)

    tf_y = tarfile.open(f"{dataset_path}/y.tar.gz", "w:gz")
    tf_z = tarfile.open(f"{dataset_path}/z.tar.gz", "w:gz")
    tf_g_src = tarfile.open(f"{dataset_path}/g_src.tar.gz", "w:gz")
    tf_g_dst = tarfile.open(f"{dataset_path}/g_dst.tar.gz", "w:gz")

    for i in tqdm.trange(0, spec.size(-1) // enc_len):
        y_slice = spec[..., i * enc_len : (i + 1) * enc_len]
        if y_slice.size(-1) < enc_len:
            y_slice = torch.nn.functional.pad(
                y_slice,
                (0, enc_len - y_slice.size(-1)),
                mode="constant",
            )
        y_slice = y_slice.numpy()
        np.save(f"{dataset_path}/y/{i}.npy", y_slice)
        tf_y.add(f"{dataset_path}/y/{i}.npy")

    z = model.enc_forward(spec)
    for i in tqdm.trange(0, z.size(-1) // dec_len):
        z_slice = z[..., i * dec_len : (i + 1) * dec_len]
        y_mask_slice = y_mask[..., i * dec_len : (i + 1) * dec_len]
        if z_slice.size(-1) < dec_len:
            z_slice = torch.nn.functional.pad(
                z_slice,
                (0, dec_len - z_slice.size(-1)),
                mode="constant",
            )

        z_slice = z_slice.numpy()
        np.save(f"{dataset_path}/z/{i}.npy", z_slice)
        np.save(f"{dataset_path}/g_src/{i}.npy", g_src.numpy())
        np.save(f"{dataset_path}/g_dst/{i}.npy", g_dst.numpy())
        
        tf_z.add(f"{dataset_path}/z/{i}.npy")
        tf_g_src.add(f"{dataset_path}/g_src/{i}.npy")
        tf_g_dst.add(f"{dataset_path}/g_dst/{i}.npy")

    tf_y.close()
    tf_z.close()
    tf_g_src.close()
    tf_g_dst.close()