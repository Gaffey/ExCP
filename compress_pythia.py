import os
import copy
import glob
import torch
import faiss
import argparse
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file

optimizer_name = ['gpt_neox.embed_in.weight', 'gpt_neox.layers.0.attention.query_key_value.weight', 'gpt_neox.layers.0.attention.dense.weight', 'gpt_neox.layers.0.mlp.dense_h_to_4h.weight', 'gpt_neox.layers.0.mlp.dense_4h_to_h.weight', 'gpt_neox.layers.1.attention.query_key_value.weight', 'gpt_neox.layers.1.attention.dense.weight', 'gpt_neox.layers.1.mlp.dense_h_to_4h.weight', 'gpt_neox.layers.1.mlp.dense_4h_to_h.weight', 'gpt_neox.layers.2.attention.query_key_value.weight', 'gpt_neox.layers.2.attention.dense.weight', 'gpt_neox.layers.2.mlp.dense_h_to_4h.weight', 'gpt_neox.layers.2.mlp.dense_4h_to_h.weight', 'gpt_neox.layers.3.attention.query_key_value.weight', 'gpt_neox.layers.3.attention.dense.weight', 'gpt_neox.layers.3.mlp.dense_h_to_4h.weight', 'gpt_neox.layers.3.mlp.dense_4h_to_h.weight', 'gpt_neox.layers.4.attention.query_key_value.weight', 'gpt_neox.layers.4.attention.dense.weight', 'gpt_neox.layers.4.mlp.dense_h_to_4h.weight', 'gpt_neox.layers.4.mlp.dense_4h_to_h.weight', 'gpt_neox.layers.5.attention.query_key_value.weight', 'gpt_neox.layers.5.attention.dense.weight', 'gpt_neox.layers.5.mlp.dense_h_to_4h.weight', 'gpt_neox.layers.5.mlp.dense_4h_to_h.weight', 'gpt_neox.layers.6.attention.query_key_value.weight', 'gpt_neox.layers.6.attention.dense.weight', 'gpt_neox.layers.6.mlp.dense_h_to_4h.weight', 'gpt_neox.layers.6.mlp.dense_4h_to_h.weight', 'gpt_neox.layers.7.attention.query_key_value.weight', 'gpt_neox.layers.7.attention.dense.weight', 'gpt_neox.layers.7.mlp.dense_h_to_4h.weight', 'gpt_neox.layers.7.mlp.dense_4h_to_h.weight', 'gpt_neox.layers.8.attention.query_key_value.weight', 'gpt_neox.layers.8.attention.dense.weight', 'gpt_neox.layers.8.mlp.dense_h_to_4h.weight', 'gpt_neox.layers.8.mlp.dense_4h_to_h.weight', 'gpt_neox.layers.9.attention.query_key_value.weight', 'gpt_neox.layers.9.attention.dense.weight', 'gpt_neox.layers.9.mlp.dense_h_to_4h.weight', 'gpt_neox.layers.9.mlp.dense_4h_to_h.weight', 'gpt_neox.layers.10.attention.query_key_value.weight', 'gpt_neox.layers.10.attention.dense.weight', 'gpt_neox.layers.10.mlp.dense_h_to_4h.weight', 'gpt_neox.layers.10.mlp.dense_4h_to_h.weight', 'gpt_neox.layers.11.attention.query_key_value.weight', 'gpt_neox.layers.11.attention.dense.weight', 'gpt_neox.layers.11.mlp.dense_h_to_4h.weight', 'gpt_neox.layers.11.mlp.dense_4h_to_h.weight', 'gpt_neox.layers.12.attention.query_key_value.weight', 'gpt_neox.layers.12.attention.dense.weight', 'gpt_neox.layers.12.mlp.dense_h_to_4h.weight', 'gpt_neox.layers.12.mlp.dense_4h_to_h.weight', 'gpt_neox.layers.13.attention.query_key_value.weight', 'gpt_neox.layers.13.attention.dense.weight', 'gpt_neox.layers.13.mlp.dense_h_to_4h.weight', 'gpt_neox.layers.13.mlp.dense_4h_to_h.weight', 'gpt_neox.layers.14.attention.query_key_value.weight', 'gpt_neox.layers.14.attention.dense.weight', 'gpt_neox.layers.14.mlp.dense_h_to_4h.weight', 'gpt_neox.layers.14.mlp.dense_4h_to_h.weight', 'gpt_neox.layers.15.attention.query_key_value.weight', 'gpt_neox.layers.15.attention.dense.weight', 'gpt_neox.layers.15.mlp.dense_h_to_4h.weight', 'gpt_neox.layers.15.mlp.dense_4h_to_h.weight', 'gpt_neox.layers.16.attention.query_key_value.weight', 'gpt_neox.layers.16.attention.dense.weight', 'gpt_neox.layers.16.mlp.dense_h_to_4h.weight', 'gpt_neox.layers.16.mlp.dense_4h_to_h.weight', 'gpt_neox.layers.17.attention.query_key_value.weight', 'gpt_neox.layers.17.attention.dense.weight', 'gpt_neox.layers.17.mlp.dense_h_to_4h.weight', 'gpt_neox.layers.17.mlp.dense_4h_to_h.weight', 'gpt_neox.layers.18.attention.query_key_value.weight', 'gpt_neox.layers.18.attention.dense.weight', 'gpt_neox.layers.18.mlp.dense_h_to_4h.weight', 'gpt_neox.layers.18.mlp.dense_4h_to_h.weight', 'gpt_neox.layers.19.attention.query_key_value.weight', 'gpt_neox.layers.19.attention.dense.weight', 'gpt_neox.layers.19.mlp.dense_h_to_4h.weight', 'gpt_neox.layers.19.mlp.dense_4h_to_h.weight', 'gpt_neox.layers.20.attention.query_key_value.weight', 'gpt_neox.layers.20.attention.dense.weight', 'gpt_neox.layers.20.mlp.dense_h_to_4h.weight', 'gpt_neox.layers.20.mlp.dense_4h_to_h.weight', 'gpt_neox.layers.21.attention.query_key_value.weight', 'gpt_neox.layers.21.attention.dense.weight', 'gpt_neox.layers.21.mlp.dense_h_to_4h.weight', 'gpt_neox.layers.21.mlp.dense_4h_to_h.weight', 'gpt_neox.layers.22.attention.query_key_value.weight', 'gpt_neox.layers.22.attention.dense.weight', 'gpt_neox.layers.22.mlp.dense_h_to_4h.weight', 'gpt_neox.layers.22.mlp.dense_4h_to_h.weight', 'gpt_neox.layers.23.attention.query_key_value.weight', 'gpt_neox.layers.23.attention.dense.weight', 'gpt_neox.layers.23.mlp.dense_h_to_4h.weight', 'gpt_neox.layers.23.mlp.dense_4h_to_h.weight', 'embed_out.weight', 'gpt_neox.layers.0.input_layernorm.weight', 'gpt_neox.layers.0.input_layernorm.bias', 'gpt_neox.layers.0.post_attention_layernorm.weight', 'gpt_neox.layers.0.post_attention_layernorm.bias', 'gpt_neox.layers.0.attention.query_key_value.bias', 'gpt_neox.layers.0.attention.dense.bias', 'gpt_neox.layers.0.mlp.dense_h_to_4h.bias', 'gpt_neox.layers.0.mlp.dense_4h_to_h.bias', 'gpt_neox.layers.1.input_layernorm.weight', 'gpt_neox.layers.1.input_layernorm.bias', 'gpt_neox.layers.1.post_attention_layernorm.weight', 'gpt_neox.layers.1.post_attention_layernorm.bias', 'gpt_neox.layers.1.attention.query_key_value.bias', 'gpt_neox.layers.1.attention.dense.bias', 'gpt_neox.layers.1.mlp.dense_h_to_4h.bias', 'gpt_neox.layers.1.mlp.dense_4h_to_h.bias', 'gpt_neox.layers.2.input_layernorm.weight', 'gpt_neox.layers.2.input_layernorm.bias', 'gpt_neox.layers.2.post_attention_layernorm.weight', 'gpt_neox.layers.2.post_attention_layernorm.bias', 'gpt_neox.layers.2.attention.query_key_value.bias', 'gpt_neox.layers.2.attention.dense.bias', 'gpt_neox.layers.2.mlp.dense_h_to_4h.bias', 'gpt_neox.layers.2.mlp.dense_4h_to_h.bias', 'gpt_neox.layers.3.input_layernorm.weight', 'gpt_neox.layers.3.input_layernorm.bias', 'gpt_neox.layers.3.post_attention_layernorm.weight', 'gpt_neox.layers.3.post_attention_layernorm.bias', 'gpt_neox.layers.3.attention.query_key_value.bias', 'gpt_neox.layers.3.attention.dense.bias', 'gpt_neox.layers.3.mlp.dense_h_to_4h.bias', 'gpt_neox.layers.3.mlp.dense_4h_to_h.bias', 'gpt_neox.layers.4.input_layernorm.weight', 'gpt_neox.layers.4.input_layernorm.bias', 'gpt_neox.layers.4.post_attention_layernorm.weight', 'gpt_neox.layers.4.post_attention_layernorm.bias', 'gpt_neox.layers.4.attention.query_key_value.bias', 'gpt_neox.layers.4.attention.dense.bias', 'gpt_neox.layers.4.mlp.dense_h_to_4h.bias', 'gpt_neox.layers.4.mlp.dense_4h_to_h.bias', 'gpt_neox.layers.5.input_layernorm.weight', 'gpt_neox.layers.5.input_layernorm.bias', 'gpt_neox.layers.5.post_attention_layernorm.weight', 'gpt_neox.layers.5.post_attention_layernorm.bias', 'gpt_neox.layers.5.attention.query_key_value.bias', 'gpt_neox.layers.5.attention.dense.bias', 'gpt_neox.layers.5.mlp.dense_h_to_4h.bias', 'gpt_neox.layers.5.mlp.dense_4h_to_h.bias', 'gpt_neox.layers.6.input_layernorm.weight', 'gpt_neox.layers.6.input_layernorm.bias', 'gpt_neox.layers.6.post_attention_layernorm.weight', 'gpt_neox.layers.6.post_attention_layernorm.bias', 'gpt_neox.layers.6.attention.query_key_value.bias', 'gpt_neox.layers.6.attention.dense.bias', 'gpt_neox.layers.6.mlp.dense_h_to_4h.bias', 'gpt_neox.layers.6.mlp.dense_4h_to_h.bias', 'gpt_neox.layers.7.input_layernorm.weight', 'gpt_neox.layers.7.input_layernorm.bias', 'gpt_neox.layers.7.post_attention_layernorm.weight', 'gpt_neox.layers.7.post_attention_layernorm.bias', 'gpt_neox.layers.7.attention.query_key_value.bias', 'gpt_neox.layers.7.attention.dense.bias', 'gpt_neox.layers.7.mlp.dense_h_to_4h.bias', 'gpt_neox.layers.7.mlp.dense_4h_to_h.bias', 'gpt_neox.layers.8.input_layernorm.weight', 'gpt_neox.layers.8.input_layernorm.bias', 'gpt_neox.layers.8.post_attention_layernorm.weight', 'gpt_neox.layers.8.post_attention_layernorm.bias', 'gpt_neox.layers.8.attention.query_key_value.bias', 'gpt_neox.layers.8.attention.dense.bias', 'gpt_neox.layers.8.mlp.dense_h_to_4h.bias', 'gpt_neox.layers.8.mlp.dense_4h_to_h.bias', 'gpt_neox.layers.9.input_layernorm.weight', 'gpt_neox.layers.9.input_layernorm.bias', 'gpt_neox.layers.9.post_attention_layernorm.weight', 'gpt_neox.layers.9.post_attention_layernorm.bias', 'gpt_neox.layers.9.attention.query_key_value.bias', 'gpt_neox.layers.9.attention.dense.bias', 'gpt_neox.layers.9.mlp.dense_h_to_4h.bias', 'gpt_neox.layers.9.mlp.dense_4h_to_h.bias', 'gpt_neox.layers.10.input_layernorm.weight', 'gpt_neox.layers.10.input_layernorm.bias', 'gpt_neox.layers.10.post_attention_layernorm.weight', 'gpt_neox.layers.10.post_attention_layernorm.bias', 'gpt_neox.layers.10.attention.query_key_value.bias', 'gpt_neox.layers.10.attention.dense.bias', 'gpt_neox.layers.10.mlp.dense_h_to_4h.bias', 'gpt_neox.layers.10.mlp.dense_4h_to_h.bias', 'gpt_neox.layers.11.input_layernorm.weight', 'gpt_neox.layers.11.input_layernorm.bias', 'gpt_neox.layers.11.post_attention_layernorm.weight', 'gpt_neox.layers.11.post_attention_layernorm.bias', 'gpt_neox.layers.11.attention.query_key_value.bias', 'gpt_neox.layers.11.attention.dense.bias', 'gpt_neox.layers.11.mlp.dense_h_to_4h.bias', 'gpt_neox.layers.11.mlp.dense_4h_to_h.bias', 'gpt_neox.layers.12.input_layernorm.weight', 'gpt_neox.layers.12.input_layernorm.bias', 'gpt_neox.layers.12.post_attention_layernorm.weight', 'gpt_neox.layers.12.post_attention_layernorm.bias', 'gpt_neox.layers.12.attention.query_key_value.bias', 'gpt_neox.layers.12.attention.dense.bias', 'gpt_neox.layers.12.mlp.dense_h_to_4h.bias', 'gpt_neox.layers.12.mlp.dense_4h_to_h.bias', 'gpt_neox.layers.13.input_layernorm.weight', 'gpt_neox.layers.13.input_layernorm.bias', 'gpt_neox.layers.13.post_attention_layernorm.weight', 'gpt_neox.layers.13.post_attention_layernorm.bias', 'gpt_neox.layers.13.attention.query_key_value.bias', 'gpt_neox.layers.13.attention.dense.bias', 'gpt_neox.layers.13.mlp.dense_h_to_4h.bias', 'gpt_neox.layers.13.mlp.dense_4h_to_h.bias', 'gpt_neox.layers.14.input_layernorm.weight', 'gpt_neox.layers.14.input_layernorm.bias', 'gpt_neox.layers.14.post_attention_layernorm.weight', 'gpt_neox.layers.14.post_attention_layernorm.bias', 'gpt_neox.layers.14.attention.query_key_value.bias', 'gpt_neox.layers.14.attention.dense.bias', 'gpt_neox.layers.14.mlp.dense_h_to_4h.bias', 'gpt_neox.layers.14.mlp.dense_4h_to_h.bias', 'gpt_neox.layers.15.input_layernorm.weight', 'gpt_neox.layers.15.input_layernorm.bias', 'gpt_neox.layers.15.post_attention_layernorm.weight', 'gpt_neox.layers.15.post_attention_layernorm.bias', 'gpt_neox.layers.15.attention.query_key_value.bias', 'gpt_neox.layers.15.attention.dense.bias', 'gpt_neox.layers.15.mlp.dense_h_to_4h.bias', 'gpt_neox.layers.15.mlp.dense_4h_to_h.bias', 'gpt_neox.layers.16.input_layernorm.weight', 'gpt_neox.layers.16.input_layernorm.bias', 'gpt_neox.layers.16.post_attention_layernorm.weight', 'gpt_neox.layers.16.post_attention_layernorm.bias', 'gpt_neox.layers.16.attention.query_key_value.bias', 'gpt_neox.layers.16.attention.dense.bias', 'gpt_neox.layers.16.mlp.dense_h_to_4h.bias', 'gpt_neox.layers.16.mlp.dense_4h_to_h.bias', 'gpt_neox.layers.17.input_layernorm.weight', 'gpt_neox.layers.17.input_layernorm.bias', 'gpt_neox.layers.17.post_attention_layernorm.weight', 'gpt_neox.layers.17.post_attention_layernorm.bias', 'gpt_neox.layers.17.attention.query_key_value.bias', 'gpt_neox.layers.17.attention.dense.bias', 'gpt_neox.layers.17.mlp.dense_h_to_4h.bias', 'gpt_neox.layers.17.mlp.dense_4h_to_h.bias', 'gpt_neox.layers.18.input_layernorm.weight', 'gpt_neox.layers.18.input_layernorm.bias', 'gpt_neox.layers.18.post_attention_layernorm.weight', 'gpt_neox.layers.18.post_attention_layernorm.bias', 'gpt_neox.layers.18.attention.query_key_value.bias', 'gpt_neox.layers.18.attention.dense.bias', 'gpt_neox.layers.18.mlp.dense_h_to_4h.bias', 'gpt_neox.layers.18.mlp.dense_4h_to_h.bias', 'gpt_neox.layers.19.input_layernorm.weight', 'gpt_neox.layers.19.input_layernorm.bias', 'gpt_neox.layers.19.post_attention_layernorm.weight', 'gpt_neox.layers.19.post_attention_layernorm.bias', 'gpt_neox.layers.19.attention.query_key_value.bias', 'gpt_neox.layers.19.attention.dense.bias', 'gpt_neox.layers.19.mlp.dense_h_to_4h.bias', 'gpt_neox.layers.19.mlp.dense_4h_to_h.bias', 'gpt_neox.layers.20.input_layernorm.weight', 'gpt_neox.layers.20.input_layernorm.bias', 'gpt_neox.layers.20.post_attention_layernorm.weight', 'gpt_neox.layers.20.post_attention_layernorm.bias', 'gpt_neox.layers.20.attention.query_key_value.bias', 'gpt_neox.layers.20.attention.dense.bias', 'gpt_neox.layers.20.mlp.dense_h_to_4h.bias', 'gpt_neox.layers.20.mlp.dense_4h_to_h.bias', 'gpt_neox.layers.21.input_layernorm.weight', 'gpt_neox.layers.21.input_layernorm.bias', 'gpt_neox.layers.21.post_attention_layernorm.weight', 'gpt_neox.layers.21.post_attention_layernorm.bias', 'gpt_neox.layers.21.attention.query_key_value.bias', 'gpt_neox.layers.21.attention.dense.bias', 'gpt_neox.layers.21.mlp.dense_h_to_4h.bias', 'gpt_neox.layers.21.mlp.dense_4h_to_h.bias', 'gpt_neox.layers.22.input_layernorm.weight', 'gpt_neox.layers.22.input_layernorm.bias', 'gpt_neox.layers.22.post_attention_layernorm.weight', 'gpt_neox.layers.22.post_attention_layernorm.bias', 'gpt_neox.layers.22.attention.query_key_value.bias', 'gpt_neox.layers.22.attention.dense.bias', 'gpt_neox.layers.22.mlp.dense_h_to_4h.bias', 'gpt_neox.layers.22.mlp.dense_4h_to_h.bias', 'gpt_neox.layers.23.input_layernorm.weight', 'gpt_neox.layers.23.input_layernorm.bias', 'gpt_neox.layers.23.post_attention_layernorm.weight', 'gpt_neox.layers.23.post_attention_layernorm.bias', 'gpt_neox.layers.23.attention.query_key_value.bias', 'gpt_neox.layers.23.attention.dense.bias', 'gpt_neox.layers.23.mlp.dense_h_to_4h.bias', 'gpt_neox.layers.23.mlp.dense_4h_to_h.bias', 'gpt_neox.final_layer_norm.weight', 'gpt_neox.final_layer_norm.bias']


def prune(tensor, mag_thres = 0.5):
    mag_thres = min(2.5, mag_thres)
    mag_thres = tensor.abs().median() * mag_thres
    tensor = torch.where(tensor.abs() > mag_thres, tensor, 0)
    return tensor, (tensor == 0).sum()


def prune_optimizer(exp_tensor, exp_sq_tensor, tensor, j = 1):
    res_exp = torch.where((exp_tensor.abs()) > j * (exp_tensor.abs()).mean(), exp_tensor, 0)
    res_exp = torch.where(tensor == 0, 0, res_exp)
    res_exp_sq = torch.where((exp_tensor.abs()) > j * (exp_tensor.abs()).mean(), exp_sq_tensor, 0)
    res_exp_sq = torch.where(tensor == 0, 0, res_exp_sq)
    return res_exp, res_exp_sq, (res_exp == 0).sum()


def quant(tensor, bit_num = 4):
    assert 8 % bit_num == 0 and bit_num <= 8
    all_labels = []
    all_codebook = []
    knn_tensor = copy.deepcopy(tensor)
    knn_tensor_shape = knn_tensor.shape
    knn_tensor_flt = knn_tensor.flatten()
    assert knn_tensor_flt.shape[0] % (8 // bit_num) == 0
    knn_tensor_nonzero = knn_tensor_flt[knn_tensor_flt != 0]
    print(knn_tensor_nonzero.shape[0])
    if knn_tensor_nonzero.shape[0] <= 2 ** bit_num - 1:
        return tensor, None
    kmeans = faiss.Kmeans(1, 2 ** bit_num - 1, gpu=False)
    kmeans.train(knn_tensor_nonzero.view(-1, 1).numpy())
    codebook = kmeans.centroids
    labels = kmeans.index.search(knn_tensor_nonzero.view(-1, 1).numpy(), 1)[1]
    knn_tensor_flt[knn_tensor_flt.nonzero().squeeze()] = torch.Tensor(labels + 1).squeeze()
    knn_tensor_flt2 = knn_tensor_flt.view(-1, 8 // bit_num)
    knn_tensor_slim = torch.zeros_like(knn_tensor_flt2[:, 0])
    for i in range(8 // bit_num):
        knn_tensor_slim += knn_tensor_flt2[:, i] * ((2 ** bit_num) ** (8 // bit_num - i - 1))
    knn_tensor_flt = knn_tensor_slim.view(-1, knn_tensor_shape[-1] // (8 // bit_num))
    return knn_tensor_flt, codebook


def unquantize(codebook, indexes, bit_num):
    recover_indexes = torch.zeros(indexes.nelement(), 8 // bit_num)
    for i in range(8 // bit_num):
        recover_indexes[:, i] = indexes.flatten() // ((2 ** bit_num) ** (8 // bit_num - i - 1))
        indexes = indexes.fmod(((2 ** bit_num) ** (8 // bit_num - i - 1)))
    recover_tensor = torch.concatenate((torch.Tensor([0]), codebook.squeeze()), dim=0)[recover_indexes.flatten().squeeze().to(torch.int64)]
    return recover_tensor
 

def recon(checkpoint, ref_weights, args):
    recon_dict = {}
    optimizer_dict = {}
    data_pt = torch.load(os.path.join(args.ref_checkpoint_path, "optimizer.pt"), map_location='cpu')
    for k in checkpoint.keys():
        ckpt = checkpoint[k]
        optim_k = optimizer_name.index(k)
        ref_shape = ref_weights.get_tensor(k).shape
        if "weights_c" in ckpt.keys():
            recover_weights = unquantize(ckpt["weights_c"], ckpt["weights_i"], args.quant_bits).view(ref_shape)
        else:
            recover_weights = ckpt["weights_i"].view(ref_shape)
        if "opt_v_c" in ckpt.keys():
            recover_opt_v = unquantize(ckpt["opt_v_c"], ckpt["opt_v_i"], args.quant_bits_opt).view(ref_shape)
        else:
            recover_opt_v = ckpt["opt_v_i"].view(ref_shape)
        if "opt_m_c" in ckpt.keys():
            recover_opt_m = unquantize(ckpt["opt_m_c"], ckpt["opt_m_i"], args.quant_bits_opt).view(ref_shape)
        else:
            recover_opt_m = ckpt["opt_m_i"].view(ref_shape)
        print(k, recover_weights.shape, ref_weights.get_tensor(k).shape)
        recon_dict[k] = (recover_weights + ref_weights.get_tensor(k)).to(torch.float32)
        optim_k = optimizer_name.index(k)
        data_pt['state'][optim_k]['exp_avg'] = recover_opt_v.to(torch.float32)
        data_pt['state'][optim_k]['exp_avg_sq'] = recover_opt_m.to(torch.float32)
    save_file(recon_dict, os.path.join(args.output, "model.safetensors"), metadata = {"format": "pt"})
    torch.save(data_pt, os.path.join(args.output, "optimizer.pt"))


def main(args):
    remove_counter_weights = 0
    remove_counter_optimizer = 0
    element_counter = 0
    import time
    torch.cuda.synchronize()
    st = time.time()
    with safe_open(os.path.join(args.ref_checkpoint_path, "model.safetensors"),\
                               framework="pt", device="cpu") as ref_weights:
        if args.only_recon:
            saved_checkpoint = torch.load(os.path.join(args.output, "compressed.pt"), map_location='cpu')
            recon(saved_checkpoint, ref_weights, args)
            return
        with safe_open(os.path.join(args.checkpoint_path, "model.safetensors"),\
                               framework="pt", device="cpu") as weights:
            optimizer = torch.load(os.path.join(args.checkpoint_path, "optimizer.pt"),\
                               map_location="cpu")
            saved_checkpoint = {}
            for ind, k in enumerate(ref_weights.keys()):
                print(k)
                if k not in optimizer_name:
                    continue
                optim_k = optimizer_name.index(k)
                residual_tensor = weights.get_tensor(k) - ref_weights.get_tensor(k)
                opt_v = optimizer['state'][optim_k]['exp_avg']
                opt_m = optimizer['state'][optim_k]['exp_avg_sq']
                print("opt_m", opt_m.mean())
                residual_tensor, remove = prune(residual_tensor, args.prune_alpha / opt_m.mean().sqrt().cpu())
                pruned_opt_v, pruned_opt_m, remove_opt = prune_optimizer(opt_v, opt_m, residual_tensor, args.prune_beta)
                residual_tensor_index, residual_tensor_codebook = quant(residual_tensor, args.quant_bits)
                remove_counter_weights += remove
                remove_counter_optimizer += remove_opt
                element_counter += residual_tensor.nelement()
                opt_v_index, opt_v_codebook = quant(pruned_opt_v, args.quant_bits_opt)
                opt_m_index, opt_m_codebook = quant(pruned_opt_m, args.quant_bits_opt)
                saved_checkpoint[k] = {}
                saved_checkpoint[k]['weights_i'] = torch.Tensor(residual_tensor_index)
                if residual_tensor_codebook is not None:
                    saved_checkpoint[k]['weights_i'] = torch.Tensor(residual_tensor_index).to(torch.uint8)
                    saved_checkpoint[k]["weights_c"] = torch.Tensor(residual_tensor_codebook).to(torch.float16)
                saved_checkpoint[k]["opt_v_i"] = torch.Tensor(opt_v_index)
                if opt_v_codebook is not None:
                    saved_checkpoint[k]["opt_v_i"] = torch.Tensor(opt_v_index).to(torch.uint8)
                    saved_checkpoint[k]["opt_v_c"] = torch.Tensor(opt_v_codebook).to(torch.float16)
                saved_checkpoint[k]["opt_m_i"] = torch.Tensor(opt_m_index)
                if opt_m_codebook is not None:
                    saved_checkpoint[k]["opt_m_i"] = torch.Tensor(opt_m_index).to(torch.uint8)
                    saved_checkpoint[k]["opt_m_c"] = torch.Tensor(opt_m_codebook).to(torch.float16)

            torch.cuda.synchronize()
            ed = time.time()
            print("compress using time: {}".format(ed - st))
            print("weights removed ratio: {}/{}({})".format(remove_counter_weights, element_counter, remove_counter_weights / element_counter))
            print("optimizer removed ratio: {}/{}({})".format(remove_counter_optimizer, element_counter, remove_counter_optimizer / element_counter))
            if not os.path.exists(args.output):
                os.makedirs(args.output, exist_ok=True)
            torch.save(saved_checkpoint, os.path.join(args.output, "compressed.pt"))
            torch.cuda.synchronize()
            st = time.time()
            if args.recon:
                recon(saved_checkpoint, ref_weights, args)
            torch.cuda.synchronize()
            ed = time.time()
            print("recon using time: {}".format(ed - st))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test a SAM')
    parser.add_argument('checkpoint_path', type=str)
    parser.add_argument('ref_checkpoint_path', type=str)
    parser.add_argument('--gpus', type=int, default=8)
    parser.add_argument('--prune_alpha', type=float, default=5e-5)
    parser.add_argument('--prune_beta', type=float, default=2.0)
    parser.add_argument('--quant_bits', type=int, default=4)
    parser.add_argument('--quant_bits_opt', type=int, default=4)
    parser.add_argument('--recon', action='store_true')
    parser.add_argument('--only_recon', action='store_true')
    parser.add_argument('--output', type=str, default="./")
    args, _ = parser.parse_known_args()
    main(args)
