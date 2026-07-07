# -*- coding: utf-8 -*-
"""导出 pp_formulanet decoder 的「带 past 单步」ONNX → engines_bin/decoder_with_past.onnx。

past 结构（probe 确认）：6层 × 4 = 24 个 tensor，每层 [self_K, self_V, cross_K, cross_V]
  self K/V : [1,16,seq,32]  seq 每步增长
  cross K/V: [1,16,144,32]  固定

路径全部经 _env 定位（权重自动下载、输出落 engines_bin/），无硬编码。
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image

import _env

os.environ.setdefault("MINERU_FORMULA_CH_SUPPORT", "True")

MFR_DIR = _env.get_mfr_weight_dir()
OUT = _env.onnx_path("decoder_with_past.onnx")
device = "cuda"  # head 内部 get_device() 硬编码 cuda，故在 GPU 导出
OPSET = _env.OPSET
N_LAYERS = _env.N_LAYERS

from mineru.model.mfr.pp_formulanet_plus_m.predict_formula import FormulaRecognizer

rec = FormulaRecognizer(MFR_DIR, device)
net = rec.net; net.eval()
head = net.head
core = head.decoder  # CustomMBartForCausalLM
core.eval()


class DecoderWithPastWrapper(torch.nn.Module):
    """单步 decoder，past 展平为扁平输入/输出，便于 ONNX/TRT。
    输入 : input_ids[B,1], attention_mask[B,L], encoder_hidden[B,144,512],
           past(24个: 每层 self_K,self_V,cross_K,cross_V)
    输出 : logits[B,1,vocab], new_past(24个)
    """
    def __init__(self, core, n_layers):
        super().__init__()
        self.core = core
        self.n = n_layers

    def forward(self, input_ids, attention_mask, encoder_hidden_states, *flat_past):
        pkv = []
        for i in range(self.n):
            pkv.append((flat_past[4*i], flat_past[4*i+1], flat_past[4*i+2], flat_past[4*i+3]))
        pkv = tuple(pkv)
        out = self.core(
            input_ids=input_ids, attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=None,
            past_key_values=pkv, use_cache=True, return_dict=True,
            output_attentions=False, output_hidden_states=False,
        )
        flat_out = []
        for layer in out.past_key_values:
            flat_out.extend([layer[0], layer[1], layer[2], layer[3]])
        return (out.logits, *flat_out)


wrapper = DecoderWithPastWrapper(core, N_LAYERS).to(device).eval()

# ---- 造真实 past（跑首步得到）----
img = np.array(Image.fromarray(((np.arange(80 * 300).reshape(80, 300)) % 255).astype("uint8")).convert("RGB"))
b = rec.pre_tfs["UniMERNetImgDecode"](imgs=[img])
b = rec.pre_tfs["UniMERNetTestTransform"](imgs=b)
b = rec.pre_tfs["LatexImageFormat"](imgs=b)
inp = torch.from_numpy(rec.pre_tfs["ToBatch"](imgs=b)[0]).to(device).float()
with torch.no_grad():
    enc_out = net.backbone(inp)
enc_hidden = enc_out["last_hidden_state"]
if head.config_decoder.hidden_size != head.encoder_hidden_size:
    enc_hidden = head.enc_to_dec_proj(enc_hidden)

# 首步得到 past（seq=1）
bos = torch.tensor([[0]], device=device)
with torch.no_grad():
    out1 = core(input_ids=bos, attention_mask=torch.ones_like(bos),
                encoder_hidden_states=enc_hidden, encoder_attention_mask=None,
                past_key_values=None, use_cache=True, return_dict=True,
                output_attentions=False, output_hidden_states=False)
pkv1 = out1.past_key_values
flat_past = []
for layer in pkv1:
    flat_past.extend([layer[0], layer[1], layer[2], layer[3]])

# 第2步输入（attention_mask 统一 int64，与首步图一致，便于 TRT）
tok2 = torch.tensor([[5]], device=device)
attn2 = torch.ones((1, 2), dtype=torch.int64, device=device)

with torch.no_grad():
    ref = wrapper(tok2, attn2, enc_hidden, *flat_past)
print("torch 参考: logits", tuple(ref[0].shape), " new_past 数", len(ref) - 1)

# ---- 动态轴：self-attn past 的 seq 维(dim2) 动态；attention_mask 的 L 动态 ----
in_names = ["input_ids", "attention_mask", "encoder_hidden_states"]
out_names = ["logits"]
dynamic_axes = {
    "input_ids": {0: "batch"},
    "attention_mask": {0: "batch", 1: "total_len"},
    "encoder_hidden_states": {0: "batch"},
    "logits": {0: "batch"},
}
for i in range(N_LAYERS):
    for k, tag in enumerate(["self_k", "self_v", "cross_k", "cross_v"]):
        nm = f"past_{i}_{tag}"
        in_names.append(nm)
        if tag in ("self_k", "self_v"):
            dynamic_axes[nm] = {0: "batch", 2: "past_len"}
        else:
            dynamic_axes[nm] = {0: "batch"}
    for k, tag in enumerate(["self_k", "self_v", "cross_k", "cross_v"]):
        nm = f"present_{i}_{tag}"
        out_names.append(nm)
        if tag in ("self_k", "self_v"):
            dynamic_axes[nm] = {0: "batch", 2: "new_len"}
        else:
            dynamic_axes[nm] = {0: "batch"}

args = (tok2, attn2, enc_hidden, *flat_past)
print("开始导出 ONNX ...")
torch.onnx.export(
    wrapper, args, OUT,
    input_names=in_names, output_names=out_names,
    dynamic_axes=dynamic_axes, opset_version=OPSET,
    do_constant_folding=True, dynamo=False
)
print("✓ 导出完成:", OUT)

# ---- onnxruntime 数值对齐 ----
try:
    import onnxruntime as ort
    sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    _onnx_inputs = {i.name for i in sess.get_inputs()}
    feed = {
        "input_ids": tok2.cpu().numpy(),
        "attention_mask": attn2.cpu().numpy(),
    }
    if "encoder_hidden_states" in _onnx_inputs:
        feed["encoder_hidden_states"] = enc_hidden.detach().cpu().numpy()
    for i in range(N_LAYERS):
        for tag, t in zip(["self_k","self_v","cross_k","cross_v"], pkv1[i]):
            feed[f"past_{i}_{tag}"] = t.detach().cpu().numpy()
    outs = sess.run(None, feed)
    d = np.abs(outs[0] - ref[0].detach().cpu().numpy())
    print(f"[logits] onnx vs torch  max|Δ|={d.max():.3e}  mean|Δ|={d.mean():.3e}")
    _rt = ref[0].detach().cpu().numpy()
    print(f"  argmax onnx={outs[0].argmax()}  torch={_rt.argmax()}  {'✓一致' if outs[0].argmax()==_rt.argmax() else '✗不同'}")
    print("数值对齐:", "✓ 通过" if d.max() < 1e-3 else "⚠ 偏大")
except Exception as e:
    import traceback; traceback.print_exc()
    print("onnxruntime 验证失败:", repr(e)[:200])
