# -*- coding: utf-8 -*-
"""导出 pp_formulanet decoder 的「首步」ONNX（decoder_model，无 past）→ engines_bin/decoder_init.onnx。

首步：input_ids(BOS) + encoder_hidden_states → logits + 24 present(含算好的 cross K/V)。
之后 present 作为 past 喂给 decoder_with_past.onnx 的后续步。

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
OUT = _env.onnx_path("decoder_init.onnx")
device = "cuda"
OPSET = _env.OPSET
N_LAYERS = _env.N_LAYERS

from mineru.model.mfr.pp_formulanet_plus_m.predict_formula import FormulaRecognizer

rec = FormulaRecognizer(MFR_DIR, device)
net = rec.net; net.eval()
head = net.head
core = head.decoder
core.eval()


class DecoderInitWrapper(torch.nn.Module):
    """首步 decoder：无 past，输出 logits + 展平 present。"""
    def __init__(self, core, n_layers):
        super().__init__()
        self.core = core
        self.n = n_layers

    def forward(self, input_ids, attention_mask, encoder_hidden_states):
        out = self.core(
            input_ids=input_ids, attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=None,
            past_key_values=None, use_cache=True, return_dict=True,
            output_attentions=False, output_hidden_states=False,
        )
        flat_out = []
        for layer in out.past_key_values:
            flat_out.extend([layer[0], layer[1], layer[2], layer[3]])
        return (out.logits, *flat_out)


wrapper = DecoderInitWrapper(core, N_LAYERS).to(device).eval()

# 真实 encoder 输出
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

bos = torch.tensor([[0]], device=device)
attn = torch.ones_like(bos)

with torch.no_grad():
    ref = wrapper(bos, attn, enc_hidden)
print("torch 参考: logits", tuple(ref[0].shape), " present 数", len(ref) - 1)

in_names = ["input_ids", "attention_mask", "encoder_hidden_states"]
out_names = ["logits"]
dynamic_axes = {
    "input_ids": {0: "batch", 1: "seq"},
    "attention_mask": {0: "batch", 1: "seq"},
    "encoder_hidden_states": {0: "batch"},
    "logits": {0: "batch", 1: "seq"},
}
for i in range(N_LAYERS):
    for tag in ["self_k", "self_v", "cross_k", "cross_v"]:
        nm = f"present_{i}_{tag}"
        out_names.append(nm)
        dynamic_axes[nm] = {0: "batch", 2: "seq"} if tag in ("self_k", "self_v") else {0: "batch"}

print("开始导出首步 ONNX ...")
torch.onnx.export(
    wrapper, (bos, attn, enc_hidden), OUT,
    input_names=in_names, output_names=out_names,
    dynamic_axes=dynamic_axes, opset_version=OPSET,
    do_constant_folding=True, dynamo=False
)
print("✓ 导出完成:", OUT)

# 数值对齐
try:
    import onnxruntime as ort
    sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    feed = {
        "input_ids": bos.cpu().numpy(),
        "attention_mask": attn.cpu().numpy(),
        "encoder_hidden_states": enc_hidden.detach().cpu().numpy(),
    }
    outs = sess.run(None, feed)
    _rt = ref[0].detach().cpu().numpy()
    d = np.abs(outs[0] - _rt)
    print(f"[logits] onnx vs torch  max|Δ|={d.max():.3e}  mean|Δ|={d.mean():.3e}")
    print(f"  argmax onnx={outs[0].argmax()}  torch={_rt.argmax()}  {'✓一致' if outs[0].argmax()==_rt.argmax() else '✗不同'}")
    print(f"  present 输出数: {len(outs)-1} (应=24)")
    print("数值对齐:", "✓ 通过" if d.max() < 1e-3 else "⚠ 偏大")
except Exception as e:
    import traceback; traceback.print_exc()
