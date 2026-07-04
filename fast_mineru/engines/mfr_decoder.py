"""MFRDecoderTRT —— pp_formulanet 自回归解码的 TensorRT 两段式引擎(全程 GPU 零拷贝)。

替代 PPFormulaNet_Head.generate_export：decoder_init(首步) + decoder_with_past(后续步)，
present KV 直接作为下一步 past 输入(data_ptr 零拷贝，无 H2D/D2H)。贪心/pad/EOS/logits_processor/
stopping_criteria 逻辑与 torch 原版逐行对齐(D:/project/MinerU rec_ppformulanet_head.py:1069)。

精度结论(真实 PDF 160 公式)：TRT 160/160 完全对齐 fp32 金标准，甚至比 torch-autocast-fp16
(158/160) 更准。速度：B=48 长公式主导组 2.47x。

**无 monkey-patch**：本类不修改 MinerU 任何全局；由 FormulaRecognizer 子类在构造期把
head.generate_export 绑成 self.generate_export(一次注入)。B>引擎上限自动拆块(48→32+16)。
"""
from __future__ import annotations

import torch

from .trt_base import TRTEngine

_N_LAYERS = 6
_TAGS = ("self_k", "self_v", "cross_k", "cross_v")


class MFRDecoderTRT:
    """持有 init/past 两引擎 + 目标 head。调用 generate_export(encoder_outputs, model_kwargs)。"""

    def __init__(self, init_engine_path: str, past_engine_path: str,
                 head=None, stream: torch.cuda.Stream | None = None,
                 debug: bool = False):
        # 两引擎共享同一条专用流，避免跨引擎跨流同步。
        self.stream = stream if stream is not None else torch.cuda.Stream()
        self.eng_init = TRTEngine(init_engine_path, stream=self.stream)
        self.eng_past = TRTEngine(past_engine_path, stream=self.stream)
        try:
            self.max_batch = self.eng_init.profile_batch_max("input_ids", 0)
        except Exception:
            self.max_batch = 32
        self.head = head
        self.debug = debug

    def attach(self, head):
        """绑定目标 head(FormulaRecognizer 构造期调用)。"""
        self.head = head
        return self

    # ---- 引擎调用(零拷贝) ---------------------------------------------------
    def _run_init(self, input_ids, attention_mask, enc_hidden):
        o = self.eng_init.run({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "encoder_hidden_states": enc_hidden,
        })
        present = {nm: o[nm] for nm in self.eng_init.outputs if nm.startswith("present_")}
        return o["logits"], present

    def _run_past(self, input_ids, attention_mask, present):
        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        for i in range(_N_LAYERS):
            for tag in _TAGS:
                feed[f"past_{i}_{tag}"] = present[f"present_{i}_{tag}"]  # 零拷贝
        o = self.eng_past.run(feed)
        present = {nm: o[nm] for nm in self.eng_past.outputs if nm.startswith("present_")}
        return o["logits"], present

    # ---- 对外：替换 head.generate_export ------------------------------------
    def generate_export(self, encoder_outputs, model_kwargs):
        """head.generate_export 的 TRT 替身。返回 input_ids[B,seq] int64(device)。"""
        head = self.head
        device = head.device
        enc_raw = encoder_outputs["last_hidden_state"]
        B = enc_raw.shape[0]

        if self.debug:
            import time as _t
            torch.cuda.synchronize(); _t0 = _t.perf_counter()
            out = self._decode(head, enc_raw, B, device)
            torch.cuda.synchronize()
            print(f"[MFR_DEC] B={B:2d} steps={out.shape[1]:4d}  "
                  f"TRT={(_t.perf_counter() - _t0) * 1000:7.1f}ms", flush=True)
            return out
        return self._decode(head, enc_raw, B, device)

    def _decode(self, head, enc_raw, B, device):
        """投影 enc_raw→[B,144,512](fp32)，B 超引擎上限则按 max_batch 拆块后拼接。"""
        if head.config_decoder.hidden_size != head.encoder_hidden_size:
            enc_hidden = head.enc_to_dec_proj(enc_raw)
        else:
            enc_hidden = enc_raw
        enc_hidden = enc_hidden.float().contiguous()  # TRT I/O 需 fp32

        if B <= self.max_batch:
            return self._decode_core(head, enc_hidden, device)

        # B > 引擎上限：按 max_batch 拆块(OCR 约定)，各块独立解码 → pad 对齐 → 拼接。
        pad_token_id = head.pad_token_id
        rows = []
        for s in range(0, B, self.max_batch):
            rows.append(self._decode_core(head, enc_hidden[s:s + self.max_batch].contiguous(), device))
        max_len = max(r.shape[1] for r in rows)
        padded = []
        for r in rows:
            if r.shape[1] < max_len:
                pad = torch.full((r.shape[0], max_len - r.shape[1]), pad_token_id,
                                 dtype=r.dtype, device=r.device)
                r = torch.cat([r, pad], dim=1)
            padded.append(r)
        return torch.cat(padded, dim=0)

    def _decode_core(self, head, enc_hidden, device):
        """单块(B<=引擎上限)两段式 TRT 贪心解码。enc_hidden: [B,144,512] fp32 → [B,seq]。"""
        B = enc_hidden.shape[0]
        pad_token_id = head.pad_token_id
        eos_token = head.eos_token_id
        max_len = head.max_seq_len

        def pick(logits, input_ids, unfinished):
            next_token_logits = logits[:, -1, :]
            scores = head.logits_processor(input_ids, next_token_logits)
            next_tokens = torch.argmax(scores, dim=-1)
            return next_tokens * unfinished + pad_token_id * (1 - unfinished)

        def all_done(input_ids):
            return bool((torch.cumsum((input_ids == eos_token).to(torch.int64), 1)[:, -1] >= 1).all())

        input_ids = torch.zeros((B, 1), dtype=torch.int64, device=device)  # decoder_start=0
        unfinished = torch.ones(B, dtype=torch.int64, device=device)

        # 首步(init 引擎)
        attn0 = torch.ones((B, 1), dtype=torch.int64, device=device)
        logits, present = self._run_init(input_ids, attn0, enc_hidden)
        next_tokens = pick(logits, input_ids, unfinished)
        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=-1)
        unfinished = unfinished & ~head.stopping_criteria(input_ids).to(torch.int64).to(device)
        if all_done(input_ids):
            return input_ids

        # 后续步(past 引擎，present 零拷贝当 past)
        i_idx = 1
        while i_idx < max_len:
            cur_len = input_ids.shape[1]  # = past_len + 1
            attn = torch.ones((B, cur_len), dtype=torch.int64, device=device)
            logits, present = self._run_past(next_tokens.unsqueeze(1), attn, present)
            next_tokens = pick(logits, input_ids, unfinished)
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=-1)
            unfinished = unfinished & ~head.stopping_criteria(input_ids).to(torch.int64).to(device)
            if all_done(input_ids):
                break
            i_idx += 1
        return input_ids
