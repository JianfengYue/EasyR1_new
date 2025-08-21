# verl/utils/sft_debug.py
import json
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from ..protocol import DataProto

IGNORE_INDEX = -100

def _get_prompt_len_tensor(data: DataProto, L: int, device: torch.device) -> torch.Tensor:
    """鲁棒地取每样本的 prompt 长度：[B] LongTensor。优先 batch，其次 non-tensor；都没有则用 responses 推断。"""
    if "raw_prompt_len" in data.batch:
        pl = data.batch["raw_prompt_len"]
        return pl.to(dtype=torch.long, device=device)
    rpl_nt = data.non_tensor_batch.get("raw_prompt_len", None)
    if rpl_nt is not None:
        rpl_list = rpl_nt.tolist() if hasattr(rpl_nt, "tolist") else list(rpl_nt)
        return torch.tensor(rpl_list, dtype=torch.long, device=device)
    # fallback: P = L - R（统一 R）
    if "responses" not in data.batch:
        raise KeyError("raw_prompt_len not found, and cannot infer from 'responses'.")
    R = int(data.batch["responses"].size(1))
    return torch.full((data.batch["input_ids"].size(0),), L - R, dtype=torch.long, device=device)

def _safe_decode(tok, ids: List[int]) -> str:
    if tok is None:
        return ""
    try:
        return tok.decode(ids)
    except Exception:
        try:
            return tok.decode([i for i in ids if isinstance(i, int) and i >= 0])
        except Exception:
            return "<DECODE_ERROR>"

@torch.no_grad()
def dump_sft_batch_debug(
    actor_module,
    data: DataProto,
    tokenizer=None,
    max_samples: int = 2,
    around: int = 8,
    do_forward: bool = False,     # True 时会在 TF 上下文做一次前向并计算窗口 CE
    file: Optional[str] = None,   # 若给路径，则把 JSON 写文件
) -> Dict:
    """
    读取并整理与 SFT 有关的关键结构，返回 JSON 友好的 dict，便于打印/保存/上报。
    - 不修改训练图；默认 no_grad。
    - 仅抽取前 max_samples 个样本，避免刷屏。
    """
    inp_ids = data.batch["input_ids"]               # [B, L]
    attn    = data.batch["attention_mask"]          # [B, L]
    pos_ids = data.batch.get("position_ids", None)
    device  = inp_ids.device
    B, L    = inp_ids.shape

    prompt_len = _get_prompt_len_tensor(data, L=L, device=device)  # [B]
    gt_lists: List[List[int]] = data.non_tensor_batch["ground_truth_input_ids"]
    mm_key = "multi_modal_inputs" if "multi_modal_inputs" in data.non_tensor_batch else (
             "multi_modal_data"   if "multi_modal_data"   in data.non_tensor_batch else None)

    # -------- 构造 TF 输入（覆盖 [P, P+R-2] 为 gt[:-1]）--------
    sft_inp = inp_ids.clone()
    pad_id = int(getattr(getattr(actor_module, "config", None), "pad_token_id", 0))

    for b in range(B):
        P = int(prompt_len[b].item())
        gt = gt_lists[b]
        R = len(gt)

        # TF 覆盖：把响应段前缀替换为 gt[:-1]
        if R > 1 and P < L:
            gt_tensor = torch.tensor(gt, dtype=torch.long, device=device)
            gt_tensor = gt_tensor.masked_fill(gt_tensor.eq(IGNORE_INDEX), pad_id)
            n_prefill = min(R - 1, L - P)
            if n_prefill > 0:
                sft_inp[b, P:P + n_prefill] = gt_tensor[:n_prefill]

    # -------- 现在再克隆 label_src（已带 TF 覆盖），并补最后一个响应 token --------
    label_src = sft_inp.clone()  # VERY IMPORTANT: clone AFTER TF overwrite
    for b in range(B):
        P = int(prompt_len[b].item())
        gt = gt_lists[b]
        R = len(gt)
        if R > 0 and P < L:
            last_pos = min(P + R - 1, L - 1)
            # 用 gt[-1] 覆盖“最后一个响应位置”，这样 shift 后窗口标签完整为 gt[0..R-1]
            last_tok = gt[-1]
            if last_tok != IGNORE_INDEX:
                label_src[b, last_pos] = int(last_tok)

    # -------- 收集可读切片（仅前 max_samples 个样本）--------
    per_sample = []
    take = min(B, max_samples)
    for b in range(B):
        P = int(prompt_len[b].item())
        gt = gt_lists[b]
        R = len(gt)

        if b < take:
            t0 = max(P - 1, 0)
            t1 = min(P + R - 2, L - 2)  # 对齐 logits[:, :-1]
            # 原输入边界切片
            prompt_tail_ids = inp_ids[b, max(P - around, 0):P].tolist()
            resp_head_ids   = inp_ids[b, P:min(P + around, L)].tolist()
            # TF 输入边界切片
            tf_prompt_tail_ids = sft_inp[b, max(P - around, 0):P].tolist()
            tf_resp_head_ids   = sft_inp[b, P:min(P + around, L)].tolist()
            # 预期标签窗口切片（来自 TF 后的 label_src shift）
            labels_shifted_full = label_src[b, 1:]  # [L-1]
            if t1 >= t0:
                label_win_ids = labels_shifted_full[t0:t1+1].tolist()
            else:
                label_win_ids = []

            per_sample.append({
                "b": b,
                "P": P, "R": R, "L": L,
                "t0": int(t0), "t1": int(t1),
                "valid_span_len": int(max(0, t1 - t0 + 1)),
                "prompt_tail_ids": prompt_tail_ids,
                "prompt_tail_str": _safe_decode(tokenizer, prompt_tail_ids),
                "resp_head_ids": resp_head_ids,
                "resp_head_str": _safe_decode(tokenizer, resp_head_ids),
                "tf_prompt_tail_ids": tf_prompt_tail_ids,
                "tf_prompt_tail_str": _safe_decode(tokenizer, tf_prompt_tail_ids),
                "tf_resp_head_ids": tf_resp_head_ids,
                "tf_resp_head_str": _safe_decode(tokenizer, tf_resp_head_ids),
                "gt_first_10": gt[:10],
                "gt_last_10": gt[-10:] if len(gt) > 10 else gt,
                "label_window_ids": label_win_ids[:20],  # 避免过长
                "label_window_str": _safe_decode(tokenizer, label_win_ids[:20]),
            })

    out = {
        "B": B, "L": L,
        "has_multi_modal": bool(mm_key),
        "multi_modal_key": mm_key,
        "pad_id": pad_id,
        "num_gt_lists": len(gt_lists),
        "prompt_len_summary": {
            "min": int(prompt_len.min().item()),
            "max": int(prompt_len.max().item()),
            "mean": float(prompt_len.float().mean().item()),
        },
        "samples": per_sample,
    }

    # -------- 可选：前向并计算窗口 CE（有助于定位 loss 异常样本）--------
    if do_forward:
        model_inputs: Dict[str, torch.Tensor] = {"input_ids": sft_inp, "attention_mask": attn}
        if pos_ids is not None:
            model_inputs["position_ids"] = pos_ids
        if mm_key:
            model_inputs[mm_key] = data.non_tensor_batch[mm_key]

        outputs = actor_module(use_cache=False, **model_inputs)
        logits = outputs.logits  # [B, L, V]
        V = logits.size(-1)
        logits_shifted = logits[:, :-1, :]
        attn_shifted = attn[:, 1:].bool()
        labels_full = torch.full((B, L - 1), IGNORE_INDEX, dtype=torch.long, device=device)

        # 写入各自窗口标签（来自 TF 后的 label_src shift）
        for b in range(B):
            P = int(prompt_len[b].item())
            R = len(gt_lists[b])
            t0 = max(P - 1, 0)
            t1 = min(P + R - 2, L - 2)
            if t1 >= t0:
                labels_full[b, t0:t1 + 1] = label_src[b, 1:][t0:t1 + 1]

        labels_full = labels_full.masked_fill(~attn_shifted, IGNORE_INDEX)
        ce_tok = F.cross_entropy(
            logits_shifted.reshape(-1, V), labels_full.reshape(-1),
            ignore_index=IGNORE_INDEX, reduction="none"
        ).view(B, L - 1)
        valid = labels_full.ne(IGNORE_INDEX)
        per_sample_ce = (ce_tok * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        out["per_sample_ce_mean"] = float(per_sample_ce.mean().item())
        out["per_sample_ce_first"] = float(per_sample_ce[0].item()) if B > 0 else None

    if file:
        try:
            with open(file, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[dump_sft_batch_debug] failed to write file={file}: {e}")

    return out
