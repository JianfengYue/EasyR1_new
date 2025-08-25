import torch
from ..protocol import DataProto

def _infer_autocast_dtype(module: torch.nn.Module):
    # 训练中一般权重全一种 dtype；若混合则优先 bf16
    dtypes = {p.dtype for p in module.parameters() if p.requires_grad}
    return torch.bfloat16 if torch.bfloat16 in dtypes else torch.float16

def compute_sft_loss_tf(actor_module: torch.nn.Module, batch: DataProto) -> torch.Tensor:
    """
    计算 SFT 交叉熵（教师强制）。要求 batch 含 sft_input_ids, sft_attention_mask, labels，
    其中 labels 的 prompt 区为 -100。
    """
    device = next(actor_module.parameters()).device

    # 尽量用 non_blocking + pin_memory
    input_ids      = batch.batch["sft_input_ids"].to(device, non_blocking=True)
    attention_mask = batch.batch["sft_attention_mask"].to(device, non_blocking=True)
    labels         = batch.batch["sft_labels"].to(device, non_blocking=True)

    if device.type == "cuda":
        with torch.autocast("cuda", dtype=_infer_autocast_dtype(actor_module)):
            outputs = actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
    else:
        outputs = actor_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )

    # 防止偶发 NaN 破坏日志/反传（一般不会，稳妥起见）
    loss = outputs.loss
    if torch.isnan(loss):
        loss = torch.nan_to_num(loss)

    return loss
