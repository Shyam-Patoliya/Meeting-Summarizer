import math
from logging import getLogger
import torch
import torch.nn.functional as F
from .utils import get_negative_inf_value, to_2tuple

logger = getLogger()


def resize_local_position_embedding(orig_pos_embed, grid_size):
    new_grid_size = to_2tuple(grid_size)
    orig_grid_size = to_2tuple(int(math.sqrt(len(orig_pos_embed) - 1)))
    new_seq_len = new_grid_size[0] * new_grid_size[1] + 1

    new_pos_emb_tok, new_pos_emb_img = (
        orig_pos_embed[:1],
        orig_pos_embed[1:],
    )
    logger.info(f"resizing position embedding grid-size from {orig_grid_size} to {new_grid_size}")

    new_pos_emb_img = new_pos_emb_img.reshape(1, orig_grid_size[0], orig_grid_size[1], -1).permute(0, 3, 1, 2)

    new_pos_emb_img = F.interpolate(
        new_pos_emb_img,
        size=new_grid_size,
        mode="bilinear",
        align_corners=True,
    )
    new_pos_emb_img = new_pos_emb_img.permute(0, 2, 3, 1).reshape(1, new_grid_size[0] * new_grid_size[1], -1)[0]
    new_pos_embed = torch.cat([new_pos_emb_tok, new_pos_emb_img], dim=0)
    return new_pos_embed


def initialize_global_position_embedding_from_local(pos_and_cls_embed, grid_size, x_scale, y_scale):
    pos_embed = pos_and_cls_embed[1:]
    cls_embed = pos_and_cls_embed[0].view(1, 1, 1, -1)
    grid_size = to_2tuple(grid_size)
    new_pos_emb_img = pos_embed.reshape(1, grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2)
    new_grid_size = (x_scale * grid_size[0], y_scale * grid_size[1])
    new_pos_emb_img = F.interpolate(
        new_pos_emb_img,
        size=new_grid_size,
        mode="bilinear",
        align_corners=True,
    )
    new_pos_emb_img = new_pos_emb_img.permute(0, 2, 3, 1)
    new_pos_emb_img = new_pos_emb_img.view(x_scale, grid_size[0], y_scale, grid_size[1], -1)
    new_pos_emb_img = new_pos_emb_img.permute(0, 2, 1, 3, 4).contiguous()
    new_pos_emb_img = new_pos_emb_img.reshape(x_scale, y_scale, grid_size[0] * grid_size[1], -1)
    cls_embed = cls_embed.expand(x_scale, y_scale, -1, -1)
    pos_and_cls_embed = torch.cat([cls_embed, new_pos_emb_img], dim=2)
    return pos_and_cls_embed


def resize_global_position_embedding(pos_and_cls_embed, grid_size, x_scale, y_scale):
    pos_embed = pos_and_cls_embed[:, :, 1:]
    cls_embed = pos_and_cls_embed[:, :, 0].unsqueeze(2)

    xs_old, ys_old, ntok, dim = pos_embed.shape
    old_grid_size = int(math.sqrt(ntok))

    pos_embed = pos_embed.view(xs_old, ys_old, old_grid_size, old_grid_size, dim)
    pos_embed = pos_embed.permute(0, 2, 1, 3, 4).contiguous()
    pos_embed = pos_embed.view(xs_old * old_grid_size, ys_old * old_grid_size, dim)
    pos_embed = pos_embed.unsqueeze(0)

    new_size = (grid_size[0] * x_scale, grid_size[1] * y_scale)
    pos_embed = pos_embed.permute(0, 3, 1, 2)
    pos_embed_resized = F.interpolate(
        pos_embed,
        size=new_size,
        mode="bilinear",
        align_corners=True,
    )
    pos_embed = pos_embed_resized.permute(0, 2, 3, 1)[0]

    pos_embed = pos_embed.view(x_scale, grid_size[0], y_scale, grid_size[1], dim)
    pos_embed = pos_embed.permute(0, 2, 1, 3, 4).contiguous()
    pos_embed = pos_embed.view(x_scale, y_scale, grid_size[0] * grid_size[1], dim)

    cls_embed = cls_embed.permute(2, 3, 0, 1)
    cls_embed_resized = F.interpolate(
        cls_embed,
        size=(x_scale, y_scale),
        mode="bilinear",
        align_corners=True,
    )
    cls_embed = cls_embed_resized.permute(2, 3, 0, 1)

    pos_and_cls_embed = torch.cat([cls_embed, pos_embed], dim=2)

    return pos_and_cls_embed


def build_encoder_attention_mask(
    x: torch.Tensor,
    ar: torch.Tensor,
    ntok: int,
    num_chunks: int,
    n_heads: int,
):

    masks = []
    for arx in ar:
        mask_i = torch.ones((num_chunks, x.shape[2], 1), dtype=x.dtype)
        mask_i[: arx[0] * arx[1], :ntok] = 0
        mask_i = mask_i.view(num_chunks * x.shape[2], -1)
        mask_i = mask_i @ mask_i.T * get_negative_inf_value(x.dtype)
        mask_i = mask_i.unsqueeze(0)
        masks.append(mask_i)
    masks = torch.stack(masks).to(x.device).expand(-1, n_heads, -1, -1)
    return masks


def expand_num_tokens_to_mult8(x):
    num_pad_tokens = 8 - (x.shape[-2] % 8)
    if num_pad_tokens == 0:
        return x, 0
    else:
        return (
            torch.cat(
                [
                    x,
                    torch.zeros(
                        (x.shape[0], x.shape[1], num_pad_tokens, x.shape[-1]),
                        dtype=x.dtype,
                        device=x.device,
                    ),
                ],
                dim=-2,
            ),
            num_pad_tokens,
        )


def contract_num_tokens_from_mult8(x, num_pad_tokens):
    if num_pad_tokens == 0:
        return x
    return x[:, :, :-num_pad_tokens]
