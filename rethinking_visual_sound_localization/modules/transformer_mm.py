# Copied and modified from https://github.com/hila-chefer/Transformer-MM-Explainability/blob/main/CLIP_explainability.ipynb
import cv2
import numpy as np
import torch

from ..eval_utils import show_cam_on_image


def get_logits(image_features, text_features, logit_scale):
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    logits_per_image = logit_scale * image_features @ text_features.t()
    return logits_per_image


def interpret(image, query_embedding, model, device, index=None):
    logits_per_image = get_logits(
        model.encode_image(image), query_embedding, model.logit_scale.exp()
    )
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    if index is None:
        index = np.argmax(logits_per_image.cpu().data.numpy(), axis=-1)
    one_hot = np.zeros((1, logits_per_image.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot * logits_per_image)
    model.zero_grad()
    one_hot.backward(retain_graph=True)
    num_layers = 10
    image_attn_blocks = list(
        dict(model.visual.transformer.resblocks.named_children()).values()
    )
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(
        num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype
    ).to(device)
    for i, blk in enumerate(image_attn_blocks):
        if i <= num_layers:
            continue
        grad = blk.attn_grad
        cam = blk.attn_probs
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)
        R += torch.matmul(cam, R)
    R[0, 0] = 0
    image_relevance = R[0, 1:]

    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(
        image_relevance, size=224, mode="bilinear"
    )
    image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (
        image_relevance.max() - image_relevance.min()
    )
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis, image_relevance
