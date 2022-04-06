from collections import defaultdict

import clip
import cv2
import numpy as np
import wav2clip
from PIL import Image
from sklearn.metrics import auc
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose(
        [
            Resize((n_px, n_px), interpolation=BICUBIC),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


preprocess = _transform(224)


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def combine_heatmap_img(img, pred):
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    img = preprocess(img).permute(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min())
    vis = show_cam_on_image(img, pred)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def cal_CIOU(infer, gtmap, thres=0.5):
    infer_map = np.zeros((224, 224))
    infer_map[infer >= thres] = 1
    ciou = np.sum(infer_map * gtmap) / (
        np.sum(gtmap) + np.sum(infer_map * (gtmap == 0))
    )
    return (
        ciou,
        np.sum(infer_map * gtmap),
        (np.sum(gtmap) + np.sum(infer_map * (gtmap == 0))),
    )


def clean_pred(pred):
    pred = normalize_img(pred)
    threshold = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1] / 2)]
    pred[pred < threshold] = 0
    return pred


def normalize_img(value, vmax=None, vmin=None):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if not (vmax - vmin) == 0:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    return value


def process_heatmap(heatmap):
    heatmap_arr = heatmap.data.cpu().numpy()
    heatmap_now = cv2.resize(
        heatmap_arr[0, 0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR
    )
    heatmap_now = normalize_img(-heatmap_now)
    pred = 1 - heatmap_now
    return pred


def compute_metrics(preds, thres=0.5, at=0.5):
    metrics = {}
    ious = [cal_CIOU(pred, gt_map, thres=thres)[0] for _, pred, gt_map in preds]

    results = []
    for i in range(21):
        result = np.sum(np.array(ious) >= 0.05 * i)
        result = result / len(ious)
        results.append(result)
    x = [0.05 * i for i in range(21)]

    metrics["auc"] = auc(x, results)
    metrics["cIoU"] = np.sum(np.array(ious) >= at) / len(ious)
    return metrics


def extract_audio_embeddings(audio, model, device="cpu"):
    return wav2clip.embed_audio(audio, model)


def extract_text_embeddings(x, model, device="cpu"):
    text = clip.tokenize(x).to(device)
    text_features = model.encode_text(text)
    return text_features
