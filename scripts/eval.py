import tqdm

from rethinking_visual_sound_localization.data import FlickrSoundNetDataset
from rethinking_visual_sound_localization.eval_utils import compute_metrics
from rethinking_visual_sound_localization.models import CLIPTran
from rethinking_visual_sound_localization.models import RCGrad

if __name__ == "__main__":
    # Download Flickr_SoundNet https://github.com/ardasnck/learning_to_localize_sound_source#preparation as data_root
    flickr_soundnet_dataset = FlickrSoundNetDataset(
        data_root="PATH_TO_Flickr_SoundNet/"
    )

    rc_grad = RCGrad()
    preds_rc_grad = []
    for ft, img, audio, gt_map in tqdm.tqdm(flickr_soundnet_dataset):
        preds_rc_grad.append((ft, rc_grad.pred_audio(img, audio), gt_map))
    metrics_rc_grad = compute_metrics(preds_rc_grad)

    clip_tran = CLIPTran()
    preds_clip_tran = []
    for ft, img, audio, gt_map in tqdm.tqdm(flickr_soundnet_dataset):
        preds_clip_tran.append((ft, clip_tran.pred_audio(img, audio), gt_map))
    metrics_clip_tran = compute_metrics(preds_clip_tran)
