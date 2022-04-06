import subprocess
from subprocess import PIPE
from subprocess import Popen

import cv2
import librosa
import skvideo.io
import tqdm
from PIL import Image

from rethinking_visual_sound_localization.eval_utils import combine_heatmap_img
from rethinking_visual_sound_localization.models import RCGrad


def get_audio(input_mp4):
    command = "ffmpeg -i {0}.mp4 -ab 160k -ac 2 -ar 44100 -vn {0}.wav".format(
        input_mp4.split(".")[0]
    )
    subprocess.call(command, shell=True)
    return "{0}.wav".format(input_mp4.split(".")[0])


def get_fps(input_mp4):
    cap = cv2.VideoCapture(input_mp4)
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps


def generate_video(pred_images, video_only_file, fps):
    ffmpeg_filter = f"minterpolate='mi_mode=mci:me=hexbs:me_mode=bidir:mc_mode=aobmc:vsbmc=1:mb_size=8:search_param=32:fps=30'"
    p = Popen(
        [
            "ffmpeg",
            "-y",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-r",
            "{}".format(fps),
            "-i",
            "-",
            "-b:v",
            "10M",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-strict",
            "-2",
            "-filter:v",
            f"{ffmpeg_filter}",
            video_only_file,
        ],
        stdin=PIPE,
    )

    for im in tqdm.tqdm(pred_images):
        Image.fromarray(im).save(p.stdin, "PNG")
    p.stdin.close()
    p.wait()


def mix_audio_video(audio_file, video_only_file, audio_video_file):
    cmd = 'ffmpeg -y -i {} -i "{}" -c:v copy -c:a aac {}'.format(
        video_only_file, audio_file, audio_video_file
    )
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    input_mp4 = "PATH_TO_MP4_VIDEO_FILE"
    output_mp4_prefix = "output"
    fps = int(get_fps(input_mp4))

    video = skvideo.io.vread(input_mp4)
    audio_file = get_audio(input_mp4)
    audio, sr = librosa.load(audio_file, sr=16000)

    rc_grad = RCGrad()

    pred_images = []
    for i in tqdm.tqdm(range(0, video.shape[0])):
        image = video[i]
        img = Image.fromarray(image)
        aud = audio[
            int(max(0, (i / fps) * sr - sr / 2)) : int(
                min((i / fps) * sr + sr / 2, len(audio))
            )
        ]
        vis = combine_heatmap_img(img, rc_grad.pred_audio(img, aud))
        pred_images.append(
            cv2.resize(
                vis,
                dsize=(video.shape[2], video.shape[1]),
                interpolation=cv2.INTER_CUBIC,
            )
        )

    generate_video(pred_images, "{}_video_only.mp4".format(output_mp4_prefix), fps)
    mix_audio_video(
        audio_file,
        "{}_video_only.mp4".format(output_mp4_prefix),
        "{}_mix_audio.mp4".format(output_mp4_prefix),
    )
