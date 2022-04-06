import clip
import torch
import wav2clip

from .eval_utils import clean_pred
from .eval_utils import extract_audio_embeddings
from .eval_utils import extract_text_embeddings
from .eval_utils import preprocess
from .modules import transformer_mm_clip
from .modules.gradcam import GradCAM
from .modules.resnet import BasicBlock
from .modules.resnet import resnet18
from .modules.resnet import ResNetSpec
from .modules.transformer_mm import interpret

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_URL = "https://github.com/hohsiangwu/rethinking-visual-sound-localization/releases/download/v0.1.0-alpha/rc_grad.ckpt"


class RCGrad:
    def __init__(self):
        super(RCGrad).__init__()

        image_encoder = resnet18(modal="vision", pretrained=False)
        audio_encoder = ResNetSpec(
            BasicBlock,
            [2, 2, 2, 2],
            pool="avgpool",
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
        )
        checkpoint = torch.hub.load_state_dict_from_url(
            MODEL_URL, map_location=device, progress=True
        )
        image_encoder.load_state_dict(
            {
                k.replace("image_encoder.", ""): v
                for k, v in checkpoint["state_dict"].items()
                if k.startswith("image_encoder")
            }
        )
        audio_encoder.load_state_dict(
            {
                k.replace("audio_encoder.", ""): v
                for k, v in checkpoint["state_dict"].items()
                if k.startswith("audio_encoder")
            }
        )

        target_layers = [image_encoder.layer4[-1]]
        self.audio_encoder = audio_encoder
        self.cam = GradCAM(
            model=image_encoder,
            target_layers=target_layers,
            use_cuda=False,
            reshape_transform=None,
        )

    def pred_audio(self, img, audio):
        grayscale_cam = self.cam(
            input_tensor=preprocess(img).unsqueeze(0),
            targets=[self.audio_encoder(torch.from_numpy(audio).unsqueeze(0))],
        )
        pred_audio = grayscale_cam[0, :]
        pred_audio = clean_pred(pred_audio)
        return pred_audio


class CLIPTran:
    def __init__(self):
        super(CLIPTran).__init__()

        clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
        self.clip_model = clip_model

        wav2clip_model = wav2clip.get_model()
        self.wav2clip_model = wav2clip_model

        transformer_mm_clip.clip._MODELS = {
            "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
            "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
        }

        transformer_mm_model, _ = transformer_mm_clip.load(
            "ViT-B/32", device=device, jit=False
        )
        self.transformer_mm_model = transformer_mm_model

    def pred_audio(self, img, audio):
        _, pred_audio = interpret(
            image=preprocess(img).unsqueeze(0).to(device),
            query_embedding=torch.from_numpy(
                extract_audio_embeddings(audio, model=self.wav2clip_model)
            ),
            model=self.transformer_mm_model,
            device=device,
            index=0,
        )
        pred_audio = clean_pred(pred_audio)
        return pred_audio

    def pred_text(self, img, text):
        _, pred_text = interpret(
            image=preprocess(img).unsqueeze(0).to(device),
            query_embedding=extract_text_embeddings(text, model=self.clip_model),
            model=self.transformer_mm_model,
            device=device,
            index=0,
        )
        pred_text = clean_pred(pred_text)
        return pred_text
