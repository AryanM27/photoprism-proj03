import open_clip
import torch
from PIL import Image
from torchvision import transforms


class Embedder:
    def __init__(self, model_name: str = "ViT-B-32", device_str: str = "auto", checkpoint_path: str = None):
        if device_str == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device_str

        self.device = torch.device(device)

        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained="openai",
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # If Milind provides a fine-tuned checkpoint, load it here
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device).eval()

        self.image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    @torch.no_grad()
    def embed_text(self, text: str) -> list[float]:
        tokens = self.tokenizer([text]).to(self.device)
        emb = self.model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb[0].cpu().tolist()

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> list[float]:
        tensor = self.image_transforms(image.convert("RGB")).unsqueeze(0).to(self.device)
        emb = self.model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb[0].cpu().tolist()
