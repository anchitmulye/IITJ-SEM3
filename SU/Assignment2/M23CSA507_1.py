import torch
import torch.nn as nn
import torch.optim as optim
from transformers import Wav2Vec2Model, Wav2Vec2Config
from peft import LoraConfig, get_peft_model
# from torchmetrics.classification import MultilabelAccuracy
import soundfile as sf
import torch
import fire
import torch.nn.functional as F
from torchaudio.transforms import Resample
from models.ecapa_tdnn import ECAPA_TDNN_SMALL

class SpeakerVerificationModel(nn.Module):
    def __init__(self, pretrained_model_name):
        super().__init__()
        # Load pre-trained model (e.g., HuBERT, Wav2Vec2)
        self.base_model = Wav2Vec2Model.from_pretrained(pretrained_model_name)

        # LoRA configuration
        lora_config = LoraConfig(
            r=16,  # Rank of low-rank adaptation
            lora_alpha=32,
            target_modules=["attention"],
            lora_dropout=0.1
        )
        self.model = get_peft_model(self.base_model, lora_config)

        # Speaker embedding layer
        self.speaker_embedding = nn.Linear(self.base_model.config.hidden_size, 512)

    def forward(self, audio_input):
        # Extract features
        outputs = self.base_model(audio_input)
        hidden_states = outputs.last_hidden_state

        # Global average pooling
        pooled_output = torch.mean(hidden_states, dim=1)

        # Speaker embedding
        speaker_embed = self.speaker_embedding(pooled_output)
        return speaker_embed


class ArcFaceLoss(nn.Module):
    def __init__(self, margin=0.5, scale=64):
        super().__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, embeddings, labels):
        # ArcFace loss implementation
        # Compute cosine similarity with angular margin
        pass


def train_speaker_verification(model, train_loader, val_loader):
    # Training loop with LoRA and ArcFace
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = ArcFaceLoss()

    # for epoch in range(num_epochs):
    #     model.train()
    #     for batch in train_loader:
    #         audio, labels = batch
    #
    #         optimizer.zero_grad()
    #         embeddings = model(audio)
    #         loss = criterion(embeddings, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    # return model


def evaluate_speaker_verification(model, test_loader):
    # Compute evaluation metrics
    model.eval()

    # Compute EER, TAR@1%FAR, Speaker Identification Accuracy
    # with torch.no_grad():
        # accuracy = MultilabelAccuracy()
        # Evaluation implementation

    # return {
    #     'EER': eer,
    #     'TAR@1%FAR': tar_at_far,
    #     'Speaker_Identification_Accuracy': accuracy
    # }


if __name__ == '__main__':
    dataset_dir = "/Users/anchitmulye/Downloads/Speech2025Datasets"
    model_weights = "/Users/anchitmulye/Downloads/wav2vec2_xlsr_SV_fixed.th"