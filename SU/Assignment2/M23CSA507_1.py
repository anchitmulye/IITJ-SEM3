import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor
)
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

# Constants
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

VOX1_PATH = "/Users/anchitmulye/Downloads/Speech2025Datasets/vox1"
VOX2_PATH = "/Users/anchitmulye/Downloads/Speech2025Datasets/vox2"
TRIALS_PATH = "/Users/anchitmulye/Downloads/Speech2025Datasets/veri_test2.txt"

# Select model - choose one from the available options
MODEL_NAME = "facebook/wav2vec2-xls-r-300m"


class SpeakerEncoder(nn.Module):
    def __init__(self, model_name, embedding_dim=256):
        super(SpeakerEncoder, self).__init__()
        self.base_model = Wav2Vec2Model.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        # self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

        for param in self.base_model.parameters():
            param.requires_grad = False

        hidden_size = self.base_model.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x, attention_mask=None):
        outputs = self.base_model(x, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            pooled = torch.sum(hidden_states * mask, dim=1) / torch.sum(mask, dim=1)
        else:
            pooled = torch.mean(hidden_states, dim=1)

        embedding = self.projector(pooled)
        return F.normalize(embedding, p=2, dim=1)  # L2 normalization

    def preprocess_audio(self, file_path, max_length=6 * SAMPLE_RATE):
        waveform, sample_rate = torchaudio.load(file_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sample_rate != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
            waveform = resampler(waveform)

        if waveform.shape[1] > max_length:
            waveform = waveform[:, :max_length]
        elif waveform.shape[1] < max_length:
            padding = torch.zeros((1, max_length - waveform.shape[1]))
            waveform = torch.cat([waveform, padding], dim=1)

        return waveform.squeeze(0)

    def extract_embedding(self, file_path):
        waveform = self.preprocess_audio(file_path)
        inputs = self.processor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            embedding = self.forward(**inputs)

        return embedding.cpu().numpy()[0]


class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_size, num_classes, scale=30.0, margin=0.5):
        super(ArcFaceLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.cos_m = torch.cos(torch.tensor(margin))
        self.sin_m = torch.sin(torch.tensor(margin))
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_size))
        nn.init.xavier_normal_(self.weight)

    def forward(self, embeddings, labels):
        normalized_weights = F.normalize(self.weight, p=2, dim=1)

        cos_theta = F.linear(embeddings, normalized_weights)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        cos_theta_with_margin = torch.where(one_hot.bool(), cos_theta_m, cos_theta)

        output = cos_theta_with_margin * self.scale
        loss = F.cross_entropy(output, labels)

        return loss


class VoxCelebDataset(Dataset):
    def __init__(self, root_dir, processor, split='train', max_speakers=None):
        self.root_dir = root_dir
        self.processor = processor
        self.sample_rate = SAMPLE_RATE
        self.max_length = 6 * self.sample_rate

        self.speaker_ids = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        if max_speakers:
            if split == 'train':
                self.speaker_ids = self.speaker_ids[:max_speakers]
            elif split == 'test':
                self.speaker_ids = self.speaker_ids[max_speakers:]

        self.file_paths = []
        self.labels = []

        self.speaker_to_idx = {speaker_id: i for i, speaker_id in enumerate(self.speaker_ids)}

        for speaker_id in tqdm(self.speaker_ids, desc="Loading dataset"):
            speaker_dir = os.path.join(root_dir, speaker_id)
            for subdir in os.listdir(speaker_dir):
                subdir_path = os.path.join(speaker_dir, subdir)
                if os.path.isdir(subdir_path):
                    for filename in os.listdir(subdir_path):
                        if filename.endswith('.wav'):
                            file_path = os.path.join(subdir_path, filename)
                            self.file_paths.append(file_path)
                            self.labels.append(self.speaker_to_idx[speaker_id])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            waveform, sample_rate = torchaudio.load(file_path)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)

            if waveform.shape[1] > self.max_length:
                start = torch.randint(0, waveform.shape[1] - self.max_length, (1,))
                waveform = waveform[:, start:start + self.max_length]
            elif waveform.shape[1] < self.max_length:
                padding = torch.zeros((1, self.max_length - waveform.shape[1]))
                waveform = torch.cat([waveform, padding], dim=1)

            waveform = waveform.squeeze(0)

            inputs = self.processor(waveform, sampling_rate=self.sample_rate, return_tensors="pt")
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}

            return inputs, label

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return self.__getitem__(0)


def apply_lora(model):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["projector.0", "projector.3"],
        lora_dropout=0.1,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    print(f"Model trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    return model


def evaluate_verification(model, trials_path, vox_path):
    print("Evaluating speaker verification...")

    with open(trials_path, 'r') as f:
        lines = f.readlines()

    scores = []
    labels = []

    for line in tqdm(lines, desc="Processing trials"):
        parts = line.strip().split()

        if len(parts) == 5:
            label, spk1, spk2, utt1, utt2 = parts
            file1 = os.path.join(vox_path, spk1, utt1, f"{utt2}.wav")
        else:
            label, file1, file2 = parts
            file1 = os.path.join(vox_path, file1)
            file2 = os.path.join(vox_path, file2)

        label = int(label)

        try:
            emb1 = model.extract_embedding(file1)
            emb2 = model.extract_embedding(file2)

            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

            scores.append(similarity)
            labels.append(label)
        except Exception as e:
            print(f"Error processing {file1} or {file2}: {e}")

    fpr, tpr, thresholds = roc_curve(labels, scores)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    idx = np.argmin(np.abs(fpr - 0.01))
    tar_at_1_far = tpr[idx]

    thresh = interp1d(fpr, thresholds)(eer)

    predictions = [1 if s >= thresh else 0 for s in scores]
    accuracy = accuracy_score(labels, predictions)

    print(f"EER: {eer * 100:.2f}%")
    print(f"TAR@1%FAR: {tar_at_1_far * 100:.2f}%")
    print(f"Speaker Identification Accuracy: {accuracy * 100:.2f}%")

    return {
        "EER": eer * 100,
        "TAR@1%FAR": tar_at_1_far * 100,
        "Accuracy": accuracy * 100
    }


def train_model(model, train_dataloader, val_dataloader, num_classes, num_epochs=10):
    model.to(DEVICE)

    criterion = ArcFaceLoss(embedding_size=256, num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': 1e-4},
        {'params': criterion.parameters(), 'lr': 1e-3}
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for inputs, labels in progress_bar:
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            embeddings = model(**inputs)
            loss = criterion(embeddings, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'train_loss': loss.item()})

        train_loss = running_loss / len(train_dataloader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in tqdm(val_dataloader, desc="Validation"):
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                labels = labels.to(DEVICE)

                embeddings = model(**inputs)
                loss = criterion(embeddings, labels)

                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'epoch': epoch
            }, 'best_speaker_model.pth')

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_curves.png')

    return model


def main():
    model = SpeakerEncoder(MODEL_NAME)
    print(f"Model initialized: {MODEL_NAME}")

    model.to(DEVICE)
    print("Evaluating pre-trained model...")
    pretrained_metrics = evaluate_verification(model, TRIALS_PATH, VOX1_PATH)

    processor = model.processor
    train_dataset = VoxCelebDataset(VOX2_PATH, processor, split='train', max_speakers=100)
    val_dataset = VoxCelebDataset(VOX2_PATH, processor, split='test', max_speakers=100)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    print(f"Training with {len(train_dataset)} samples, validating with {len(val_dataset)} samples")

    lora_model = apply_lora(model)

    print("Fine-tuning model...")
    trained_model = train_model(
        lora_model,
        train_dataloader,
        val_dataloader,
        num_classes=len(train_dataset.speaker_to_idx),
        num_epochs=10
    )

    print("Evaluating fine-tuned model...")
    finetuned_metrics = evaluate_verification(trained_model, TRIALS_PATH, VOX1_PATH)

    print("\nPerformance Comparison:")
    print("-" * 50)
    print(f"{'Metric':<20} {'Pre-trained':<15} {'Fine-tuned':<15} {'Improvement':<15}")
    print("-" * 50)

    for metric in ["EER", "TAR@1%FAR", "Accuracy"]:
        pre = pretrained_metrics[metric]
        fine = finetuned_metrics[metric]

        if metric == "EER":
            improvement = pre - fine
            print(f"{metric:<20} {pre:.2f}%{' ':<10} {fine:.2f}%{' ':<10} {improvement:.2f}%{' ':<10}")
        else:
            improvement = fine - pre
            print(f"{metric:<20} {pre:.2f}%{' ':<10} {fine:.2f}%{' ':<10} {improvement:.2f}%{' ':<10}")

    print("-" * 50)


if __name__ == "__main__":
    main()
