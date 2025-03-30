import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

DATASET_PATH = "/Users/anchitmulye/Downloads/LanguageDetectionDataset"
SELECTED_LANGUAGES = ["Hindi", "Marathi", "Gujarati"]


# Task A: MFCC Feature Extraction
class MFCCExtractor:
    def __init__(self, sample_rate=16000, n_mfcc=13):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': 400,
                'n_mels': 128,
                'hop_length': 160,
                'mel_scale': 'htk',
            }
        )

    def extract_features(self, file_path):
        try:
            waveform, sample_rate = torchaudio.load(file_path)

            if sample_rate != self.sample_rate:
                resampler = T.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            mfcc_features = self.mfcc_transform(waveform)
            mfcc_np = mfcc_features.squeeze().numpy()

            return mfcc_np
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None


def visualize_mfcc(mfcc_features, title="MFCC Spectrogram"):
    plt.figure(figsize=(10, 4))
    plt.imshow(mfcc_features, aspect='auto', origin='lower', interpolation='none')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.ylabel('MFCC Coefficients')
    plt.xlabel('Time Frames')
    plt.tight_layout()
    plt.show()


def calculate_mfcc_statistics(mfcc_features):
    stats = {
        'mean': np.mean(mfcc_features, axis=1),
        'variance': np.var(mfcc_features, axis=1)
    }
    return stats


def process_dataset(extractor):
    features = {lang: [] for lang in SELECTED_LANGUAGES}
    all_features = []
    all_labels = []

    for idx, language in enumerate(SELECTED_LANGUAGES):
        language_dir = os.path.join(DATASET_PATH, language)

        audio_files = [os.path.join(language_dir, f) for f in os.listdir(language_dir)
                       if f.endswith('.wav') or f.endswith('.mp3')]

        for audio_file in audio_files:
            mfcc_features = extractor.extract_features(audio_file)

            if mfcc_features is not None:
                features[language].append(mfcc_features)
                feature_vector = np.mean(mfcc_features, axis=1)
                all_features.append(feature_vector)
                all_labels.append(idx)

    return features, np.array(all_features), np.array(all_labels)


def visualize_representative_samples(features):
    for language in SELECTED_LANGUAGES:
        representative_mfcc = features[language][0]
        visualize_mfcc(representative_mfcc, f"MFCC Spectrogram - {language}")

        stats = calculate_mfcc_statistics(representative_mfcc)
        print(f"Statistics for {language}:")
        print(f"Mean: {stats['mean']}")
        print(f"Variance: {stats['variance']}")
        print("-" * 50)


def compare_language_mfccs(features):
    language_stats = {}

    for language in SELECTED_LANGUAGES:
        all_mfccs = np.concatenate([mfcc for mfcc in features[language]], axis=1)
        language_stats[language] = calculate_mfcc_statistics(all_mfccs)

    plt.figure(figsize=(12, 6))
    for language in SELECTED_LANGUAGES:
        plt.plot(language_stats[language]['mean'], label=language)

    plt.title("Comparison of Mean MFCC Coefficients Across Languages")
    plt.xlabel("MFCC Coefficient Index")
    plt.ylabel("Mean Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    for language in SELECTED_LANGUAGES:
        plt.plot(language_stats[language]['variance'], label=language)

    plt.title("Comparison of Variance in MFCC Coefficients Across Languages")
    plt.xlabel("MFCC Coefficient Index")
    plt.ylabel("Variance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


class LanguageDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class LanguageClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LanguageClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def train_model(train_loader, val_loader, input_size, hidden_size=64, num_classes=3, epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LanguageClassifier(input_size, hidden_size, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")
            model.eval()
            correct = 0
            total = 0
            val_loss = 0.0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f"Validation - Loss: {val_loss / len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

    print("Training complete")
    return model


def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, all_labels, all_preds


def display_results(y_true, y_pred):
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=SELECTED_LANGUAGES))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=SELECTED_LANGUAGES, yticklabels=SELECTED_LANGUAGES)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()


def main():
    # Task A: Extract and analyze MFCC features
    print("Initializing MFCC extractor...")
    extractor = MFCCExtractor()

    print("Extracting MFCC features...")
    features, X, y = process_dataset(extractor)

    print("Visualizing representative samples...")
    visualize_representative_samples(features)

    print("Comparing MFCC features across languages...")
    compare_language_mfccs(features)

    # Task B: Train and evaluate classifier with PyTorch
    print("Preparing data for PyTorch...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create datasets and dataloaders
    train_dataset = LanguageDataset(X_train, y_train)
    test_dataset = LanguageDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Train model
    print("Training PyTorch model...")
    input_size = X_train.shape[1]  # Number of MFCC coefficients
    model = train_model(train_loader, test_loader, input_size)

    # Evaluate model
    print("Evaluating model...")
    accuracy, y_true, y_pred = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Display results
    display_results(y_true, y_pred)

    # # Save model
    # torch.save(model.state_dict(), "language_classifier.pth")
    # print("Model saved as 'language_classifier.pth'")


if __name__ == "__main__":
    main()
