import os
import librosa
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns


class Solution3:
    def __init__(self):
        self.dataset = "Dataset"

    def _extract_formants(self, y, sr):
        lpc_order = 12
        a = librosa.lpc(y, order=lpc_order)
        roots = np.roots(a)
        roots = [r for r in roots if np.imag(r) >= 0]
        angles = np.angle(roots)
        freqs = np.sort(angles * (sr / (2 * np.pi)))
        return freqs[:3] if len(freqs) >= 3 else [0, 0, 0]

    def _extract_f0(self, y, sr):
        autocorr = np.correlate(y, y, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        peaks, _ = scipy.signal.find_peaks(autocorr, height=0)
        if len(peaks) > 1:
            return sr / peaks[1]
        return 0

    def extract_features(self, file_path, vowel_label):
        y, sr = librosa.load(file_path, sr=None)
        f1, f2, f3 = self._extract_formants(y, sr)
        f0 = self._extract_f0(y, sr)
        return {"F1": f1, "F2": f2, "F3": f3, "F0": f0, "Vowel": vowel_label}

    def process_dataset(self):
        feature_data = []
        for gender in ["Male", "Female"]:
            gender_path = os.path.join(self.dataset, gender)
            for vowel in ["a", "e", "i", "o", "u"]:
                vowel_path = os.path.join(gender_path, vowel)
                for file in os.listdir(vowel_path):
                    if file.endswith(".wav"):
                        file_path = os.path.join(vowel_path, file)
                        feature_data.append(self.extract_features(file_path, vowel))
        return pd.DataFrame(feature_data)

    def train(self, data_df):
        X = data_df[["F1", "F2", "F3", "F0"]]
        y = data_df["Vowel"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

        # Train KNN classifier
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        print("Confusion Matrix:")
        print(conf_matrix)
        print("Accuracy KNN:", accuracy)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=data_df["F1"], y=data_df["F2"], hue=data_df["Vowel"], palette="viridis")
        plt.xlabel("F1 (Hz)")
        plt.ylabel("F2 (Hz)")
        plt.title("Vowel Space")
        plt.legend(title="Vowel")
        plt.show()

    def train2(self, data_df):
        X = data_df[["F1", "F2", "F3", "F0"]]
        y = data_df["Vowel"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=45)

        # Train Gaussian Mixture Model (GMM)
        gmm = GaussianMixture(n_components=len(np.unique(y_train)), covariance_type='full', random_state=45)
        gmm.fit(X_train)

        y_pred = gmm.predict(X_test)
        y_pred_labels = label_encoder.inverse_transform(y_pred)
        y_test_labels = label_encoder.inverse_transform(y_test)

        conf_matrix = confusion_matrix(y_test_labels, y_pred_labels, labels=label_encoder.classes_)
        accuracy = accuracy_score(y_test_labels, y_pred_labels)

        print("Confusion Matrix:\n", conf_matrix)
        print("Accuracy:", accuracy)


if __name__ == "__main__":
    solution = Solution3()
    data_df = solution.process_dataset()
    solution.train(data_df)
    solution.train2(data_df)
