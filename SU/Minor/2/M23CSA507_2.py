import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


class Solution2:
    def __init__(self, sample_dir):
        self.audio_dir = sample_dir

    def _extract_features(self, file_path):
        y, sr = librosa.load(file_path, sr=None)

        # Zero-Crossing Rate (ZCR)
        zcr = librosa.feature.zero_crossing_rate(y)
        avg_zcr = np.mean(zcr)

        # Short-Time Energy (STE)
        frame_length = 2048
        hop_length = 512
        ste = np.array([np.sum(np.abs(y[i:i + frame_length] ** 2)) for i in range(0, len(y), hop_length)])
        avg_ste = np.mean(ste)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)   # first 13 coefficients
        avg_mfccs = np.mean(mfccs, axis=1)

        return {
            "File Name": os.path.basename(file_path),
            "Zero-Crossing Rate": avg_zcr,
            "Short-Time Energy": avg_ste,
            **{f"MFCC {i + 1}": avg_mfccs[i] for i in range(13)}
        }

    def solve(self):
        file_list = [os.path.join(self.audio_dir, f) for f in os.listdir(self.audio_dir) if f.endswith(".mp3")]
        feature_data = []
        for file in file_list:
            features = self._extract_features(file)
            feature_data.append(features)
        return pd.DataFrame(feature_data)

    def compare_audio(self, file1, file2):
        df = self.solve()
        df = df[df["File Name"].isin([file1, file2])]
        print(df)

        # Plot STE
        plt.bar(df["File Name"], df["Short-Time Energy"], color=['blue', 'red'])
        plt.ylabel("Short-Time Energy")
        plt.title("Energy Comparison")
        plt.show()

        # Plot ZCR
        plt.bar(df["File Name"], df["Zero-Crossing Rate"], color=['blue', 'red'])
        plt.ylabel("Zero-Crossing Rate")
        plt.title("ZCR Comparison")
        plt.show()


if __name__ == "__main__":
    solution = Solution2("Dataset")
    ans = solution.solve()
    print(ans)
    solution.compare_audio("PM Shri Narendra Modi.mp3", "Lalu Yadav.mp3")