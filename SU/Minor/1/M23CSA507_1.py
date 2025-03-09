import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd


class Solution1:
    def __init__(self, num_samples, sample_dir, plotting_enabled=False):
        self.sample_rate = 44100
        self.num_samples = num_samples
        self.audio_dir = sample_dir
        self.data = list()
        self.plotting_enabled = plotting_enabled

    def _analyze_audio(self, filename):
        y, sr = librosa.load(self.audio_dir + '/' + filename, sr=self.sample_rate)
        time = np.linspace(0, len(y) / sr, num=len(y))

        amplitude = np.abs(y)                                         # Amplitude
        rms_energy = np.sqrt(np.mean(y ** 2))                         # RMS Energy
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch = np.max(pitches)                                       # Pitch
        fft_spectrum = np.fft.fft(y)
        freqs = np.fft.fftfreq(len(fft_spectrum), 1 / sr)
        peak_freq = np.abs(freqs[np.argmax(np.abs(fft_spectrum))])    # Frequency

        if self.plotting_enabled:
            self._plot(time, y, sr, filename)

        self.data.append({
            "File Name": filename,
            "Max Amplitude": max(amplitude),
            "RMS Energy": rms_energy,
            "Approximate Pitch (Hz)": pitch,
            "Dominant Frequency (Hz)": peak_freq
        })

    def _plot(self, time, y, sr, filename):
        # Plot waveform
        plt.figure(figsize=(10, 4))
        plt.plot(time, y, label='Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'Waveform of {filename}')
        plt.legend()
        plt.show()

        # Plot spectrogram
        plt.figure(figsize=(10, 4))
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram of {filename}')
        plt.show()

    def solve(self):
        for i in range(1, self.num_samples + 1):
            self._analyze_audio(f"{i}.m4a")

    def result(self):
        df = pd.DataFrame(self.data)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)
        print("\nFinal Analysis Data:")
        print(df)


if __name__ == "__main__":
    graph_enabled = False    # This is used to generated spectrograms and waveforms
    solution = Solution1(10, "recordings", graph_enabled)
    solution.solve()
    solution.result()

