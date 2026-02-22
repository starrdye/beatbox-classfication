import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import platform
import subprocess
from scipy.stats import iqr

def analyze_peeters_features(file_path, frame_size=2048, hop_length=512, show_plot=True):
    """
    Analyzes an audio file to extract the 4 Independent Feature Families 
    defined in Peeters et al. (2011).
    
    Args:
        file_path (str): Path to the .wav file.
        frame_size (int): STFT window size (default 2048).
        hop_length (int): STFT hop size (default 512).
        show_plot (bool): Whether to generate and open the dashboard image.
        
    Returns:
        dict: A dictionary containing the 4 global descriptor values.
    """
    
    # 1. LOAD AUDIO
    print(f"Analyzing: {file_path}...")
    try:
        y, sr = librosa.load(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

    # 2. EXTRACT TIME-SERIES VECTORS
    # A. Spectral Centroid (Time-Series)
    centroid_series = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_size, hop_length=hop_length)[0]

    # B. RMS Energy (Time-Series)
    rms_series = librosa.feature.rms(y=y, frame_length=frame_size, hop_length=hop_length)[0]

    # C. Harmonic-Percussive Separation (For Noisiness Calculation)
    y_harm, y_perc = librosa.effects.hpss(y)
    noise_rms_series = librosa.feature.rms(y=y_perc, frame_length=frame_size, hop_length=hop_length)[0]


    # 3. CALCULATE THE 4 TRULY INDEPENDENT PEETERS FAMILIES
    # ---------------------------------------------------------

    # --- FAMILY 1: SPECTRAL CENTER (Blue Cluster) ---
    # Text: "median descriptors for spectral... [e.g., Centroid (med)]"
    fam1_centroid_median = np.median(centroid_series)

    # --- FAMILY 2: SPECTRAL VARIABILITY (Green Cluster) ---
    # Text: "interquartile range spectral descriptors [e.g., Centroid (iqr)]"
    fam2_centroid_iqr = iqr(centroid_series)

    # --- FAMILY 3: TEMPORAL/ENERGY (Orange Cluster) ---
    # Text: "temporal descriptors... and energetic descriptors... RMSEnv(med)"
    fam3_energy_median = np.median(rms_series)

    # --- FAMILY 4: PERIODICITY (Cyan Cluster) ---
    # Text: "signal periodicity... (e.g., F0 and noisiness)"
    # Formula: Noise Energy / Total Energy
    fam4_noisiness_series = noise_rms_series / (rms_series + 1e-6)
    fam4_noisiness_median = np.median(fam4_noisiness_series)
    
    # Pack results into a dictionary
    results = {
        "Spectral Center (Median)": fam1_centroid_median,
        "Spectral Variability (IQR)": fam2_centroid_iqr,
        "Temporal Energy (Median)": fam3_energy_median,
        "Periodicity Noisiness (Median)": fam4_noisiness_median
    }

    # 4. PRINT RESULTS
    print("-" * 50)
    print(f"RESULTS FOR: {os.path.basename(file_path)}")
    print("-" * 50)
    print(f"1. SPECTRAL CENTER (Brightness):      {results['Spectral Center (Median)']:.2f} Hz")
    print(f"2. SPECTRAL VARIABILITY (Stability):  {results['Spectral Variability (IQR)']:.2f} Hz")
    print(f"3. TEMPORAL/ENERGY (Volume):          {results['Temporal Energy (Median)']:.4f}")
    print(f"4. PERIODICITY (Noisiness Ratio):     {results['Periodicity Noisiness (Median)']:.4f}")
    print("-" * 50)

    # 5. VISUALIZATION DASHBOARD
    if show_plot:
        frames = range(len(rms_series))
        t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

        plt.figure(figsize=(12, 10))

        # Plot 1: Waveform
        plt.subplot(4, 1, 1)
        librosa.display.waveshow(y, sr=sr, alpha=0.5)
        plt.title(f'1. Raw Waveform: {os.path.basename(file_path)}')
        plt.ylabel('Amp')
        plt.xticks([])

        # Plot 2: Family 3 (Energy)
        plt.subplot(4, 1, 2)
        plt.plot(t, rms_series, color='g', label=f'Median: {fam3_energy_median:.3f}')
        plt.axhline(y=fam3_energy_median, color='black', linestyle='--', label='Median')
        plt.title('2. Family 3 (Orange): Temporal/Energy')
        plt.ylabel('Energy')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.xticks([])

        # Plot 3: Families 1 & 2 (Spectral Centroid)
        plt.subplot(4, 1, 3)
        plt.semilogy(t, centroid_series, color='r', label='Spectral Centroid')
        plt.axhline(y=fam1_centroid_median, color='black', linestyle='--', label=f'Fam 1: Median')
        plt.axhline(y=fam1_centroid_median + (fam2_centroid_iqr/2), color='orange', linestyle=':', alpha=0.7)
        plt.axhline(y=fam1_centroid_median - (fam2_centroid_iqr/2), color='orange', linestyle=':', alpha=0.7, label=f'Fam 2: IQR Range')
        plt.title('3. Family 1 (Blue) & Family 2 (Green): Center & Variability')
        plt.ylabel('Hz (Log)')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.xticks([])

        # Plot 4: Family 4 (Noisiness)
        plt.subplot(4, 1, 4)
        plt.plot(t, fam4_noisiness_series, color='c', label=f'Median: {fam4_noisiness_median:.3f}')
        plt.axhline(y=fam4_noisiness_median, color='black', linestyle='--', label='Median')
        plt.title('4. Family 4 (Cyan): Periodicity/Noisiness')
        plt.ylabel('Ratio')
        plt.ylim(0, 1.0)
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.xlabel('Time (s)')

        plt.tight_layout()
        
        # Save unique filename based on input
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        plot_filename = f'dashboard_{base_name}.png'
        plt.savefig(plot_filename)
        print(f"Dashboard saved as '{plot_filename}'")
        
        # Auto-open
        if platform.system() == 'Windows':
            os.startfile(plot_filename)
        elif platform.system() == 'Darwin':
            subprocess.call(('open', plot_filename))
        elif platform.system() == 'Linux':
            subprocess.call(('xdg-open', plot_filename))
            
    return results

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Example usage:
    # You can now analyze ANY file by changing the path below.
    target_file = 'ode_to_joy_complex.wav'
    target_file2 = 'ode_to_joy_sine.wav'
    
    # Run the function
    analyze_peeters_features(target_file)
    analyze_peeters_features(target_file2)