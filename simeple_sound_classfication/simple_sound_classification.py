import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io.wavfile import write
from scipy.stats import iqr
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os
import random
import platform
import subprocess

# ==========================================
# PART 1: THE DATASET GENERATOR
# ==========================================
SAMPLE_RATE = 44100
AMPLITUDE = 0.45 

def generate_note_file(filename, freq, duration=0.5):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    
    # Timbre Recipe (Single Note, Rich Timbre)
    weights = [1.0, 0.6, 0.4, 0.2]
    wave = np.zeros_like(t)
    for i, w in enumerate(weights):
        harmonic_freq = freq * (i + 1)
        wave += w * np.sin(2 * np.pi * harmonic_freq * t)
        
    # --- RESTORED SAFETY LOGIC ---
    # We re-enable this to prevent digital distortion (wrapping).
    # Distortion destroys pitch info, which ruins classification accuracy.
    wave *= AMPLITUDE
    
    # # Envelope (Fade In/Out)
    # fade_len = int(SAMPLE_RATE * 0.02)
    # wave[:fade_len] *= np.linspace(0, 1, fade_len)
    # wave[-fade_len:] *= np.linspace(1, 0, fade_len)
    
    # Noise (Random Factor)
    # Kept your value of 0.0005 to create realistic cluster spread
    # noise = np.random.normal(0, 0.0005, wave.shape)
    # wave += noise
    
    scaled = np.int16(wave * 32767)
    write(filename, SAMPLE_RATE, scaled)

# Setup
notes_map = {
    'Do': 261.63, 
    'Re': 293.66,
    'Mi': 329.63,
    'Fa': 349.23
}
filenames = []
labels_ground_truth = []

# Generate 80 Samples (20 of each note)
print("--- 1. GENERATING DATASET ---")
num_samples = 80
for i in range(num_samples):
    label = random.choice(['Do', 'Re', 'Mi', 'Fa'])
    fname = f"sample_{i}_{label}.wav"
    generate_note_file(fname, notes_map[label])
    filenames.append(fname)
    labels_ground_truth.append(label)
    
print(f"Generated {num_samples} audio files.")


# ==========================================
# PART 2: FEATURE EXTRACTION
# ==========================================
print("\n--- 2. EXTRACTING FEATURES ---")

def get_features(filepath):
    y, sr = librosa.load(filepath, sr=None)
    FRAME_SIZE, HOP_LENGTH = 2048, 512
    
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    rms = librosa.feature.rms(y=y, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    y_harm, y_perc = librosa.effects.hpss(y)
    noise_rms = librosa.feature.rms(y=y_perc, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    
    fam1 = np.median(centroid)          # Brightness (Pitch)
    fam2 = iqr(centroid)                # Stability
    fam3 = np.median(rms)               # Energy
    fam4 = np.median(noise_rms / (rms + 1e-9)) # Noisiness
    
    return [fam1, fam2, fam3, fam4]

data = []
for fname in filenames:
    data.append(get_features(fname))

df = pd.DataFrame(data, columns=['Fam1_Brightness', 'Fam2_Stability', 'Fam3_Energy', 'Fam4_Noisiness'])
df['Label'] = labels_ground_truth


# ==========================================
# PART 3: SUPERVISED CLASSIFICATION
# ==========================================
print("\n--- 3. TRAINING NEAREST CENTROID CLASSIFIER ---")

X = df[['Fam1_Brightness', 'Fam2_Stability', 'Fam3_Energy', 'Fam4_Noisiness']]
y = df['Label']

# Split: 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Scale Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train (Learn the Centers)
clf = NearestCentroid()
clf.fit(X_train_scaled, y_train)

# Test
y_pred = clf.predict(X_test_scaled)


# ==========================================
# PART 4: EVALUATION & CENTROID PRINTING
# ==========================================
print("\n--- 4. FINAL RESULTS ---")

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# --- NEW: PRINT THE LEARNED PROFILES (CENTROIDS) ---
print("\n" + "="*60)
print("LEARNED PROFILES (The 'Average' for each Note)")
print("Notice how Brightness (Hz) climbs with Pitch.")
print("="*60)

# Get centroids and unscale them back to Hz/Energy
centroids_scaled = clf.centroids_
centroids_real = scaler.inverse_transform(centroids_scaled)

centroid_df = pd.DataFrame(centroids_real, columns=['Brightness (Hz)', 'Stability (Hz)', 'Energy', 'Noisiness'])
centroid_df['Note'] = clf.classes_
centroid_df = centroid_df.set_index('Note')

# Sort by Brightness to reveal the physics (Do < Re < Mi < Fa)
print(centroid_df.sort_values('Brightness (Hz)').round(4).to_string())
print("="*60)


# --- VISUALIZATION ---
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

colors = {'Do': 'blue', 'Re': 'red', 'Mi': 'green', 'Fa': 'purple'}

# Plot Test Data
for label, color in colors.items():
    mask = (y_test == label)
    subset = X_test[mask]
    # Multiply noisiness size for visibility
    sizes = subset['Fam4_Noisiness'] * 10000 + 50 
    ax.scatter(subset['Fam1_Brightness'], subset['Fam2_Stability'], subset['Fam3_Energy'],
               c=color, marker='^', s=sizes, alpha=1.0, edgecolors='k', label=f'Test: {label}')

# Plot Centroids
for i, label in enumerate(clf.classes_):
    c_vals = centroids_real[i]
    ax.scatter(c_vals[0], c_vals[1], c_vals[2], 
               c='black', marker='x', s=300, linewidth=3, 
               label=f'Centroid' if i==0 else "")

ax.set_xlabel('Fam 1: Brightness (Hz)')
ax.set_ylabel('Fam 2: Stability (Hz)')
ax.set_zlabel('Fam 3: Energy')
ax.set_title(f'Classification of Do-Re-Mi-Fa (Simple Notes)\nAccuracy: {accuracy*100:.0f}%')
plt.legend()
plt.tight_layout()

plot_file = 'supervised_classification.png'
plt.savefig(plot_file)
print(f"\nSaved visualization to '{plot_file}'")

if platform.system() == 'Windows':
    os.startfile(plot_file)
elif platform.system() == 'Darwin':
    subprocess.call(('open', plot_file))
elif platform.system() == 'Linux':
    subprocess.call(('xdg-open', plot_file))

# Cleanup
for f in filenames:
    if os.path.exists(f):
        os.remove(f)