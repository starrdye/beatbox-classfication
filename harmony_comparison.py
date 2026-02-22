import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy.fft import fft, fftfreq

# ==========================================
# 1. SETUP
# ==========================================
SAMPLE_RATE = 44100
DURATION = 2.0       # Seconds
FUNDAMENTAL = 261.63 # C4 (Middle C)
AMPLITUDE = 0.3      # Lower amplitude to allow stacking 10 harmonics without clipping

def generate_harmonic_stack(freq, duration, missing_indices=[]):
    """
    Generates a tone with 10 harmonics.
    missing_indices: List of harmonic ranks to remove (e.g., [2] removes the 2nd harmonic).
    """
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    
    # Start with silence
    mixed_audio = np.zeros_like(t)
    
    # We create harmonics 1 through 10
    # Weights follow the 1/n rule (Sawtooth-like natural decay)
    # 1st = 1.0, 2nd = 0.5, 3rd = 0.33, etc.
    print(f"\n--- Generating Stack (Missing: {missing_indices}) ---")
    
    for h in range(1, 11): # 1 to 10
        if h in missing_indices:
            print(f"   Skipping Harmonic {h} ({freq*h:.2f} Hz)")
            continue
            
        weight = 1.0 / h
        harmonic_freq = freq * h
        
        # Add the sine wave
        mixed_audio += weight * np.sin(2 * np.pi * harmonic_freq * t)
        print(f"   Added Harmonic {h}: {freq*h:.2f} Hz (Vol: {weight:.2f})")

    # Apply Master Volume
    mixed_audio *= AMPLITUDE
    
    # Envelope (Fade In/Out to avoid clicking)
    fade_len = int(SAMPLE_RATE * 0.05) # 50ms fade
    mixed_audio[:fade_len] *= np.linspace(0, 1, fade_len)
    mixed_audio[-fade_len:] *= np.linspace(1, 0, fade_len)
    
    return mixed_audio

# ==========================================
# 2. GENERATE THE SOUNDS
# ==========================================

# A. Full Harmony (10 Harmonics)
print("1. Generating Full Stack...")
audio_full = generate_harmonic_stack(FUNDAMENTAL, DURATION, missing_indices=[])
scaled_full = np.int16(audio_full * 32767)
write('harmony_full_10.wav', SAMPLE_RATE, scaled_full)

# B. Missing 2nd Multiple (9 Harmonics)
# We remove harmonic #2 (523.26 Hz). This is the "Octave".
# Removing the octave often makes the sound "hollow" or "woody" (like a clarinet).
print("2. Generating Missing 2nd...")
audio_missing = generate_harmonic_stack(FUNDAMENTAL, DURATION, missing_indices=[2])
scaled_missing = np.int16(audio_missing * 32767)
write('harmony_missing_2nd.wav', SAMPLE_RATE, scaled_missing)


# ==========================================
# 3. VISUALIZATION (FREQUENCY CHECK)
# ==========================================
# We perform a Fast Fourier Transform (FFT) to prove the harmonic is gone.

def plot_spectrum(audio, title, subplot_pos):
    # Number of samples
    N = len(audio)
    # FFT calculation
    yf = fft(audio)
    xf = fftfreq(N, 1 / SAMPLE_RATE)
    
    # We only care about positive frequencies up to 3000Hz (where our 10 harmonics are)
    idx_limit = int(3000 * N / SAMPLE_RATE)
    
    plt.subplot(2, 1, subplot_pos)
    plt.plot(xf[:idx_limit], np.abs(yf[:idx_limit]))
    plt.title(title)
    plt.ylabel("Magnitude")
    plt.grid(True)
    
    # Mark the spot where the 2nd harmonic should be
    second_harmonic = FUNDAMENTAL * 2
    plt.axvline(x=second_harmonic, color='r', linestyle='--', alpha=0.5, label='2nd Harmonic Position')
    plt.legend()

plt.figure(figsize=(10, 8))

plot_spectrum(audio_full, "1. Full Harmony (Note the peak at red line)", 1)
plot_spectrum(audio_missing, "2. Missing 2nd Harmonic (Note the GAP at red line)", 2)

plt.xlabel("Frequency (Hz)")
plt.tight_layout()
plt.savefig('harmony_test_spectrum.png')
print("\nSpectrum plot saved as 'harmony_test_spectrum.png'")
print("Done.")