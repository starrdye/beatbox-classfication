import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# 1. SETUP PARAMETERS
# We set this to 0.45. 
# Since our harmonic weights sum to 2.2, the peak will be 0.45 * 2.2 = 0.99.
# This maximizes energy without hitting the 1.0 distortion ceiling.
AMPLITUDE = 0.15 
SAMPLE_RATE = 44100

# 2. DEFINE FREQUENCIES
notes = {
    'C4': 261.63,
    'D4': 293.66,
    'E4': 329.63,
    'F4': 349.23,
    'G4': 392.00,
    'REST': 0.0
}

# 3. MANUAL COMPLEX GENERATOR (No Normalization Logic)
def generate_loud_wave(freq, duration_sec):
    t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec), endpoint=False)
    
    if freq == 0:
        return np.zeros_like(t)
    
    # --- THE RECIPE ---
    # We include all 4 harmonics for complexity.
    # We DO NOT normalize (divide) them. We trust our AMPLITUDE (0.45) to keep them safe.
    weights = [1.0, 0.6, 0.4, 0.2] 
    
    mixed_audio = np.zeros_like(t)
    
    for i, w in enumerate(weights):
        harmonic_freq = freq * (i + 1)
        # Add the layer directly
        mixed_audio += w * np.sin(2 * np.pi * harmonic_freq * t)
        
    # Apply Master Volume
    mixed_audio *= AMPLITUDE
    
    # SAFETY CHECK: If we messed up the math, warn the user.
    if np.max(np.abs(mixed_audio)) > 1.0:
        print("WARNING: Distortion detected! Your Amplitude is too high for these harmonics.")
        
    # # Apply Envelope (Fade)
    # fade_len = int(SAMPLE_RATE * 0.02)
    # if len(mixed_audio) > 2 * fade_len:
    #     mixed_audio[:fade_len] *= np.linspace(0, 1, fade_len)
    #     mixed_audio[-fade_len:] *= np.linspace(1, 0, fade_len)

    return mixed_audio

# 4. THE SCORE
melody = [
    ('E4', 0.5), ('E4', 0.5), ('F4', 0.5), ('G4', 0.5),
    ('G4', 0.5), ('F4', 0.5), ('E4', 0.5), ('D4', 0.5),
    ('C4', 0.5), ('C4', 0.5), ('D4', 0.5), ('E4', 0.5),
    ('E4', 0.75), ('D4', 0.25), ('D4', 1.0) 
]

# 5. BUILD
full_song = []
for note_name, duration in melody:
    freq = notes[note_name]
    wave = generate_loud_wave(freq, duration)
    full_song.append(wave)

combined_audio = np.concatenate(full_song)

# 6. EXPORT
scaled_audio = np.int16(combined_audio * 32767)
write('ode_to_joy_complex.wav', SAMPLE_RATE, scaled_audio)
print("File 'ode_to_joy_loud.wav' generated. (High Energy / No Normalization)")

# 7. VISUALIZE
plt.figure(figsize=(10, 4))
plt.plot(combined_audio[2000:3000]) 
plt.title("Visual Check: Waveform Amplitude (Peaks near 1.0)")
plt.ylabel("Amplitude")
plt.grid(True)
# Save plot to confirm we didn't clip
plt.savefig('ode_to_joy_complex.png')
print("Plot saved as 'ode_to_joy_complex.png'")