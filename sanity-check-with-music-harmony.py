import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# =========================
# 1. GLOBAL SETTINGS
# =========================
SAMPLE_RATE = 44100
AMPLITUDE = 0.45

# =========================
# 2. NOTE FREQUENCIES
# =========================
notes = {
    'C4': 261.63,
    'D4': 293.66,
    'E4': 329.63,
    'F4': 349.23,
    'G4': 392.00,
    'A4': 440.00,
    'REST': 0.0
}

# =========================
# 3. HARMONIC NOTE GENERATOR
# (Sound-science harmony)
# =========================
def harmonic_note(freq, duration_sec):
    t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec), endpoint=False)

    if freq == 0:
        return np.zeros_like(t)

    harmonic_weights = [1.0, 0.6, 0.4, 0.2]
    wave = np.zeros_like(t)

    for i, w in enumerate(harmonic_weights):
        harmonic_freq = freq * (i + 1)
        wave += w * np.sin(2 * np.pi * harmonic_freq * t)

    wave /= np.sum(harmonic_weights)
    wave *= AMPLITUDE

    # Envelope (anti-click)
    # fade_len = int(SAMPLE_RATE * 0.02)
    # wave[:fade_len] *= np.linspace(0, 1, fade_len)
    # wave[-fade_len:] *= np.linspace(1, 0, fade_len)

    return wave

# =========================
# 4. CHORD GENERATOR
# (Music-definition harmony)
# =========================
def harmonic_chord(note_names, duration_sec):
    waves = []

    for name in note_names:
        freq = notes[name]
        waves.append(harmonic_note(freq, duration_sec))

    chord = np.sum(waves, axis=0)
    chord /= len(waves)
    return chord

# =========================
# 5. SCORE (MELODY + HARMONY)
# =========================
# Format: (chord_notes, duration)

score = [
    (['E4', 'G4', 'C4'], 0.5),
    (['E4', 'G4', 'C4'], 0.5),
    (['F4', 'A4', 'C4'], 0.5),
    (['G4', 'B4', 'D4'] if 'B4' in notes else ['G4', 'D4'], 0.5),

    (['G4', 'C4', 'E4'], 0.5),
    (['F4', 'A4', 'C4'], 0.5),
    (['E4', 'G4', 'C4'], 0.5),
    (['D4', 'F4', 'A4'], 0.5),

    (['C4', 'E4', 'G4'], 1.0),
]

# =========================
# 6. BUILD SONG
# =========================
song = []

for chord_notes, dur in score:
    song.append(harmonic_chord(chord_notes, dur))

audio = np.concatenate(song)

# =========================
# 7. EXPORT WAV
# =========================
scaled = np.int16(audio * 32767)
write("ode_to_joy_harmonic_harmony.wav", SAMPLE_RATE, scaled)
print("Generated: ode_to_joy_harmonic_harmony.wav")

# =========================
# 8. VISUAL CHECK
# =========================
plt.figure(figsize=(10, 4))
plt.plot(audio[2000:3000])
plt.title("Waveform: Harmonic Timbre + Musical Harmony")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid(True)
plt.savefig("harmonic_harmony_waveform.png")
print("Saved waveform plot.")
