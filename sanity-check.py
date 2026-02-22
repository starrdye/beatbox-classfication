import numpy as np                  # Numerical library for arrays and math (sine, linspace, etc.)
import matplotlib.pyplot as plt     # Used to visualize the waveform (sanity check)
from scipy.io.wavfile import write  # Writes raw audio data into a .wav file

# ============================================================
# 1. GLOBAL AUDIO PARAMETERS
# ============================================================

SAMPLE_RATE = 44100  
# Number of samples per second.
# This means the waveform is measured 44,100 times every second.
# Higher sample rate = smoother, more accurate sound.

AMPLITUDE = 0.15   
# Maximum height of the waveform.
# Controls loudness: waveform will range from -0.45 to +0.45.
# Must stay below 1.0 to avoid clipping distortion.

# ============================================================
# 2. NOTE FREQUENCIES (PHYSICS OF MUSIC)
# ============================================================

notes = {
    'C4': 261.63,
    'D4': 293.66,
    'E4': 329.63,
    'F4': 349.23,
    'G4': 392.00,
    'A4': 440.00,  # <-- Added (Standard Tuning Reference)
    'B4': 493.88,  # <-- Added (2 semitones up from A4)
    'REST': 0.0
}

# ============================================================
# 3. SINE WAVE GENERATOR (CORE MATH)
# ============================================================

def generate_sine_wave(freq, duration_sec):
    """
    Generates a sine wave using the physical sound equation:
        y(t) = A * sin(2πft)

    A  = amplitude (volume)
    f  = frequency (pitch)
    t  = time (seconds)
    """

    # --------------------------------------------------------
    # Create the time axis (t)
    # --------------------------------------------------------

    t = np.linspace(
        0,
        duration_sec,
        int(SAMPLE_RATE * duration_sec),
        endpoint=False
    )
    # Generates evenly spaced time values from 0 to duration_sec.
    # Example (0.5s note):
    #   SAMPLE_RATE * 0.5 = 22050 samples
    # Each value in t represents one moment in time.

    # --------------------------------------------------------
    # Handle silence (REST)
    # --------------------------------------------------------

    if freq == 0:
        # For silence, the waveform is always 0
        # No vibration = no sound
        audio = np.zeros_like(t)

    else:
        # ----------------------------------------------------
        # Generate the sine wave
        # ----------------------------------------------------

        audio = AMPLITUDE * np.sin(2 * np.pi * freq * t)
        # Step-by-step:
        # 1) freq * t
        #    → how many cycles have passed at each time value
        #
        # 2) 2 * π * freq * t
        #    → converts cycles into radians (sine uses radians)
        #
        # 3) sin(...)
        #    → produces values between -1 and +1
        #
        # 4) multiply by AMPLITUDE
        #    → scales the wave to the desired loudness
        #
        # Result:
        # audio is an array where each value represents
        # speaker position at a specific moment in time.

        # ----------------------------------------------------
        # Optional fade-in / fade-out (prevents clicking)
        # ----------------------------------------------------

        # fade_len = int(SAMPLE_RATE * 0.01)
        # # 10 ms worth of samples.
        # # Clicking happens when waveform jumps abruptly
        # # from 0 to non-zero amplitude.

        # if len(audio) > 2 * fade_len:
        #     # Gradually increase volume at the start
        #     audio[:fade_len] *= np.linspace(0, 1, fade_len)

        #     # Gradually decrease volume at the end
        #     audio[-fade_len:] *= np.linspace(1, 0, fade_len)

    return audio
    # Returns a floating-point waveform in range [-AMPLITUDE, +AMPLITUDE]

# ============================================================
# 4. MELODY DEFINITION (SCORE)
# ============================================================

melody = [
    # Each tuple is: (note name, duration in seconds)

    ('E4', 0.5), ('E4', 0.5), ('F4', 0.5), ('G4', 0.5),
    ('G4', 0.5), ('F4', 0.5), ('E4', 0.5), ('D4', 0.5),

    ('C4', 0.5), ('C4', 0.5), ('D4', 0.5), ('E4', 0.5),
    ('E4', 0.75), ('D4', 0.25), ('D4', 1.0)
]

melody2 = [
    # Each tuple is: (note name, duration in seconds)

    ('C4', 0.5), ('D4', 0.5), ('E4', 0.5), ('C4', 0.5),
    ('C4', 0.5), ('D4', 0.5), ('E4', 0.5), ('C4', 0.5),

    ('E4', 0.5), ('F4', 0.5), ('G4', 0.5),
    ('E4', 0.5), ('F4', 0.5), ('G4', 0.5),
]

# ============================================================
# 5. BUILD THE FULL SONG
# ============================================================

full_song = []  # Will store waveform arrays for each note

for note_name, duration in melody:
    freq = notes[note_name]                 # Convert note name → frequency
    wave = generate_sine_wave(freq, duration)  # Generate math-based waveform
    full_song.append(wave)                  # Store the waveform

# Join all note waveforms end-to-end into one long array
combined_audio = np.concatenate(full_song)

# ============================================================
# 6. EXPORT AS WAV FILE
# ============================================================

scaled_audio = np.int16(combined_audio * 32767)
# WAV files store audio as 16-bit integers.
# Floating-point range [-1.0, 1.0] is scaled to [-32767, 32767].

write('ode_to_joy_sine.wav', SAMPLE_RATE, scaled_audio)
# Writes the waveform to disk with the given sample rate.

print("Audio file 'ode_to_joy_sine.wav' generated successfully.")

# ============================================================
# 7. VISUAL SANITY CHECK
# ============================================================

plt.figure(figsize=(10, 4))
plt.plot(combined_audio[:1000])
# Plot the first 1000 samples (~20 ms).
# Should look like a smooth sine wave.

plt.title("Visual Sanity Check: First 20ms of Audio")
plt.xlabel("Sample Number")
plt.ylabel("Amplitude")
plt.grid(True)
plt.savefig('sine_wave_check.png')

print("Plot saved as 'sine_wave_check.png'")