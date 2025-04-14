import wave
import numpy as np

# Open the .wav file
try:
    wav_file = wave.open('StarWars4.wav', 'rb')
except wave.Error as e:
    print(f"Error opening file: {e}")
    exit()

# Get basic audio parameters
num_channels = wav_file.getnchannels()
frame_rate = wav_file.getframerate()
num_frames = wav_file.getnframes()
sample_width = wav_file.getsampwidth()

print(f"Number of channels: {num_channels}")
print(f"Frame rate: {frame_rate} Hz")
print(f"Number of frames: {num_frames}")
print(f"Sample width: {sample_width} bytes")

# Read the audio data as bytes
audio_data = wav_file.readframes(num_frames)

# Convert byte data to numerical array
if sample_width == 1:
    dtype = np.uint8  # 8-bit unsigned integer
elif sample_width == 2:
    dtype = np.int16 # 16-bit signed integer
elif sample_width == 4:
    dtype = np.int32  # 32-bit signed integer
else:
     raise ValueError("Unsupported sample width")

audio_array = np.frombuffer(audio_data, dtype=dtype)

# Reshape array if stereo
if num_channels > 1:
    audio_array = audio_array.reshape(-1, num_channels)

# Close the file
wav_file.close()

# Now you can work with the audio_array
# For example, print the first 10 samples
print("First 50 samples:", audio_array[:50])