import numpy as np
import matplotlib.pyplot as plt

# Load your segmented data
segments_array = np.load('/Users/aidantang/Desktop/Gesture_recog/raw_dataV2/S1_raw_ungrouped.npy')
# segments_array shape: (num_segments, 400, 17)

# Extract the EMG part: columns 5 to 16 (12 channels)
emg_segments = segments_array[:, :, 5:]  # shape: (num_segments, 400, 12)

# Reshape to flatten all segments into one long time series
emg_flat = emg_segments.reshape(-1, 12)  # shape: (num_segments * 400, 12)

# Get the first 2000 datapoints
emg_2000 = emg_flat[:2000, :]
time = np.arange(2000)

# Plot only channel 12 (index 11), using black color
plt.figure(figsize=(12, 6))
plt.plot(time, emg_2000[:, 11], color='black', label='Channel 12')

# Set x-axis ticks every 200 samples
plt.xticks(np.arange(0, 2001, 200))

plt.xlabel("Time (samples)")
plt.ylabel("Normalized EMG Amplitude")
plt.title("First 2000 Data Points for Channel 12 (Patient 1)")
plt.legend()
plt.tight_layout()
plt.show()
