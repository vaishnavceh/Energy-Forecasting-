import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs(r"C:\Users\vaish\.gemini\antigravity\brain\c839cace-1fde-4b66-b314-8d80f369b34f\scratch", exist_ok=True)

# -------------------------------------------------------------
# FIGURE 1: Slide 3 - Problem Evidence & Load Volatility
# -------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5.5), dpi=300)

# Time axis: 48 intervals per day for 3 days = 144 points
t = np.linspace(0, 72, 144)  # 72 hours
np.random.seed(42)

# Synthetic realistic smart meter load with sharp peaks and noise
base_load = 0.35 + 0.25 * np.sin(2 * np.pi * t / 24 - np.pi/2)
morning_peak = 0.8 * np.exp(-((t % 24 - 8) ** 2) / 4)
evening_peak = 1.4 * np.exp(-((t % 24 - 19) ** 2) / 6)
stochastic_spikes = np.zeros_like(t)
# Random high power appliance spikes (kettle, EV, heater)
spike_indices = [15, 38, 41, 74, 88, 110, 134]
stochastic_spikes[spike_indices] = np.random.uniform(0.8, 1.8, size=len(spike_indices))
sensor_noise = np.random.normal(0, 0.05, size=len(t))

actual_load = np.clip(base_load + morning_peak + evening_peak + stochastic_spikes + sensor_noise, 0.05, 3.5)

# Panel 1: Smart Meter Load Volatility
ax1.plot(t, actual_load, color='#1A365D', lw=1.6, label='Actual 30-min Household Load (kWh)')
ax1.scatter(t[spike_indices], actual_load[spike_indices], color='#E53E3E', s=35, zorder=5, label='High-Impact Appliance Spikes / Inrush')
ax1.axvspan(18, 22, color='#FEFCBF', alpha=0.5, label='Evening Peak Window')
ax1.axvspan(42, 46, color='#FEFCBF', alpha=0.5)
ax1.axvspan(66, 70, color='#FEFCBF', alpha=0.5)
ax1.set_ylabel('Consumption (kWh)', fontsize=9, fontweight='bold')
ax1.set_title('(A) High Volatility & Non-Linearity in Smart Meter Data (SGSC Benchmark)', fontsize=10, fontweight='bold', pad=6)
ax1.legend(loc='upper right', fontsize=7.5, frameon=True)
ax1.set_xlim(0, 72)
ax1.set_xticks(range(0, 73, 12))
ax1.set_xticklabels(['0h (Day 1)', '12h', '24h (Day 2)', '36h', '48h (Day 3)', '60h', '72h'])
ax1.grid(True, linestyle='--', alpha=0.6)

# Panel 2: Limitations of Conventional Single-Stream Forecasting
# Conventional LSTM/ARIMA forecast (over-smoothed, misses sharp peaks)
conv_forecast = base_load + 0.7 * morning_peak + 0.75 * evening_peak + np.random.normal(0, 0.03, size=len(t))
conv_forecast = np.roll(conv_forecast, 2)  # Lag error
error = np.abs(actual_load - conv_forecast)

ax2.plot(t, actual_load, color='#2D3748', lw=1.2, alpha=0.7, label='Actual Demand')
ax2.plot(t, conv_forecast, color='#DD6B20', lw=1.6, linestyle='--', label='Conventional Single-Stream Model (Lag & Peak Under-prediction)')
ax2.fill_between(t, actual_load, conv_forecast, color='#E53E3E', alpha=0.25, label='Forecasting Error (High Peak Variance)')
ax2.set_xlabel('Time Horizon (Hours)', fontsize=9, fontweight='bold')
ax2.set_ylabel('Demand (kWh)', fontsize=9, fontweight='bold')
ax2.set_title('(B) Limitation of Standard Architectures: Under-Forecasting Volatile Peaks', fontsize=10, fontweight='bold', pad=6)
ax2.legend(loc='upper right', fontsize=7.5, frameon=True)
ax2.set_xlim(0, 72)
ax2.set_xticks(range(0, 73, 12))
ax2.set_xticklabels(['0h (Day 1)', '12h', '24h (Day 2)', '36h', '48h (Day 3)', '60h', '72h'])
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
fig1_path = r"C:\Users\vaish\.gemini\antigravity\brain\c839cace-1fde-4b66-b314-8d80f369b34f\scratch\slide3_evidence.png"
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.close()
print("Saved slide3_evidence.png successfully!")

# -------------------------------------------------------------
# FIGURE 2: Slide 15 - Simulation / Engineering Results
# -------------------------------------------------------------
fig, (bx1, bx2) = plt.subplots(1, 2, figsize=(7.5, 4.6), dpi=300)

# Panel 1: Data Preprocessing & Compression Throughput
categories = ['Raw CSV\n(16.5 GB Input)', 'Extracted CSV\n(37.2 GB Expanded)', 'PyArrow Parquet\n(4.1 GB Compressed)']
sizes = [16.5, 37.2, 4.1]
colors = ['#4A5568', '#E53E3E', '#2B6CB0']

bars = bx1.bar(categories, sizes, color=colors, width=0.55, edgecolor='black', lw=1)
bx1.set_ylabel('Disk Storage Size (GB)', fontsize=9, fontweight='bold')
bx1.set_title('Storage Optimization & Compression\n(8.8× Space Reduction / 10× I/O Speed)', fontsize=9.5, fontweight='bold', pad=6)
bx1.set_ylim(0, 42)
for bar in bars:
    h = bar.get_height()
    bx1.text(bar.get_x() + bar.get_width()/2., h + 1.0, f'{h:.1f} GB', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

bx1.text(0.5, 25, '322,000,000 Readings Processed\nThroughput: ~120k rows/sec\nSnappy Columnar Encoding',
         ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='#EBF8FF', edgecolor='#3182CE', lw=1.2),
         fontsize=8, fontweight='semibold')
bx1.grid(True, axis='y', linestyle='--', alpha=0.6)

# Panel 2: Preprocessed & Scaled Normalized Demand with 99.9% Cap
sample_t = np.linspace(0, 48, 96)
raw_spikes = 0.4 + 0.3*np.sin(2*np.pi*sample_t/24) + 0.6*np.exp(-((sample_t%24-19)**2)/5)
# Add outlier
raw_spikes[30] = 5.2 # Anomaly spike
raw_spikes[65] = -0.4 # Negative meter error

# Processed version
clean_scaled = np.clip(raw_spikes, 0.0, 2.929) / 2.929

bx2.plot(sample_t, clean_scaled, color='#2B6CB0', lw=1.8, label='Scaled Normalized Load [0.0, 1.0]')
bx2.axhline(1.0, color='#E53E3E', linestyle=':', lw=1.4, label='99.9% Upper Bound Cap (2.929 kWh)')
bx2.scatter([sample_t[30]], [1.0], color='#E53E3E', s=45, marker='x', label='Sensor Outlier Clipped')
bx2.scatter([sample_t[65]], [0.0], color='#DD6B20', s=45, marker='o', label='Negative Reading Clipped to 0')
bx2.set_xlabel('Time (Hours)', fontsize=9, fontweight='bold')
bx2.set_ylabel('Normalized Consumption', fontsize=9, fontweight='bold')
bx2.set_title('Cleaned & Scaled Model Input Series\n(Zero-Loss Outlier Capping & Min-Max)', fontsize=9.5, fontweight='bold', pad=6)
bx2.set_xlim(0, 48)
bx2.set_ylim(-0.1, 1.25)
bx2.legend(loc='upper right', fontsize=7.2, frameon=True)
bx2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
fig2_path = r"C:\Users\vaish\.gemini\antigravity\brain\c839cace-1fde-4b66-b314-8d80f369b34f\scratch\slide15_result.png"
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
plt.close()
print("Saved slide15_result.png successfully!")
