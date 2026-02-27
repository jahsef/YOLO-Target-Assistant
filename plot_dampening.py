import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#IMPORTANT
#THIS SCRIPT IS USED TO VISUALIZE LEAD TARGET INTERNALS
#src.aimbot.data_parsing.targetselector.lead_target() writes to dampening_factors.csv
#
# CSV columns:
# zero_ratio_factor, jitter_factor, dampening_factor, wma_factor, raw_lead_sens, lead_sensitivity(ema),
# wma_velocity, cos_sim, jitter_score, zero_ratio, lead_pixels_x, lead_pixels_y, raw_delta_x, raw_delta_y

sns.set_theme(style="darkgrid")
palette = sns.color_palette("tab10")

data = np.genfromtxt('dampening_factors.csv', delimiter=',')
(zero_ratio_factor, jitter_factor, dampening_factor, wma_factor, raw_lead_sens, lead_sens_ema,
 wma_velocity, cos_sim, jitter_score, zero_ratio, lead_px_x, lead_px_y,
 raw_delta_x, raw_delta_y) = data.T
frames = np.arange(len(data))

# compute EMA of raw deltas for plotting
ema_alpha = 0.3
ema_x, ema_y = np.empty_like(raw_delta_x), np.empty_like(raw_delta_y)
ema_x[0], ema_y[0] = raw_delta_x[0], raw_delta_y[0]
for i in range(1, len(frames)):
    ema_x[i] = ema_x[i-1] + ema_alpha * (raw_delta_x[i] - ema_x[i-1])
    ema_y[i] = ema_y[i-1] + ema_alpha * (raw_delta_y[i] - ema_y[i-1])

fig, axes = plt.subplots(5, 1, figsize=(14, 20), sharex=True)

# --- Row 1: Gate multipliers (all 0-1 range, same axis) ---
ax = axes[0]
ax.set_title('Gate Multipliers (0 = fully gated, 1 = no gating)', fontweight='bold')
ax.set_ylabel('Multiplier')
ax.plot(frames, zero_ratio_factor, color=palette[0], label='zero_ratio_factor', linewidth=1.5)
ax.plot(frames, jitter_factor, color=palette[1], label='jitter_factor', linewidth=1.5)
ax.plot(frames, dampening_factor, color=palette[2], label='dampening_factor (harmonic)', linewidth=1.5)
ax.plot(frames, wma_factor, color=palette[3], label='wma_factor (velocity)', linewidth=1.5)
ax.set_ylim(-0.05, 1.15)
ax.legend(loc='upper right')

# --- Row 2: Lead sensitivity (raw vs EMA) ---
ax = axes[1]
ax.set_title('Lead Sensitivity', fontweight='bold')
ax.set_ylabel('Sensitivity')
ax.plot(frames, raw_lead_sens, color=palette[4], label='raw (all gates applied)', linewidth=1, alpha=0.6)
ax.plot(frames, lead_sens_ema, color=palette[3], label='EMA smoothed (final)', linewidth=2)
ax.legend(loc='upper right')

# --- Row 3: Raw inputs (wma_velocity, jitter_score, cos_sim, zero_ratio) ---
ax = axes[2]
ax.set_title('Raw Inputs', fontweight='bold')
ax.set_ylabel('Value')
ax.plot(frames, wma_velocity, color=palette[5], label='wma_velocity', linewidth=1.5)
ax.plot(frames, jitter_score, color=palette[6], label='jitter_score', linewidth=1.5)
ax2 = ax.twinx()
ax2.set_ylabel('Ratio / Similarity')
ax2.plot(frames, cos_sim, color=palette[7], label='cos_sim', linewidth=1.5, linestyle='--')
ax2.plot(frames, zero_ratio, color=palette[8], label='zero_ratio', linewidth=1.5, linestyle='--')
ax2.set_ylim(-1.1, 1.1)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# --- Row 4: Raw deltas (buffer input) with EMA ---
ax = axes[3]
ax.set_title('Buffer Input (raw deltas)', fontweight='bold')
ax.set_ylabel('Pixels')
ax.plot(frames, raw_delta_x, color=palette[0], label='raw_delta_x', linewidth=1, alpha=0.4)
ax.plot(frames, raw_delta_y, color=palette[3], label='raw_delta_y', linewidth=1, alpha=0.4)
ax.plot(frames, ema_x, color=palette[0], label=f'ema_x (α={ema_alpha})', linewidth=2)
ax.plot(frames, ema_y, color=palette[3], label=f'ema_y (α={ema_alpha})', linewidth=2)
ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
ax.legend(loc='upper right')

# --- Row 5: Lead output pixels ---
ax = axes[4]
ax.set_title('Lead Pixel Output', fontweight='bold')
ax.set_xlabel('Frame')
ax.set_ylabel('Pixels')
ax.plot(frames, lead_px_x, color=palette[0], label='lead_pixels_x', linewidth=1.5)
ax.plot(frames, lead_px_y, color=palette[3], label='lead_pixels_y', linewidth=1.5)
ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
ax.legend(loc='upper right')

plt.suptitle('Lead Target Internals', fontsize=16, fontweight='bold', y=1.0)
plt.tight_layout()
plt.show()
