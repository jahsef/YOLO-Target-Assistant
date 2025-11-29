import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#IMPORTANT
#THIS SCRIPT IS USED TO VISUALIZE DAMPENING FACTOR AND LEAD SENSITIVITYY
#src.aimbot.data_parsing.target_selector.get_lead() writes to file, it should be uncommented to use this script

# Set seaborn style
sns.set_theme(style="darkgrid")

# Read CSV data
data = np.genfromtxt('dampening_factors.csv', delimiter=',')
dampening_factors = data[:, 0]
lead_sensitivities = data[:, 1]
frames = np.arange(len(dampening_factors))

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot dampening factor on primary y-axis
color1 = sns.color_palette()[0]
ax1.set_xlabel('Frame', fontsize=12)
ax1.set_ylabel('Dampening Factor', color=color1, fontsize=12)
sns.lineplot(x=frames, y=dampening_factors, ax=ax1, color=color1, label='Dampening Factor', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color1)

# Create secondary y-axis for lead sensitivity
ax2 = ax1.twinx()
color2 = sns.color_palette()[3]
ax2.set_ylabel('Lead Sensitivity', color=color2, fontsize=12)
sns.lineplot(x=frames, y=lead_sensitivities, ax=ax2, color=color2, label='Lead Sensitivity', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color2)

# Add title and legends
plt.title('Dampening Factor and Lead Sensitivity Over Time', fontsize=14, fontweight='bold')
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9), frameon=True)
plt.tight_layout()
plt.show()
