import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define the COLORMAP as provided
COLORMAP = [
    (0.0, 0.0, 0.0),
    (174.0, 199.0, 232.0),
    (152.0, 223.0, 138.0),
    (31.0, 119.0, 180.0),
    (255.0, 187.0, 120.0),
    (188.0, 189.0, 34.0),
    (140.0, 86.0, 75.0),
    (255.0, 152.0, 150.0),
    (214.0, 39.0, 40.0),
    (197.0, 176.0, 213.0),
    (148.0, 103.0, 189.0),
    (196.0, 156.0, 148.0),
    (23.0, 190.0, 207.0),
    (247.0, 182.0, 210.0),
    (219.0, 219.0, 141.0),
    (255.0, 127.0, 14.0),
    (158.0, 218.0, 229.0),
    (44.0, 160.0, 44.0),
    (112.0, 128.0, 144.0),
    (227.0, 119.0, 194.0),
    (213.0, 92.0, 176.0),
    (94.0, 106.0, 211.0),
    (82.0, 84.0, 163.0),
    (100.0, 85.0, 144.0),
    (66.0, 188.0, 102.0),
    (140.0, 57.0, 197.0),
    (202.0, 185.0, 52.0),
    (51.0, 176.0, 203.0),
    (200.0, 54.0, 131.0),
    (92.0, 193.0, 61.0),
    (78.0, 71.0, 183.0),
    (172.0, 114.0, 82.0),
    (91.0, 163.0, 138.0),
    (153.0, 98.0, 156.0),
    (140.0, 153.0, 101.0),
    (100.0, 125.0, 154.0),
    (178.0, 127.0, 135.0),
    (146.0, 111.0, 194.0),
    (96.0, 207.0, 209.0),
]

# Define the labels and their corresponding label_ids
labels = ['invalid', 'wall', 'floor', 'cabinet', 'bed', 'table', 'desk', 'curtain', 'toilet',
          'counter', 'refrigerator', 'sink', 'chair', 'picture', 'window', 'door']
label_ids = [0, 1, 2, 3, 4, 7, 14, 16, 33, 12, 24, 34, 5, 11, 9, 8]

# Create the figure and axis
fig, ax = plt.subplots(figsize=(17, 2))  # Reduced height from 4 to 2.5
ax.set_xlim(0, 17)  
ax.set_ylim(-0.3, 1.2)  # Adjusted to fit the new y coordinates
ax.axis('off')      # Hide axes for a clean look

# Plot each label with its color block
for i, label in enumerate(labels):
    # Determine position: first row (i < 7) at y=0.5, second row at y=-0.05
    if i < 8:
        x = i * 2.1      # Increased spacing by 10% (from 2.0 to 2.2)
        y = 0.5          # Top row centered at y=0.5
    else:
        x = (i - 8) * 2.1  # Increased spacing by 10% (from 2.0 to 2.2)
        y = -0.05          # Bottom row centered at y=-0.05, creating a gap of 0.05 (10% of 0.5)
    
    # Get the color and normalize to [0, 1] for matplotlib
    label_id = label_ids[i]
    color = COLORMAP[label_id]
    color = tuple(c / 255.0 for c in color)  # Convert from [0, 255] to [0, 1]
    
    # Add the color block (a rectangle)
    rect = patches.Rectangle((x, y), 0.7, 0.5, facecolor=color)
    ax.add_patch(rect)
    
    # Add the label text to the right
    ax.text(x + 0.8, y+0.25, label, ha='left', va='center', fontsize=12)

# Display the plot
plt.show()

# Optional: Save the image to a file
plt.savefig('label_color_blocks.png', bbox_inches='tight', dpi=1600)
plt.savefig('label_color_blocks.svg', bbox_inches='tight', dpi=1600)