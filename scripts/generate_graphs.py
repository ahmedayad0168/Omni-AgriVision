"""
Generate training performance graphs for documentation
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create docs directory if it doesn't exist
docs_dir = Path("docs")
docs_dir.mkdir(exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# 1. Detection Training Progress
epochs = [1, 2, 3, 4, 5]
mAP50 = [0.658, 0.710, 0.830, 0.822, 0.890]
mAP50_95 = [0.499, 0.533, 0.690, 0.676, 0.768]
precision = [0.869, 0.566, 0.591, 0.791, 0.714]
recall = [0.382, 0.775, 0.784, 0.665, 0.889]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs, mAP50, 'o-', label='mAP50', linewidth=2, markersize=8)
ax1.plot(epochs, mAP50_95, 's-', label='mAP50-95', linewidth=2, markersize=8)
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('mAP Score', fontsize=12, fontweight='bold')
ax1.set_title('Detection Training Progress', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.plot(epochs, precision, 'o-', label='Precision', linewidth=2, markersize=8)
ax2.plot(epochs, recall, 's-', label='Recall', linewidth=2, markersize=8)
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
ax2.set_title('Precision vs Recall', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(docs_dir / 'detection_training.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Classification Training Progress
epochs = [1, 2, 3, 4, 5]
train_loss = [0.0309, 0.0230, 0.0198, 0.0187, 0.0187]
val_loss = [0.0289, 0.0224, 0.0176, 0.0160, 0.0152]
val_acc = [99.02, 99.33, 99.42, 99.49, 99.58]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs, train_loss, 'o-', label='Train Loss', linewidth=2, markersize=8)
ax1.plot(epochs, val_loss, 's-', label='Val Loss', linewidth=2, markersize=8)
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax1.set_title('Classification Loss Progress', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.plot(epochs, val_acc, 'o-', label='Validation Accuracy', linewidth=2, markersize=8, color='green')
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Classification Accuracy Progress', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([98, 100])

plt.tight_layout()
plt.savefig(docs_dir / 'classification_training.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Segmentation Training Progress
epochs = [1, 2, 3, 4, 5]
train_loss = [0.1361, 0.0463, 0.0396, 0.0304, 0.0182]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(epochs, train_loss, 'o-', linewidth=2, markersize=8, color='purple')
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
ax.set_title('Segmentation Training Progress (DeepLabV3+)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(docs_dir / 'segmentation_training.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Per-Class Performance
classes = ['Bacterial Spot', 'Early Blight', 'Yellow Leaf Curl', 'Healthy', 'Mosaic Virus']
precision_vals = [0.857, 0.605, 0.185, 0.954, 0.970]
recall_vals = [0.591, 0.854, 1.000, 1.000, 1.000]
mAP50_vals = [0.832, 0.792, 0.837, 0.995, 0.995]

x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width, precision_vals, width, label='Precision', alpha=0.8)
rects2 = ax.bar(x, recall_vals, width, label='Recall', alpha=0.8)
rects3 = ax.bar(x + width, mAP50_vals, width, label='mAP50', alpha=0.8)

ax.set_xlabel('Disease Class', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Per-Class Detection Performance', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=45, ha='right')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(docs_dir / 'per_class_performance.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Overall Performance Summary
metrics = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1-Score']
values = [0.890, 0.768, 0.714, 0.889, 0.791]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Overall Detection Performance Metrics', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1.0])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(docs_dir / 'overall_performance.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Training Time Comparison
models = ['YOLO11m Detection', 'EfficientNet Classification', 'DeepLabV3+ Segmentation']
times = [4.3, 0.4, 0.5]  # hours
colors = ['#e74c3c', '#3498db', '#2ecc71']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(models, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Training Time (hours)', fontsize=12, fontweight='bold')
ax.set_title('Model Training Time Comparison', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}h',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(docs_dir / 'training_time_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Training graphs generated successfully!")
print(f"Graphs saved to: {docs_dir.absolute()}")
print("\nGenerated files:")
print("  - detection_training.png")
print("  - classification_training.png")
print("  - segmentation_training.png")
print("  - per_class_performance.png")
print("  - overall_performance.png")
print("  - training_time_comparison.png")