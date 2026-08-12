"""Generate all figures for the hand-tracking-ablation report."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT = '/rlwrld3/home/seungjun/projects/hand-tracking-ablation/assets'
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

# ============================================================
# 1. Tar Dataset Statistics (bar chart)
# ============================================================
tar_datasets = [
    ('FreiHAND',      130_240,  0.25),
    ('InterHand26M', 1_424_632, 0.25),
    ('MTC',           363_947,  0.10),
    ('COCO-W',         78_666,  0.10),
    ('DEX',           406_888,  0.05),
    ('H2O3D',         121_996,  0.05),
    ('HO3D',           83_325,  0.05),
    ('RHD',            61_705,  0.05),
    ('HALPE',          34_289,  0.05),
    ('MPIINZSL',       15_184,  0.05),
]

names = [d[0] for d in tar_datasets]
samples = [d[1] for d in tar_datasets]
weights = [d[2] for d in tar_datasets]

fig, ax1 = plt.subplots(figsize=(10, 5))
x = np.arange(len(names))
bars = ax1.bar(x, [s/1000 for s in samples], color='#4C72B0', alpha=0.85, label='Samples (K)')
ax1.set_ylabel('Samples (thousands)')
ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=35, ha='right')
ax1.set_title('Tar-Based Training Datasets (WebDataset)')

ax2 = ax1.twinx()
ax2.plot(x, weights, 'D-', color='#C44E52', markersize=7, linewidth=2, label='Sampling Weight')
ax2.set_ylabel('Sampling Weight')
ax2.set_ylim(0, 0.35)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
plt.tight_layout()
plt.savefig(f'{OUT}/tar_dataset_stats.png')
plt.close()
print('  tar_dataset_stats.png')

# ============================================================
# 2. Clip Dataset Statistics (bar chart)
# ============================================================
clip_datasets = [
    ('ARCTIC-Ego\n(train)',   2_007,   192_383),
    ('ARCTIC-Exo\n(train)',  16_056, 1_539_064),
    ('DexYCB\n(train)',       6_400,   465_504),
    ('HO3D-Clip\n(train)',      899,    83_325),
    ('H2O-Clip\n(train)',     1_278,   121_996),
    ('InterHand26M\n(train)',114_082, 7_301_143),
]

cnames  = [d[0] for d in clip_datasets]
clips   = [d[1] for d in clip_datasets]
frames  = [d[2] for d in clip_datasets]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(len(cnames))
ax1.bar(x, [c/1000 for c in clips], color='#55A868', alpha=0.85)
ax1.set_ylabel('Clips (thousands)')
ax1.set_xticks(x)
ax1.set_xticklabels(cnames, fontsize=9)
ax1.set_title('Clip Counts')

ax2.bar(x, [f/1_000_000 for f in frames], color='#8172B2', alpha=0.85)
ax2.set_ylabel('Frames (millions)')
ax2.set_xticks(x)
ax2.set_xticklabels(cnames, fontsize=9)
ax2.set_title('Total Frames')

fig.suptitle('Clip-Based Training Datasets', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/clip_dataset_stats.png')
plt.close()
print('  clip_dataset_stats.png')

# ============================================================
# 3. Model Variants Progression (timeline diagram)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(-0.5, 6.5)
ax.set_ylim(-1, 5)
ax.axis('off')
ax.set_title('Model Variant Progression', fontsize=15, pad=20)

boxes = [
    (0, 2, 'HaMeR\n(Base)', '#4C72B0',
     'ViT backbone\nDirect MANO head\nAdversarial (P+S)\nsmplx MANO'),
    (1, 2, 'WiLoR', '#55A868',
     '+ RefineNet\n  iterative refinement\n+ Two-stage pipeline'),
    (2, 2, 'WiLoR\n+MANOTorch', '#C44E52',
     '+ MANOTorch IK\n+ Anatomy constraints\n+ Joint angle limits'),
    (3, 2, 'WiLoR+MT\n+FixShape', '#8172B2',
     '+ Zero betas\n+ Pose-only disc.\n+ Normalized 3D loss'),
    (4, 2, 'WiLoR+MT+FS\n+Euler', '#CCB974',
     '+ Euler angles\n+ Per-joint DoF\n+ Active mask'),
    (5.5, 3.5, 'HaWoR\n(Clip)', '#64B5CD',
     'Transformer decoder\nST module\nMotion module\nNo discriminator\nVideo input (B,T,C,H,W)'),
]

for i, (x_pos, y_pos, title, color, desc) in enumerate(boxes):
    rect = mpatches.FancyBboxPatch((x_pos - 0.42, y_pos - 0.7), 0.84, 1.4,
                                    boxstyle="round,pad=0.08",
                                    facecolor=color, alpha=0.2,
                                    edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(x_pos, y_pos + 0.25, title, ha='center', va='center',
            fontweight='bold', fontsize=9, color=color)
    ax.text(x_pos, y_pos - 0.35, desc, ha='center', va='top',
            fontsize=7, color='#333333', linespacing=1.3)

# Arrows between sequential models
for i in range(4):
    ax.annotate('', xy=(i + 0.6, 2), xytext=(i + 0.4, 2),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

# Arrow from WiLoR to HaWoR (branching)
ax.annotate('', xy=(5.08, 3.5), xytext=(4.42, 2.5),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, linestyle='--'))

# Labels
ax.text(2, -0.5, 'Image-based models (single frame)', ha='center',
        fontsize=10, style='italic', color='#666666')
ax.text(5.5, -0.5, 'Video-based\n(temporal)', ha='center',
        fontsize=10, style='italic', color='#666666')

plt.savefig(f'{OUT}/model_progression.png')
plt.close()
print('  model_progression.png')

# ============================================================
# 4. Data Loading Pipeline Comparison (flow diagram)
# ============================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

# --- Tar pipeline ---
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 3)
ax1.axis('off')
ax1.set_title('Tar-Based Pipeline (WebDataset) - Single Frame', fontsize=12, fontweight='bold', color='#4C72B0')

tar_steps = [
    (0.7, 1.5, '.tar Shards\n(JPEG + pyd)', '#E8E8E8'),
    (2.5, 1.5, 'WebDataset\nStream', '#D4E6F1'),
    (4.3, 1.5, 'RandomMix\n(by weight)', '#D5F5E3'),
    (6.1, 1.5, 'Augment\n(IID/image)', '#FADBD8'),
    (7.9, 1.5, 'Batch\n(B,C,H,W)', '#F9E79F'),
    (9.3, 1.5, 'Model', '#D7BDE2'),
]
for x_pos, y_pos, label, color in tar_steps:
    rect = mpatches.FancyBboxPatch((x_pos - 0.55, y_pos - 0.45), 1.1, 0.9,
                                    boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor='#888888', linewidth=1.2)
    ax1.add_patch(rect)
    ax1.text(x_pos, y_pos, label, ha='center', va='center', fontsize=8)
for i in range(len(tar_steps) - 1):
    ax1.annotate('', xy=(tar_steps[i+1][0] - 0.55, 1.5),
                 xytext=(tar_steps[i][0] + 0.55, 1.5),
                 arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))

ax1.text(6.1, 0.6, 'Each image augmented independently\n(random rot, flip, crop, color)',
         ha='center', fontsize=8, style='italic', color='#666')

# --- Clip pipeline ---
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 3)
ax2.axis('off')
ax2.set_title('Clip-Based Pipeline (ClipDataset) - Temporal Sequence', fontsize=12, fontweight='bold', color='#55A868')

clip_steps = [
    (0.7, 1.5, 'NPZ Labels\n+ Image Dirs', '#E8E8E8'),
    (2.3, 1.5, 'Frame\nSelection\n(stride dt)', '#D4E6F1'),
    (3.9, 1.5, 'Load T\nFrames', '#D5F5E3'),
    (5.5, 1.5, 'Correlated\nAugment\n(IID_AUG)', '#FADBD8'),
    (7.3, 1.5, 'Collate\n_collate_clip()', '#FEF9E7'),
    (8.7, 1.5, 'Batch\n(B,T,C,H,W)', '#F9E79F'),
    (9.7, 1.5, 'Model', '#D7BDE2'),
]
for x_pos, y_pos, label, color in clip_steps:
    w = 0.9 if x_pos < 7 else 0.7
    rect = mpatches.FancyBboxPatch((x_pos - w/2, y_pos - 0.45), w, 0.9,
                                    boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor='#888888', linewidth=1.2)
    ax2.add_patch(rect)
    ax2.text(x_pos, y_pos, label, ha='center', va='center', fontsize=7.5)
for i in range(len(clip_steps) - 1):
    x1 = clip_steps[i][0] + (0.45 if clip_steps[i][0] < 7 else 0.35)
    x2 = clip_steps[i+1][0] - (0.45 if clip_steps[i+1][0] < 7 else 0.35)
    ax2.annotate('', xy=(x2, 1.5), xytext=(x1, 1.5),
                 arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))

ax2.text(5.5, 0.5, 'IID_AUG=0: all frames same aug | IID_AUG=1: independent | IID_AUG=2: correlated rotation',
         ha='center', fontsize=8, style='italic', color='#666')

plt.tight_layout()
plt.savefig(f'{OUT}/data_pipeline_comparison.png')
plt.close()
print('  data_pipeline_comparison.png')

# ============================================================
# 5. Augmentation Comparison Diagram
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, (mode, title, desc) in zip(axes, [
    (0, 'IID_AUG = 0\n(Fully Correlated)', 'All T frames share\nidentical augmentation'),
    (1, 'IID_AUG = 1\n(Fully Independent)', 'Each frame gets\nindependent augmentation'),
    (2, 'IID_AUG = 2\n(Correlated Rotation)', 'Rotation shared,\ncolor/scale independent'),
]):
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_title(title, fontsize=10, fontweight='bold')

    colors_rot = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
    colors_col = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']

    for i in range(4):
        y = 3.2 - i * 0.8
        # Frame box
        if mode == 0:
            c_rot = colors_rot[0]
            c_col = colors_col[0]
        elif mode == 1:
            c_rot = colors_rot[i]
            c_col = colors_col[i]
        else:  # mode 2
            c_rot = colors_rot[0]
            c_col = colors_col[i]

        ax.text(0.3, y, f't={i}', fontsize=9, va='center')
        # Rotation box
        rect = mpatches.FancyBboxPatch((0.8, y - 0.25), 1.5, 0.5,
                                        boxstyle="round,pad=0.03",
                                        facecolor=c_rot, alpha=0.3,
                                        edgecolor=c_rot, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(1.55, y, 'rot', ha='center', va='center', fontsize=8)

        # Color/scale box
        rect = mpatches.FancyBboxPatch((2.6, y - 0.25), 1.5, 0.5,
                                        boxstyle="round,pad=0.03",
                                        facecolor=c_col, alpha=0.3,
                                        edgecolor=c_col, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(3.35, y, 'color/scale', ha='center', va='center', fontsize=8)

    ax.text(2.5, -0.2, desc, ha='center', fontsize=8, style='italic', color='#666')

fig.suptitle('Temporal Augmentation Modes for Clip Dataset', fontsize=13, y=1.05)
plt.tight_layout()
plt.savefig(f'{OUT}/augmentation_modes.png')
plt.close()
print('  augmentation_modes.png')

# ============================================================
# 6. Architecture Comparison Table (as figure)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('off')

col_labels = ['HaMeR', 'WiLoR', '+MANOTorch', '+FixShape', '+Euler', 'HaWoR']
row_labels = ['MANO Type', 'Head', 'Discriminator', 'Fixed Shape',
              'Rotation Rep.', 'Temporal', 'Loss Type']
cell_text = [
    ['smplx',      'smplx',     'MANOTorch',   'MANOTorch',   'MANOTorch',   'smplx'],
    ['Direct',     'RefineNet', 'RefineNet',   'RefineNet',   'RefineNet',   'TransfDec'],
    ['Pose+Shape', 'Pose+Shape','Pose+Shape',  'Pose-only',   'Pose-only',   'None'],
    ['No',         'No',        'No',          'Yes',         'Yes',         'No'],
    ['6D',         '6D',        '6D',          '6D',          'Euler (45D)', '6D'],
    ['None',       'None',      'None',        'None',        'None',        'ST+Motion'],
    ['Standard',   'Standard',  'Standard',    'Normalized',  'Normalized',  'Standard'],
]

colors = []
for row in cell_text:
    row_colors = []
    for val in row:
        if val in ('Yes', 'MANOTorch', 'Euler (45D)', 'Normalized', 'ST+Motion', 'TransfDec', 'Pose-only'):
            row_colors.append('#D5F5E3')
        elif val == 'None' or val == 'No':
            row_colors.append('#F8F8F8')
        else:
            row_colors.append('#EBF5FB')
    colors.append(row_colors)

table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
                 cellColours=colors, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#4C72B0')
        cell.set_text_props(color='white', fontweight='bold')
    if col == -1:
        cell.set_text_props(fontweight='bold')

ax.set_title('Architecture Comparison Across Model Variants', fontsize=13, pad=20)
plt.savefig(f'{OUT}/architecture_comparison.png')
plt.close()
print('  architecture_comparison.png')

print('\nAll figures generated successfully.')
