import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# import dataset
df = pd.read_csv('doench2016_simulated.csv')
print("Veri yüklendi:", df.shape)
print(df.head())

# basic feature engineering

def gc_content(seq):
    """GC content: proportion of G and C bases."""
    return (seq.count('G') + seq.count('C')) / len(seq)

def has_tttt(seq):
    """Returns 1 if 'TTTT' motif exists, otherwise 0. May stop RNA Polymerase III."""
    return int('TTTT' in seq)

def tm_approx(seq):
    """Approximate melting temperature using Wallace rule."""
    a = seq.count('A'); t = seq.count('T')
    g = seq.count('G'); c = seq.count('C')
    return 2 * (a + t) + 4 * (g + c)

df['gc']          = df['guide'].apply(gc_content)
df['has_tttt']    = df['guide'].apply(has_tttt)
df['tm']          = df['guide'].apply(tm_approx)
df['pam_perfect'] = (df['pam'] == 'GG').astype(int)

# Label: high efficiency (1) if ≥ 0.6, otherwise low (0)
df['label'] = (df['efficiency'] >= 0.6).astype(int)

print("\nFeatures created:")
print(df[['guide','gc','tm','has_tttt','pam_perfect','efficiency','label']].head())

# --- 3. EDA VISUALIZATIONS ---
TEAL   = '#1D9E75'
CORAL  = '#D85A30'
PURPLE = '#7F77DD'
AMBER  = '#BA7517'
GRAY   = '#888780'

fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor('#fafafa')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1. Efficiency distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df['efficiency'], bins=35, color=TEAL, alpha=0.85,
         edgecolor='white', linewidth=0.4)
ax1.axvline(0.6, color=CORAL, linewidth=1.5, linestyle='--', label='threshold = 0.6')
ax1.set_title('Efficiency Score Distribution', fontsize=11, fontweight='500')
ax1.set_xlabel('Efficiency Score'); ax1.set_ylabel('Number of Guide RNAs')
ax1.legend(fontsize=8)
ax1.spines[['top','right']].set_visible(False)

# 2. GC content vs efficiency
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(df['gc'], df['efficiency'], alpha=0.25, s=10, color=PURPLE)
z = np.polyfit(df['gc'], df['efficiency'], 2)
xfit = np.linspace(df['gc'].min(), df['gc'].max(), 100)
ax2.plot(xfit, np.poly1d(z)(xfit), color=AMBER, linewidth=2, label='trend')
ax2.set_title('GC Content → Efficiency', fontsize=11, fontweight='500')
ax2.set_xlabel('GC Content (0–1)'); ax2.set_ylabel('Efficiency Score')
ax2.legend(fontsize=8)
ax2.spines[['top','right']].set_visible(False)

# 3. Effect of TTTT motif
ax3 = fig.add_subplot(gs[0, 2])
groups = [df[df['has_tttt']==0]['efficiency'],
          df[df['has_tttt']==1]['efficiency']]
bp = ax3.boxplot(groups, patch_artist=True, widths=0.5,
                 medianprops=dict(color='white', linewidth=2))
bp['boxes'][0].set_facecolor(TEAL)
bp['boxes'][1].set_facecolor(CORAL)
for w in bp['whiskers'] + bp['caps']: w.set_color(GRAY)
for f in bp['fliers']: f.set(marker='o', markersize=3, alpha=0.3, color=GRAY)
ax3.set_xticklabels(['No TTTT', 'Has TTTT'], fontsize=9)
ax3.set_title('Effect of TTTT Motif', fontsize=11, fontweight='500')
ax3.set_ylabel('Efficiency Score')
ax3.spines[['top','right']].set_visible(False)

# 4. Tm distribution by class
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(df[df['label']==1]['tm'], bins=25, alpha=0.7,
         color=TEAL, label='High (≥0.6)', density=True)
ax4.hist(df[df['label']==0]['tm'], bins=25, alpha=0.7,
         color=CORAL, label='Low (<0.6)', density=True)
ax4.set_title('Tm Distribution (by Class)', fontsize=11, fontweight='500')
ax4.set_xlabel('Tm (°C)'); ax4.set_ylabel('Density')
ax4.legend(fontsize=8)
ax4.spines[['top','right']].set_visible(False)

# 5. Class balance
ax5 = fig.add_subplot(gs[1, 1])
counts = df['label'].value_counts()
ax5.pie(counts, labels=['High Efficiency', 'Low Efficiency'],
        colors=[TEAL, CORAL], autopct='%1.0f%%', startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=1.5),
        textprops=dict(fontsize=9))
ax5.set_title('Class Balance', fontsize=11, fontweight='500')

# 6. Correlation heatmap
ax6 = fig.add_subplot(gs[1, 2])
corr_cols = ['gc','tm','has_tttt','pam_perfect','efficiency']
corr = df[corr_cols].corr()
labels_map = {'gc':'GC','tm':'Tm','has_tttt':'TTTT',
              'pam_perfect':'PAM','efficiency':'Efficiency'}
im = ax6.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
ticks = range(len(corr_cols))
ax6.set_xticks(ticks); ax6.set_yticks(ticks)
ax6.set_xticklabels([labels_map[c] for c in corr_cols],
                    fontsize=8, rotation=30)
ax6.set_yticklabels([labels_map[c] for c in corr_cols], fontsize=8)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax6.text(j, i, f'{corr.iloc[i,j]:.2f}',
                 ha='center', va='center', fontsize=7.5,
                 color='black' if abs(corr.iloc[i,j]) < 0.7 else 'white')
ax6.set_title('Feature Correlations', fontsize=11, fontweight='500')
plt.colorbar(im, ax=ax6, shrink=0.8)

fig.suptitle('CRISPR Cas9 — EDA', fontsize=13, fontweight='500', y=1.01)
plt.savefig('eda_output.png', dpi=150, bbox_inches='tight', facecolor='#fafafa')
print("\nPlot saved: eda_output.png")

#Summary
print("\n=== Week 1 Findings ===")
print(f"Total guide RNAs: {len(df)}")
print(f"High efficiency (≥0.6): {df['label'].sum()} ({df['label'].mean()*100:.0f}%)")
print(f"GC–efficiency correlation: {df['gc'].corr(df['efficiency']):.3f}")
print(f"Average efficiency (with TTTT):    {df[df['has_tttt']==1]['efficiency'].mean():.3f}")
print(f"Average efficiency (without TTTT): {df[df['has_tttt']==0]['efficiency'].mean():.3f}")
