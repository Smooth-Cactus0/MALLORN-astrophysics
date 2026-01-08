import pickle
from pathlib import Path

base_path = Path(__file__).parent.parent
data = pickle.load(open(base_path / 'data/processed/models_v23_tde.pkl', 'rb'))

print(f"OOF F1: {data['oof_f1']:.4f}")
print(f"Threshold: {data['best_thresh']:.2f}")
print(f"Features: {len(data['feature_cols'])}")

# Show new TDE features importance
importance_df = data['importance_df']
tde_spec = ['r_decay_alpha', 'r_decay_residual', 'r_decay_tde_like', 'g_decay_alpha',
            'ri_slope_50d', 'ri_slope_100d', 'gr_color_150d', 'ri_color_150d',
            'gr_curvature', 'gr_flux_ratio_evolution']

print("\nNew TDE-specific feature rankings:")
for _, row in importance_df.iterrows():
    if row['feature'] in tde_spec:
        rank = importance_df[importance_df['importance'] >= row['importance']].shape[0]
        print(f"  {row['feature']}: {row['importance']:.4f} (rank {rank}/{len(importance_df)})")

print("\nTop 10 overall features:")
for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
    marker = "[NEW]" if row['feature'] in tde_spec else "     "
    print(f"  {i+1}. {marker} {row['feature']}: {row['importance']:.4f}")
