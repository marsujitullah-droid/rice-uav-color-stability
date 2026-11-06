# ---- CELL: run ----
import time
t0 = time.time()
os.makedirs(OUT_DIR, exist_ok=True)

print("Step 1/6  Reading metadata ...")
md = build_metadata(IMAGE_ROOT, CFG["SESSION_GAP_MIN"])
md = apply_qc(md, CFG)
md.drop(columns=["path"]).to_csv(os.path.join(OUT_DIR, "01_image_metadata_qc.csv"), index=False)
print(md.groupby("session").agg(images=("file", "size"), kept=("keep", "sum"),
                                alt_min=("rel_alt_m", "min"), alt_max=("rel_alt_m", "max"),
                                start=("datetime", "min"), end=("datetime", "max")))
print("Saturated-pixel fraction per image (quantiles):")
print(md["saturated_frac"].quantile([0, .1, .25, .5, .75, .9, 1]).round(3).to_string())
print(md.loc[~md["keep"], "reason"].str.split(";").explode().replace("", np.nan).dropna().value_counts())

print("\nStep 2/6  Registering images within each session ...")
reg_tables, all_real, all_sim = [], [], []
for s, sess in md[md["keep"]].groupby("session"):
    if len(sess) < 2:
        continue
    sess_r, Hs, ref_file = register_session(sess, CFG)
    if sess_r is None:
        print(f"  {s}: no nadir image, skipped"); continue
    reg_tables.append(sess_r.drop(columns=["path"]).assign(reference=ref_file))
    print(f"  {s}: {int(sess_r['registered'].sum())}/{len(sess_r)} images registered to {ref_file}")
    print(f"  {s}: extracting tile features ...")
    all_real.append(extract_session(sess_r, Hs, ref_file, CFG))
    all_sim.append(simulate_session(sess_r, Hs, ref_file, CFG))
if not reg_tables:
    raise RuntimeError("No session could be registered: check the exclusion reasons printed in Step 1.")
reg = pd.concat(reg_tables, ignore_index=True)
reg.to_csv(os.path.join(OUT_DIR, "02_registration.csv"), index=False)
real = pd.concat(all_real, ignore_index=True)
sim = pd.concat(all_sim, ignore_index=True) if any(len(x) for x in all_sim) else pd.DataFrame()
real.to_csv(os.path.join(OUT_DIR, "03_tile_features_real.csv"), index=False)
sim.to_csv(os.path.join(OUT_DIR, "04_tile_features_simulated.csv"), index=False)
multi = real.groupby("tile").size()
print(f"  {len(real)} tile observations, {real['tile'].nunique()} tiles, "
      f"{int((multi >= 2).sum())} tiles seen in 2+ images")

print("\nStep 3/6  Real vs simulated degradation ...")
rvs = real_vs_simulated(real, sim, FEATURES)
rvs.to_csv(os.path.join(OUT_DIR, "05_real_vs_simulated.csv"), index=False)
print(rvs.round(3).to_string(index=False))

print("\nStep 4/6  Hue stability index and sensitivity ratios ...")
hsi = hue_stability(real)
hsi.to_csv(os.path.join(OUT_DIR, "06_hue_stability_index.csv"), index=False)
sens = sensitivity_ratio(real, FEATURES)
sens.to_csv(os.path.join(OUT_DIR, "07_sensitivity_ratio.csv"), index=False)
print(hsi[["HSI_H_deg", "HSI_H_veg_deg", "HSI_H_tm_deg"]].describe().round(2))
print(sens.round(3).to_string(index=False))

print("\nStep 5/6  Mixed-effects models ...")
try:
    lmm = fit_mixed_models(real, FEATURES)
    lmm.to_csv(os.path.join(OUT_DIR, "08_mixed_models.csv"), index=False)
    print(lmm.round(4).to_string(index=False))
except ImportError:
    lmm = pd.DataFrame()
    print("  statsmodels not installed: run  !pip install statsmodels  and re-run this cell")
except Exception as e:  # never lose the figures because of the model step
    lmm = pd.DataFrame()
    print("  mixed models failed:", repr(e))

print("\nStep 6/6  Figures ...")
make_figures(md, real, sim, hsi, sens, os.path.join(OUT_DIR, "figures"))

summary = {
    "n_files_unique": int(len(md)), "n_kept": int(md["keep"].sum()),
    "n_registered": int(reg["registered"].sum()), "sessions": sorted(md["session"].unique().tolist()),
    "n_tile_obs": int(len(real)), "n_tiles": int(real["tile"].nunique()),
    "n_tiles_multi": int((multi >= 2).sum()), "n_sim_obs": int(len(sim)),
    "config": CFG, "runtime_min": round((time.time() - t0) / 60, 1),
}
with open(os.path.join(OUT_DIR, "00_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2, default=str)
print("\nDone in", summary["runtime_min"], "min. Results are in", OUT_DIR)
