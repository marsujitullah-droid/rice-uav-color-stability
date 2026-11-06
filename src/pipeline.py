# ==== Paper 3 analysis pipeline: acquisition effects on color-space features ====
# Everything below is pure Python (numpy, OpenCV, pandas, scipy, matplotlib).
# The mixed-model step uses statsmodels, which is preinstalled on Google Colab.

import os, re, glob, json, math, hashlib, warnings
import numpy as np
import pandas as pd
import cv2
from PIL import Image

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------- 1. metadata
XMP_KEYS = {
    "RelativeAltitude": "rel_alt_m",
    "AbsoluteAltitude": "abs_alt_m",
    "GimbalPitchDegree": "gimbal_pitch",
    "GimbalYawDegree": "gimbal_yaw",
    "FlightYawDegree": "flight_yaw",
}


def _xmp_value(raw, key):
    m = re.search(rb'drone-dji:%s\s*=\s*"([^"]+)"' % key.encode(), raw)
    if m is None:
        m = re.search(rb"<drone-dji:%s>([^<]+)</drone-dji:%s>" % (key.encode(), key.encode()), raw)
    if m is None:
        return np.nan
    try:
        return float(m.group(1).decode().replace("+", ""))
    except ValueError:
        return np.nan


def _dms_to_deg(dms, ref):
    try:
        d, m, s = [float(x) for x in dms]
        v = d + m / 60 + s / 3600
        return -v if ref in ("S", "W") else v
    except Exception:
        return np.nan


def read_metadata(path):
    with open(path, "rb") as f:
        raw = f.read(200_000)  # XMP sits in the first APP segments
    rec = {"path": path, "file": os.path.basename(path), "bytes": os.path.getsize(path)}
    for k, col in XMP_KEYS.items():
        rec[col] = _xmp_value(raw, k)
    with Image.open(path) as im:
        rec["width"], rec["height"] = im.size
        ex = im.getexif()
        ifd = ex.get_ifd(0x8769)  # Exif IFD
        gps = ex.get_ifd(0x8825)
        rec["datetime"] = ifd.get(36867) or ex.get(306)
        fl = ifd.get(37386)
        rec["focal_mm"] = float(fl) if fl is not None else np.nan
        rec["model"] = ex.get(272)
        rec["lat"] = _dms_to_deg(gps.get(2), gps.get(1)) if gps.get(2) else np.nan
        rec["lon"] = _dms_to_deg(gps.get(4), gps.get(3)) if gps.get(4) else np.nan
    # content fingerprint for de-duplication (same photo stored in several folders)
    rec["fingerprint"] = hashlib.md5(raw[:65536]).hexdigest() + f"_{rec['bytes']}"
    return rec


def find_originals(root):
    files = []
    for ext in ("*.JPG", "*.jpg", "*.JPEG", "*.jpeg"):
        files += glob.glob(os.path.join(root, "**", ext), recursive=True)
    # Roboflow exports (".rf.") are resized and recompressed: never use them
    files = [f for f in files if ".rf." not in os.path.basename(f)]
    return sorted(set(files))


def build_metadata(root, session_gap_min=60):
    recs = [read_metadata(p) for p in find_originals(root)]
    md = pd.DataFrame(recs)
    n_all = len(md)
    md = md.drop_duplicates("fingerprint").reset_index(drop=True)
    md["datetime"] = pd.to_datetime(md["datetime"], format="%Y:%m:%d %H:%M:%S", errors="coerce")
    md = md.sort_values("datetime").reset_index(drop=True)
    gap = md["datetime"].diff().dt.total_seconds().div(60).fillna(1e9)
    md["session"] = (gap > session_gap_min).cumsum().map(lambda i: f"S{i}")
    print(f"{n_all} original JPEG files found, {len(md)} unique after de-duplication, "
          f"{md['session'].nunique()} sessions")
    return md


# ---------------------------------------------------------------- 2. geometry
def gsd_cm(alt_m, focal_mm, sensor_w_mm, img_w_px, pitch_deg):
    """GSD at the image centre, flat terrain. pitch -90 = nadir."""
    off_nadir = np.radians(90.0 - np.abs(pitch_deg))
    slant = alt_m / np.clip(np.cos(off_nadir), 0.2, 1.0)
    return sensor_w_mm * slant * 100.0 / (focal_mm * img_w_px)


def pitch_class(p):
    if p <= -80:
        return "nadir"
    if p <= -60:
        return "moderate_oblique"
    if p <= -30:
        return "strong_oblique"
    return "excluded"


# ---------------------------------------------------------------- 3. quality control
def quality_metrics(img_bgr, work_w=1000):
    s = work_w / img_bgr.shape[1]
    small = cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    sat = float((img_bgr[::4, ::4] >= 254).any(axis=2).mean())
    return sharp, sat


def apply_qc(md, cfg):
    md = md.copy()
    md["pitch_class"] = md["gimbal_pitch"].apply(pitch_class)
    md["gsd_cm"] = gsd_cm(md["rel_alt_m"], md["focal_mm"].fillna(cfg["FOCAL_MM"]),
                          cfg["SENSOR_W_MM"], md["width"], md["gimbal_pitch"])
    sharp, sat = [], []
    for p in md["path"]:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        a, b = quality_metrics(img)
        sharp.append(a); sat.append(b)
    md["sharpness"] = sharp
    md["saturated_frac"] = sat
    # sharpness rises naturally with altitude, so compare each image only with
    # images of the same session taken at a similar GSD (within a factor 1.4)
    med = []
    for i, r in md.iterrows():
        peers = md[(md["session"] == r["session"]) & (md.index != i) &
                   (md["gsd_cm"].between(r["gsd_cm"] / 1.4, r["gsd_cm"] * 1.4))]
        med.append(peers["sharpness"].median() if len(peers) >= 2 else 0.0)
    med = pd.Series(med, index=md.index)
    md["reason"] = ""
    md.loc[md["rel_alt_m"] < cfg["MIN_ALT_M"], "reason"] += "low_altitude;"
    md.loc[md["pitch_class"] == "excluded", "reason"] += "horizon_or_near_horizontal;"
    md.loc[md["saturated_frac"] > cfg["MAX_SATURATED"], "reason"] += "saturated;"
    md.loc[md["sharpness"] < cfg["BLUR_FRACTION"] * med, "reason"] += "blurred;"
    md.loc[md["rel_alt_m"].isna() | md["gimbal_pitch"].isna(), "reason"] += "missing_xmp;"
    md["keep"] = md["reason"] == ""
    return md


# ---------------------------------------------------------------- 4. registration
_sift = None


def _features(img_bgr, work_w):
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create(nfeatures=8000)
    s = work_w / img_bgr.shape[1]
    small = cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    kp, des = _sift.detectAndCompute(gray, None)
    pts = np.float32([k.pt for k in kp]) / s if kp else np.zeros((0, 2), np.float32)
    return pts, des


def match_homography(fa, fb, min_inliers, min_ratio):
    """Homography mapping full-res pixels of image A onto image B."""
    pa, da = fa; pb, db = fb
    if da is None or db is None or len(da) < 10 or len(db) < 10:
        return None, 0
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < min_inliers:
        return None, len(good)
    src = pa[[m.queryIdx for m in good]]
    dst = pb[[m.trainIdx for m in good]]
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 8.0)
    if H is None:
        return None, 0
    n_in = int(mask.sum())
    if n_in < min_inliers or n_in / len(good) < min_ratio or np.linalg.det(H[:2, :2]) <= 0:
        return None, n_in
    return H, n_in


def register_session(sess, cfg):
    """Register every kept image of a session into the frame of a reference
    nadir image (the highest kept nadir image, which covers the widest area).
    Images are chained by altitude so that each match spans a modest scale step."""
    sess = sess.sort_values("rel_alt_m", ascending=False).reset_index(drop=True)
    nadir = sess[sess["pitch_class"] == "nadir"]
    if nadir.empty:
        return None, {}, None
    ref_idx = nadir.index[0]
    feats = {}
    for i, r in sess.iterrows():
        feats[i] = _features(cv2.imread(r["path"]), cfg["REG_WORK_W"])
    H_to_ref = {ref_idx: np.eye(3)}
    inl = {ref_idx: -1}
    pending = [i for i in sess.index if i != ref_idx]
    progress = True
    while pending and progress:
        progress = False
        for i in list(pending):
            # try already-registered images, closest in altitude first
            cands = sorted(H_to_ref, key=lambda j: abs(sess.at[j, "rel_alt_m"] - sess.at[i, "rel_alt_m"]))
            for j in cands[: cfg["REG_MAX_TRIES"]]:
                H, n = match_homography(feats[i], feats[j], cfg["REG_MIN_INLIERS"], cfg["REG_MIN_RATIO"])
                if H is not None:
                    H_to_ref[i] = H_to_ref[j] @ H
                    inl[i] = n
                    pending.remove(i)
                    progress = True
                    break
    sess["registered"] = sess.index.isin(list(H_to_ref))
    sess["inliers"] = sess.index.map(lambda k: inl.get(k, 0))
    return sess, {sess.at[k, "file"]: H_to_ref[k] for k in H_to_ref}, sess.at[ref_idx, "file"]


# ---------------------------------------------------------------- 5. color features
FEATURES = ["H_deg", "H_veg_deg", "S", "V", "L", "a", "b",
            "ExG", "VARI", "GLI", "NGRDI", "veg_frac",
            "H_tm_deg", "a_tm", "b_tm", "ExG_tm", "VARI_tm", "GLI_tm", "NGRDI_tm"]


def circ_mean_deg(h):
    if h.size == 0:
        return np.nan, np.nan
    r = np.radians(h)
    c, s = np.cos(r).mean(), np.sin(r).mean()
    return (np.degrees(np.arctan2(s, c)) % 360), float(np.hypot(c, s))


def _indices(R, G, B):
    eps = 1e-6
    tot = R + G + B + eps
    r, g, b = R / tot, G / tot, B / tot
    exg = 2 * g - r - b
    den_v = G + R - B
    vari = np.where(np.abs(den_v) > 0.05, (G - R) / np.where(np.abs(den_v) > 0.05, den_v, 1), np.nan)
    gli = (2 * G - R - B) / (2 * G + R + B + eps)
    ngrdi = (G - R) / (G + R + eps)
    return exg, vari, gli, ngrdi


def image_color_planes(img_bgr):
    """Per-pixel planes computed once per image (float32, sRGB in 0-1)."""
    f = img_bgr.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)      # H in degrees for float input
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2Lab)      # L 0-100, a/b approx -127..127, D65
    B, G, R = f[..., 0], f[..., 1], f[..., 2]
    exg, vari, gli, ngrdi = _indices(R, G, B)
    # vegetation mask: Otsu threshold on ExG for this image
    e8 = np.clip((exg + 1) * 127.5, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(e8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    veg = e8 > t
    sat = (img_bgr >= 254).any(axis=2)            # clipped pixels (sun glint on water, bright bunds)
    return dict(R=R, G=G, B=B, H=hsv[..., 0], S=hsv[..., 1], V=hsv[..., 2],
                L=lab[..., 0], a=lab[..., 1], b=lab[..., 2],
                ExG=exg, VARI=vari, GLI=gli, NGRDI=ngrdi, veg=veg, sat=sat)


TILE_MAX_SATURATED = 0.10   # a tile is dropped if more than 10% of its pixels are clipped


def tile_features(planes, mask):
    n0 = int(mask.sum())
    if n0 < 50:
        return None
    sat_frac = float(planes["sat"][mask].mean())
    if sat_frac > TILE_MAX_SATURATED:
        return None
    m = mask & ~planes["sat"]                   # clipped pixels never enter the features
    n = int(m.sum())
    out = {"n_px": n, "sat_frac": sat_frac}
    if n < 50:
        return None
    h, rbar = circ_mean_deg(planes["H"][m])
    out["H_deg"], out["H_R"] = h, rbar
    mv = m & planes["veg"]
    out["veg_frac"] = mv.sum() / n
    out["H_veg_deg"] = circ_mean_deg(planes["H"][mv])[0] if mv.sum() >= 50 else np.nan
    for k in ["S", "V", "L", "a", "b", "ExG", "GLI", "NGRDI"]:
        out[k] = float(np.nanmean(planes[k][m]))
    out["VARI"] = float(np.nanmedian(planes["VARI"][m]))
    # features of the tile-mean colour (insensitive to pure area-averaging)
    R, G, B = (float(planes[c][m].mean()) for c in "RGB")
    px = np.float32([[[B, G, R]]])
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0]
    lab = cv2.cvtColor(px, cv2.COLOR_BGR2Lab)[0, 0]
    out["H_tm_deg"], out["a_tm"], out["b_tm"] = float(hsv[0]), float(lab[1]), float(lab[2])
    exg, vari, gli, ngrdi = _indices(np.array(R), np.array(G), np.array(B))
    out["ExG_tm"], out["VARI_tm"], out["GLI_tm"], out["NGRDI_tm"] = map(float, (exg, vari, gli, ngrdi))
    return out


# ---------------------------------------------------------------- 6. tiles
def make_tiles(ref_row, tile_m, margin_frac=0.03):
    """Square ground tiles of tile_m metres laid out in the reference image."""
    w, h = ref_row["width"], ref_row["height"]
    step = tile_m * 100.0 / ref_row["gsd_cm"]   # pixels per tile in the reference
    mx, my = margin_frac * w, margin_frac * h
    tiles = []
    y = my
    k = 0
    while y + step <= h - my:
        x = mx
        while x + step <= w - mx:
            tiles.append((k, np.float32([[x, y], [x + step, y], [x + step, y + step], [x, y + step]])))
            k += 1
            x += step
        y += step
    return tiles


def extract_session(sess, H_to_ref, ref_file, cfg):
    ref = sess[sess["file"] == ref_file].iloc[0]
    tiles = make_tiles(ref, cfg["TILE_M"])
    rows = []
    for _, r in sess[sess["registered"]].iterrows():
        img = cv2.imread(r["path"])
        planes = image_color_planes(img)
        H_img_from_ref = np.linalg.inv(H_to_ref[r["file"]])
        hgt, wdt = img.shape[:2]
        for tid, poly in tiles:
            q = cv2.perspectiveTransform(poly[None], H_img_from_ref)[0]
            if (q[:, 0].min() < 5 or q[:, 1].min() < 5 or q[:, 0].max() > wdt - 5 or q[:, 1].max() > hgt - 5):
                continue  # tile not fully inside this image
            x0, y0 = np.floor(q.min(0)).astype(int)
            x1, y1 = np.ceil(q.max(0)).astype(int) + 1
            sub = {k: v[y0:y1, x0:x1] for k, v in planes.items()}
            mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillConvexPoly(mask, np.int32(np.round(q - [x0, y0])), 1)
            # inward buffer so bunds and misregistration at the edge are excluded
            bpx = max(1, int(round(cfg["EDGE_BUFFER_M"] * 100.0 / r["gsd_cm"])))
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=bpx) > 0
            feat = tile_features(sub, mask)
            if feat is None:
                continue
            feat.update(session=r["session"], file=r["file"], tile=f"{r['session']}_T{tid:04d}",
                        source="real", rel_alt_m=r["rel_alt_m"], gimbal_pitch=r["gimbal_pitch"],
                        pitch_class=r["pitch_class"], gsd_cm=r["gsd_cm"])
            rows.append(feat)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 7. simulated degradation
def simulate_session(sess, H_to_ref, ref_file, cfg):
    """Area-average the finest kept nadir image to coarser GSDs and extract the
    same tiles, so the effect of resolution alone can be isolated."""
    nad = sess[(sess["registered"]) & (sess["pitch_class"] == "nadir")]
    if nad.empty:
        return pd.DataFrame()
    base = nad.sort_values("gsd_cm").iloc[0]
    ref = sess[sess["file"] == ref_file].iloc[0]
    tiles = make_tiles(ref, cfg["TILE_M"])
    img0 = cv2.imread(base["path"])
    H0 = np.linalg.inv(H_to_ref[base["file"]])
    rows = []
    for target in cfg["SIM_GSD_CM"]:
        if target < base["gsd_cm"]:
            continue
        f = base["gsd_cm"] / target
        img = cv2.resize(img0, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else img0
        S = np.diag([f, f, 1.0]) if f < 1 else np.eye(3)
        planes = image_color_planes(img)
        hgt, wdt = img.shape[:2]
        for tid, poly in tiles:
            q = cv2.perspectiveTransform(poly[None], S @ H0)[0]
            if (q[:, 0].min() < 2 or q[:, 1].min() < 2 or q[:, 0].max() > wdt - 2 or q[:, 1].max() > hgt - 2):
                continue
            x0, y0 = np.floor(q.min(0)).astype(int)
            x1, y1 = np.ceil(q.max(0)).astype(int) + 1
            sub = {k: v[y0:y1, x0:x1] for k, v in planes.items()}
            mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillConvexPoly(mask, np.int32(np.round(q - [x0, y0])), 1)
            bpx = max(1, int(round(cfg["EDGE_BUFFER_M"] * 100.0 / target)))
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=bpx) > 0
            feat = tile_features(sub, mask)
            if feat is None:
                continue
            feat.update(session=base["session"], file=base["file"], tile=f"{base['session']}_T{tid:04d}",
                        source="simulated", rel_alt_m=target * base["rel_alt_m"] / base["gsd_cm"],
                        gimbal_pitch=base["gimbal_pitch"], pitch_class="nadir", gsd_cm=float(target))
            rows.append(feat)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 8. stability summaries
ANGULAR = {"H_deg", "H_veg_deg", "H_tm_deg"}


def circ_sd_deg(h):
    h = np.asarray(h, float)
    h = h[~np.isnan(h)]
    if h.size < 2:
        return np.nan
    _, R = circ_mean_deg(h)
    return float(np.degrees(np.sqrt(-2 * np.log(max(R, 1e-12)))))


def hue_stability(real):
    """Hue stability index (HSI) per tile = circular SD of tile hue across the
    acquisition settings in which that tile was observed."""
    g = real.groupby("tile")
    out = pd.DataFrame({
        "session": g["session"].first(),
        "n_obs": g.size(),
        "alt_range_m": g["rel_alt_m"].agg(lambda s: s.max() - s.min()),
        "n_pitch_classes": g["pitch_class"].nunique(),
        "HSI_H_deg": g["H_deg"].agg(circ_sd_deg),
        "HSI_H_veg_deg": g["H_veg_deg"].agg(circ_sd_deg),
        "HSI_H_tm_deg": g["H_tm_deg"].agg(circ_sd_deg),
    }).reset_index()
    return out[out["n_obs"] >= 2]


def _spread(x, angular):
    return circ_sd_deg(x) if angular else float(np.nanstd(x, ddof=1)) if np.isfinite(x).sum() > 1 else np.nan


def sensitivity_ratio(real, features):
    """Within-tile spread across acquisition settings divided by the spread of
    session means. Sessions are weeks apart, so session means stand in for
    growth-stage differences until stage labels are available."""
    rows = []
    multi = real[real.groupby("tile")["tile"].transform("size") >= 2]
    for f in features:
        ang = f in ANGULAR
        within = multi.groupby("tile")[f].agg(lambda s: _spread(s.values, ang)).median()
        if ang:
            sess_means = real.groupby("session")[f].agg(lambda s: circ_mean_deg(s.dropna().values)[0])
        else:
            sess_means = real.groupby("session")[f].mean()
        between = _spread(sess_means.values, ang)
        rows.append({"feature": f, "within_tile_spread": within, "between_session_spread": between,
                     "sensitivity_ratio": within / between if between and np.isfinite(between) else np.nan})
    return pd.DataFrame(rows).sort_values("sensitivity_ratio")


def real_vs_simulated(real, sim, features, gsd_tol=0.25):
    """Pair each real observation with the simulated observation of the same tile
    at the nearest GSD (within ±25 %), and report agreement per feature."""
    rows = []
    if sim.empty:
        return pd.DataFrame()
    for f in features:
        a, b = [], []
        for tile, grp in real[real["pitch_class"] == "nadir"].groupby("tile"):
            s = sim[sim["tile"] == tile]
            if s.empty:
                continue
            for _, r in grp.iterrows():
                d = (s["gsd_cm"] - r["gsd_cm"]).abs()
                j = d.idxmin()
                if d[j] <= gsd_tol * r["gsd_cm"]:
                    a.append(r[f]); b.append(s.at[j, f])
        a, b = np.array(a, float), np.array(b, float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3:
            continue
        diff = a[ok] - b[ok]
        if f in ANGULAR:
            diff = (diff + 180) % 360 - 180
        rows.append({"feature": f, "n_pairs": int(ok.sum()), "pearson_r": float(np.corrcoef(a[ok], b[ok])[0, 1]),
                     "mean_diff_real_minus_sim": float(diff.mean()), "mae": float(np.abs(diff).mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 9. mixed models
def fit_mixed_models(real, features):
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
    d = real.copy()
    d["pitch_class"] = d["pitch_class"].astype(str)
    d = d[d.groupby("tile")["tile"].transform("size") >= 2]
    rows = []
    for f in features:
        y = d[f].copy()
        if f in ANGULAR:  # unwrap hue around its circular mean so the linear model is valid
            mu = circ_mean_deg(y.dropna().values)[0]
            y = (y - mu + 180) % 360 - 180
        dd = d.assign(y=y).dropna(subset=["y", "gsd_cm"])
        if dd["tile"].nunique() < 5:
            continue
        terms = "gsd_cm"
        if dd["pitch_class"].nunique() > 1:
            terms += " + C(pitch_class, Treatment('nadir'))"
        if dd["session"].nunique() > 1:
            terms += " + C(session)"
        try:
            m = smf.mixedlm(f"y ~ {terms}", dd, groups=dd["tile"]).fit(reml=True, method="lbfgs")
        except Exception as e:
            print(f"  {f}: model failed ({e})")
            continue
        for term in m.params.index:
            if term.startswith("gsd_cm") or term.startswith("C(pitch_class"):
                rows.append({"feature": f, "term": term, "estimate": m.params[term],
                             "std_err": m.bse[term], "p_value": m.pvalues[term],
                             "n_obs": int(m.nobs), "n_tiles": dd["tile"].nunique()})
    res = pd.DataFrame(rows)
    if not res.empty:
        res["p_adj_BH"] = multipletests(res["p_value"], method="fdr_bh")[1]
        res["significant"] = res["p_adj_BH"] < 0.05
    return res


# ---------------------------------------------------------------- 10. figures
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def make_figures(md, real, sim, hsi, sens, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    sessions = sorted(real["session"].unique(), key=lambda s: int(s[1:]))
    color = {s: SERIES[i % len(SERIES)] for i, s in enumerate(sessions)}

    # Fig 1: acquisition settings of all kept images
    fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=300)
    _style(ax)
    k = md[md["keep"]]
    for s in sessions:
        q = k[k["session"] == s]
        ax.scatter(q["rel_alt_m"], q["gimbal_pitch"], s=14, color=color[s], label=s,
                   edgecolor=SURF, linewidth=0.6)
    ax.set_xlabel("Relative flight altitude (m)", color=INK2, fontsize=9)
    ax.set_ylabel("Gimbal pitch (°)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=7, ncol=len(sessions), loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "Fig1_acquisition_settings.png")); plt.close(fig)

    # Fig 2: feature vs GSD for nadir tiles (real, by session) with the simulated curve
    feats = [("H_deg", "Hue (°)"), ("a", "CIELAB a*"), ("ExG", "ExG"), ("VARI", "VARI")]
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.2), dpi=300, sharex=True)
    nad = real[real["pitch_class"] == "nadir"]
    for ax, (f, lab) in zip(axes.flat, feats):
        _style(ax)
        for s in sessions:
            q = nad[nad["session"] == s]
            ax.scatter(q["gsd_cm"], q[f], s=6, color=color[s], alpha=0.55, linewidth=0)
            if not sim.empty:
                ss = sim[sim["session"] == s].groupby("gsd_cm")[f].mean()
                if len(ss) > 1:
                    ax.plot(ss.index, ss.values, color=color[s], lw=1.5)
        ax.set_title(lab, fontsize=9, color=INK, loc="left")
    for ax in axes[1]:
        ax.set_xlabel("Ground sampling distance (cm/pixel)", color=INK2, fontsize=8)
    handles = [plt.Line2D([], [], color=color[s], marker="o", lw=1.5, ms=4, label=s) for s in sessions]
    fig.legend(handles=handles, frameon=False, fontsize=7, ncol=len(sessions), loc="upper center")
    fig.text(0.5, 0.005, "Dots: real images. Lines: same tiles, simulated by area-averaging the finest nadir image.",
             ha="center", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94)); fig.savefig(os.path.join(outdir, "Fig2_features_vs_GSD.png")); plt.close(fig)

    # Fig 3: hue by pitch class
    fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=300)
    _style(ax)
    order = ["nadir", "moderate_oblique", "strong_oblique"]
    present = [p for p in order if p in set(real["pitch_class"])]
    data = []
    for p in present:  # deviation of each observation from its tile's nadir mean
        dev = []
        for tile, g in real.groupby("tile"):
            base = g[g["pitch_class"] == "nadir"]["H_deg"]
            if base.empty:
                continue
            mu = circ_mean_deg(base.dropna().values)[0]
            v = g[g["pitch_class"] == p]["H_deg"].dropna().values
            dev += list((v - mu + 180) % 360 - 180)
        data.append(dev if dev else [np.nan])
    bp = ax.boxplot(data, widths=0.45, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set(facecolor=SERIES[0], alpha=0.35, edgecolor=SERIES[0])
    for el in ("whiskers", "caps", "medians"):
        for line in bp[el]:
            line.set(color=INK2, lw=1)
    ax.set_xticks(range(1, len(present) + 1), [p.replace("_", " ") for p in present])
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_ylabel("Hue deviation from tile nadir mean (°)", color=INK2, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "Fig3_hue_by_camera_angle.png")); plt.close(fig)

    # Fig 4: sensitivity ratio per feature (lower = more robust)
    if not sens.empty:
        s = sens.dropna(subset=["sensitivity_ratio"]).sort_values("sensitivity_ratio")
        fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(s) + 1), dpi=300)
        _style(ax); ax.grid(axis="x", color=GRID, lw=0.6); ax.grid(axis="y", visible=False)
        ax.barh(s["feature"], s["sensitivity_ratio"], color=SERIES[0], height=0.6)
        ax.axvline(1, color=AXIS, lw=0.8, ls="--")
        ax.invert_yaxis()
        ax.set_xlabel("Acquisition sensitivity ratio (within-tile spread ÷ between-session spread)", color=INK2, fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "Fig4_sensitivity_ratio.png")); plt.close(fig)
    print("Figures written to", outdir)
