import csv, math, pickle, numpy as np, os

PKL = "/sessions/nice-determined-sagan/mnt/SMILify/3D_model_prep/SMILy_STICK.pkl"
OUT = "/sessions/nice-determined-sagan/mnt/SMILify/docs/joint_dofs/stick_insect_joint_limits.csv"

d = pickle.load(open(PKL, "rb"), encoding="latin1")
names = list(d["J_names"])
idx = {n: i for i, n in enumerate(names)}

GIVE = 10.0  # "locked" axes get +-10 deg per user spec

T15 = "Theunissen, Bekemeier & Durr 2015 JEB 218:340-352"
G22 = "Guschlbauer et al. 2022 Curr Biol 32:2334-2340"
D16 = "Dallmann, Durr & Schmitz 2016 Proc R Soc B 283:20151708"

rows = []
def add(joint, anat, axis, dof, lo, hi, src, note):
    rows.append(dict(
        joint_index=idx[joint], repo_joint=joint, anatomical_joint=anat,
        axis=axis, anatomical_dof=dof[0], dof_status=dof[1],
        min_deg=round(lo, 1), max_deg=round(hi, 1),
        min_rad=round(math.radians(lo), 5), max_rad=round(math.radians(hi), 5),
        source=src, notes=note))

LOCK = ("(no joint / weak secondary axis)", "locked")

# ---- ThC (coxa bone) : 3 DOF -------------------------------------------------
# 0 deg = leg perpendicular to its thorax segment; + = protraction (T15 Fig.9A).
# Envelope = union of T15 Fig.9/10 plotted ranges and G22 ThF ranges re-zeroed
# to the perpendicular (ThF 90 deg == 0 deg here; pro/retraction = 90 - ThF).
THC_PRO = {
    "1": (-20.0,  88.0, f"{T15} Fig.9A/Fig.10 (front -20..+80); {G22} Fig.1 FL ThF 2-104 deg -> -14..+88",
          "front leg. T15 Table 2 gives FL protraction range 45 deg (Carausius). Envelope of walking + passive-rest angles."),
    "2": (-49.0,  71.0, f"{T15} Fig.9A/Fig.10 (middle -40..+60); {G22} Fig.1 ML ThF 19-139 deg -> -49..+71",
          "middle leg. T15 Table 2 ML protraction mean 0 deg. Largest pro/retraction amplitude of all legs (T15)."),
    "3": (-80.0,  20.0, f"{T15} Fig.9A/Fig.10 (hind -80..+20); {G22} Fig.1 HL ThF 77-164 deg -> -74..+13",
          "hind leg. Two independent datasets agree to within ~6 deg."),
}
for L in ("1", "2", "3"):
    lo, hi, src, note = THC_PRO[L]
    for side in ("r", "l"):
        j = f"l_{L}_co_{side}"
        a, b = (lo, hi) if side == "r" else (-hi, -lo)
        add(j, "thorax-coxa (ThC)", "z", ("protraction(+)/retraction(-)", "free"), a, b, src,
            note + ("" if side == "r" else " | LEFT: sign mirrored, see repo precedent l_2_tr_l in OmniAnt_25PCs_joint_limited.pkl"))
        add(j, "thorax-coxa (ThC)", "y", ("supination(+)/pronation(-)", "free"), -60.0, 60.0,
            f"{T15} Fig.10 supination axis -60..+60 deg (Fig.9B time courses span -40..+40)",
            "T15 measured pro/supination independently instead of assuming a fixed slanted axis; "
            f"{D16} instead lumps it into one slanted ThC axis (theta=30 deg from vertical, 1 d.f.).")
        add(j, "thorax-coxa (ThC)", "x", ("levation/depression at ThC", "locked"), -GIVE, GIVE,
            f"{T15} Fig.9C legend (ThC+CTr levation lumped); {D16} sec.2c (CTr is the 1-d.f. lev/dep hinge)",
            "Levation/depression is assigned to the CTr joint (l_*_tr_*); T15 notes 'slack of the rotation axis in the ThC joint, which is commonly considered fixed'.")

# ---- CTr (trochanter bone) : 1 DOF hinge ------------------------------------
for L in ("1", "2", "3"):
    for side in ("r", "l"):
        j = f"l_{L}_tr_{side}"
        lo, hi = (-40.0, 60.0) if side == "r" else (-60.0, 40.0)
        add(j, "coxa-trochanter (CTr)", "x", ("levation(+)/depression(-)", "free"), lo, hi,
            f"{T15} Fig.9C/Fig.10 levation axis -40..+60 deg; {D16} sec.2c 'CTr approximated as a hinge with 1 d.f.'",
            "Levation ranges were 'generally similar among species and all legs' (T15). "
            f"{D16}: CTr torques point to depression throughout stance and carry body weight + propulsion."
            + ("" if side == "r" else " | LEFT: sign mirrored."))
        add(j, "coxa-trochanter (CTr)", "y", LOCK, -GIVE, GIVE, f"{D16} sec.2c (1 d.f. hinge)", "Bone long axis / roll. +-10 deg give.")
        add(j, "coxa-trochanter (CTr)", "z", LOCK, -GIVE, GIVE, f"{D16} sec.2c (1 d.f. hinge)", "+-10 deg give.")

# ---- trochanterofemur : FUSED, no joint -------------------------------------
for L in ("1", "2", "3"):
    for side in ("r", "l"):
        for ax in ("x", "y", "z"):
            add(f"l_{L}_fe_{side}", "trochanter-femur (FUSED)", ax,
                ("(none - segments fused)", "locked"), -GIVE, GIVE,
                f"{T15} Fig.9 legend: 'the femur is fused with the trochanter in these species, without a movable joint in between'",
                "No anatomical joint in Phasmatodea. Kept at +-10 deg only as cuticular give. "
                f"{D16} likewise models a single trochanterofemur link.")

# ---- FTi (tibia bone) : 1 DOF hinge -----------------------------------------
for L in ("1", "2", "3"):
    for side in ("r", "l"):
        j = f"l_{L}_ti_{side}"
        lo, hi = (-50.0, 70.0) if side == "r" else (-70.0, 50.0)
        add(j, "femur-tibia (FTi)", "x", ("extension(+)/flexion(-)", "free"), lo, hi,
            f"{T15} Fig.9D absolute FTi angle 40-160 deg (Fig.10 axis 40-140); {D16} fig.2a Ext up to 150 deg, neutral Ext=90 deg",
            "Expressed RELATIVE to the 90 deg neutral tibia posture identified by D16 ('torques counteract a deviation from an angle of 90 deg relative to the femur'). "
            "If the SMILy_STICK rest pose is not at FTi=90 deg, re-zero: min=40-rest, max=160-rest. "
            f"{T15} Table 2: HL flexion mean 100 deg (Carausius)."
            + ("" if side == "r" else " | LEFT: sign mirrored."))
        add(j, "femur-tibia (FTi)", "y", LOCK, -GIVE, GIVE, f"{D16} sec.2c 'FTi approximated as a hinge with 1 d.f.'", "Bone long axis / roll. +-10 deg give.")
        add(j, "femur-tibia (FTi)", "z", LOCK, -GIVE, GIVE, f"{D16} sec.2c 'FTi approximated as a hinge with 1 d.f.'", "+-10 deg give.")

# ---- TiTa / pretarsus : not measured ----------------------------------------
for L in ("1", "2", "3"):
    for side in ("r", "l"):
        add(f"l_{L}_ta_{side}", "tibia-tarsus (TiTa)", "x", ("flexion/extension", "free (ASSUMED)"), -60.0, 60.0,
            f"NOT MEASURED. {D16} fig.1a names TiTa but uses it only as a tarsus-position estimate ('markers cannot be placed on the tarsus without restraining movements')",
            "No published range in these 3 papers. -60..+60 is a permissive placeholder, not a citation.")
        for ax in ("y", "z"):
            add(f"l_{L}_ta_{side}", "tibia-tarsus (TiTa)", ax, LOCK, -GIVE, GIVE, "NOT MEASURED (assumed hinge)", "+-10 deg give.")
        for ax in ("x", "y", "z"):
            add(f"l_{L}_pt_{side}", "pretarsus / claw", ax, ("multi-axis", "free (ASSUMED)"), -180.0, 180.0,
                "NOT MEASURED in any of the 3 papers", "Left wide open (matches OmniAnt_25PCs_joint_limited.pkl precedent).")

# ---- body ---------------------------------------------------------------
for ax in ("x", "y", "z"):
    add("b_t", "root (metathorax)", ax, ("global orientation", "root - not limited"), 0.0, 0.0,
        f"Root convention (joint_limits[0] == 0 in OmniAnt_25PCs_joint_limited.pkl). {T15} Fig.4: metathorax pitch is a global body attitude, handled by global rotation.",
        "Root row must stay all-zero.")
for i, jn in enumerate(["b_a_1", "b_a_2", "b_a_3", "b_a_4", "b_a_5"], 1):
    for ax, dofn in (("x", "roll"), ("y", "pitch"), ("z", "yaw")):
        add(jn, f"abdominal segment {i}", ax, (dofn, "free (NOT MEASURED)"), -180.0, 180.0,
            "NOT MEASURED. All 3 papers track thorax/head/legs only; abdomen enters only via CoM and length ratios (T15 Table 2).",
            "Left free. Note T15: the 1st abdominal segment is fused to the metathorax in stick insects.")
add("b_h", "head-prothorax (neck)", "x", ("pitch / levation-depression", "free"), -15.0, 15.0,
    f"{T15} Table 2: 'Head levation range 30 deg' (Carausius; 15 deg Medauroidea, 45 deg Aretaon)",
    "30 deg total range split symmetrically about rest. T15 Fig.5D: head orientation is NOT gaze-stabilised - it tracks body pitch.")
add("b_h", "head-prothorax (neck)", "y", ("roll (bone long axis)", "locked"), -GIVE, GIVE,
    f"NOT MEASURED. {T15} computed head roll/pitch/yaw but reports only levation (pitch) ranges.", "+-10 deg give.")
add("b_h", "head-prothorax (neck)", "z", ("yaw", "free (NOT MEASURED)"), -30.0, 30.0,
    f"NOT MEASURED numerically. {T15}: 'head orientation varied almost' as much as the body.", "Placeholder, not a citation.")

# ---- mandibles / antennae / wings -------------------------------------------
for side in ("r", "l"):
    add(f"ma_{side}", "mandible", "z", ("open(+)/close(-)", "free"), -70.0, 70.0,
        "NOT IN THESE PAPERS. Value carried over from repo prior 3D_model_prep/OmniAnt_25PCs_joint_limited.pkl",
        "These 3 papers are locomotion studies; no mandible kinematics. Ant-derived prior.")
    add(f"ma_{side}", "mandible", "x", ("weak secondary swing", "locked"), -GIVE, GIVE,
        "NOT IN THESE PAPERS (repo prior had +-30 deg)", "Reduced to +-10 deg per the 'little give' rule.")
    add(f"ma_{side}", "mandible", "y", LOCK, -GIVE, GIVE, "NOT IN THESE PAPERS (repo prior +-10 deg)", "Hinge joint: +-10 deg give.")
    add(f"an_1_{side}", "head-scape (HS)", "x", ("elevation/depression", "free (NOT MEASURED)"), -60.0, 60.0,
        f"NOT MEASURED. {T15} reports antennal LENGTHS (Table 2, Ant:FL ratio) and that all species moved antennae continuously, but no joint angles.",
        "HS is a 2-DOF joint in stick insects; ranges need Duerr et al. antennal-movement literature, not these 3 papers.")
    add(f"an_1_{side}", "head-scape (HS)", "z", ("adduction/abduction", "free (NOT MEASURED)"), -60.0, 60.0, "NOT MEASURED", "See above.")
    add(f"an_1_{side}", "head-scape (HS)", "y", LOCK, -GIVE, GIVE, "NOT MEASURED", "Scape long axis. +-10 deg give.")
    add(f"an_2_{side}", "scape-pedicel (SP)", "x", ("rotation about SP axis", "free (NOT MEASURED)"), -80.0, 80.0, "NOT MEASURED", "SP is a 1-DOF joint; range not in these papers.")
    for ax in ("y", "z"):
        add(f"an_2_{side}", "scape-pedicel (SP)", ax, LOCK, -GIVE, GIVE, "NOT MEASURED", "+-10 deg give.")
    for ax in ("x", "y", "z"):
        add(f"an_3_{side}", "flagellum", ax, ("(none - passive flagellum)", "locked"), -GIVE, GIVE,
            "NOT MEASURED. The flagellum has no muscles in Phasmatodea.", "+-10 deg give (passive bending only).")
    for w in ("w_1", "w_2"):
        for ax in ("x", "y", "z"):
            add(f"{w}_{side}", "wing (N/A)", ax, ("(none)", "locked"), -GIVE, GIVE,
                "N/A - Carausius morosus and the other species studied are apterous/brachypterous; no wing data.",
                "Locked; delete the bone or leave at +-10 deg.")

rows.sort(key=lambda r: (r["joint_index"], "xyz".index(r["axis"])))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# verification
seen = {r["repo_joint"] for r in rows}
missing = [n for n in names if n not in seen]
print("rows:", len(rows), "expected:", 55*3)
print("missing joints:", missing)
from collections import Counter
c = Counter((r["repo_joint"], r["axis"]) for r in rows)
print("dупes:", [k for k, v in c.items() if v != 1])
print("index/name mismatches:", [r["repo_joint"] for r in rows if names[r["joint_index"]] != r["repo_joint"]])
print("out of +-180:", [(r["repo_joint"], r["axis"]) for r in rows if not (-180 <= r["min_deg"] <= r["max_deg"] <= 180)])
