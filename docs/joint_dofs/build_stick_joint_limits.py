"""Author joint_limits for SMILy_STICK_authored.pkl from the literature table.

Sources: Theunissen+2015 JEB 218:340-352 (T15); Guschlbauer+2022 Curr Biol 32:2334-2340
(G22); Dallmann+2016 Proc R Soc B 283:20151708 (D16). See stick_insect_joint_dofs.md.

Limits are written in the MODEL/GLOBAL frame (+x anterior, +y left, +z dorsal), which is
what joint_limits stores. Bone-local values for Blender are derived with the verified
uniform mapping XL=XG, YL=ZG, ZL=-YG and exported to the CSV alongside.
"""
import csv, math, pickle, os, numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKL  = os.path.join(REPO, "3D_model_prep", "SMILy_STICK_authored.pkl")
CSV  = os.path.join(REPO, "docs", "joint_dofs", "stick_insect_joint_limits.csv")
GIVE, FREE = 10.0, 180.0

d = pickle.load(open(PKL, "rb"), encoding="latin1")
names = list(d["J_names"]); J = np.array(d["J"], float)
idx = {m: i for i, m in enumerate(names)}
u = lambda v: v / np.linalg.norm(v)

T15="Theunissen+2015 JEB 218:340-352"; G22="Guschlbauer+2022 CurrBiol 32:2334-2340"
D16="Dallmann+2016 ProcRSocB 283:20151708"

# ---- anatomical axes, recomputed from THIS file's rest pose ------------------
legax = {}
for L in (1, 2, 3):
    for s in ("r", "l"):
        v = J[idx[f"l_{L}_ta_{s}"]] - J[idx[f"l_{L}_co_{s}"]]
        h = u(np.array([v[0], v[1], 0.0]))          # limb horizontal dir = supination axis
        legax[(L, s)] = dict(pro=np.array([0.,0.,1.]), sup=h, lev=u(np.cross(h,[0,0,1.])),
                             splay=math.degrees(math.acos(min(1, abs(h @ np.array([0.,1.,0.]))))))

def env(dofs):
    """Minkowski sum of theta*axis -> conservative axis-aligned box, global frame."""
    lo, hi = np.zeros(3), np.zeros(3)
    for ax, (a, b) in dofs:
        for i in range(3):
            c = (a*ax[i], b*ax[i]); lo[i] += min(c); hi[i] += max(c)
    return [(round(lo[i],2), round(hi[i],2)) for i in range(3)]

THC = {1:(-20.,88.,f"{T15} Fig.9A/10 front -20..+80; {G22} FL ThF 2-104deg -> -14..+88"),
       2:(-49.,71.,f"{T15} Fig.9A/10 middle -40..+60; {G22} ML ThF 19-139deg -> -49..+71"),
       3:(-80.,20.,f"{T15} Fig.9A/10 hind -80..+20; {G22} HL ThF 77-164deg -> -74..+13")}
SUP,LEV,FTI,TITA = (-60.,60.), (-40.,60.), (-50.,70.), (-60.,60.)

G = {}; META = {}
def put(j, box, dof, st, src, note, cond):
    G[j] = box; META[j] = (dof, st, src, note, cond)

lock3 = [(-GIVE,GIVE)]*3; free3 = [(-FREE,FREE)]*3

for L in (1,2,3):
    for s in ("r","l"):
        A = legax[(L,s)]; sp = A["splay"]
        mir = lambda t: t if s=="r" else (-t[1], -t[0])
        cond = f"protraction exact on ZG; XG/YG conservative (limb splay {sp:.1f}deg)"
        lo,hi,src = THC[L]
        put(f"l_{L}_co_{s}", env([(A["pro"],mir((lo,hi))),(A["sup"],mir(SUP))]),
            ("supination comp.","protraction/retraction","supination comp."), ("free",)*3,
            src+f" | supination {T15} Fig.10 -60..+60",
            f"ThC 3-DOF per {T15}; {D16} models it as one slanted axis (theta=30deg).", cond)
        put(f"l_{L}_tr_{s}", env([(A["lev"],mir(LEV))]),
            ("levation comp.","(none)","levation comp."), ("free","locked","free"),
            f"{T15} Fig.9C/10 levation -40..+60; {D16} sec.2c 'CTr approximated as a hinge with 1 d.f.'",
            "1-DOF hinge; axis not rig-aligned, do not hard-lock XG/YG here.", cond)
        put(f"l_{L}_fe_{s}", lock3, ("(fused)",)*3, ("locked",)*3,
            f"{T15} Fig.9 legend: 'the femur is fused with the trochanter in these species, without a movable joint in between'",
            "No anatomical joint in Phasmatodea; +-10deg cuticular give only.", "n/a (locked)")
        put(f"l_{L}_ti_{s}", env([(A["lev"],mir(FTI))]),
            ("flex/ext comp.","(none)","flex/ext comp."), ("free","locked","free"),
            f"{T15} Fig.9D absolute 40-160deg; {D16} Ext<=150deg, neutral Ext=90deg; {D16} sec.2c 'both joints move in the same plane, the leg plane'",
            "Relative to the 90deg neutral tibia posture (D16). SIGN NOT VERIFIED - confirm extension sense with R X X.", cond)
        put(f"l_{L}_ta_{s}", env([(A["lev"],mir(TITA))]),
            ("flex/ext comp.","(none)","flex/ext comp."), ("free (ASSUMED)","locked","free (ASSUMED)"),
            f"NOT MEASURED. {D16} uses TiTa only as a tarsus-position estimate",
            "-60..+60 is a permissive placeholder, not a citation.", cond)
        put(f"l_{L}_pt_{s}", free3, ("multi-axis",)*3, ("free (ASSUMED)",)*3,
            "NOT MEASURED in any of the 3 papers", "Left wide open.", "n/a (free)")

put("b_t", [(0.,0.)]*3, ("global orientation",)*3, ("root - not limited",)*3,
    "Root convention (joint_limits[0]==0)", "Root row must stay all-zero.", "n/a")
for i,jn in enumerate(["b_a_1","b_a_2","b_a_3","b_a_4","b_a_5"],1):
    put(jn, free3, ("roll","pitch","yaw"), ("free (NOT MEASURED)",)*3,
        "NOT MEASURED. All 3 papers track thorax/head/legs only.",
        "b_a_1 authored row corrected: Y was [-180,-180] (zero-width) -> [-180,180] per user.", "n/a (free)")
put("b_h", [(-GIVE,GIVE),(-15.,15.),(-30.,30.)], ("roll","pitch / levation","yaw"),
    ("locked","free","free (NOT MEASURED)"),
    f"{T15} Table 2 'Head levation range 30deg' (Carausius; 15 Medauroidea, 45 Aretaon)",
    f"{T15} Fig.5D: head is NOT gaze-stabilised. Yaw is a placeholder.", "exact")
for s in ("r","l"):
    put(f"ma_{s}", [(-GIVE,GIVE),(-GIVE,GIVE),(-70.,70.)],
        ("weak secondary","(none)","open/close"), ("locked","locked","free"),
        "NOT IN THESE PAPERS. Carried over from OmniAnt_25PCs_joint_limited.pkl",
        "Locomotion papers only; no mandible kinematics.", "n/a")
    put(f"an_1_{s}", [(-GIVE,GIVE),(-60.,60.),(-60.,60.)],
        ("(none)","elevation/depression","adduction/abduction"),
        ("locked","free (NOT MEASURED)","free (NOT MEASURED)"),
        f"NOT MEASURED. {T15} reports antennal LENGTHS only (Table 2)",
        "HS is 2-DOF; ranges need the Durr-lab antennal literature.", "n/a")
    put(f"an_2_{s}", [(-GIVE,GIVE),(-80.,80.),(-GIVE,GIVE)],
        ("(none)","rotation about SP axis","(none)"), ("locked","free (NOT MEASURED)","locked"),
        "NOT MEASURED", "SP is 1-DOF; range not in these papers.", "n/a")
    put(f"an_3_{s}", lock3, ("(passive flagellum)",)*3, ("locked",)*3,
        "NOT MEASURED. The flagellum has no muscles in Phasmatodea.", "Passive bending only.", "n/a")
    for w in ("w_1","w_2"):
        put(f"{w}_{s}", lock3, ("(none)",)*3, ("locked",)*3,
            "N/A - the species studied are apterous/brachypterous", "Locked.", "n/a")

# ---- give floor -------------------------------------------------------------
# Projecting a horizontal hinge axis onto ZG yields exactly 0, i.e. a zero-width
# hard lock. Real cuticular joints have slack and a hard zero makes the hinge loss
# brittle, so every non-root axis is widened to at least +-GIVE.
for j,box in G.items():
    if j == "b_t": continue
    G[j] = [(min(lo,-GIVE), max(hi,GIVE)) for lo,hi in box]

# ---- build array (global frame, radians) ------------------------------------
jl = np.zeros((len(names),3,2))
for j,box in G.items():
    for i,(lo,hi) in enumerate(box):
        jl[idx[j],i] = (math.radians(lo), math.radians(hi))

# ---- verify -----------------------------------------------------------------
assert set(G)==set(names), set(names)^set(G)
assert jl.shape==(55,3,2)
assert np.allclose(jl[0],0.0), "root must be zero"
assert (jl[...,0]<=jl[...,1]).all(), "min>max somewhere"
assert (np.abs(jl)<=math.pi+1e-9).all(), "outside +-180deg"
assert not np.allclose(np.abs(np.degrees(jl)),180.0), "all-free would be a no-op prior"
w = np.degrees(jl[...,1]-jl[...,0]); w[idx["b_t"]] = np.inf
assert (w >= 2*GIVE-1e-6).all(), f"zero-width/underwide axis: {[(names[i],a) for i,a in zip(*np.where(w<2*GIVE-1e-6))]}"
deg=np.degrees(jl)
assert np.allclose(deg[idx["b_a_1"]], [[-180,180]]*3), "b_a_1 fix not applied"
for L in (1,2,3):                     # protraction must land exactly on ZG
    lo,hi,_=THC[L]
    assert abs(deg[idx[f"l_{L}_co_r"],2,0]-lo)<0.01 and abs(deg[idx[f"l_{L}_co_r"],2,1]-hi)<0.01
    assert abs(deg[idx[f"l_{L}_co_l"],2,0]+hi)<0.01 and abs(deg[idx[f"l_{L}_co_l"],2,1]+lo)<0.01

d["joint_limits"]=jl
with open(PKL,"wb") as f: pickle.dump(d,f,protocol=2)

# round-trip the written file
chk=pickle.load(open(PKL,"rb"),encoding="latin1")
assert np.allclose(np.array(chk["joint_limits"],float),jl)
assert np.array(chk["v_template"]).shape==(3015,3) and np.array(chk["f"]).shape==(6009,3)
assert list(chk["J_names"])==names and np.allclose(np.array(chk["J"],float),J)

# ---- CSV, both frames -------------------------------------------------------
rows=[]
for j in sorted(G,key=lambda x: idx[x]):
    (dof,st,src,note,cond)=META[j]; box=G[j]
    loc={"x":box[0], "y":box[2], "z":(-box[1][1],-box[1][0])}   # XL=XG, YL=ZG, ZL=-YG
    ldof={"x":dof[0],"y":dof[2],"z":dof[1]}; lst={"x":st[0],"y":st[2],"z":st[1]}
    for ax in "xyz":
        g={"x":box[0],"y":box[1],"z":box[2]}[ax]
        rows.append(dict(joint_index=idx[j], repo_joint=j,
            anatomical_joint=j, axis_local=ax,
            local_min_deg=round(loc[ax][0],2), local_max_deg=round(loc[ax][1],2),
            local_min_rad=round(math.radians(loc[ax][0]),5), local_max_rad=round(math.radians(loc[ax][1]),5),
            maps_to_global={"x":"XG","y":"ZG","z":"-YG"}[ax],
            global_axis=ax.upper()+"G", global_min_deg=round(g[0],2), global_max_deg=round(g[1],2),
            anatomical_dof_local=ldof[ax], dof_status_local=lst[ax],
            axis_conditioning=cond, source=src, notes=note))
with open(CSV,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
assert len(rows)==165

print(f"OK  wrote joint_limits {jl.shape} to {os.path.basename(PKL)}  (mesh untouched: 3015 v / 6009 f)")
print(f"OK  wrote {len(rows)} rows to {os.path.basename(CSV)}")
print(f"    free axes: {int(((np.abs(deg[...,0])>GIVE+.01)|(np.abs(deg[...,1])>GIVE+.01)).sum())}/165"
      f"   fully-free joints: {sum(1 for i in range(55) if np.allclose(np.abs(deg[i]),180))}/55")
print("\nsample (global frame, deg):")
for j in ["b_a_1","l_1_co_r","l_1_co_l","l_2_tr_r","l_3_ti_r","l_1_fe_r","b_h"]:
    r=np.round(deg[idx[j]],1)
    print(f"  {j:10s} X[{r[0,0]:7.1f},{r[0,1]:7.1f}] Y[{r[1,0]:7.1f},{r[1,1]:7.1f}] Z[{r[2,0]:7.1f},{r[2,1]:7.1f}]")
