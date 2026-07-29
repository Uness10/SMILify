import numpy as np
from smal_model.smal_torch import load_smal_model   # chumpy-safe loader
dd = load_smal_model("3D_model_prep/SMIL_OmniAnt_authored.pkl")
jl = np.asarray(dd["joint_limits"])
print(jl.shape)                          # -> (J, 3, 2)
i = dd["J_names"].index("w_1_l")
print(jl[i])                             # your limits, in RADIANS
print(jl[0])   