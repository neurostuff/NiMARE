"""The atlas ladder: coarse networks to fine functional modes."""
from nilearn import datasets

ATLASES = {
    "Yeo-7":        lambda: datasets.fetch_atlas_yeo_2011(n_networks=7, thickness="thick"),
    "MSDL-39":      lambda: datasets.fetch_atlas_msdl(),
    "HarvOx-48":    lambda: datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm"),
    "DiFuMo-64":    lambda: datasets.fetch_atlas_difumo(dimension=64, resolution_mm=3),
    "AAL-116":      lambda: datasets.fetch_atlas_aal(),
    "DiFuMo-128":   lambda: datasets.fetch_atlas_difumo(dimension=128, resolution_mm=3),
    "Destrieux":    lambda: datasets.fetch_atlas_destrieux_2009(),
    "Schaefer-200": lambda: datasets.fetch_atlas_schaefer_2018(n_rois=200, resolution_mm=2),
    "DiFuMo-256":   lambda: datasets.fetch_atlas_difumo(dimension=256, resolution_mm=3),
    "Schaefer-400": lambda: datasets.fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2),
    "DiFuMo-512":   lambda: datasets.fetch_atlas_difumo(dimension=512, resolution_mm=3),
}
