"""Meta-analytic clustering workflow."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.metrics import silhouette_score

from .. import references
from ..dataset import Dataset
from ..due import due
from ..io import convert_sleuth_to_dataset
from ..meta.kernel import ALEKernel, KDAKernel, MKDAKernel, Peaks2MapsKernel


@due.dcite(
    references.META_CLUSTER,
    description="Introduces meta-analytic clustering analysis; "
    "hierarchically clustering face paradigms.",
)
@due.dcite(
    references.META_CLUSTER2,
    description="Performs the specific meta-analytic clustering approach implemented here.",
)
def meta_cluster_workflow(
    dataset_file,
    output_dir=None,
    output_prefix=None,
    kernel="ALEKernel",
    coord=True,
    algorithm="kmeans",
    clust_range=(2, 10),
):
    """Perform a meta-analytic clustering analysis on a dataset file.

    Warnings
    --------
    This method is not yet implemented.
    """

    def VI(X, Y):
        from math import log

        # from https://gist.github.com/jwcarr/626cbc80e0006b526688
        n = float(sum([len(x) for x in X]))
        sigma = 0.0
        for x in X:
            p = len(x) / n
            for y in Y:
                q = len(y) / n
                r = len(set(x) & set(y)) / n
                if r > 0.0:
                    sigma += r * (log(r / p, 2) + log(r / q, 2))
        return abs(sigma)

    if dataset_file.endswith(".json"):
        dset = Dataset(dataset_file, target="mni152_2mm")
    elif dataset_file.endswith(".txt"):
        dset = convert_sleuth_to_dataset(dataset_file, target="mni152_2mm")
    else:
        dset = Dataset.load(dataset_file)

    if coord:
        if kernel == "ALEKernel":
            kern = ALEKernel(dset.coordinates, "template_img")
        elif kernel == "MKDAKernel":
            kern = MKDAKernel(dset.coordinates, "template_img")
        elif kernel == "KDAKernel":
            kern = KDAKernel(dset.coordinates, "template_img")
        elif kernel == "Peaks2MapsKernel":
            kern = Peaks2MapsKernel(dset.coordinates, "template_img")
        imgs = kern.transform(dset.ids)
    imgs_arr = []
    for i in np.arange(0, len(imgs)):
        imgs_arr.append(np.ravel(imgs[i].get_fdata(), order="C"))
    labels = pd.DataFrame(index=dset.ids)
    k = np.arange(clust_range[0], (clust_range[1] + 1))
    for i in k:
        if algorithm == "kmeans":
            clustering = KMeans(i, init="k-means++", precompute_distances="auto")
        if algorithm == "spectral":
            clustering = SpectralClustering(
                n_clusters=i,
                eigen_solver=None,
                random_state=None,
                n_init=300,
                gamma=1.0,
                affinity="rbf",
                n_neighbors=10,
                eigen_tol=0.0,
                assign_labels="discretize",
                degree=3,
                coef0=1,
                kernel_params=None,
            )
        if algorithm == "dbscan":
            min = len(dset.ids) / (i - 1)
            clustering = DBSCAN(
                eps=0.1,
                min_samples=min,
                metric="euclidean",
                metric_params=None,
                algorithm="auto",
                leaf_size=30,
                p=None,
            )
        labels[i] = clustering.fit_predict(imgs_arr)
    labels.to_csv("{0}/{1}_labels.csv".format(output_dir, output_prefix))

    silhouette_scores = {}
    for i in k:
        j = i - 2
        silhouette = silhouette_score(imgs_arr, labels[i], metric="correlation", random_state=None)
        silhouette_scores[i] = silhouette
    silhouettes = pd.Series(silhouette_scores, name="Average Silhouette Scores")

    clusters_idx = {}
    for i in k:
        clusters = []
        for j in range(i):
            clusters.append(list(np.where(labels[i] == j)[0]))
        clusters_idx["Solution {0}".format(i)] = clusters

    variation_of_information = {}
    for i in k[:-1]:
        j = clusters_idx["Solution {0}".format(i)]
        z = clusters_idx["Solution {0}".format(i + 1)]
        var_info = VI(j, z)
        variation_of_information[i + 1] = var_info

    vi = pd.Series(variation_of_information, name="Variation of Information")

    metrics = pd.concat([vi, silhouettes], axis=1)
    metrics.to_csv("{0}/{1}_metrics.csv".format(output_dir, output_prefix))

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    g = sns.lineplot(metrics.index, metrics["Average Silhouette Scores"], ax=ax[0])
    g.set_title("Silhouette Scores")
    g = sns.lineplot(metrics.index, metrics["Variation of Information"], ax=ax[1])
    g.set_title("Variation of Information")
    fig.savefig("{0}/{1}_metrics.png".format(output_dir, output_prefix), dpi=300)
