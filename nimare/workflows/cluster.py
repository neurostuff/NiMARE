import click
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.metrics import silhouette_score

from ..due import due
from .. import references
from ..meta.cbma.kernel import ALEKernel, MKDAKernel, KDAKernel, Peaks2MapsKernel
from ..io import convert_sleuth_to_dataset


@click.command(name='metacluster',
               short_help='clusters experiments based on similarity'
                          'of activation patterns, to investigate '
                          'heterogeneity across a meta-analytic dataset',
               help='Method for investigating recurrent patterns of activation accross a '
                    'meta-analytic dataset, thus identifying trends across a collection of '
                    'experiments.')
@click.argument('database', required=True, type=click.Path(exists=True, readable=True))
@click.option('--output_dir', required=True, type=click.Path(exists=True), help='Directory into which clustering results will be written.')
@click.option('--output_prefix', default='metacluster', type=str, help='Common prefix for output clustering results.')
@click.option('--kernel', default='ALEKernel', type=click.Choice(['ALEKernel', 'MKDAKernel', 'KDAKernel', 'Peaks2MapsKernel']), help='Kernel estimator, for coordinate-based metaclustering.')
@click.option('--coord/--img', required=True, default=False, help='Is input data image- or coordinate-based?')
@click.option('--algorithm', '-a', default='kmeans', type=click.Choice(['kmeans', 'dbscan', 'spectral']), help='Clustering algorithm to be used, from sklearn.cluster.')
@click.option('--clust_range', nargs=2, type=int, help='Select a range for k over which clustering solutions will be evaluated (e.g., 2 10 will evaluate solutions with k = 2 clusters to k = 10 clusters).')
@due.dcite(references.META_CLUSTER,
           description='Introduces meta-analytic clustering analysis; hierarchically clusering face paradigms.')
@due.dcite(references.META_CLUSTER2,
           description='Performs the specific meta-analytic clustering approach included here.')
def meta_cluster_workflow(database, output_dir=None, output_prefix=None,
                          kernel='ALEKernel', coord=True, algorithm='kmeans',
                          clust_range=(2, 10)):
    """
    Perform a meta-analytic clustering analysis on a database file.

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
    # template_file = get_template(space='mni152_1mm', mask=None)
    if database.endswith('.json'):
        db = database  # how do I read in a generic database file? do I need options for source type?
        ids = db.ids
        dset = db.get_dataset(ids, target='mni152_2mm')
    elif database.endswith('.txt'):
        db = convert_sleuth_to_dataset(database)
        dset = db.get_dataset(target='mni152_2mm')
    else:
        raise click.BadParameter('You\'ve provided a database that metacluster can\'t read. :(', param_hint='database')
    # imgs = dset.images
    if coord:
        if kernel == 'ALEKernel':
            kern = ALEKernel(dset.coordinates, 'template_img')
        elif kernel == 'MKDAKernel':
            kern = MKDAKernel(dset.coordinates, 'template_img')
        elif kernel == 'KDAKernel':
            kern = KDAKernel(dset.coordinates, 'template_img')
        elif kernel == 'Peaks2MapsKernel':
            kern = Peaks2MapsKernel(dset.coordinates, 'template_img')
        imgs = kern.transform(dset.ids)
    imgs_arr = []
    for i in np.arange(0, len(imgs)):
        imgs_arr.append(np.ravel(imgs[i].get_data(), order='C'))
    labels = pd.DataFrame(index=dset.ids)
    k = np.arange(clust_range[0], (clust_range[1] + 1))
    for i in k:
        if algorithm == 'kmeans':
            clustering = KMeans(i, init='k-means++', precompute_distances='auto')
        if algorithm == 'spectral':
            clustering = SpectralClustering(
                n_clusters=i, eigen_solver=None, random_state=None, n_init=300,
                gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0,
                assign_labels='discretize', degree=3, coef0=1,
                kernel_params=None)
        if algorithm == 'dbscan':
            min = len(dset.ids) / (i - 1)
            clustering = DBSCAN(eps=0.1, min_samples=min, metric='euclidean',
                                metric_params=None, algorithm='auto',
                                leaf_size=30, p=None)
        labels[i] = clustering.fit_predict(imgs_arr)
    labels.to_csv('{0}/{1}_labels.csv'.format(output_dir, output_prefix))

    silhouette_scores = {}
    for i in k:
        j = i - 2
        silhouette = silhouette_score(imgs_arr, labels[i], metric='correlation', random_state=None)
        silhouette_scores[i] = silhouette
    silhouettes = pd.Series(silhouette_scores, name='Average Silhouette Scores')

    clusters_idx = {}
    for i in k:
        clusters = []
        for j in range(i):
            clusters.append(list(np.where(labels[i] == j)[0]))
        clusters_idx['Solution {0}'.format(i)] = clusters

    variation_of_infofmation = {}
    for i in k[:-1]:
        j = clusters_idx['Solution {0}'.format(i)]
        z = clusters_idx['Solution {0}'.format(i + 1)]
        var_info = VI(j, z)
        variation_of_infofmation[i + 1] = var_info

    vi = pd.Series(variation_of_infofmation, name='Variation of Information')

    metrics = pd.concat([vi, silhouettes], axis=1)
    metrics.to_csv('{0}/{1}_metrics.csv'.format(output_dir, output_prefix))

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    g = sns.lineplot(metrics.index, metrics['Average Silhouette Scores'], ax=ax[0])
    g.set_title('Silhouette Scores')
    g = sns.lineplot(metrics.index, metrics['Variation of Information'], ax=ax[1])
    g.set_title('Variation of Information')
    fig.savefig('{0}/{1}_metrics.png'.format(output_dir, output_prefix), dpi=300)
