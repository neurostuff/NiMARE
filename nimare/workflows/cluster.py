from ..meta.cbma import kernel
from ..due import due, Doi

from nimare.dataset.extract import convert_sleuth_to_database
from nimare.meta.cbma.ale import ALE
from nimare.due import due, Doi
from sklearn import cluster
import pandas as pd
#from ..dataset.extract import convert_sleuth_to_database
#from ..meta.cbma.ale import ALE
#from ..due import due, Doi
import click

@click.command(name=metacluster)
@click.argument('database', required=True, type=click.Path(exists=True, readable=True), help='NiMARE database or Sleuth text file containing meta-analytic data to be clustered')
@click.argument('output_dir', required=True, type=click.Path(), help='Directory into which clustering results will be written.')
@click.argument('output_prefix', default='metacluster', type=string, help='Basename for written out clustering results.')
@click.argument('kernel', default='ALEKernel' type=click.Choice(['ALEKernel', 'MKDAKernel', 'KDAKernel', 'Peaks2MapsKernel']), help='Kernel estimator, for coordinate-based metaclustering.')
@click.option('--img/--coord', '-i/-c', required=True, default=False, help='Is input data image- or coordinate-based?')
@click.option('--algorithm', '-a', default='kmeans', type=click.Choice(['kmeans', 'dbscan', 'spectral']), help='Clustering algorithm to be used, from sklearn.cluster.')
@click.option('--clust_range', n_args=2, type=float, help='Select a range for k over which clustering solutions will be evaluated (e.g., 2 10 will evaluate solutions with k = 2 clusters to k = 10 clusters).')


@due.dcite(Doi('10.1016/j.neuroimage.2015.06.044'),
           description='Introduces meta-analytic clustering analysis; hierarchically clusering face paradigms.')
@due.dcite(Doi('10.1162/netn_a_00050'),
           description='Performs the specific meta-analytic clustering approach included here.')

def meta_cluster_workflow(database, output_dir, output_prefix, kernel, img, algorithm, clust_range):
    def VI(X, Y):
        from math import log
        #from https://gist.github.com/jwcarr/626cbc80e0006b526688
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
    if database.endswith('.json'):
        db = x ##how do I read in a generic database file? do I need options for source type?
        dset = db.get_dataset(ids, target='mni152_2mm')
    elif database.endswith('.txt'):
        db = convert_sleuth_to_database(database)
        dset = db.get_dataset(target='mni152_2mm')
    elif:
        raise click.BadParameter('You\'ve provided a database that metacluster can\'t read. :(', param_hint='database')
    if img:
        imgs = dset.images
    if coord:
        if kernel == 'ALEKernel':
            kern = kernel.ALEKernel(dset.coordinates, template_img)
        elif kernel == 'MKDAKernel':
            kern = kernel.MKDAKernel(dset.coordinates, r=6, value=1, template_img)
        elif kernel == 'KDAKernel':
            kern = kernel.KDAKernel(dset.coordinates, r=6, value=1, template_img)
        elif kernel == 'Peaks2MapsKernel':
            kern = kernel.Peaks2MapsKerneltransform(dset.coordinates, template_img)
        imgs = kern.transform(dset.ids)
    for i in np.arange(0,len(imgs)):
        imgs_arr.append(np.ravel(imgs[i].get_data(), order='C'))
    labels = pd.DataFrame(index=dset.ids)
    k = np.arange(n_clusters[0], (n_clusters[1] + 1)
    for i in k:
        if algorithm == 'kmeans':
            clustering = KMeans(i, sample_weight=None, init='k-means++', precompute_distances='auto', n_init=300, max_iter=1000, verbose=False, tol=0.0001, random_state=None, copy_x=True, n_jobs=2, algorithm='auto', return_n_iter=False)
        if algorithm == 'spectral':
            clustering = SpectralClustering(n_clusters=i, eigen_solver=None, random_state=None, n_init=300, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='discretize', degree=3, coef0=1, kernel_params=None, n_jobs=2)
        if algorithm == 'dbscan':
            min = len(dset.ids)/(i-1)
            clustering = DBSCAN(eps=0.1, min_samples=min, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)
        labels[i] = clustering.fit_predict(imgs_arr)
    labels.to_csv('{0}/{1}_labels.csv'.format(output_dir, output_prefix))

    silhouette_scores = {}
    for i in k:
        j = i-2
        silhouette = silhouette_score(imgs_arr, labels[i].values, metric='correlation', random_state=None)
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
        z = clusters_idx['Solution {0}'.format(i+1)]
        var_info = VI(j, z)
        variation_of_infofmation[i+1] = var_info

    vi = pd.Series(vi_k, name='Variation of Information')

    metrics = pd.concat([kmeans_vi, silhouettes], axis=1)
    metrics.to_csv('{0}/{1}_metrics.csv'.format(output_dir, output_prefix))

    fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    g = sns.lineplot(metrics.index, metrics['Silhouette Scores'], ax=ax[0])
    g.set_title('Silhouette Scores')
    g = sns.lineplot(metrics.index, metrics['Variation of Information'], ax=ax[1])
    g.set_title('Variation of Information')
    fig.savefig('{0}/{1}_metrics.png'.format(output_dir, output_prefix), dpi=300)

if __name__ == '__main__':
    meta_cluster_workflow()
