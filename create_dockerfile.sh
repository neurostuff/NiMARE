docker run --rm kaczmarj/neurodocker:v0.3.2 generate -b neurodebian:stretch-non-free -p apt \
    --install fsl \
    --user=neuro \
    --add-to-entrypoint "source /etc/fsl/fsl.sh" \
    --miniconda env_name=nimare \
    miniconda_version=4.3.31 \
    conda_install="python=3.6 jupyter jupyterlab jupyter_contrib_nbextensions
                   matplotlib scikit-learn seaborn numpy scipy pandas
                   statsmodels nibabel nipype" \
    pip_install="nilearn pybids duecredit pymc3" \
    activate=true \
    --run 'mkdir -p ~/.jupyter && echo c.NotebookApp.ip = \"0.0.0.0\" > ~/.jupyter/jupyter_notebook_config.py' \
    --workdir /home/neuro --no-check-urls > Dockerfile
