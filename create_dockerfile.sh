docker run --rm kaczmarj/neurodocker:0.4.3 generate docker -b neurodebian:stretch-non-free -p apt \
    --install fsl gcc g++ software-properties-common \
    --user=neuro \
    --add-to-entrypoint "source /etc/fsl/fsl.sh" \
    --copy . /src/NiMARE/ \
    --miniconda create_env=nimare \
    miniconda_version=4.3.31 \
    conda_install="python=3.6 jupyter jupyterlab jupyter_contrib_nbextensions seaborn" \
    pip_install="/src/NiMARE/" \
    activate=true \
    --run 'mkdir -p ~/.jupyter && echo c.NotebookApp.ip = \"0.0.0.0\" > ~/.jupyter/jupyter_notebook_config.py' \
    --user=root \
    --run 'apt-get update && add-apt-repository -y ppa:openjdk-r/ppa && apt-get install -y --no-install-recommends openjdk-8-jdk openjdk-8-jre' \
    --run 'update-alternatives --config java && update-alternatives --config javac' \
    --run 'curl -o mallet-2.0.7.tar.gz http://mallet.cs.umass.edu/dist/mallet-2.0.7.tar.gz && tar xzf mallet-2.0.7.tar.gz && rm mallet-2.0.7.tar.gz && mkdir /home/neuro/resources && mv mallet-2.0.7 /home/neuro/resources/mallet' \
    --workdir /home/neuro > Dockerfile
