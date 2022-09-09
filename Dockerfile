# Use kubeflow:jovyan_base, which is the base docker-stack image built on nvhpc image
FROM ulissigroup/kubeflow:extras-notebook

# Add channels for pytorch geometric requirements, pytorch, etc
RUN conda config --add channels pytorch
RUN conda config --add channels pyg
RUN conda config --add channels nvidia

USER $NB_UID

# Install requirements needed to use OCP requirements
RUN mamba install --quiet --yes \
    'conda-merge' \
    'conda-build' && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Install dask/dask-kubernetes, and necessary pytorch_geometric requirements along with updated CUDA
RUN mamba install --quiet --yes \
    'pytorch=1.10' \
    'pyg' && \
    mamba clean --all -f -y && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

RUN mamba install --quiet --yes \
    'schema' \
    'dask-kubernetes' \
    'ipywidgets' \
    'joblib' \
    'numba' \
    'pynvml' \
    'dask' \
    'dask-cuda' \
    'jupyter-server-proxy' \
    'networkx' \
    'future' \
    'fireworks' \
    'sqlalchemy' \
    'jsonpickle' \
    'spglib' \
    'scikit-learn' \
    'cerberus' \
    'scikit-image' \
    'matplotlib' \
    'python-kaleido' \
    'monty' \
    'maggma' \
    'requests' \
    'typing-extensions' \
    'pydantic' \
    'numba' \
    'sphinx' \
    'graph-tool' \
    'pandoc' \
    'black' \
    'pyyaml' \
    'pre-commit' \
    'tensorboard' \
    'jupyter-book' \
    'tqdm' \
    'wandb' \
    'pillow' \
    'pytest' \
    'lmdb' \
    'python-lmdb' \
    'submitit' \
    'pympler' \
    'backoff' \
    'pymatgen>=2022.0.17' \
    'ase>=3.22.1' && \
    mamba clean --all -f -y && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# Needed to get dimenet++ working since pyg 2.0.4 broke it
#RUN pip install --no-deps git+https://github.com/pyg-team/pytorch_geometric.git@a7e6be4
RUN pip install --no-deps git+https://github.com/superstar54/x3dase.git
RUN pip install --no-deps git+https://github.com/ulissigroup/CatKit.git
RUN pip install --no-deps git+https://github.com/brookwander/Open-Catalyst-Dataset.git
RUN pip install --no-deps git+https://github.com/lab-cosmo/chemiscope.git
RUN pip install mp-api 


# Add OCP
WORKDIR /home/jovyan
RUN git clone https://github.com/Open-Catalyst-Project/ocp.git && \
    python ocp/setup.py develop
CMD ["sh","-c", "jupyter notebook --notebook-dir=/home/jovyan --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.base_url=${NB_PREFIX}"]
ENV PYTHONPATH=/home/jovyan/ocp/:/home/jovyan/catlas/
