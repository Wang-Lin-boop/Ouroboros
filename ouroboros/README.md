
Installation
====

```
conda activate -n Ouroboros python=3.9

pip install six scipy tqdm dill pyarrow matplotlib pandas dask
pip intstall oddt scikit-learn rdkit fair-esm umap-learn selfies 

pip install graphein==1.7.7

pip install dgl==2.1.0 -f https://data.dgl.ai/wheels/cu121/repo.html
pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
pip install dgllife==0.3.2

pip3 install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
--index-url https://download.pytorch.org/whl/cu121

pip install torchtriton --extra-index-url "https://download.pytorch.org/whl/nightly/cu121"

```



