<h1 align="center">  Ouroboros  </h1>
<h3 align="center"> Directed Chemical Evolution via Navigating Molecular Encoding Space </h3>
<p align="center">
  📃 <a href="https://onlinelibrary.wiley.com/doi/10.1002/advs.202403998" target="_blank">Paper</a> ·  🤗 <a href="https://huggingface.co/AlphaMWang/Ouroboros" target="_blank">Model</a> ·  📕 <a href="https://zenodo.org/records/10450788" target="_blank">Data</a><br>
</p>

<p align="center">
  <img style="float: center" src="imgs/ouroboros.jpg" alt="alt text" width="650px" align="center"/>
</p>

This repository provides the official implementation of the Ouroboros model and utilities. Ouroboros aims to bridge the gap between representation learning and generative AI models and **facilitate chemical space navigation for directed chemical evolution**. Ouroboros first employs representation learning to encode molecular graphs into **1D vectors**, which are then **decoded independently** into molecular property prediction and molecular structure production. Please also refer to our [paper]() for a detailed description of Ouroboros.     

## 💗 Motivation  

Artificial neural networks have rapidly advanced representation learning and generative modeling in molecular pharmaceutics. However, a gap persists between the representation learning and molecular generation, hindering the application of AI and deep-learning techniques in drug discovery. We introduce Ouroboros, a unified framework that seamlessly integrates representation learning with molecular generation and therefore **allows efficient chemical space exploration through pre-trained molecular encodings**. By reframing the directed chemical evolution as a process of encoding space compression and decompression, the strategy overcomes the challenges associated with iterative molecular optimization, enabling optimal molecular optimization directly within the encoding space.    

## 💡 Highlight

**Overview:**

* Ouroboros introduces a new protocol to **unify the representative learning and molecular generation**.      
* Ouroboros towards directed chemical evolution in **encoding space**, which was designed to mimic the chemical space.     
* Ouroboros provides a molecular representation that can be flexibly encoded and decoded, **allowing new models built upon it to be easily applied to molecular generation**.     

**Chemical Space Modeling:**

* Ouroboros projects molecules of different sizes and structures into a **1D Vector** and ensures that the similarities of this **1D Vector** are pharmacophore meaningful.     
* Ouroboros' **1D Vector** can be reconstructed to the original chemical structure.     

**Chemical Foundational Model:**

* Ouroboros not only provides **molecule-to-encoding** transformation, but also **encoding-to-molecule** reconstruction.     
* You can practice **directed chemical evolution on any downstream model built on Ouroboros**.     

**Prospective:**

* The Ouroboros framework can **accommodate other molecular representation learning strategies (chemicals vs multi-omics, phenotype)** to hold more information in the encoding space.       
* Predictive models of protein-ligand recognition, durg-target affinity or other chemical propteries trained using Ouroboros will be promising for target-based molecule generation.      

## 😫 Limitations

* Ouroboro is not a model trained on a large-scale dataset, and you can expect **scaling up** to improve its performance.       
* Reconstruction of the 1D Vector to a chemical structure is challenging and **100% reconstruction** has not yet been achieved.        

## 📕 Installation

To set up the Ouroboros model, we recommend using conda for Python environment configuration.   
If you encounter any problems with the installation, please feel free to post an issue or discussion it.    

<details>
<summary>Environment Setup</summary>
<br>

> Installing MiniConda (skip if conda was installed)   

``` shell
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    sh Miniconda3-latest-Linux-x86_64.sh
```

> Creating Ouroboros environment   

``` shell
    conda create -n Ouroboros python=3.9
    conda activate Ouroboros
```

> Setting up Ouroboros PATH and configuration   
 
``` shell
    git clone https://github.com/Wang-Lin-boop/Ouroboros
    cd Ouroboros/
    echo "# Ouroboros" >> ~/.bashrc
    echo "export PATH=\"${PWD}:\${PATH}\"" >> ~/.bashrc # optional, not required in the current version
    echo "export Ouroboros=\"${PWD}\"" >> ~/.bashrc
    source ~/.bashrc
    echo "export ouroboros_app=\"${Ouroboros}/ouroboros\"" >> ~/.bashrc # Ouroboros applications     
    echo "export ouroboros_lib=\"${Ouroboros}/models\"" >> ~/.bashrc # Ouroboros models 
    echo "export ouroboros_dataset=\"${Ouroboros}/datasets\"" >> ~/.bashrc # Ouroboros datasets 
    source ~/.bashrc
```

Before running Ouroboros, you need to install dependency packages.   

```
conda activate -n Ouroboros python=3.9

pip install pandas six scipy tqdm dill pyarrow matplotlib
pip install oddt scikit-learn rdkit umap-learn selfies

pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
pip install dglgo -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
pip install dgllife

pip install torchtriton --extra-index-url "https://download.pytorch.org/whl/nightly/cu121"
```

<br>
</details>   

<details>
<summary>Download datasets and Ouroboros models</summary>
<br>

In this repository, we provide the pre-trained Ouroboros models and useful chemical datasets, you can download them via [ZhangLab WebPage](https://zhanglab.comp.nus.edu.sg/Ouroboros/). Then, we need place the models to the `${ouroboros_lib}`, and place the chemical datasets to `${ouroboros_dataset}`.       

We provide three different versions of the model, all of them trained based on the strategy reported in the paper, with the difference that:    
```
1. M0 was trained and tested strictly according to the methodology section of our paper;      
2. M1c and M1d: training datasets used for their molecular decoders consisting more complex sources and with SMILES (c) and SELFIES (d) as the chemical language.      
```
In this GitHub repository, we update the latest version of the code and models for Ouroboros (they usually have better performance). If your goal is only to reproduce the results in the article, please use the original model and source code provided on [ZhangLab WebPage](https://zhanglab.comp.nus.edu.sg/Ouroboros/), or use the 0.1.0 release of the repository. If the current version does not meet the demands of your drug discovery program, feel free to contact [us](Wanglin1102@outlook.com) to try our in-house version.         

<br>
</details>   

## ⭐ Citing This Work

coming soon.... 

## ✅ License

Ouroboros is released under the **Apache** Licence, which permits non-profit use, modification and distribution free of charge.     

## 💌 Get in Touch

We welcome community contributions of extension tools based on the Ouroboros model, etc. If you have any questions not covered in this overview, please contact the Ouroboros Developer Team via Wanglin1102@outlook.com. We would like to hear your feedback and understand how Ouroboros has been useful in your research. Share your stories with [us](Wanglin1102@outlook.com).    

You are welcome to contact our development team via Wanglin1102@outlook.com in the event that you would like to use our in-house version under development for optimal performance. Our model aims to provide thoughts on the design of novel chemotypes for medicinal chemists.    

## 😃 Acknowledgements

Ouroboros communicates with and/or references the following separate libraries and packages, we thank all their contributors and maintainers!    

*  [_RDKit_](https://www.rdkit.org/)
*  [_PyTorch_](https://pytorch.org/)
*  [_DGL-Life_](https://lifesci.dgl.ai/)
*  [_ODDT_](https://oddt.readthedocs.io/en/latest/)
*  [_SciPy_](https://scipy.org/)
*  [_scikit-learn_](https://scikit-learn.org/stable/)
*  [_matplotlib_](https://matplotlib.org/)
*  [_SELFIES_](https://github.com/aspuru-guzik-group/selfies)

