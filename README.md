## Preparation
- Install requirements via [environment.yml](environment.yml) or [requirements.txt](requirements.txt)
- Download benchmark datasets from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) to ```./PatchTST/PatchTST_supervised/dataset```


## Create time lag data example
- [Jupyter notebook for creating the time lag data example](thesis_create_data/shower.ipynb)


## Scripts for running the experiments
- [Folder with scripts for Linear](PatchTST/PatchTST_supervised/scripts/Linear_SK)
- [Folder with scripts for ModernTCN](PatchTST/PatchTST_supervised/scripts/ModernTCN_SK)
- [Folder with sripts for PatchTST](PatchTST/PatchTST_supervised/scripts/PatchTST_SK)


## Visualization of results
- [Jupyter notebook for the visualization of predictions](visualization/visualize_predictions_final.ipynb)
- [Jupyter notebook for the visualization of results](visualization/visualize_results_final.ipynb)
- The respective illustrations and error tables end up in ```./visualization/plots/Thesis```


## Model Files
- [Linear](PatchTST/PatchTST_supervised/models/_Linear_final.py)
- [ModernTCN](PatchTST/PatchTST_supervised/models/_ModernTCN.py)
- [PatchTST](PatchTST/PatchTST_supervised/models/PatchTST.py)


## Acknowledgement
This repository is based on the official implementation of [PatchTST](https://github.com/yuqinie98/PatchTST) and adopts code from the official implementation of [ModernTCN](https://github.com/luodhhh/ModernTCN)
