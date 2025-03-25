# IDE-Net
Source code and supplementary materials for the paper "Inferring Individual Direct Causal Effects Under Heterogeneous Peer Influence" published in Springer Nature Machine Learning Journal 2025 by Shishir Adhikari and Elena Zheleva .

Please use the following citation:

```
@article{adhikari-mlj2025,
  title={Inferring individual direct causal effects under heterogeneous peer influence},
  author={Adhikari, Shishir and Zheleva, Elena},
  journal={Machine Learning},
  volume={114},
  number={4},
  pages={113},
  year={2025},
  publisher={Springer}
}
```

Here is the brief overview of the code structure:
- `configs`: Configuration file for different experimental setup
- `data`: BlogCatalog and Flicker network data with node attributes that is used for semisynthetic data generation
- `models`: A simple network structural causal model (NSCM) for synthetic data generation
- `src`: Source code for experiments including proposed method, baselines, and data generation
- `environment.yml`: Anaconda environment for all dependencies used

### Running IDE-Net and baselines
> Note: IDE-Net is referred as `INE_TARNet` in the source code as it could be adapted to estimate any network effects although paper focuses on individual direct effects.

```
cd src
python run_gnn_tarnet_synthetic.py --config ../configs/<synthetic_experiment_config>.yaml --estimator INE_TARNet  --outfolder <folder_to_output_results>

python run_gnn_tarnet_semisynthetic.py --config ../configs/<semisynthetic_experiment_config>.yaml --net <BlogCatalog or Flickr> --exposure het --outfolder <folder_to_output_results>
```
Use `python <filename.py> --help` to view all options.

For example:
```
python run_gnn_tarnet_synthetic.py --help
usage: run_gnn_tarnet_synthetic.py [-h]
                                   [--estimator {INE_TARNet,INE_TARNet_ONLY,INE_TARNet_INT,INE_TARNet_MOTIFS,INE_CFR,INE_CFR_INT,INE_CFR_MOTIFS,GCN_TARNet,GCN_TARNet_INT,GCN_TARNet_MOTIFS,GCN_CFR,GCN_CFR_INT,GCN_CFR_MOTIFS}]
                                   [--config CONFIG] [--maxiter MAXITER] [--val VAL] [--lr LR]
                                   [--lrest LREST] [--lrstep LRSTEP] [--lrgamma LRGAMMA]
                                   [--weight_decay WEIGHT_DECAY] [--clip CLIP]
                                   [--max_patience MAX_PATIENCE] [--fdim FDIM] [--edim EDIM]
                                   [--dropout DROPOUT] [--inlayers INLAYERS] [--alpha ALPHA]
                                   [--normY {0,1}] [--verbose VERBOSE] [--isolated {0,1}] [--reg {0,1}]
                                   [--outfolder OUTFOLDER]

Run GNN-based Heterogeneous Network Effects Experiments

optional arguments:
  -h, --help            show this help message and exit
  --estimator {INE_TARNet,INE_TARNet_ONLY,INE_TARNet_INT,INE_TARNet_MOTIFS,INE_CFR,INE_CFR_INT,INE_CFR_MOTIFS,GCN_TARNet,GCN_TARNet_INT,GCN_TARNet_MOTIFS,GCN_CFR,GCN_CFR_INT,GCN_CFR_MOTIFS}
                        Estimator
  --config CONFIG       YAML file with config for experiments
  --maxiter MAXITER     Maximum epochs
  --val VAL             Fraction of nodes used for validation and early stopping
  --lr LR               Learning rate for encoder
  --lrest LREST         Learning rate for learner
  --lrstep LRSTEP       Change LR after N steps
  --lrgamma LRGAMMA     lr=gamma*lr after N steps
  --weight_decay WEIGHT_DECAY
                        Regularization
  --clip CLIP           Clip gradient
  --max_patience MAX_PATIENCE
                        Early stopping if loss not improved
  --fdim FDIM           Hidden layer (node embedding) dimension
  --edim EDIM           Hidden layer (edge embedding) dimension
  --dropout DROPOUT     Dropout
  --inlayers INLAYERS   MLP layers for feature/exposure mapping
  --alpha ALPHA         Alpha like CFR estimator
  --normY {0,1}         Normalize Y
  --verbose VERBOSE     Verbose for parameters tuning
  --isolated {0,1}      Evaluate isolated direct effects
  --reg {0,1}           Variance smoothing regularization
  --outfolder OUTFOLDER
                        Folder to output records
```
#### Source files for baselines
- BART: `IDE-Net/src/main_baseline_synthetic_bart.py`
- 1GNN-HSIC: `IDE-Net/src/main_baseline_synthetic_gnn_hsic.py` and `IDE-Net/src/main_baseline_semisynthetic_gnn_hsic.py`
- DWR: `IDE-Net/src/main_baseline_synthetic_dwr.py` and `IDE-Net/src/main_baseline_semisynthetic_dwr.py`
- Others: `IDE-Net/src/main_baseline.py` and `IDE-Net/src/main_baseline_semisynthetic.py`

> Note: This repository is shared to facilitate reproducibility of the work. Code clean up and optimization is ongoing and contributions are welcome.
