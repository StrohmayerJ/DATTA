# DATTA: Domain-Adversarial Test-Time Adaptation for Cross-Domain WiFi-Based Human Activity Recognition

### Paper
**Strohmayer, J., Sterzinger, R., Wödlinger, M., and Kampel, M.** (2024). DATTA: Domain-Adversarial Test-Time Adaptation for Cross-Domain WiFi-Based Human Activity Recognition. arXiv. doi: https://doi.org/10.48550/arXiv.2411.13284.

BibTeX:
```BibTeX
@misc{strohmayer2024datta,
      title={DATTA: Domain-Adversarial Test-Time Adaptation for Cross-Domain WiFi-Based Human Activity Recognition}, 
      author={Julian Strohmayer and Rafael Sterzinger and Matthias Wödlinger and Martin Kampel},
      year={2024},
      eprint={2411.13284},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.13284}, 
      }
```

### Prerequisites
```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Widar3.0-G6D Dataset

Download the *Widar3.0* dataset from: https://tns.thss.tsinghua.edu.cn/widar3.0/. It is recommended to use an FTP client (such as FileZilla): 166.111.80.127:40121 (username: widarftp, password: widar2019) 

To create the *Widar3.0-G6D* dataset as described in our paper, only the following four files are required:

```
/CSI/CSI_20181130.zip
/CSI/CSI_20181211.zip
/CSI/CSI_20181209.zip
/CSI/CSI_20181204.zip
```

Download and unzip them in the `/data/widar3g6d` directory (the original .zip file can be deleted). Then run the following command from the directory root (~16GB RAM required): 

```
python3 utils/createWidar3g6d.py --mode TRAIN; python3 utils/createWidar3g6d.py --mode TEST
```

The training and testing CSI cache files (`widar3-g6_csi_domain_train_cache.pkl` and `widar3-g6_csi_domain_test_cache.pkl`) should now be present in the `/data/widar3g6d` directory. 


### Domain-Adversarial Training (DAT) 

**Training** | Example command for DAT of three [*WiFlexFormer*](https://github.com/StrohmayerJ/WiFlexFormer) models (*wdat_1*, *wdat_2*, and *wdat_3*) on the *Widar3.0-G6D* training subset, using our optimal DAT hyperparameters (default argument values):

```
python3 trainDAT.py --name wdat --num 3 --device 0
```
Model checkpoints for the lowest validation loss and highest validation F1-Score are stored in the corresponding run directories `runs/wdat_*`.

**Testing** | Example command for testing the trained models *wdat_\**:

```
python3 testDAT.py --name wdat --device 0
```

### Test-Time Adaptation (TTA) 
Example command for TTA (and testing) of the trained models *wdat_\** on the *Widar3.0-G6D* test subset, using our optimal TTA hyperparameters (default argument values):

```
python3 DATTA.py --name wdat --device 0
```


### Logging
To use wandb logging, set the credentials in *trainDAT.py* and *DATTA.py*, and pass the `--log` flag.

```
# enable/disable wandb
if opt.log:
      wandb.init(project=f"DATTA",entity="XXXX",name=opt.exp_name)
else:
      wandb.init(mode="disabled")
```
