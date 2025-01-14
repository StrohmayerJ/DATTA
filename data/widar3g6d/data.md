Download the Widar3.0 dataset from: https://tns.thss.tsinghua.edu.cn/widar3.0/. 
It is recommended to use an FTP client (such as FileZilla): 166.111.80.127:40121 (username: widarftp, password: widar2019) 

To create the *Widar3.0-G6D* dataset as described in our paper, only the following four files are required:

```
/CSI/CSI_20181130.zip
/CSI/CSI_20181211.zip
/CSI/CSI_20181209.zip
/CSI/CSI_20181204.zip
```

Download and unzip them in this directory (the original .zip file can be deleted). Then run the following command from the directory root: 

Then run the following command (~16GB RAM required):

```
python3 utils/createWidar3g6d.py --mode TRAIN; python3 utils/createWidar3g6d.py --mode TEST
```

The training and testing cache files (`widar3-g6_csi_domain_train_cache.pkl` and `widar3-g6_csi_domain_test_cache.pkl`) should now be present in this directory. 
