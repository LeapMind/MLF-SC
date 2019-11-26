# MLF-SC: Incorporating Multi-Layer Features to Sparse Coding for Unsupervised Anomaly Detection

MLF-SC (Multi-Layer Feature Sparse Coding) is an anomaly detection method that incorporates multipscale features to sparse coding.
This is a PyTorch implementation for [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad/) texture datasets (Carpet, Grid, Leather, Tile, Wood).

## Install Python Requirements
```
pip3 install -r requirements.txt
```

## Download Dataset
```
wget ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz
tar -xvf mvtec_anomaly_detection.tar.xz 
```

## Usage
You can train and test for each texture dataset.

### Set dataset path to `config.yml`
Set `root` in `config.yml` to texture dataset path (like `path/to/mvtec_anomaly_detection/carpet/`).  
If you set `Grid` dataset , set `gray2rgb` in `config.yml` to `True`. 

### Train
```
$ python3 main.py tran config.yaml
```

### Test
```
$ python3 main.py test config.yaml
```

## Contribution

Table. AUROC of each method in anomaly detection experiment using 5 texture category of MVTec data set.

| Category | AE(SSIM) | AE(L2) | AnoGAN | CNN <br>Feature Dictionary | Texture<br>Inspection | Sparse<br>Coding | MLF-SC<br>(Proposed) |
|:--------:|:--------:|:------:|:------:|:--------------------------:|:---------------------:|:----------------:|:--------------------:|
|  Carpet  |   0.87   |  0.59  |  0.54  |            0.72            |          0.88         |       0.58       |       **0.99**       |
|   Grid   |   0.94   |  0.90  |  0.58  |            0.59            |          0.72         |       0.89       |       **0.97**       |
|  Leather |   0.78   |  0.75  |  0.64  |            0.87            |          0.97         |       0.95       |       **0.99**       |
|   Tile   |   0.59   |  0.51  |  0.50  |          **0.93**          |          0.41         |       0.86       |         0.92         |
|   Wood   |   0.73   |  0.73  |  0.62  |            0.91            |          0.78         |       0.97       |       **0.99**       |
|    Avg.   |   0.78   |  0.70  |  0.58  |            0.80            |          0.75         |       0.85       |       **0.97**       |



## License
Non-commercial, research purposes only

## License of dependent libraries
See `LICENSE` directory.
