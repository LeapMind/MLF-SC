# MLF-SC: Incorporating Multi-Layer Features to Sparse Coding for Unsupervised Anomaly Detection

[![Actions Status](https://github.com/LeapMind/MLF-SC/workflows/MLF-SC/badge.svg)](https://github.com/LeapMind/MLF-SC/actions)

MLF-SC (Multi-Layer Feature Sparse Coding) is an anomaly detection method that incorporates multi-scale features to sparse coding.
This is a PyTorch implementation for [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad/) texture datasets (Carpet, Grid, Leather, Tile, Wood).

## Output visualization of MVTec texture dataset (Carpet)

![000-900](https://user-images.githubusercontent.com/46925310/79444545-4399a500-8016-11ea-9ef0-14f72f2c47ee.png)
![000-774](https://user-images.githubusercontent.com/46925310/79444575-4bf1e000-8016-11ea-803a-410cc9623621.png)
![001-768](https://user-images.githubusercontent.com/46925310/79444703-7cd21500-8016-11ea-9250-b51a7f6fae09.png)

## Quick Start
```
git clone git@github.com:LeapMind/MLF-SC.git
pip3 install -r requirements.txt
python3 main.py train sample_config.yml
python3 main.py test sample_config.yml
python3 main.py visualize sample_comfig.yml
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
$ python3 main.py train config.yml
```

### Test
```
$ python3 main.py test config.yml
```

### Visualize
```
$ python3 main.py visualize config.yml
```

## Contribution

Anomaly detection performance for the texture categories of the MVTec AD dataset. For each cell in the R1 / R2 columns, the ratio of correctly classified samples of normal R1 and that of anomalous images R2 are shown with R1 / R2 notation.  The maximum averages (R1 + R2) / 2 are marked with boldface. The performance for the non-sparse-coding-based methods are cited from Table 2 of (Bergmann et al., 2019). The AUROC columns show only sparse coding and MLF-SC.

|   | R1 / R2 |  |  |  |  |  |  | AUROC |  |
| :---: | :---: | --- | --- | --- | --- | --- | --- | :---: | --- |
|  Category | AE (SSIM) | AE (L2) | AnoGAN | CNN<br/>Feature Dictionary | Texture<br/>Inspection | Sparse<br/>Coding | MLF-SC<br/>(Proposed) | Sparse<br/>Coding | MLF-SC<br/>(Proposed) |
|  Carpet | 0.43 / 0.90 | 0.57 / 0.42 | 0.82 / 0.16 | 0.89 / 0.36 | 0.57 / 0.61 | 0.43 / 0.79 | **1.00 / 0.98** | 0.58 | **0.99** |
|  Grid | 0.38 / 1.00 | 0.57 / 0.98 | 0.90 / 0.12 | 0.57 / 0.33 | 1.00 / 0.05 | 0.76 / 0.72 | **1.00 / 0.88** | 0.89 | **0.97** |
|  Leather | 0.00 / 0.92 | 0.06 / 0.82 | 0.91 / 0.12 | 0.63 / 0.71 | 0.00 / 0.99 | 0.84 / 0.96 | **0.97 / 0.97** | 0.95 | **0.99** |
|  Tile | 1.00 / 0.04 | 1.00 / 0.54 | 0.97 / 0.05 | 0.97 / 0.44 | 1.00 / 0.43 | 0.94 / 0.60 | **0.94 / 0.76** | 0.86 | **0.92** |
|  Wood | 0.84 / 0.82 | 1.00 / 0.47 | 0.89 / 0.47 | 0.79 / 0.88 | 0.42 / 1.00 | 0.84 / 0.60 | **0.95 / 0.98** | 0.97 | **0.99** |
|  Average | 0.53 / 0.74 | 0.64 / 0.65 | 0.90 / 0.18 | 0.77 / 0.54 | 0.60 / 0.62 | 0.７６ / 0.81 | **0.97 / 0.91** | 0.85 | **0.97** |

## License
Non-commercial, research purposes only

## License of dependent libraries
See `LICENSE` directory.
