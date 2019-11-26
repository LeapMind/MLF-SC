# MLF-SC: Incorporating Multi-Layer Features to Sparse Coding for Unsupervised Anomaly Detection

MLF-SC (Multi-Layer Feature Sparse Coding) is an anomaly detection method that incorporates multipscale features to sparse coding.
This is a PyTorch implementation for [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad/) texture datas (Carpet, Grid, Leather, Tile, Wood).

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

## License
Non-commercial, research purposes only

## License of dependent libraries
See `LICENSE` directory.
