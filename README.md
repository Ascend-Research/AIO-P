# Expanding Neural Performance Predictors Beyond Image Classification

Repository for the paper
> AIO-P: Expanding Neural Performance Predictors Beyond Image Classification \
> Keith G. Mills, Di Niu, Mohammad Salameh, Weichen Qiu, Fred X. Han, Puyuan Liu, Jialin Zhang, Wei Lu and Shangling Jui \
> AAAI-23

## Setup
### Dependencies
- Machine with an NVIDIA GPU and CUDA (>=10.2) support
- Python 3.7
- System: Ubuntu 20.04.4 LTS
- Conda is installed

### Installing packages
First create a conda environment
```bash
conda create -n aiop python=3.7
conda activate aiop
```
Install conda packages
```
$ conda install -c anaconda tensorflow-gpu=1.15.0 cudatoolkit
```

Install pip packages (can use conda instead, but this worked for us)
```
$ pip install --trusted-host pytorch.org --trusted-host download.pytorch.org  torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install --trusted-host pytorch-geometric.com torch-scatter==2.0.8 torch-sparse==0.6.11 torch-cluster==1.5.9 torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.8.1.html
$ pip install torch_geometric==1.7.2 opencv-python thop
$ pip install git+https://github.com/facebookresearch/detectron2.git git+https://github.com/cocodataset/panopticapi.git
```
These commands worked for us, but your mileage may vary. We note that for Detectron2 to work properly, torch and torchvision should be compiled with a version of CUDA which is reflected in their respective `__version__` fields.

### Downloading Datasets
- Download Computational Graph (CG) caches:
    - Download `cache.zip` from the [shared google drive](https://drive.google.com/drive/folders/1CxHYLAH8AsFz138325xWo1aIlfB-KPx4) and place all `.pkl` files in `/cache/`.
      - Caches with `ind` in their name are individually-trained architectures. `shared` refers to architectures fine-tuned by a shared head.
      - `deeplab' or `slim' caches refer to model zoos.
      - Caches without either are classifiction CGs.
    - Download `sample_pbs.zip` from the [shared google drive](https://drive.google.com/drive/folders/1CxHYLAH8AsFz138325xWo1aIlfB-KPx4) for examples of ResNet-18 and EfficientNet-B0.
- Download the following datasets:
    - [LSP](http://sam.johnson.io/research/lsp.html)
        - Unzip contents to `data/HPE/lsp`
    - [LSP Extended](http://sam.johnson.io/research/lspet.html)
        - Unzip contents to `data/HPE/lsp_extended`
    - [MPII](http://human-pose.mpi-inf.mpg.de/#download)
        - Unzip and place the annot and images folders under `data/HPE/mpii` 
    - [COCO](https://cocodataset.org/#download)
        - Set up the detectron2 COCO dataset: https://github.com/facebookresearch/detectron2/tree/main/datasets

### Setting up the repository
1. Download and unpack CG_data.zip
2. Place `.pkl` files in `cache` folder
3. Set the environment variable `DETECTRON2_DATASETS` to the directory containing the coco datasets:
    ```
    export DETECTRON2_DATASETS=/home/...
    ```

## Experiments
### Training individual architectures
> The `run_train_cgs_on_task.py` will train individual architectures to be used as test architectures that represent the ground truth.\
> Inside the *cache* folder, this script will output a subfolder with a `.txt` file that contains the logs and a `.pkl` file with the trained architectures 
#### Example for Detectron2
```
python run_train_cgs_on_task.py -family ofa_mbv3 -task detectron2 -tag individual -start_idx 0 -num_archs 10 -skip --num-gpus 2 --config-file tasks/detectron2/COCO_PanSeg_FPN_Adapted_Head.yml
```
- `-family` is the OFA family to train, select from one of `ofa_pn`, `ofa_mbv3`, and `ofa_resnet`.
- `-task detectron2` will execute the detectron2 code
- `-tag` can be any string that labels the output folder
- `-start_idx` is the start index of the architectures to train
- `-num_archs` is the number of architectures to train
- `-skip` uses skip connections
- `--num-gpus 2` uses 2 GPUs. We use Tesla V100 GPUs with 32GB of VRAM each, so depending on your computer resources, you may need to increase number of GPUs to avoid CUDA Out of memory errors
- `--config-file` is the path to the detectron2 config file

#### For HPE
```
python run_train_cgs_on_task.py -family ofa_mbv3 -task hpe2d -tag individual -start_idx 0 -num_archs 10 --num_epochs 140 --data_dir data/HPE
```
- See the `tasks/pose_hg_3d/lib/opts.py` file for the full list of flags on HPE
- `-family` is the OFA family to train, select from one of `ofa_pn`, `ofa_mbv3`, and `ofa_resnet`.
- `-task hpe2d` will execute the hpe2d code
- `-tag` can be any string that labels the output folder
- `-start_idx` is the start index of the architectures to train
- `-num_archs` is the number of architectures to train
- `--num_epochs` is the number of epochs to train for
- `--data_dir` is the folder path of the data directory that contains the HPE data and it should contain `lsp`, `lsp_extended` and `mpii` subfolders



### Generating HPE latent representation caches
> These caches are needed for the HPE training shared head experiments\
> The caches are passed in at the `--cache_file` flag for `run_train_head_on_task.py`


#### For HPE
```
python tasks/pose_hg_3d/lsp_dataloader.py --family mbv3 --data_dir data/HPE/
```
You might need run `export PYTHONPATH=$PYTHONPATH:/path/to/this/directory/`
- `--family` is the OFA family to train, select from one of `pn`, `mbv3`, and `resnet`.
- `--data_dir` is the folder path of the data directory that contains the HPE data and it should contain both `lsp` and `lsp_extended` subfolders

### Training the shared head weights
> `run_train_head_on_task.py` will train the shared head.\
> The script will produce a `.pkl` file that contains the shared head weights.
#### For Detectron2
We do not generate caches as COCO is too big, but sample latent representations on the fly.
```
python run_train_head_on_task.py -family ofa_mbv3 -task detectron2 -tag sampled -skip --num-gpus 1 --config-file tasks/detectron2/COCO_PanSeg_FPN_Adapted_Head.yml -sample_n 3 SOLVER.MAX_ITER 250000 SOLVER.STEPS 166000,222000 SOLVER.IMS_PER_BATCH 8
```
- `-family` is the OFA family to train, select from one of `ofa_pn`, `ofa_mbv3`, and `ofa_resnet`.
- `-task detectron2` will execute the detectron2 code
- `-tag` can be any string that labels the output folder
- `-skip` uses skip connections
- `-sample_n` is the number of architectures per bin
- `--num-gpus 1` uses 1 GPU
- `--config-file` is the path to the detectron2 config file
- `SOLVER.MAX_ITER` is the maximum number of iterations
- `SOLVER.STEPS` are the steps at which the learning rate will be decreased
- `SOLVER.IMS_PER_BATCH` is the number of architectures per batch
 
 **Hyperparameters**
| OFA family | SOLVER.MAX_ITER | SOLVER.STEPS  | SOLVER.IMS_PER_BATCH |
| ---------- | --------------- | ------------- | -------------------- |
| PN         | 250000          | 166000,222000 | 8                    |
| MBv3       | 250000          | 166000,222000 | 8                    |
| ResNet     | 250000          | 166000,222000 | 5                    |

#### For HPE
```
python run_train_head_on_task.py -family ofa_mbv3 --family mbv3 -task hpe2d -tag sampled --dataset lsp_cache --num_epochs 5000 --batch_size 256 -swap 10 --lr_cosine --cache_file cache/ofa_mbv3_cache_dict_n5 --data_dir data/HPE
```
- `-family` is the OFA family to train, select from one of `ofa_pn`, `ofa_mbv3`, and `ofa_resnet`
- `--family` is the OFA family to train, select from one of `pn`, `mbv3`, and `resnet`, it should be the same as `-family` except without the `ofa_` prefix
- `-task hpe2d` will execute the hpe2d code
- `-tag` can be any string that labels the output folder
- `--dataset lsp_cache` indicates that it should use the LSP dataset with a cache file
- `--num_epochs` is the number of epochs to train for
- `--batch_size` is the number of architectures per batch
- `--lr_cosine` uses a cosine learning rate
- `--cache_file` is the prefix of the directory containing the latent representation caches
- `--data_dir` is the folder path of the data directory that contains the HPE data and it should contain both `lsp` and `mpii` subfolders


### Training shared head architectures
#### For Detectron2
```
python run_train_cgs_on_task.py -family ofa_mbv3 -task detectron2 -tag shared -start_idx 10 -num_archs 15 -skip --num-gpus 2 --config-file tasks/detectron2/COCO_PanSeg_FPN_Adapted_Head.yml -chkpt cache/ofa_mbv3_detectron2_sampled/head_weights.pkl SOLVER.MAX_ITER 750 SOLVER.STEPS 465,635
```
- `-family` is the OFA family to train, select from one of `ofa_pn`, `ofa_mbv3`, and `ofa_resnet`.
- `-task detectron2` will execute the detectron2 code
- `-tag` can be any string that labels the output folder
- `-start_idx` is the start index of the architectures to train
- `-num_archs` is the number of architectures to train
- `-skip` uses skip connections
- `-chkpt` is the file path of the shared head weights
- `--num-gpus 2` uses 2 GPUs. Depending on your computer resources, you may need to increase number of GPUs to avoid CUDA Out of memory errors
- `--config-file` is the path to the detectron2 config file
- `SOLVER.MAX_ITER` is the maximum number of iterations
- `SOLVER.STEPS` are the steps at which the learning rate will be decreased

**Hyperparameters**
| OFA family | SOLVER.MAX_ITER | SOLVER.STEPS  | SOLVER.IMS_PER_BATCH |
| ---------- | --------------- | ------------- | -------------------- |
| PN         | 750             | 465,635       | -                    |
| MBv3       | 750             | 465,635       | -                    |
| ResNet     | 1000            | 620,850       | 12                   |



#### For HPE
```
python run_train_cgs_on_task.py -family ofa_mbv3 -task hpe2d -tag shared --lr_cosine --num_epochs 10 -start_idx 10 -num_archs 15 --dataset lsp_extended -chkpt saved_models/ofa_mbv3_hpe2d_sampled_head_head.pt --data_dir data/HPE
```
- `-family` is the OFA family to train, select from one of `ofa_pn`, `ofa_mbv3`, and `ofa_resnet`.
- `-task hpe2d` will execute the hpe2d code
- `-tag` can be any string that labels the output folder
- `-start_idx` is the start index of the architectures to train
- `-num_archs` is the number of architectures to train
- `--dataset lsp_extended` indicates that it should use the LSP extended dataset
- `--num_epochs` is the number of epochs to train for
- `--batch_size` is the number of architectures per batch
- `--lr_cosine` uses a cosine learning rate
- `-chkpt` is the file path of the shared head weights
- `--data_dir` is the folder path of the data directory that contains the HPE data and it should contain both `lsp` and `mpii` subfolders


### Profiler
> Run this profiling script to get the FLOPs and Params of all the architectures in a .pkl cache file\
> `run_profiler.py` will take a cache file containing architectures, profile those architectures, and then overwrite the cache file(s) with new file(s) containing flops and params

#### For Detectron2
```
python run_profiler.py -task detectron -profiler flops params -reprofile --config-file tasks/detectron2/COCO_PanSeg_FPN_Adapted_Head.yml -cache_file cache/FOLDER
```
- `-task detectron` will execute the detectron2 code
- `-reprofile` will profile the architectures even if it has already been profiled
- `-profiler` selects the metrics to profile for
- `-cache_file` is the path to the folder of `.pkl` files to profile
- `--config-file` is the path to the detectron2 config file

#### For HPE
```
python run_profiler.py -task hpe2d -profiler flops params -reprofile --data_dir data/HPE -cache_file cache/FOLDER
```
- `-task hpe2d` will execute the hpe2d code
- `-reprofile` will profile the architectures even if it has already been profiled
- `-profiler` selects the metrics to profile for
- `--data_dir` is the folder path of the data directory that contains the HPE data and it should contain both `lsp` and `mpii` subfolders
- `-cache_file` is the path to the folder of `.pkl` files to profile


### Make cg caches from .pkl files
> This scripts takes the output folders from run_train_cgs_on_task.py, which contains multiple `.pkl` files and combines it into a single `.pkl` file\
> This script outputs a single file named "gpi_ofa_{family}_{test_metric}_{suffix}_comp_graph_cache.pkl"

#### For Detectron2
```bash
python make_cg_task_cache.py -cache_dir cache/FOLDER -family ofa_mbv3 -suffix SUFFIX -test_metric "obj_det"
python make_cg_task_cache.py -cache_dir cache/FOLDER -family ofa_mbv3 -suffix SUFFIX -test_metric "inst_seg"
python make_cg_task_cache.py -cache_dir cache/FOLDER -family ofa_mbv3 -suffix SUFFIX -test_metric "sem_seg"
python make_cg_task_cache.py -cache_dir cache/FOLDER -family ofa_mbv3 -suffix SUFFIX -test_metric "pan_seg"
```
- `-cache_dir` is the path to the folder contains all the `.pkl` files to combine
- `-family` is the OFA family to combine, select from one of `ofa_pn`, `ofa_mbv3`, and `ofa_resnet`.
- `-suffix` is any string to uniquely identify the output cache file
- `-test_metric` is the metric/task you wish to make a cache for. Select from: `obj_det`, `inst_seg`, `sem_seg`, and `pan_seg`

#### For HPE
```bash
python make_cg_task_cache.py -cache_dir cache/FOLDER -family ofa_mbv3 -suffix SUFFIX -test_metric "val_PCK"
```
- `-cache_dir` is the path to the folder contains all the `.pkl` files to combine
- `-family` is the OFA family to combine, select from one of `ofa_pn`, `ofa_mbv3`, and `ofa_resnet`.
- `-suffix` is any string to uniquely identify the output cache file, usually `lsp` or `mpii`
- `-test_metric` is the metric on which accuracy is evaluated. For hpe, the test metric is just `val_PCK`


### Training predictor
>
> This script will output a model of the trained predictor as a `.pt` file to the `saved_models` folder
```
python run_gpi_acc_predictor.py -model_name MODEL_NAME -family_train nb101 -family_test ofa_mbv3_val_PCK_lsp_ind#20+ofa_mbv3_val_PCK_mpii_ind#20+ofa_mbv3_obj_det_coco_ind#20+ofa_mbv3_inst_seg_coco_ind#20+ofa_mbv3_sem_seg_coco_ind#20+ofa_mbv3_pan_seg_coco_ind#20 -fine_tune_epochs 100 -epochs 40 -num_seeds 5 -k_adapt 1 -k_epochs 100 -family_k ofa_mbv3_val_PCK_lsp_shared -tar_norm stand_flops 
```
- `-model_name` is any string to uniquely identify the model
- `-family_train` is the family to train on
- `-family_test` is the list of families to test on
    - Each family is separated by a `+`
    - The names of the tests refer to the middle text in the filename: gpi_*_comp_graph_cache.pkl
    - #20 means set aside 20 archs for calculating standardization mean/s.dev and fine-tunin.
- `-fine_tune_epochs` is the number of epochs to fine tune for
- `-epochs` is the number of epochs to train for
- `-num_seeds` is how many times the same code will be executed at different seed values
- `-e_chk` is the path to the checkpoint file
- `-k_adapt` is the k-adapter
- `-k_epochs` is the number of epochs to train the k-adapter
- `-family_k` is the family to train the k-adapter on
- `-tar_norm` will apply a transform, should be `stand` or `stand_flops`

### Misc. Files
We also include some files for making new compute graphs from .pb files and visualizing them.

#### Making new CGs from .pb files
> See `make_cg.py`\
> We provide sample .pb files for EfficientNet-b0 and ResNet18. 

#### Visualization of CGs
> See `visualize_cgs.py`\
> Need graphviz library.\
> Saves CGs as images which you can then view.\
> E.g., print pictures for the models we provided `.pb` files for, then compare to the actual model using [Netron](https://netron.app/).
