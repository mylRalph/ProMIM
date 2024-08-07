# ProMIM - Multi-level Interaction Modeling for Protein Mutational Effect Prediction

## Environment Setups
For the `promim` environment, we suggest installing the `unicore` package from [Uni-Core](https://github.com/dptech-corp/Uni-Core/releases). The `ppiref` environment is used for parsing the PPIRef50K dataset. Please refer to the official [PPIRef](https://github.com/anton-bushuiev/PPIRef/tree/main) repo for detailed instructions. 
```bash
conda create -n promim python=3.8
conda activate promim
pip install -r requirements.txt
conda create -n ppiref python=3.10
```

## Datasets
**1. Get PPIRef50K**

Download PPIRef50K dataset, and split it into training and validation sets by running:
```bash
conda activate ppiref
python ./data/get_ppiref50k.py
```
Use the following command to preprocess PPIRef50K:
```bash
conda activate promim
python ./src/datasets/ppiref50k.py
```

**2. Get SKEMPI2**
```bash
cd data
bash ./get_skempi_v2.sh
cd ..
```

## Inference

**1. Predicting Mutational Effects on Protein-Protein Binding**

```bash
bash ./scripts/test/test_promim_skempi.sh
```

Remember to specify the `idx_cvfolds` parameter to evaluate on the particular fold. You can chose from `0,1,2`.
```bash
ckpt=./trained_models/promim_skempi_cvfold_2.pt
device=cuda:0
idx_cvfolds=2

python test_promim_skempi.py \
    --ckpt $ckpt \
    --device $device \
    --idx_cvfolds $idx_cvfolds
```

**2. Predicting Mutational Effects on Binding Affinity of SARS-CoV-2 RBD**

```bash
bash ./scripts/test/test_promim_6m0j.sh
```

**3. Optimization of Human Antibodies against SARS-CoV-2**

```bash
bash ./scripts/test/test_promim_7fae.sh
```

## Training
**1. Train ProMIM**

```bash
bash ./scripts/train/train_promim.sh
```
You can set the `wandb` flag and `wandb_entity` parameter in `train_promim.sh` to use [Weights & Biases](https://wandb.ai/site) for logging or use TensorBoard by default.

```bash
nproc_per_node=4
world_size=4
master_port=20888
config_path=./configs/train/promim.yml

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$nproc_per_node --master_port=$master_port train_promim.py \
    --config $config_path \
    --world_size $world_size \
    --wandb \
    --wandb_entity your_wandb_username
```

**2. Train ProMIM DDG Predictor**
```bash
bash ./scripts/train/train_promim_skempi.sh
```
