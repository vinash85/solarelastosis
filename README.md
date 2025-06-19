# HistoPath-Fusion

## 1. CLAM Base Information

- **Source Repository**: Cloned from [CLAM GitHub](https://github.com/mahmoodlab/CLAM)  
- **License & Copyright**:  
  - All rights reserved by the original authors at Dr. Faisal Mahmood Lab  
  - Reference paper: *An End-to-End Weakly Supervised Framework for Whole Slide Image Analysis* (Nature Biomedical Engineering, 2020)  
  - Lab website: [clam.mahmoodlab.org](http://clam.mahmoodlab.org/)  
- **Usage**: Follow the original CLAM documentation for patch extraction, feature generation, and heatmap visualization  
- **Extension**: Added support for multimodal inputs (image + clinical data via an MLP)

---

## 2. Environment Setup

### 2.1 Docker Build & Run

1. Change directory:  
   ```bash
   cd solarelastosis/docker

2. Build Docker image:

   ```bash
   docker build . -t solarel:v1 --rm
   ```

3. Run container (Jupyter on port 9090):

   ```bash
   docker run --gpus=all --restart=always --shm-size=10g \
     --name solarelastosis \
     -p 9090:8888 \
     -v /local/path/solarelastosis:/home/tailab/solarelastosis \
     -v /local/path/data:/home/tailab/data \
     -dit solarel:v1
   ```

4. Enter container shell:

   ```bash
   docker exec -it solarelastosis /bin/bash
   ```


## 3. Running the Code

> **Note**: First execute CLAM’s patch extraction, feature generation, and heatmap steps as per official guide.

### 3.1 Training

#### 3.1.1 Unimodal—Image Only

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --drop_out 0.25 \
  --early_stopping \
  --lr 2e-4 \
  --k 10 \
  --exp_code task_1_tumor_vs_normal_CLAM_100 \
  --weighted_sample \
  --bag_loss ce \
  --inst_loss svm \
  --task task_1_tumor_vs_normal \
  --model_type clam_sb \
  --log_data \
  --csv_path /home/tailab/data/gem_blgm/resnet_features/tumor_vs_normal_dummy_clean.csv \
  --data_root_dir ~/data/gem_blgm/concat_resnet_tab_features_mm/ \
  --embed_dim 1024 \
  --modality unimodal \
  --results_dir ./results/mildvmarked_concat_post_resnet \
  --split_dir mildvmarkedtask_1_tumor_vs_normal_100/
```

#### 3.1.2 Multimodal—Image + Clinical

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --drop_out 0.25 \
  --early_stopping \
  --lr 2e-4 \
  --k 10 \
  --exp_code task_1_tumor_vs_normal_CLAM_100 \
  --weighted_sample \
  --bag_loss ce \
  --inst_loss svm \
  --task task_1_tumor_vs_normal \
  --model_type clam_sb \
  --log_data \
  --csv_path /home/tailab/data/gem_blgm/concat_resnet_tab_features_mm/tumor_vs_normal_dummy_clean.csv \
  --data_root_dir ~/data/gem_blgm/concat_resnet_tab_features_mm/ \
  --embed_dim 1024 \
  --clinical_dim 59 \
  --modality multimodal \
  --results_dir ./results/mildvmarked_concat_post_resnet \
  --split_dir mildvmarkedtask_1_tumor_vs_normal_100/
```

#### 3.1.3 Unimodal—Clinical Only

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --drop_out 0.25 \
  --early_stopping \
  --lr 2e-4 \
  --k 10 \
  --exp_code task_1_tumor_vs_normal_MLP_100 \
  --weighted_sample \
  --bag_loss ce \
  --inst_loss svm \
  --task task_1_tumor_vs_normal \
  --model_type mlp \
  --log_data \
  --data_root_dir ~/data/gem_blgm/clinical_pathology/ \
  --embed_dim 59 \
  --csv_path /home/tailab/data/gem_blgm/clinical_features/tumor_vs_normal_dummy_clean.csv \
  --results_dir results/clinical_mm_only \
  --split_dir mildvmarkedtask_1_tumor_vs_normal_100/ \
  --modality unimodal
```

#### 3.1.4 MultiModal—Clinical High-Confidence Threshold

```bash
CUDA_VISIBLE_DEVICES=0 python3 baggingVarianceCutMain.py \
  --data_root_dir /home/tailab/se/data/differenetMethodsFeatures/differenetMethodsFeaturesBasedOnCLAM/autoML_three_ways_features \
  --csv_path /home/tailab/se/data/csv_script_and_csv/mild_vs_absent.csv \
  --k 10 \
  --k_start -1 \
  --k_end -1 \
  --task task_1_tumor_vs_normal \
  --max_epochs 200 \
  --lr 2e-4 \
  --reg 1e-5 \
  --label_frac 1.0 \
  --bag_loss ce \
  --seed 1 \
  --model_type clam_sb \
  --model_size small \
  --drop_out 0.25 \
  --weighted_sample \
  --opt adam \
  --modality multimodal \
  --results_dir baggingVarianceResults \
  --bag_weight 0.7 \
  --inst_loss svm \
  --B 8 \
  --exp_code task1_mild_vs_normal_CLAM_100_baggingVariance \
  --clinical_dim 3 \
  --early_stopping \
  --log_data \
  --split_dir /home/tailab/se/code/solarelastosisFusionBagging/code/splits/mild_vs_absenttask_1_tumor_vs_normal_100
```

#### 3.1.5 MIL—Image Baseline

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --drop_out 0.25 \
  --early_stopping \
  --lr 2e-4 \
  --k 10 \
  --exp_code task_1_tumor_vs_normal_MIL_100 \
  --weighted_sample \
  --bag_loss ce \
  --inst_loss svm \
  --task task_1_tumor_vs_normal \
  --model_type mlp \
  --log_data \
  --data_root_dir ~/data/gem_blgm/Resnet_features/ \
  --embed_dim 59 \
  --csv_path /home/taiblab/data/gem_blgm/concat_resnet_tab_features_mm/tumor_vs_normal_dummy_clean.csv \
  --results_dir results/clinical_mm_only \
  --split_dir mildvmarkedtask_1_tumor_vs_normal_100/ \
  --modality unimodal
```

#### 3.1.6 Image-Only Bagging

```bash
CUDA_VISIBLE_DEVICES=0 python3 bagging_main.py \
  --drop_out 0.25 \
  --early_stopping \
  --lr 2e-4 \
  --k 10 \
  --exp_code task_1_tumor_vs_normal_CLAM_100 \
  --weighted_sample \
  --bag_loss ce \
  --inst_loss svm \
  --task task_1_tumor_vs_normal \
  --model_type clam_sb \
  --log_data \
  --csv_path /home/tailab/se/data/mild_vs_absent.csv \
  --data_root_dir /home/tailab/se/data/resnet_features_solarelastosis/tumor_subtyping_resnet_features \
  --embed_dim 1024 \
  --modality unimodal \
  --results_dir mild_vs_absent_bagging_only_images \
  --split_dir mild_vs_absenttask_1_tumor_vs_normal_100
```

#### 3.1.7 Clinical Bagging

```bash
CUDA_VISIBLE_DEVICES=0 python3 bagging_main.py \
  --drop_out 0.25 \
  --early_stopping \
  --lr 2e-4 \
  --k 10 \
  --exp_code task_1_tumor_vs_normal_CLAM_100 \
  --weighted_sample \
  --bag_loss ce \
  --inst_loss svm \
  --task task_1_tumor_vs_normal \
  --model_type clam_sb \
  --log_data \
  --csv_path /home/tailab/se/data/marked_vs_mild.csv \
  --data_root_dir /home/tailab/se/data/three_ways_features \
  --clinical_dim 3 \
  --embed_dim 1024 \
  --modality product_fusion \
  --results_dir marked_vs_mild_bagging_dot_product \
  --split_dir marked_vs_mildtask_1_tumor_vs_normal_100
```

------

### 3.2 Evaluation

- **TCGA Melanoma Prediction**

  ```bash
  CUDA_VISIBLE_DEVICES=0 python3 eval.py \
    --k 10 \
    --models_exp_code severe_vs_other_image_only \
    --save_exp_code tcga_mel_solar_so \
    --task task_1_tumor_vs_normal \
    --model_type clam_sb \
    --results_dir results \
    --data_root_dir ~/data/tcgamel_clam/ \
    --embed_dim 1024 \
    --mode predict
  ```

- **SHAP Analysis**

  ```bash
  CUDA_VISIBLE_DEVICES=0 python3 eval.py \
    --k 10 \
    --models_exp_code severe_vs_other_image_only \
    --save_exp_code tcga_mel_solar_so \
    --task task_1_tumor_vs_normal \
    --model_type mlp \
    --results_dir results \
    --data_root_dir ~/data/gem_blgm/clinical_pathology \
    --embed_dim 59 \
    --return_probs True \
    --mode shapley
  ```

------

## 4. Docker Directory Structure

```
/home/tailab
├── solarelastosis
│   └── docker
└── Data
    └── gem_blgm
        └── gem_results

```
