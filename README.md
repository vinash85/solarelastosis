# HistoPath-Fusion

## [CLAM has been cloned from here] (https://github.com/mahmoodlab/CLAM)
1. All license and copy rights to the Original CLAM belongs to the [original authors] (https://www.nature.com/articles/s41551-020-00682-w) from Dr.Faisal Mahmood Lab [http://clam.mahmoodlab.org/]
2. Usage instruction for HistoPath-Fusion remains same as the instructions for CLAM in above link.
3. The additional code useage for multimodal data, clinical data MLP is given below

## Requirements
Docker usage:
1. cd solarelatosis/docker
2. Create image-command: "docker build . -t solarel:v1 --rm"
3. Run the image[port 9090 for jupyter notebooks]-command: "docker run --gpus=all --restart=always --shm-size=10g --name solarelastosis -p 9090:8888 -v path_on_local_system/:/home/tailab/solarelastosis -v path_on_local_system/:/home/tailab/data/  -dit solarel:v1"
4. Enter the docker:"docker --exec -it solarelastosis /bin/bash"

(Container to be released  soon)
## Running code inside docker
**PLEASE REFER TO CLAM LINK ABOVE TO EXTRACT PATCHES, FEATURES AND HEATMAP**
### Training
1. UNIMODAL-IMAGE(Example:Mild vs Marked):
"CUDA_VISIBLE_DEVICES=0 python3 main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --exp_code task_1_tumor_vs_normal_CLAM_100 --weighted_sample --bag_loss ce --inst_loss svm 
--task task_1_tumor_vs_normal --model_type clam_sb --log_data --csv_path /home/tailab/data/gem_blgm/resnet_features/tumor_vs_normal_dummy_clean.csv
--data_root_dir ~/data/gem_blgm/concat_resnet_tab_features_mm/ --embed_dim 1024  --modality unimodal 
--results_dir ./results/mildvmarked_concat_post_resnet --split_dir mildvmarkedtask_1_tumor_vs_normal_100/"

2. MULTIMODAL (Example:Mild vs Marked):
"CUDA_VISIBLE_DEVICES=0 python3 main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --exp_code task_1_tumor_vs_normal_CLAM_100 --weighted_sample --bag_loss ce --inst_loss svm 
--task task_1_tumor_vs_normal --model_type clam_sb --log_data --csv_path /home/tailab/data/gem_blgm/concat_resnet_tab_features_mm/tumor_vs_normal_dummy_clean.csv
--data_root_dir ~/data/gem_blgm/concat_resnet_tab_features_mm/ --embed_dim 1024 --clinical_dim 59 --modality multimodal 
--results_dir ./results/mildvmarked_concat_post_resnet --split_dir mildvmarkedtask_1_tumor_vs_normal_100/"

3. UNIMODAL-Clinical (Example:Mild vs Marked):
CUDA_VISIBLE_DEVICES=0 python3 main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --exp_code task_1_tumor_vs_normal_MLP_100 --weighted_sample --bag_loss ce --inst_loss svm 
--task task_1_tumor_vs_normal --model_type mlp --log_data --data_root_dir ~/data/gem_blgm/clinical_pathology/ --embed_dim 59  
--csv_path /home/tailab/data/gem_blgm/clinical_features/tumor_vs_normal_dummy_clean.csv --results_dir results/clinical_mm_only 
--split_dir mildvmarkedtask_1_tumor_vs_normal_100/ --modality unimodal"

4. MIL-Image ONLY (Baseline):
CUDA_VISIBLE_DEVICES=0 python3 main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --exp_code task_1_tumor_vs_normal_MIL_100 --weighted_sample --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --model_type mlp --log_data --data_root_dir ~/data/gem_blgm/Resnet_features/ --embed_dim 59  --csv_path /home/taiblab/data/gem_blgm/concat_resnet_tab_features_mm/tumor_vs_normal_dummy_clean.csv --results_dir results/clinical_mm_only --split_dir mildvmarkedtask_1_tumor_vs_normal_100/ --modality unimodal

5. Image only Bagging
CUDA_VISIBLE_DEVICES=0 python3 bagging_main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --exp_code task_1_tumor_vs_normal_CLAM_100 --weighted_sample --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --model_type clam_sb --log_data --csv_path /home/tailab/se/data/mild_vs_absent.csv --data_root_dir /home/tailab/se/data/resnet_features_solarelastosis/tumor_subtyping_resnet_features  --embed_dim 1024  --modality unimodal --results_dir mild_vs_absent_bagging_only_images --split_dir mild_vs_absenttask_1_tumor_vs_normal_100

6. Clinical Bagging
CUDA_VISIBLE_DEVICES=0 python3 bagging_main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --exp_code task_1_tumor_vs_normal_CLAM_100 --weighted_sample --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --model_type clam_sb --log_data --csv_path /home/tailab/se/data/marked_vs_mild.csv --data_root_dir /home/tailab/se/data/three_ways_features --clinical_dim 3  --embed_dim 1024  --modality product_fusion --results_dir marked_vs_mild_bagging_dot_product --split_dir marked_vs_mildtask_1_tumor_vs_normal_100

### Evaluation

1. TCGA Melanoma:
"CUDA_VISIBLE_DEVICES=0 python3 eval.py --k 10 --models_exp_code severe_vs_other_image_only  --save_exp_code tcga_mel_solar_so --task task_1_tumor_vs_normal --model_type clam_sb --results_dir results --data_root_dir ~/data/tcgamel_clam/ --embed_dim 1024 --mode predict"

2. SHAPLey:
"CUDA_VISIBLE_DEVICES=0 python3 eval.py --k 10 --models_exp_code severe_vs_other_image_only  --save_exp_code tcga_mel_solar_so --task task_1_tumor_vs_normal --model_type mlp --results_dir results --data_root_dir ~/data/gem_blgm/clinical_pathology --embed_dim 59 
--return_probs True --mode shapley"


## Folder Structure in docker
/home/tailab

           |

            solarelastosis -- code

           |               |__docker
            
            Data -- gem_blgm
                   
                   |__gem_results      
