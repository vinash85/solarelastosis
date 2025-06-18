import numpy as np
import torch
from sklearn.utils import resample
import torch.utils
from utils.utils import *
import os
from dataset_modules.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc, MLP
from models.model_clam import CLAM_MB, CLAM_SB, BoostingModel
from models.AM_Former import FTTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
import random
import argparse
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, Generic_Split
from utils.core_utils import train_loop_clam, validate_clam
import matplotlib.pyplot as plt
from matplotlib.figure import Figure      # 你已 import 过，可跳过
import json                                   # ⬅︎ 用来保存阈值


def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

class BaggingEnsembleTrainer:
    def __init__(self, base_model, args, n_estimators=5):
        self.base_model = base_model
        self.args = args
        self.n_estimators = n_estimators
        self.models = []
        self.bootstrap_splits = []
        self.thr_candidates, self.fold_auc_mat, self.fold_cov_mat = None, [], []
                # —— 方差阈值相关 ——
        self.var_thr_candidates = np.logspace(-9, -1, 200) 
        self.fold_var_auc_mat, self.fold_var_cov_mat = [], []
    # ① ————————————
    @staticmethod
    def _get_loader_stats(models, loader):
        """返回 (sample_means, sample_vars, labels, avg_probs)"""
        device = next(models[0].parameters()).device
        with torch.no_grad():
            all_probs = []
            labels = None
            for model in models:
                model.eval()
                probs = []
                batch_labels  = []   
                for data, label in loader:
                    data, label = data.to(device), label.to(device)
                    _, y_prob, *_ = model(data)
                    probs.append(y_prob.cpu().numpy())
                    # 只在第一轮记录 label
                    if labels is None:
                        batch_labels.append(label.cpu().numpy())

                all_probs.append(np.vstack(probs))
                if labels is None:                  # 首次循环后一次性拼好
                    labels = np.hstack(batch_labels)

        all_probs = np.array(all_probs)                     # (K, N, C)
        n_estimators, n_samples, n_classes = all_probs.shape
        mean_probs = all_probs.mean(axis=0)                 # (N, C)
        pred_cls   = mean_probs.argmax(1)                   # (N,)
        selected   = np.stack([all_probs[i, np.arange(n_samples), pred_cls]
                               for i in range(n_estimators)], axis=0)
        sample_means = selected.mean(0)                     # (N,)
        sample_vars  = selected.var(0)                      # (N,)
        return sample_means, sample_vars, labels, mean_probs

    # ② ————————————
    @staticmethod
    def _scan_threshold(sample_means, labels, probs,
                        writer=None, tag_prefix='th_scan', cur=0,
                        return_curve=True):
        if args.n_classes == 2:
            thresholds = np.linspace(0.5, 0.99, 50)
        else:   # 多分类时，取 0.1 到 0.9 的阈值
            thresholds = np.linspace(0.33, 0.99, 66)
        cov, auc = [], []
        for t in thresholds:
            mask = sample_means >= t
            cov.append(mask.mean())
            if mask.sum():                      # 至少留下一些样本
                if args.n_classes == 2:
                    auc.append(safe_roc_auc(labels[mask], probs[mask, 1]))
                else:
                    try:
                        auc.append(roc_auc_score(labels[mask], probs[mask], multi_class='ovr'))
                    except ValueError as e:
                        print(f"Error calculating AUC for multiclass: {e}")
                        auc.append(np.nan)
            else:
                auc.append(np.nan)

        # 覆盖-AUC 曲线
        fig = Figure(figsize=(6,5), dpi=100)
        ax  = fig.add_subplot(1,1,1)
        ax.plot(cov, auc, marker='o')
        ax.set_xlabel('Coverage (≥thr)')
        ax.set_ylabel('AUC')
        ax.set_title('Coverage-AUC curve')
        ax.grid(True, linestyle='--', linewidth=0.5)
        if writer:
            writer.add_figure(f'{tag_prefix}/cov_auc', fig, global_step=cur)

        # 直方图
        if writer:
            writer.add_histogram(f'{tag_prefix}/conf_hist', sample_means, cur)

        valid_idx = np.where(~np.isnan(auc))[0]
        if len(valid_idx) == 0:        # 全 NaN，退回默认 0.5
            return 0.5
        best_idx = valid_idx[np.argmax([a if cov[i] >= 0.6 else -np.inf
                                        for i, a in enumerate(np.array(auc)[valid_idx])])]
        if return_curve:
            return thresholds, np.array(cov), np.array(auc)
        else:
            return thresholds[best_idx]

    # ③ ————————————
    @staticmethod
    def _save_thr(path, thr):
        with open(path, 'w') as f:
            json.dump({'thr': float(thr)}, f)
    
        # ②·b  ————————————
    @staticmethod
    def _scan_var_threshold(thresholds, sample_vars, labels, probs,
                            writer=None, tag_prefix='var_thr_scan', cur=0,
                            return_curve=True):
        """
        与 _scan_threshold 类似，但使用 'sample_vars ≤ thr' 作为筛选条件。
        阈值越小 → 留下“高置信”样本越少。
        """
        # 先用对数刻度把 [min-var, max-var] 均匀切 50 段
        v_min, v_max = sample_vars.min(), sample_vars.max()
        # # 避免 log(0)
        # v_min = max(v_min, 1e-6)
        # thresholds = np.exp(np.linspace(np.log(v_min), np.log(v_max), 50))

        cov, auc = [], []
        for t in thresholds:
            mask = sample_vars <= t          # ⬅︎ 方差小于阈值时保留
            cov.append(mask.mean())
            if mask.sum():
                if args.n_classes == 2:
                    auc.append(safe_roc_auc(labels[mask], probs[mask, 1]))
                else:
                    try:
                        auc.append(roc_auc_score(labels[mask], probs[mask], multi_class='ovr'))
                    except ValueError:
                        auc.append(np.nan)
            else:
                auc.append(np.nan)

        # —— 画 Coverage-AUC 曲线到 TensorBoard ——
        if writer:
            from matplotlib.figure import Figure
            fig = Figure(figsize=(6,5), dpi=100)
            ax  = fig.add_subplot(1,1,1)
            ax.plot(cov, auc, marker='o')
            ax.set_xlabel('Coverage (var ≤ thr)')
            ax.set_ylabel('AUC')
            ax.set_title('Coverage-AUC curve (variance filter)')
            ax.grid(True, linestyle='--', linewidth=0.5)
            writer.add_figure(f'{tag_prefix}/cov_auc', fig, global_step=cur)
            # variance 直方图
            writer.add_histogram(f'{tag_prefix}/var_hist', sample_vars, cur)

        valid_idx = np.where(~np.isnan(auc))[0]
        if len(valid_idx) == 0:
            return  thresholds, np.array(cov), np.array(auc) if return_curve else v_max
        # 要求 coverage ≥ 0.6（或自行调整），再取 AUC 最大
        best_idx = valid_idx[np.argmax([a if cov[i] >= 0.6 else -np.inf
                                        for i,a in enumerate(np.array(auc)[valid_idx])])]
        return (thresholds, np.array(cov), np.array(auc)) if return_curve else thresholds[best_idx]


    def create_bootstrap_samples(self, dataset):
        # Create bootstrap samples by resampling indices and constructing new dataset instances
        for i in range(self.n_estimators):
            seed = self.args.seed + i  # Different seed for each estimator
            np.random.seed(seed)
            indices = np.random.choice(len(dataset), size=len(dataset), replace=True)
            slide_data = dataset.slide_data.iloc[indices].reset_index(drop=True)
            bootstrap_sample = Generic_Split(slide_data=slide_data, data_dir=dataset.data_dir, num_classes=dataset.num_classes)
            self.bootstrap_splits.append(bootstrap_sample)


    def train(self, datasets, cur):
        """
        Train multiple models on different bootstrap samples.
        """
        print('\nTraining Fold {} with Bagging Ensemble!'.format(cur))
        writer_dir = os.path.join(self.args.results_dir, str(cur))
        if not os.path.isdir(writer_dir):
            os.mkdir(writer_dir)

        if self.args.log_data:
            from tensorboardX import SummaryWriter
            writer = SummaryWriter(writer_dir, flush_secs=15)
        else:
            writer = None

        print('\nInit train/val/test splits...', end=' ')
        train_split, val_split, test_split = datasets
        save_splits(datasets, ['train', 'val', 'test'], os.path.join(self.args.results_dir, 'splits_{}.csv'.format(cur)))
        print('Done!')
        print("Training on {} samples".format(len(train_split)))
        print("Validating on {} samples".format(len(val_split)))
        print("Testing on {} samples".format(len(test_split)))

        # Create bootstrap samples from training data
        self.create_bootstrap_samples(train_split)

        print('\nInit loss function...', end=' ')
        if self.args.bag_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            loss_fn = SmoothTop1SVM(n_classes=self.args.n_classes).to(device)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
        print('Done!')

        print('\nSetup EarlyStopping...', end=' ')
        if args.early_stopping:
            early_stopping = EarlyStopping(patience = 10, stop_epoch=30, verbose = True)

        else:
            early_stopping = None
        print('Done!')

        for i in range(self.n_estimators):
            print(f"\nTraining model {i + 1}/{self.n_estimators} on bootstrap sample...")
            model = self.base_model(gate=True, size_arg=self.args.model_size, dropout=self.args.drop_out, k_sample=self.args.B, n_classes=self.args.n_classes, instance_loss_fn=torch.nn.CrossEntropyLoss(), subtyping=self.args.subtyping, num_cate=self.args.num_cate, embed_dim=self.args.embed_dim, modality=self.args.modality, clinical_dim=self.args.clinical_dim).to(device)
            optimizer, warmup_scheduler, cosine_scheduler = get_optim(model, self.args)

            train_loader = get_split_loader(self.bootstrap_splits[i], training=True, testing=self.args.testing, weighted=self.args.weighted_sample)
            val_loader = get_split_loader(val_split, testing=self.args.testing)
            test_loader = get_split_loader(test_split, testing=self.args.testing)

            # Train individual model
            for epoch in range(self.args.max_epochs):
                train_loop_clam(args, epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn, warmup_scheduler, cosine_scheduler)
                stop = validate_clam(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
                if stop:
                    break

            # Save trained model
            self.models.append(model)
            torch.save(model.state_dict(), os.path.join(self.args.results_dir, f"s_{cur}_model_{i}.pt"))
        
                # —— 选阈值（验证集）—————————————
        val_loader = get_split_loader(val_split, testing=self.args.testing)
        s_mean, s_var, y_val, mean_probs_val = self._get_loader_stats(self.models, val_loader)

        # 拿到整条曲线
        thr_list, cov_arr, auc_arr = self._scan_threshold(
            s_mean, y_val, mean_probs_val, writer, 'val_thr_scan', cur,
            return_curve=True)

        # 初始化候选阈值列表，只会在第一折运行一次
        if self.thr_candidates is None:
            self.thr_candidates = thr_list

        # 累加到全局曲线矩阵
        self.fold_auc_mat.append(auc_arr)
        self.fold_cov_mat.append(cov_arr)

        # 本折最佳阈值（用 coverage≥0.6 且 AUC 最大的规则）
        valid_idx = np.where(~np.isnan(auc_arr))[0]
        best_idx  = valid_idx[np.argmax([a if cov_arr[i] >= 0.1 else -np.inf
                                         for i, a in enumerate(auc_arr[valid_idx])])]
        best_thr = thr_list[best_idx]
        print(f'Fold {cur}: best confidence threshold = {best_thr:.3f}')
        self._save_thr(os.path.join(self.args.results_dir, f"thr_fold{cur}.json"), best_thr)
        # —— 选方差阈值（验证集）—————————————
        s_mean, s_var, y_val, mean_probs_val = self._get_loader_stats(self.models, val_loader)

        var_thr_list, var_cov_arr, var_auc_arr = self._scan_var_threshold(
            self.var_thr_candidates, # <-- 使用固定列表
            s_var, y_val, mean_probs_val, writer, 'val_var_thr_scan', cur,
            return_curve=True)

        # # 仅第一次初始化候选
        # if self.var_thr_candidates is None:
        #     self.var_thr_candidates = var_thr_list
        self.fold_var_auc_mat.append(var_auc_arr)
        self.fold_var_cov_mat.append(var_cov_arr)

        # 每折最佳方差阈值
        valid_idx = np.where(~np.isnan(var_auc_arr))[0]
        best_idx  = valid_idx[np.argmax([a if var_cov_arr[i] >= 0.6 else -np.inf
                                         for i,a in enumerate(var_auc_arr[valid_idx])])]
        best_var_thr = var_thr_list[best_idx]
        print(f'Fold {cur}: best variance threshold = {best_var_thr:.6f}')
        self._save_thr(os.path.join(self.args.results_dir, f"var_thr_fold{cur}.json"),
                       best_var_thr)

        # Evaluate ensemble
        self.evaluate_ensemble(test_loader, cur, writer)

    def evaluate_ensemble(self, test_loader, cur, writer):
        print("\nEvaluating Bagging Ensemble...")
        all_probs = []
        all_labels = []

        # 收集各模型的预测概率
        with torch.no_grad():
            for model in self.models:
                model.eval()
                probs, labels = [], []
                for data, label in test_loader:
                    data, label = data.to(device), label.to(device)
                    _, Y_prob, *_ = model(data)
                    probs.append(Y_prob.cpu().numpy())
                    labels.append(label.cpu().numpy())
                all_probs.append(np.vstack(probs))
                all_labels = np.hstack(labels)

        all_probs = np.array(all_probs)  # shape: (n_estimators, n_samples, n_classes)
        n_estimators, n_samples, n_classes = all_probs.shape

       # 2’) Compute per-estimator variance around the ensemble mean for the chosen class
        mean_probs = np.mean(all_probs, axis=0)            # (n_samples, n_classes)
        pred_classes = np.argmax(mean_probs, axis=1)       # (n_samples,)
        mean_selected = mean_probs[np.arange(n_samples), pred_classes]

        variances = []
        means = []
        for i in range(n_estimators):
            probs_i = all_probs[i]                         # (n_samples, n_classes)
            selected_i = probs_i[np.arange(n_samples), pred_classes]
            # variance relative to the ensemble’s mean prediction for that class
            means.append(np.mean(selected_i))
            var_i = np.mean((selected_i - mean_selected) ** 2)
            variances.append(var_i)
        variances = np.array(variances)
        means = np.array(means)
                # 1) 线性拟合 (y = kx + b)
        k, b = np.polyfit(means, variances, deg=1)
        x_line = np.linspace(means.min(), means.max(), 200)
        y_line = k * x_line + b

        # 2) 构造 Figure（完全不用 pyplot）
        fig = Figure(figsize=(6, 5), dpi=100)
        ax  = fig.add_subplot(1, 1, 1)

        # --- 散点 ---
        ax.scatter(means, variances, alpha=0.75, label='estimators')

        # --- 拟合直线 ---
        ax.plot(x_line, y_line, linestyle='--', linewidth=1.5,
                label=f'fit: y = {k:.3f}·x + {b:.3f}')

        # --- 轴 & 标题 ---
        ax.set_xlabel('Mean predicted probability', fontsize=12)
        ax.set_ylabel('Variance', fontsize=12)
        ax.set_title('Per-Estimator Mean vs Variance', fontsize=14)
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.legend(fontsize=9)

        # 3) 直接写入 TensorBoard
        if writer:
            writer.add_figure('variance_filter/mean_vs_variance', fig, global_step=cur)
        
        # ---------------------------------------------------
        # 1) 计算每个样本在预测类别上的 mean & variance
        # mean_selected 已经是 shape (n_samples,) 的平均概率
        # 我们再算一下 sample-level 的方差

        # 把每个 estimator 在样本 j 上、预测类别对应的概率挑出来
        # selected_probs: shape (n_estimators, n_samples)
        selected_probs = np.stack([
            all_probs[i][np.arange(n_samples), pred_classes]
            for i in range(n_estimators)
        ], axis=0)

        # 每个样本的 mean & var
        sample_means = selected_probs.mean(axis=0)    # (n_samples,)
        sample_vars  = selected_probs.var(axis=0)     # (n_samples,)

        # 2) 用 TensorBoard 画直方图，查看分布
        if writer:
            writer.add_histogram('confidence/sample_means_hist', sample_means, global_step=cur)
            writer.add_histogram('confidence/sample_vars_hist',  sample_vars,  global_step=cur)

        # 3) 用 TensorBoard 画散点图，查看 mean vs var
        fig = Figure(figsize=(6,5), dpi=100)
        ax  = fig.add_subplot(1,1,1)
        ax.scatter(sample_means, sample_vars, alpha=0.6)
        ax.set_xlabel('Sample Mean Probability')
        ax.set_ylabel('Sample Variance')
        ax.set_title('Per-sample Confidence Distribution')
        ax.grid(True, linestyle='--', linewidth=0.5)

        if writer:
            writer.add_figure('confidence/mean_vs_var_scatter', fig, global_step=cur)
        # ---------------------------------------------------


        # 把 estimator 按方差从大到小排序
        order = np.argsort(variances)[::-1]

        # 2) 对多种保留比例，计算筛选后 ensemble 的 AUC
        percentages = np.linspace(0.2, 1.0, 5)  # 20%, 40%, …, 100%
        aucs = []
        for p in percentages:
            k = max(1, int(n_estimators * p))
            selected = order[:k]
            avg_probs = np.mean(all_probs[selected], axis=0)  # shape: (n_samples, n_classes)

            # 计算 AUC-ROC
            if n_classes == 2:
                auc = safe_roc_auc(all_labels, avg_probs[:, 1])
            else:
                try:
                    auc = roc_auc_score(all_labels, avg_probs, multi_class='ovr')
                except ValueError as e:
                    print(f"Error calculating AUC for multiclass: {e}")
                    auc = np.nan

            aucs.append(auc)

            # 将每个比例的指标都写入 TensorBoard
            if writer:
                writer.add_scalar(f'variance_filter/auc_{int(p*100)}%', auc, cur)
                writer.add_scalar(f'variance_filter/keep_{int(p*100)}%_n', k, cur)

        # 4) Directly plot in TensorBoard via add_scalars
        #    main_tag: 'variance_filter/auc_vs_keep'
        #    tag_scalar_dict: {'10%': 0.71, '20%': 0.72, …}
        scalar_dict = {f'{int(p*100)}%': auc for p, auc in zip(percentages, aucs)}
        if writer:
            writer.add_scalars('variance_filter/auc_vs_keep', scalar_dict, cur)


        # 原有的平均所有 estimator 的 AUC 也保留
        avg_probs_all = np.mean(all_probs, axis=0)
        if n_classes == 2:
            auc_all = safe_roc_auc(all_labels, avg_probs_all[:, 1])
            auc_pr_all = safe_pr_auc(all_labels, avg_probs_all[:, 1])
        else:
            try:
                auc_all = roc_auc_score(all_labels, avg_probs_all, multi_class='ovr')
                auc_pr_all = average_precision_score(all_labels, avg_probs_all, average='macro')
            except ValueError as e:
                print(f"Error calculating AUC for multiclass: {e}")
                auc_all = np.nan
                auc_pr_all = np.nan

        print(f'All estimators ensemble AUC: {auc_all:.4f}, AUC-PR: {auc_pr_all:.4f}')
        if writer:
            writer.add_scalar('final/ensemble_auc', auc_all, cur)
            writer.add_scalar('final/ensemble_auc_pr', auc_pr_all, cur)
        print("\nEvaluating Bagging Ensemble...")
        all_probs = []
        all_labels = []

        # Collect predictions from each model
        with torch.no_grad():
            for model in self.models:
                model.eval()
                probs = []
                labels = []
                for batch_idx, (data, label) in enumerate(test_loader):
                    data, label = data.to(device), label.to(device)
                    _, Y_prob, _, _, _ = model(data)
                    probs.append(Y_prob.cpu().numpy())
                    labels.append(label.cpu().numpy())
                all_probs.append(np.vstack(probs))
                all_labels = np.hstack(labels)  # Assuming all models see the same labels

        # Average predictions for final ensemble output
        avg_probs = np.mean(all_probs, axis=0)
        
        # Compute AUC-ROC
        if self.args.n_classes == 2:
            auc = safe_roc_auc(all_labels, avg_probs[:, 1])
            auc_pr = safe_pr_auc(all_labels, avg_probs[:, 1])  # AUC-PR for binary
        else:
            try:
                auc = roc_auc_score(all_labels, avg_probs, multi_class='ovr')
                auc_pr = average_precision_score(all_labels, avg_probs, average='macro')  # AUC-PR for multiclass
            except ValueError as e:
                print(f"Error calculating AUC for multiclass: {e}")
                auc = np.nan
                auc_pr = np.nan
        print(f'Test AUC of Bagging Ensemble: {auc:.4f}')
        print(f'Test AUC-PR of Bagging Ensemble: {auc_pr:.4f}')
        
        # Log to writer if available
        if writer:
            writer.add_scalar('final/ensemble_auc', auc, cur)
            writer.add_scalar('final/ensemble_auc_pr', auc_pr, cur)
        # —— 利用阈值做选择性预测 ——————————
        # ---------------------------------------------------
        # 额外：把 test 的 s_mean / labels / probs 存起来，便于全局阈值评测
        s_mean_test, s_var_test, _, _ = self._get_loader_stats(self.models, test_loader)
        np.savez(os.path.join(self.args.results_dir,
                              f"test_stats_fold{cur}.npz"),
                 s_mean=s_mean_test,
                 s_var = s_var_test,
                 labels=all_labels,
                 probs=avg_probs)

        thr_file = os.path.join(self.args.results_dir, f"thr_fold{cur}.json")
        if os.path.isfile(thr_file):
            thr = json.load(open(thr_file))['thr']
            s_mean, _, _, _ = self._get_loader_stats(self.models, test_loader)
            mask = s_mean >= thr
            if mask.sum():
                if args.n_classes == 2:
                    auc_conf = safe_roc_auc(all_labels[mask], avg_probs[mask, 1])
                else:
                    try:
                        auc_conf = roc_auc_score(all_labels[mask], avg_probs[mask], multi_class='ovr')
                    except ValueError as e:
                        print(f"Error calculating AUC for multiclass with threshold: {e}")
                        auc_conf = np.nan
                coverage = mask.mean()
                print(f'High-confidence AUC={auc_conf:.4f} @ coverage={coverage:.2%}')
                if writer:
                    writer.add_scalar('selective/auc_conf', auc_conf, cur)
                    writer.add_scalar('selective/coverage', coverage, cur)
        # —— 2) 方差阈值 selective prediction ——————————
        var_thr_file = os.path.join(self.args.results_dir, f"var_thr_fold{cur}.json")
        if os.path.isfile(var_thr_file):
            var_thr = json.load(open(var_thr_file))['thr']
            _, s_var_test, _, _ = self._get_loader_stats(self.models, test_loader)
            mask_var = s_var_test <= var_thr
            if mask_var.sum():
                if args.n_classes == 2:
                    auc_var = safe_roc_auc(all_labels[mask_var], avg_probs[mask_var, 1])
                else:
                    try:
                        auc_var = roc_auc_score(all_labels[mask_var], avg_probs[mask_var], multi_class='ovr')
                    except ValueError:
                        auc_var = np.nan
                coverage_var = mask_var.mean()
                print(f'Low-variance AUC={auc_var:.4f} @ coverage={coverage_var:.2%}')
                if writer:
                    writer.add_scalar('selective/auc_var', auc_var, cur)
                    writer.add_scalar('selective/coverage_var', coverage_var, cur)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configurations for WSI Training')
    parser.add_argument('--data_root_dir', type=str, default=None, 
                        help='data directory')
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--clinical_dim', type=int, default=59)
    parser.add_argument('--max_epochs', type=int, default=200,
                        help='maximum number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--label_frac', type=float, default=1.0,
                        help='fraction of training labels (default: 1.0)')
    parser.add_argument('--reg', type=float, default=1e-5,
                        help='weight decay (default: 1e-5)')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Number of warmup epochs')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Minimum learning rate for cosine annealing')
    parser.add_argument('--seed', type=int, default=1, 
                        help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
    parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
    parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
    parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
    parser.add_argument('--csv_path', default='/home/tailab/se/data/tumor_vs_normal_dummy_clean_tabnet.csv', help='csv file path')
    parser.add_argument('--split_dir', type=str, default=None, 
                        help='manually specify the set of splits to use, ' 
                        +'instead of infering from the task and label_frac argument (default: None)')
    parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
    parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
    parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
    parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
    parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
    parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='svm',
                        help='slide-level classification loss function (default: ce)')
    parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil','mlp'], default='clam_sb', 
                        help='type of model (default: clam_sb, clam w/ single attention branch)')
    parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
    parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
    parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
    parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping'])
    ### CLAM specific options
    parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                        help='disable instance-level clustering')
    parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                        help='instance-level clustering loss function (default: None)')
    parser.add_argument('--subtyping', action='store_true', default=False, 
                        help='subtyping problem')
    parser.add_argument('--bag_weight', type=float, default=0.7,
                        help='clam: weight coefficient for bag-level loss (default: 0.7)')
    parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
    parser.add_argument('--modality', type=str, default='multimodal', choices=['unimodal', 'multimodal', 'fusion', 'Former', 'product_fusion', 'sum_fusion', 'concat_fusion', 'FiLM', 'gated_fusion'])
    parser.add_argument('--config', type=str, default='models/config/run/ours_fttrans-hcdr.yaml')
    parser.add_argument('--num_cate', type=int, default=54, help='numbr of category columns for tabular data')

    args = parser.parse_args()

    print("\nLoad Dataset")

    if args.task == 'task_1_tumor_vs_normal':
        args.n_classes=2
        dataset = Generic_MIL_Dataset(csv_path = args.csv_path,
                                data_dir= args.data_root_dir,
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                # label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                                label_dict = {0:0, 1:1},
                                patient_strat=False,
                                ignore=[])

    elif args.task == 'task_2_tumor_subtyping':
        args.n_classes=3
        dataset = Generic_MIL_Dataset(csv_path = args.csv_path,
                                data_dir= args.data_root_dir,
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict = {1:0, 2:1, 3:2},
                                patient_strat= False,
                                ignore=[])

        if args.model_type in ['clam_sb', 'clam_mb']:
            print(1)
            # assert args.subtyping 
            
    else:
        raise NotImplementedError
        
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.split_dir is None:
        args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
    else:
        args.split_dir = os.path.join('splits', args.split_dir)

    print('split_dir: ', args.split_dir)
    assert os.path.isdir(args.split_dir)

    settings = {'num_splits': args.k, 
                'k_start': args.k_start,
                'k_end': args.k_end,
                'task': args.task,
                'max_epochs': args.max_epochs, 
                'results_dir': args.results_dir, 
                'lr': args.lr,
                'experiment': args.exp_code,
                'reg': args.reg,
                'label_frac': args.label_frac,
                'bag_loss': args.bag_loss,
                'seed': args.seed,
                'model_type': args.model_type,
                'model_size': args.model_size,
                "use_drop_out": args.drop_out,
                'weighted_sample': args.weighted_sample,
                'opt': args.opt,
                'modality' : args.modality,
                'csv_path': args.csv_path,
                'data_root_dir': args.data_root_dir}

    if args.model_type in ['clam_sb', 'clam_mb']:
        settings.update({'bag_weight': args.bag_weight,
                        'inst_loss': args.inst_loss,
                        'B': args.B})

    with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
        print(settings, file=f)
    f.close()

    print("################# Settings ###################")
    for key, val in settings.items():
        print("{}:  {}".format(key, val))        

    print("\nLoad Dataset")

    # Instantiate BaggingEnsembleTrainer
    trainer = BaggingEnsembleTrainer(base_model=CLAM_SB if args.model_type == 'clam_sb' else CLAM_MB, args=args, n_estimators=5)
    folds = np.arange(args.k_start if args.k_start != -1 else 0, args.k_end if args.k_end != -1 else args.k)

    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                                                                         csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        datasets = (train_dataset, val_dataset, test_dataset)
        trainer.train(datasets, cur=i)
    # === 训练全部折后：计算平均 AUC-Coverage 曲线并选全局阈值 ===
    fold_auc_mat = np.stack(trainer.fold_auc_mat, axis=0)   # (k, T)
    fold_cov_mat = np.stack(trainer.fold_cov_mat, axis=0)   # (k, T)

    mean_auc = np.nanmean(fold_auc_mat, axis=0)             # (T,)
    mean_cov = np.nanmean(fold_cov_mat, axis=0)

    best_idx_global = np.nanargmax(mean_auc)
    global_best_thr = trainer.thr_candidates[best_idx_global]
    print(f'\n>>> Global best threshold across {args.k} folds = {global_best_thr:.3f}')


    # 画 AUC-Coverage 曲线
    plt.figure(figsize=(6,5), dpi=100)
    plt.plot(mean_cov, mean_auc, marker='o')
    plt.xlabel("Coverage (≥thr)")
    plt.ylabel("AUC")
    plt.title("Average Val AUC-Coverage across folds")
    plt.grid(ls='--', lw=0.5)
    fig_path = os.path.join(args.results_dir, "val_auc_vs_coverage.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {fig_path}")
        # === 用 global_best_thr 评测 10 折 test 集 ===
    glob_auc_list, cov_list = [], []
    for i in folds:
        stat_path = os.path.join(args.results_dir, f"test_stats_fold{i}.npz")
        stat = np.load(stat_path)
        s_mean  = stat["s_mean"]
        labels  = stat["labels"]
        probs   = stat["probs"]

        mask = s_mean >= global_best_thr
        coverage = mask.mean()
        if coverage == 0:
            auc_sel = np.nan
        else:
            if args.n_classes == 2:
                auc_sel = safe_roc_auc(labels[mask], probs[mask, 1])
            else:
                auc_sel = roc_auc_score(labels[mask], probs[mask], multi_class="ovr")

        glob_auc_list.append(auc_sel)
        cov_list.append(coverage)
        print(f"Fold {i}: selective AUC = {auc_sel:.4f}  @ coverage = {coverage:.2%}")

    print("\n>>> Mean selective AUC across folds: "
          f"{np.nanmean(glob_auc_list):.4f} ± {np.nanstd(glob_auc_list):.4f}")
    print(">>> Mean coverage: "
          f"{np.nanmean(cov_list):.2%}")
    global_thr_path = os.path.join(args.results_dir, "global_thr_auc_and_coverage.json")
    with open(global_thr_path, "w") as f:
        json.dump({"thr": global_best_thr,"global_test_auc":np.nanmean(glob_auc_list),"global_test_coverage":np.nanmean(cov_list)}, f)
    print(f"Saved to {global_thr_path}")

        # ========== 方差阈值的全局汇总 ==========  <<< 新增 >>>
    fold_var_auc_mat = np.stack(trainer.fold_var_auc_mat, axis=0)   # (k, T)
    fold_var_cov_mat = np.stack(trainer.fold_var_cov_mat, axis=0)

    mean_var_auc = np.nanmean(fold_var_auc_mat, axis=0)
    mean_var_cov = np.nanmean(fold_var_cov_mat, axis=0)

    best_idx_var = np.nanargmax(mean_var_auc)
    global_best_var_thr = trainer.var_thr_candidates[best_idx_var]
    print(f'\n>>> Global best VAR threshold across {args.k} folds = {global_best_var_thr:.6f}')

    # 画方差 Coverage–AUC 曲线
    plt.figure(figsize=(6,5), dpi=100)
    plt.plot(mean_var_cov, mean_var_auc, marker='o')
    plt.xlabel("Coverage (var ≤ thr)")
    plt.ylabel("AUC")
    plt.title("Average Val AUC-Coverage (variance filter)")
    plt.grid(ls='--', lw=0.5)
    var_fig_path = os.path.join(args.results_dir, "val_auc_vs_coverage_var.png")
    plt.savefig(var_fig_path, bbox_inches='tight')
    plt.close()
    print(f"Variance-curve figure saved to {var_fig_path}")

    # === 用 global_best_var_thr 评测 test 集 ===
    glob_var_auc_list, var_cov_list = [], []
    for i in folds:
        stat_path = os.path.join(args.results_dir, f"test_stats_fold{i}.npz")
        stat = np.load(stat_path)
        s_var   = stat["s_var"]          # 方差
        labels  = stat["labels"]
        probs   = stat["probs"]

        mask = s_var <= global_best_var_thr
        coverage_var = mask.mean()
        if coverage_var == 0:
            auc_sel = np.nan
        else:
            if args.n_classes == 2:
                auc_sel = safe_roc_auc(labels[mask], probs[mask, 1])
            else:
                auc_sel = roc_auc_score(labels[mask], probs[mask], multi_class="ovr")

        glob_var_auc_list.append(auc_sel)
        var_cov_list.append(coverage_var)
        print(f"Fold {i}: selective AUC (var) = {auc_sel:.4f}  @ coverage = {coverage_var:.2%}")

    print("\n>>> Mean selective AUC (var) across folds: "
          f"{np.nanmean(glob_var_auc_list):.4f} ± {np.nanstd(glob_var_auc_list):.4f}")
    print(">>> Mean coverage (var): "
          f"{np.nanmean(var_cov_list):.2%}")

    global_var_thr_path = os.path.join(args.results_dir, "global_var_thr_auc_and_coverage.json")
    with open(global_var_thr_path, "w") as f:
        json.dump({"var_thr": global_best_var_thr,
                   "global_test_auc_var": np.nanmean(glob_var_auc_list),
                   "global_test_coverage_var": np.nanmean(var_cov_list)}, f)
    print(f"Saved to {global_var_thr_path}")


    print("finished!")
    print("end script")