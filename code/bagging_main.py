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
    def __init__(self, base_model, args, n_estimators=10):
        self.base_model = base_model
        self.args = args
        self.n_estimators = n_estimators
        self.models = []
        self.bootstrap_splits = []

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

        # Evaluate ensemble
        self.evaluate_ensemble(test_loader, cur, writer)

    def evaluate_ensemble(self, test_loader, cur, writer):
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
            auc = roc_auc_score(all_labels, avg_probs[:, 1])
            auc_pr = average_precision_score(all_labels, avg_probs[:, 1])  # AUC-PR for binary
        else:
            auc = roc_auc_score(all_labels, avg_probs, multi_class='ovr')
            auc_pr = average_precision_score(all_labels, avg_probs, average='macro')  # AUC-PR for multiclass

        print(f'Test AUC of Bagging Ensemble: {auc:.4f}')
        print(f'Test AUC-PR of Bagging Ensemble: {auc_pr:.4f}')
        
        # Log to writer if available
        if writer:
            writer.add_scalar('final/ensemble_auc', auc, cur)
            writer.add_scalar('final/ensemble_auc_pr', auc_pr, cur)

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
                                label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
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

    print("finished!")
    print("end script")