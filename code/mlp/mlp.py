# Train on MNIST
import csv
import torch.nn as nn
import torch.optim as optim
import torch

from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import torchvision.transforms as transforms
from sklearn.preprocessing import MinMaxScaler

feature_names = ''
class CSVDataset(Dataset):
    def __init__(self, filepath):
        # Load the data
        self.data = pd.read_csv(filepath)
        # Remove the first 3 columns
        columns_to_remove =['MRN','status','Indicator for being a control to case patient','Deepest Lesion (Recent & Previous) SOLAREL',
                 'Deepest Lesion (Recent & Previous) LYMPHCT_new',
                 'blgm_1','blgm_2','Deepest Lesion (Recent & Previous) LYMPHCT', 'Deepest Lesion (Recent & Previous) Solar_new',
                 'Solar Elastosis - Control/Case Defining Lesion - Most Recent Dx', 'Solar Elastosis Lesion 1 recoded with absent=0', 
                 'ulc1', 'ulc2', 'Pathologist performing Slide Review - Control/Case Defining Lesion - Most Recent Dx',
                 'Deepest Lesion (Recent & Previous) reviewed?','Dx most recent - calculated from DOB and path report'
                ]
        final_data = self.data.drop(columns=columns_to_remove)

        # Extract features and labels
        scales = MinMaxScaler(feature_range=(0, 1))
        self.features = (torch.tensor(
            scales.fit_transform(final_data.values)).float())
        # 创造一个映射从FeatureX到实际的列名
        global feature_names
        feature_names = final_data.columns

        # 假设shap_values_df是你的一个DataFrame，其中包含SHAP值，列名为Feat0 
        self.labels = torch.tensor(
            self.data['Deepest Lesion (Recent & Previous) SOLAREL'].values).long()-1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Example usage
dataset = CSVDataset(r'Gem_processed_new_two_classes.csv')
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_size, hidden_sizes[0]), nn.ReLU()]
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Define loss
criterion = nn.CrossEntropyLoss()
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm
import shap
from sklearn.model_selection import train_test_split

## Define the KFold cross-validator
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Define the model path

# Early stopping and model performance tracking
patience = 90
best_val_acc = 0.0


# best_model_path = "best_model.pth"
best_model_path = "best_mlp_v2.pth"

def predict(data):
    return model(data).detach().cpu().numpy()


for fold, (train_val_idx, test_idx) in enumerate(kf.split(dataset)):
    # Further split the train_val set into train and validation sets
    # Ensuring approximately 90% for train and 10% for validation of the train_val set
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.11, random_state=42)  # 0.11 of 90% is about 10% of the whole
    # Define model
    model = MLP(60, [16], 3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to('cpu')
    counter = 0
    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    train_subset = Subset(dataset, train_idx)
    valid_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)

    trainloader = DataLoader(train_subset, batch_size=256, shuffle=True)
    valloader = DataLoader(valid_subset, batch_size=256, shuffle=False)
    testloader = DataLoader(test_subset, batch_size=256, shuffle=False)

    # Prepare to log metrics
    train_losses, val_losses, test_losses = [], [], []
    train_accuracies, val_accuracies, test_accuracies = [], [], []
    #model.load_state_dict(torch.load("best_mlp_v2.pth"))
    #model.eval()

    # 提取所有测试数据
    #X_test = torch.cat([data for data, _ in testloader], 0)
    #y_test = torch.cat([target for _, target in testloader], 0)

    # 选择一个合适的背景数据集（这里我们用X_test的一部分）
    #background = X_test  # 可以根据实际情况调整数量

    # 创建SHAP DeepExplainer对象
    #explainer = shap.DeepExplainer(model, background)

    # 计算SHAP值
    #shap_values = explainer.shap_values(X_test)
    # 生成摘要图
    #shap.summary_plot(shap_values, X_test.numpy(), show=False, feature_names=feature_names)
    #plt.gcf().set_size_inches(30, 30)  # 更改图表尺寸为12x8英寸

    # 旋转列名标签和缩小字号
    #plt.xticks(rotation=45, fontsize=4)  # 旋转45度，设置字号为8
    #plt.savefig('summary_plot.svg', format='svg')  # 保存为SVG格式
    # break

    for epoch in range(4000):
        # Train
        model.train()
        epoch_loss, epoch_accuracy = 0, 0
        with tqdm(trainloader, desc=f"Fold {fold+1} Epoch {epoch+1}") as pbar:
            for features, labels in pbar:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(features)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                accuracy = (output.argmax(dim=1) == labels).float().mean()
                epoch_loss += loss.item()
                epoch_accuracy += accuracy.item()
                pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item())

        train_losses.append(epoch_loss / len(trainloader))
        train_accuracies.append(epoch_accuracy / len(trainloader))

        # Validation
        model.eval()
        val_loss, val_accuracy = 0, 0
        with torch.no_grad():
            for features, labels in valloader:
                features, labels = features.to(device), labels.to(device)
                output = model(features)
                val_loss += criterion(output, labels).item()
                val_accuracy += (output.argmax(dim=1) == labels).float().mean().item()

        val_loss /= len(valloader)
        val_accuracy /= len(valloader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        # test
        test_loss, test_accuracy = 0, 0
        with torch.no_grad():
            for features, labels in testloader:
                features, labels = features.to(device), labels.to(device)
                output = model(features)
                test_loss += criterion(output, labels).item()
                test_accuracy += (output.argmax(dim=1) == labels).float().mean().item()

        test_loss /= len(testloader)
        test_accuracy /= len(testloader)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        # Log epoch results
        print(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]}, Train Accuracy: {train_accuracies[-1]}")
        print(f"Epoch {epoch+1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")
        print(f"Epoch {epoch+1}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
        # Early stopping and learning rate scheduler
        scheduler.step(val_loss)
        if val_accuracy > best_val_acc:
            print("Saving new best model")
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            best_val_acc = val_accuracy
            best_test_acc = test_accuracy
    with open('training_metrics.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Fold', 'Val Accuracy', 'Test Accuracy'])
        writer.writerow([fold, max(val_accuracies), max(test_accuracies)])
        #     counter = 0
        # else:
        #     counter += 1
        #     if counter >= patience:
        #         print("Early stopping triggered")
        #         break

    # Optionally, save the logged metrics to a file or analyze them further
