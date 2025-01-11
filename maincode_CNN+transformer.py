import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from torch_geometric.nn import GCNConv
from torch_geometric.nn.dense import Linear as DenseLinear
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AdamW
from torchvision.transforms import transforms
from tqdm import tqdm
import torch.cuda
import xml.etree.ElementTree as ET
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import dgl
import numpy as np
import torch as th
from dgl.nn import RelGraphConv
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN
from torch.nn import MultiheadAttention
from scipy.stats import uniform, randint
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt



#############################################定义 BERT 模型和 tokenizer##############################################

#导入Biobert
model_path = './model_path/biobert'                     #这个要用相对路径，不要用绝对路径
biobert_tokenizer = AutoTokenizer.from_pretrained(model_path)
biobert_model = AutoModel.from_pretrained(model_path)


# #导入bert
# model_path_1 = './model_path/bert_pretrain'                     #这个要用相对路径，不要用绝对路径
# bert_tokenizer = AutoTokenizer.from_pretrained(model_path_1)
# bert_model = AutoModel.from_pretrained(model_path_1)



####################################################################################################################

#############################################读取数据################################################################

df_train = pd.read_csv('./data/ddi2013ms/augmented_train_data.tsv', sep='\t')
df_dev = pd.read_csv('./data/ddi2013ms/dev.tsv', sep='\t')
df_test = pd.read_csv('./data/ddi2013ms/test.tsv', sep='\t')
print("read")

# print("训练集数据量：", df_train.shape)
# print("验证集数据量：", df_dev.shape)
# print("测试集数据量：", df_test.shape)

####################################################################################################################

#######################################################定义模型参数##################################################
#定义训练设备，默认为GPU，若没有GPU则在CPU上训练
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备

num_label=5

# 定义模型参数
max_length = 300
batch_size = 32


# #############################################定义数据集和数据加载器###################################################
# # 定义数据集类
# 定义标签到整数的映射字典
label_map = {
    'DDI-false': 0,
    'DDI-effect': 1,
    'DDI-mechanism': 2,
    'DDI-advise': 3,
    'DDI-int': 4
    # 可以根据你的实际标签情况添加更多映射关系
}

# 定义数据集类
class DDIDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = str(self.data['sentence'][idx])
        label_str = self.data['label'][idx]
        label = label_map[label_str]

        encoding = self.tokenizer(sentence, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')

        # # 输出检查语句
        # print(f"txt_intra_matrix shape: {txt_intra_matrix.shape}")

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 定义数据加载器
def create_data_loader(df, tokenizer, max_length, batch_size):
    dataset = DDIDataset(
        dataframe=df,
        tokenizer=tokenizer,

        max_length=max_length
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True  # 设置 drop_last=True 来丢弃最后一个不满足批次大小的批次,因为我们在LSTM和GCN维度转换时，出现了维度不匹配问题，找了很久原因，发现是在最后batch时，数据只有4条，导致维度出错
    )


# # 加载数据集和数据加载器
train_data_loader = create_data_loader(df_train, biobert_tokenizer, max_length, batch_size)
dev_data_loader = create_data_loader(df_dev, biobert_tokenizer, max_length, batch_size)
test_data_loader = create_data_loader(df_test, biobert_tokenizer, max_length, batch_size)

# for batch in test_data_loader:
#     print(batch)
#     break  # 这将打印第一批数据并中断循环。


class BioMedRelationExtractor(nn.Module):
    def __init__(self, d_model=768, cnn_out_channels=256, num_heads=8, num_classes=5, dropout=0.2):
        super(BioMedRelationExtractor, self).__init__()

        # BioBERT 嵌入层
        self.bert = biobert_model  # 你的 BioBERT 模型
        
        # CNN 层：用于提取局部特征
        self.cnn1 = nn.Conv1d(in_channels=d_model, out_channels=cnn_out_channels, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv1d(in_channels=cnn_out_channels, out_channels=cnn_out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Dropout层：防止CNN层过拟合
        self.dropout_cnn = nn.Dropout(dropout)
        
        # 自定义的 Transformer 层：用于提取全局上下文
        self.attn_layer = nn.MultiheadAttention(embed_dim=cnn_out_channels, num_heads=num_heads, dropout=dropout)

        # Dropout层：防止Transformer层过拟合
        self.dropout_transformer = nn.Dropout(dropout)
        
        # 输出分类层
        self.fc = nn.Linear(cnn_out_channels, num_classes)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, labels):
        # Step 1: 获取 BioBERT 输出
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output[0]  # shape: [batch_size, seq_len, hidden_size]
        
        # Step 2: CNN 层提取局部特征
        sequence_output = sequence_output.permute(0, 2, 1)  # [batch_size, hidden_size, seq_len]
        cnn_output = F.relu(self.cnn1(sequence_output))
        cnn_output = self.pool(cnn_output)  # Pooling after CNN
        cnn_output = F.relu(self.cnn2(cnn_output))
        
        # Step 3: Transformer 层提取全局上下文
        cnn_output = cnn_output.permute(2, 0, 1)  # [seq_len, batch_size, cnn_out_channels]
        
        # 使用 MultiheadAttention 计算注意力
        attn_output, attn_weights = self.attn_layer(cnn_output, cnn_output, cnn_output)
        
        # Step 4: 池化操作
        attn_output = attn_output.mean(dim=0)  # [batch_size, cnn_out_channels]
        
        # Step 5: 分类层
        logits = self.fc(attn_output)

        return logits, attn_weights




##################################################################################################

# 计算FocalLoss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, outputs, labels):
        ce_loss = F.cross_entropy(outputs, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            at = torch.tensor([self.alpha[i] for i in labels], device=labels.device)
            focal_loss = at * focal_loss

        return focal_loss.mean()

# 计算类分布并调整alpha
num_samples = [21745, 2972, 2326, 1450, 354]
inverse_samples = [1 / s for s in num_samples]
total_weight = sum(inverse_samples)
normalized_alpha = [weight / total_weight for weight in inverse_samples]
print("Alpha weights:", normalized_alpha)

# 实例化损失函数
criterion = FocalLoss(gamma=2, alpha=normalized_alpha)

####################################################################################################


#分类器
class Classifier(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=5):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, features):
        logits = self.fc(features)  
        return logits

# Visualize the attention weights
def plot_attention_weights(attn_weights, sentence):
    sns.heatmap(attn_weights.cpu().detach().numpy(), cmap='viridis')
    plt.title(f'Attention Weights for: {sentence}')
    plt.xlabel('Token')
    plt.ylabel('Token')
    plt.show()



# 在训练和测试之前定义 true_labels 和 predicted_probs
true_labels = []
predicted_probs = []


# 训练代码
def train_model(model, train_data_loader, optimizer, criterion, device):

    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    epoch_true_labels = []
    epoch_pred_labels = []

    for batch in train_data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)


        optimizer.zero_grad()
        logits, attn_scores = model(input_ids, attention_mask, labels)    # 从模型中获取 outputs 和 attn_scores

        loss = criterion(logits, labels)  
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total_preds += labels.size(0)
        correct_preds += (predicted == labels).sum().item()
        
        epoch_true_labels.extend(labels.cpu().numpy())
        epoch_pred_labels.extend(predicted.cpu().numpy())

        # 记录每个 batch 的真实标签和预测概率
        true_labels.extend(labels.cpu().numpy())
        predicted_probs.extend(F.softmax(logits, dim=1).detach().cpu().numpy())  # Use detach() here

    train_loss = running_loss / len(train_data_loader)
    train_acc = correct_preds / total_preds
    
    # 计算混淆矩阵和 F1 值
    conf_matrix = confusion_matrix(epoch_true_labels, epoch_pred_labels)  # Use epoch_pred_labels here
    accuracy = accuracy_score(epoch_true_labels, epoch_pred_labels)
    precision = precision_score(epoch_true_labels, epoch_pred_labels, average='weighted', zero_division=1)
    recall = recall_score(epoch_true_labels, epoch_pred_labels, average='weighted', zero_division=1)
    f1 = 2*precision*recall/(precision+recall)
    
    return train_loss, train_acc, conf_matrix, f1


# 测试代码
def test_model(model, test_data_loader, criterion, device):

    model.eval()
    epoch_true_labels = []
    epoch_pred_labels = []

    with torch.no_grad():
        for batch in test_data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits, attn_scores = model(input_ids, attention_mask, labels)
            _, predicted = torch.max(logits, 1)

            epoch_true_labels.extend(labels.cpu().numpy())
            epoch_pred_labels.extend(predicted.cpu().numpy())

    # 计算混淆矩阵、准确率、精确率、召回率和 F1 值
    conf_matrix = confusion_matrix(epoch_true_labels, epoch_pred_labels)
    accuracy = accuracy_score(epoch_true_labels, epoch_pred_labels)
    precision = precision_score(epoch_true_labels, epoch_pred_labels, average='weighted', zero_division=1)
    recall = recall_score(epoch_true_labels, epoch_pred_labels, average='weighted', zero_division=1)
    f1 = 2*precision*recall/(precision+recall)

    # 计算每个类别的F1值
    class_report = classification_report(epoch_true_labels, epoch_pred_labels, output_dict=True, zero_division=1)
    f1_per_class = {label: metrics['f1-score'] for label, metrics in class_report.items() if label.isdigit()}
    
    return conf_matrix, accuracy, precision, recall, f1, f1_per_class


#模型实例化
model = BioMedRelationExtractor().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
#criterion = nn.CrossEntropyLoss()
criterion = FocalLoss(gamma=2, alpha=None)

# 训练模型
num_epochs = 20

# 存储训练过程中每个 epoch 的结果
epoch_train_losses = []
epoch_train_accuracies = []
epoch_train_f1_scores = []
epoch_train_conf_matrices = []

# 打开文件，以追加模式（'a'）写入
with open('./figure/training_results.txt', 'a') as f:
    for epoch in tqdm(range(num_epochs), desc="Training Progress"):

        train_loss, train_acc, conf_matrix, f1 = train_model(model, train_data_loader, optimizer, criterion, device)

        #保存结果文件
        f.write(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, F1 Score: {f1:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(conf_matrix) + '\n')

        #打印输出
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, F1 Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        
        # 保存每个 epoch 的结果用于后续可视化
        epoch_train_losses.append(train_loss)
        epoch_train_accuracies.append(train_acc)
        epoch_train_f1_scores.append(f1)
        epoch_train_conf_matrices.append(conf_matrix)



with open('./figure/test_results.txt', 'w') as f:
    test_conf_matrix, test_accuracy, test_precision, test_recall, test_f1, test_f1_per_class = test_model(model, test_data_loader, criterion, device)
    f.write("Test Results:\n")
    f.write("Confusion Matrix:\n")
    f.write(str(test_conf_matrix) + '\n')
    f.write("Accuracy: " + str(test_accuracy) + '\n')
    f.write("Precision: " + str(test_precision) + '\n')
    f.write("Recall: " + str(test_recall) + '\n')
    f.write("F1 Score: " + str(test_f1) + '\n')
    f.write("F1 Score per Class:\n")
    for label, f1 in test_f1_per_class.items():
        f.write(f"Class {label}: {f1:.4f}\n")
    print("Test Results:")
    print("Confusion Matrix:")
    print(test_conf_matrix)
    print("Accuracy:", test_accuracy)
    print("Precision:", test_precision)
    print("Recall:", test_recall)
    print("F1 Score:", test_f1)
    print("F1 Score per Class:")
    for label, f1 in test_f1_per_class.items():
        print(f"Class {label}: {f1:.4f}")



##############################################画图####################################################
# 计算每个类别的 AUC
# 假设你有 `true_labels` 和 `predicted_probs` 以及 `label_map`
num_classes = 5  # 根据你的情况调整
fpr = dict()
tpr = dict()
roc_auc = dict()

# 对于每个类别，计算fpr, tpr和AUC
for i in range(num_classes):
    # 获取每个类的真值和预测概率
    class_true_labels = [1 if true_label == i else 0 for true_label in true_labels]
    class_predicted_probs = [probs[i] for probs in predicted_probs]

    # 计算 fpr 和 tpr
    fpr[i], tpr[i], _ = roc_curve(class_true_labels, class_predicted_probs)
    roc_auc[i] = auc(fpr[i], tpr[i])


##################################训练集结果画图####################################################
# 画训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_train_losses, marker='o', label='Train Loss')
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.title('Training Loss Over Epochs', fontsize=20)
plt.legend(fontsize=18)
plt.grid(True, linestyle="--", alpha=0.8)
plt.tight_layout()
plt.savefig('./figure/training_loss_over_epochs.png', dpi=300)
plt.show()

# 画训练准确率曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_train_accuracies, marker='o', label='Train Accuracy')
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.title('Training Accuracy Over Epochs', fontsize=20)
plt.legend(fontsize=18)
plt.grid(True, linestyle="--", alpha=0.8)
plt.tight_layout()
plt.savefig('./figure/training_accuracy_over_epochs.png', dpi=300)
plt.show()

# 画训练 F1 分数曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_train_f1_scores, marker='o', label='Train F1 Score')
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('F1 Score', fontsize=20)
plt.title('Training F1 Score Over Epochs', fontsize=20)
plt.legend(fontsize=18)
plt.grid(True, linestyle="--", alpha=0.8)
plt.tight_layout()
plt.savefig('./figure/training_f1_score_over_epochs.png', dpi=300)
plt.show()




################################################测试集结果画图##############################
# 画混淆矩阵热力图
plt.figure(figsize=(10, 8))

#颜色方案：Blues：蓝色；viridis：视觉效果良好的色系；coolwarm：显示对比强烈的颜色；YlGnBu：绿色-蓝色渐变；RdBu：红-蓝渐变
sns.heatmap(test_conf_matrix, annot=True, fmt="d", cmap="YlGnBu", xticklabels=label_map.keys(), yticklabels=label_map.keys(), annot_kws={"size": 20})
plt.xlabel('Predicted Label', fontsize=20)
plt.ylabel('True Label', fontsize=20)
plt.title('Confusion Matrix', fontsize=20)
plt.xticks(rotation=45, ha='right')   # 设置x轴标签，旋转一定角度以避免重叠（如果需要）
plt.yticks(rotation=0)           # 设置y轴标签水平显示
plt.tight_layout()  # 调整子图布局以适应标签
plt.legend(fontsize=20)
plt.savefig('./figure/confusion_matrix_heatmap.png', dpi=300)  # 保存混淆矩阵热力图
plt.show()

# 画准确率
plt.figure(figsize=(10, 8))
bar_width = 0.3  # # 设置柱子的宽度,可以根据需要调整这个值
plt.bar(range(len(label_map)), [test_accuracy]*len(label_map), color='skyblue', width=bar_width, align='center')
plt.xlabel('Class', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.title('', fontsize=20)
plt.xticks(range(len(label_map)), label_map.keys(), rotation=45, ha='right')
plt.tight_layout()  # 调整子图布局以适应标签
plt.savefig('./figure/accuracy_by_class.png', dpi=300)  # 保存准确率图
plt.show()

#画AUC曲线图
plt.figure(figsize=(10, 6))

for i, label in enumerate(label_map):
    plt.plot(fpr[i], tpr[i], label=f'{label} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)  # 随机分类器的线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.title('Receiver Operating Characteristic (ROC)', fontsize=20)
plt.legend(loc='lower right')
plt.tight_layout()  # 调整子图布局以适应标签
plt.savefig('./figure/auc_by_class.png', dpi=300)
plt.show()
