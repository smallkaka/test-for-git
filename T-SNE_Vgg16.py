import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image

# 定义数据目录和域列表
data_dir = '/home/shiya.xu/domain-github/data/EX'  # 替换为您的数据目录
domains = ['domain1', 'domain2', 'domain3']

# 定义颜色映射，为每个域分配不同的颜色
domain_colors = {'domain1': 'r', 'domain2': 'g', 'domain3': 'b'}

# 定义数据转换
image_size = (960, 960)
batch_size = 16

data_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # 使用ImageNet的均值和标准差
])


# 自定义数据集类
class CustomDataset(Dataset):

    def __init__(self, data_dir, domain, data_type, transform=None):
        self.data_dir = os.path.join(data_dir, domain, data_type, 'image')
        self.transform = transform
        self.image_paths = [
            os.path.join(self.data_dir, filename)
            for filename in os.listdir(self.data_dir)
        ]
        self.domain = domain  # 记录域信息

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.domain  # 返回图像和域信息


# 创建数据加载器
all_loaders = []
all_colors = []  # 记录每个样本的颜色

for domain in domains:
    domain_loaders = {}
    for data_type in ['train', 'test']:
        dataset = CustomDataset(data_dir,
                                domain,
                                data_type,
                                transform=data_transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        domain_loaders[data_type] = data_loader

        # 为每个样本分配相应的颜色和域标签
        all_colors.extend([domain_colors[domain]] * len(dataset))

    all_loaders.append(domain_loaders)

# 加载预训练的VGG16模型
model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
model = model.features

# 准备特征
all_features = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.eval()

# 多次运行t-SNE并平均的方法
n_runs = 10  # 运行次数
tsne_features_all = []

for _ in range(n_runs):
    all_features = []  # 重新初始化特征列表
    with torch.no_grad():
        for domain_loader in all_loaders:
            for data_loader in domain_loader.values():
                for images, _ in data_loader:  # 忽略域信息
                    images = images.to(device)  # 16, 3, 960, 960
                    features = model(images).cpu().numpy()
                    all_features.append(features)

    # 合并特征并进行t-SNE降维
    all_features = np.concatenate(all_features, axis=0)
    all_features_2d = np.reshape(all_features,
                                 (all_features.shape[0], -1))  # 将特征展平为2D
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_features = tsne.fit_transform(all_features_2d)
    tsne_features_all.append(tsne_features)

# 计算多次运行的t-SNE结果的平均值
tsne_features_all = np.array(tsne_features_all)
tsne_features_mean = np.mean(tsne_features_all, axis=0)

# 可视化t-SNE结果，为每个样本分配相应的颜色和域标签
plt.figure(figsize=(10, 8))
plt.scatter(tsne_features_mean[:, 0], tsne_features_mean[:, 1], c=all_colors)

# 在图的旁边标注域信息
for i, domain in enumerate(domains):
    plt.scatter([], [], color=domain_colors[domain], label=f'Domain {i + 1}')

plt.legend()
plt.title(
    't-SNE Visualization of Image Features from Multiple Domains (Averaged)')
plt.show()

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# import torch
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, Dataset
# from torchvision.datasets import ImageFolder
# from PIL import Image

# # 定义数据目录和域列表
# data_dir = '/home/shiya.xu/domain-github/data/EX'  # 替换为您的数据目录
# domains = ['domain1', 'domain2', 'domain3']

# # 定义颜色映射，为每个域分配不同的颜色
# domain_colors = {'domain1': 'r', 'domain2': 'g', 'domain3': 'b'}

# # 定义数据转换
# image_size = (960, 960)
# batch_size = 16

# data_transform = transforms.Compose([
#     transforms.Resize(image_size),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])  # 使用ImageNet的均值和标准差
# ])

# # 自定义数据集类
# class CustomDataset(Dataset):

#     def __init__(self, data_dir, domain, data_type, transform=None):
#         self.data_dir = os.path.join(data_dir, domain, data_type, 'image')
#         self.transform = transform
#         self.image_paths = [
#             os.path.join(self.data_dir, filename)
#             for filename in os.listdir(self.data_dir)
#         ]
#         self.domain = domain  # 记录域信息

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         img = Image.open(img_path).convert('RGB')
#         if self.transform:
#             img = self.transform(img)
#         return img, self.domain  # 返回图像和域信息

# # 创建数据加载器
# all_loaders = []
# all_colors = []  # 记录每个样本的颜色

# for domain in domains:
#     domain_loaders = {}
#     for data_type in ['train', 'test']:
#         dataset = CustomDataset(data_dir,
#                                 domain,
#                                 data_type,
#                                 transform=data_transform)
#         data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#         domain_loaders[data_type] = data_loader

#         # 为每个样本分配相应的颜色
#         all_colors.extend([domain_colors[domain]] * len(dataset))

#     all_loaders.append(domain_loaders)

# # 加载预训练的VGG16模型
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
# model = model.features

# # 准备特征
# all_features = []

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# model.eval()
# with torch.no_grad():
#     for domain_loader in all_loaders:
#         for data_loader in domain_loader.values():
#             for images, _ in data_loader:  # 忽略域信息
#                 images = images.to(device)  # 16, 3, 512, 512
#                 features = model(images).cpu().numpy()
#                 all_features.append(features)

# # 合并特征并进行t-SNE降维
# all_features = np.concatenate(all_features, axis=0)
# all_features_2d = np.reshape(all_features,
#                              (all_features.shape[0], -1))  # 将特征展平为2D
# tsne = TSNE(n_components=2, perplexity=30, random_state=42)
# tsne_features = tsne.fit_transform(all_features_2d)

# # 可视化t-SNE结果，为每个样本分配相应的颜色
# plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=all_colors)
# plt.title('t-SNE Visualization of Image Features from Multiple Domains')
# plt.show()
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# import torch
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, Dataset
# from torchvision.datasets import ImageFolder
# from PIL import Image

# # 定义数据目录和域列表
# data_dir = '/home/shiya.xu/domain-github/data/EX'  # 替换为您的数据目录
# domains = ['domain1', 'domain2', 'domain3']

# # 定义颜色映射，为每个域分配不同的颜色
# domain_colors = {'domain1': 'r', 'domain2': 'g', 'domain3': 'b'}

# # 定义数据转换
# image_size = (960, 960)
# batch_size = 16

# data_transform = transforms.Compose([
#     transforms.Resize(image_size),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]``)  # 使用ImageNet的均值和标准差
# ])

# # 自定义数据集类
# class CustomDataset(Dataset):

#     def __init__(self, data_dir, domain, data_type, transform=None):
#         self.data_dir = os.path.join(data_dir, domain, data_type, 'image')
#         self.transform = transform
#         self.image_paths = [
#             os.path.join(self.data_dir, filename)
#             for filename in os.listdir(self.data_dir)
#         ]
#         self.domain = domain  # 记录域信息

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         img = Image.open(img_path).convert('RGB')
#         if self.transform:
#             img = self.transform(img)
#         return img, self.domain  # 返回图像和域信息

# # 创建数据加载器
# all_loaders = []
# all_colors = []  # 记录每个样本的颜色

# for domain in domains:
#     domain_loaders = {}
#     for data_type in ['train', 'test']:
#         dataset = CustomDataset(data_dir,
#                                 domain,
#                                 data_type,
#                                 transform=data_transform)
#         data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#         domain_loaders[data_type] = data_loader

#         # 为每个样本分配相应的颜色和域标签
#         all_colors.extend([domain_colors[domain]] * len(dataset))

#     all_loaders.append(domain_loaders)

# # 加载预训练的VGG16模型
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
# model = model.features

# # 准备特征
# all_features = []

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# model.eval()
# with torch.no_grad():
#     for domain_loader in all_loaders:
#         for data_loader in domain_loader.values():
#             for images, _ in data_loader:  # 忽略域信息
#                 images = images.to(device)  # 16, 3, 512, 512
#                 features = model(images).cpu().numpy()
#                 all_features.append(features)

# # 合并特征并进行t-SNE降维
# all_features = np.concatenate(all_features, axis=0)
# all_features_2d = np.reshape(all_features,
#                              (all_features.shape[0], -1))  # 将特征展平为2D
# tsne = TSNE(n_components=2, perplexity=30, random_state=42)
# tsne_features = tsne.fit_transform(all_features_2d)

# # 可视化t-SNE结果，为每个样本分配相应的颜色和域标签
# plt.figure(figsize=(10, 8))
# plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=all_colors)

# # 在图的旁边标注域信息
# for i, domain in enumerate(domains):
#     plt.scatter([], [], color=domain_colors[domain], label=f'Domain {i + 1}')

# plt.legend()
# plt.title('t-SNE Visualization of Image Features from Multiple Domains')
# plt.show()
