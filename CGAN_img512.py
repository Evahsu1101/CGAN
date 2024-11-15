# -*- coding: utf-8 -*-
"""
Created on Tue May 28 22:02:46 2024

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 06:01:38 2024

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 00:55:14 2023

@author: User
"""

import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
import os
# import pandas as pd  
# import numpy as np  
# import cv2  

# from scipy.stats import entropy
# from torchvision.models import inception_v3
# from torchvision.transforms import ToPILImage
# from scipy.linalg import sqrtm
from torchvision.models import inception_v3
import numpy as np
from scipy.stats import entropy
from scipy.linalg import sqrtm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

 
    
    
# 數據集定義
class CustomDataset(Dataset):
    # 初始化
    def __init__(self, root_folder, transform=None):
        self.root_folder = root_folder
        self.transform = transform

        self.image_files = []
        self.labels = []

        # 加載數據和標籤
        for label in os.listdir(root_folder):
            label_path = os.path.join(root_folder, label)
            if os.path.isdir(label_path):
                for image_file in os.listdir(label_path):
                    self.image_files.append(os.path.join(label_path, image_file))
                    self.labels.append(int(label))  # 文件夾名稱為0 OR 1

    # 獲取數據及長度
    def __len__(self):
        return len(self.image_files)

    # 獲取數據項
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# 圖像預處理256*256
transform = transforms.Compose([
    transforms.Resize((512, 512)),  
    transforms.ToTensor(), 
])


# 定義路徑
root_folder = r"./classification" #資料集輸入
output_folder = r"./generation_test" #模型測試時觀察生成情況路徑
model_save_dir = r"./model" #模型存檔路徑
custom_dataset = CustomDataset(root_folder, transform=transform)

# dataloader
batch_size = 258 # 258/n 最好是整除比較不會出問題
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)


# 定義生成器網絡
class Generator(nn.Module):
    def __init__(self, latent_dim, condition_dim, img_channels, img_size):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.img_channels = img_channels
        self.img_size = img_size

        # 嵌入層將類別條件轉換為條件向量

        self.fc = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, img_channels * img_size * img_size),
            nn.Tanh()
        )

    def forward(self, noise, conditions):
        # 將類別條件轉換為條件向量

        # 將隨機噪聲和條件連接起來
        x = torch.cat((noise, conditions), dim=1)
        x = self.fc(x)
        # 重新調整輸出為圖像大小
        x = x.view(x.size(0), self.img_channels, self.img_size, self.img_size)
        return x

# 定義判別器網絡
class Discriminator(nn.Module):
    def __init__(self, condition_dim, img_channels, img_size):
        super(Discriminator, self).__init__()

        self.condition_dim = condition_dim
        self.img_channels = img_channels
        self.img_size = img_size

        # 嵌入層將類別條件轉換為條件向量

        self.fc = nn.Sequential(
            nn.Linear(img_channels * img_size * img_size + condition_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, conditions):
        # 將類別條件轉換為條件向量


        # 將圖像和條件連接起來
        x = img.view(img.size(0), -1)
        x = torch.cat((x, conditions), dim=1)
        x = self.fc(x)
        return x


    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    



# 超參數設置
latent_size = 200
latent_dim = 200
condition_dim = 1 #需更改為條件數量
img_channels = 3
img_size = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 創建及初始化生成器及判別器
generator = Generator(latent_dim, condition_dim, img_channels, img_size).to(device)
discriminator = Discriminator(condition_dim, img_channels, img_size).to(device)

# 定義損失函數和優化器
criterion = nn.BCELoss().to(device)
import torch.optim as optim

# 初始化生成器和判别器的优化器，加入weight decay
optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))




def plot_generator_loss(generator_losses, title):
    plt.figure(figsize=(12, 8))
    plt.plot(generator_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


    
    
    
# 初始化紀錄的列表
overall_accuracies = []
class_0_accuracies = []
class_1_accuracies = []
generator_losses = []
discriminator_losses = []
is_scores_list = []  
fid_scores_list = [] 
lpips_scores = []  

#訓練參數
num_classes = 2
epochs = 10000
sample_interval = 100  # 10個EPOCH生成一張圖
num_images_to_generate = 100
sample_interval1 = 100
#訓練迴圈
for epoch in range(epochs):
    for batch_idx, (images, labels) in enumerate(data_loader):
        real_images = images.to(device)

        conditions = labels.to(device)
        conditions = labels.unsqueeze(1).to(device)

        # 將判別器梯度清為0，以便新一輪的反向傳播和優化
        optimizer_D.zero_grad()

        #生成假圖片
        z = torch.randn(batch_size, latent_size).to(device) # 生成隨機的噪音張量(10,106)
        fake_images = generator(z, conditions)#將隨機噪音z和條件傳回GENERATOR 生成假圖片
        inputdata = torch.cat((z, conditions), dim=1)
        # 判別器對於真實圖像的輸出
        real_outputs = discriminator(real_images, conditions)#判別器對於Real_images進行評估conditions一起放進去輔助評估
        real_targets = torch.ones_like(real_outputs)#將Real_outputs鑑別器的輸出為1

        # 判別器對於假圖像的輸出
        fake_outputs = discriminator(fake_images.detach(), conditions)#判別器對於fake_images進行評估conditions一起放進去輔助評估
        fake_targets = torch.zeros_like(fake_outputs)#將fake_outputs鑑別器的輸出為0

        #判別器的損失函數
        loss_real = criterion(real_outputs, real_targets)#判別器對於真實圖圖像的輸出與真實標籤的差異
        loss_fake = criterion(fake_outputs, fake_targets)#判別器對於假圖像的輸出與假標籤的差異
        loss_D = loss_real + loss_fake #總LOSS判別器
        #使判別器更好區分真實及生成圖像
        loss_D.backward()
        optimizer_D.step()#更新判別器的參數

        #將生成器梯度清為0，以便新一輪的反向傳播和優化
        optimizer_G.zero_grad()

        # 計算假圖像的判別結果
        fake_outputs = discriminator(fake_images, conditions)
        real_targets = torch.ones_like(fake_outputs)  # 生成器希望生成的圖被判別器定義為真實的

        # 生成器的損失
        loss_G = criterion(fake_outputs, real_targets)

        loss_G.backward()
        optimizer_G.step()
        generator_losses.append(loss_G.item())
        discriminator_losses.append(loss_D.item())
        # 每個EPOCH計算準確率
       # 每个epoch的信息输出
        print('Epoch [{}/{}], Step [{}/{}]'.format(epoch+1, epochs, epoch+1, len(data_loader)))


        # 保存生成器的生成的圖像和標籤
        if epoch % sample_interval == 0:
            with torch.no_grad():
             # 將生成的圖像和標籤丟入變量中
               samples_with_labels = [{'image': fake_images[i], 'label': conditions[i].cpu().numpy()} for i in range(batch_size)]
               save_image(fake_images.data[:1], "{}/Test2epoch_{}.png".format(output_folder, epoch), nrow=1, normalize=True)
               
        # 每1000个Epoch保存一次模型
        if (epoch + 1) % 1000 == 0:
            generator_save_path = os.path.join(model_save_dir, 'CGAN C T2 generator_epoch_{}.pth'.format(epoch + 1))

            torch.save(generator.state_dict(), generator_save_path)
            
            print("Models saved at epoch {}".format(epoch + 1))




# 定义保存模型的文件夹
save_folder = r"./model"
os.makedirs(save_folder, exist_ok=True)  # 确保文件夹存在

# 定义保存的文件路径
generator_path = os.path.join(save_folder, 'CGAN T2 generator EPOCH5000.pth')


# 保存模型
torch.save(generator.state_dict(), generator_path)


print("Generator model saved at {}".format(generator_path))


              
                
#訓練結束畫圖      
plot_generator_loss(generator_losses, 'CGAN Training Losses')


# 保存模型參數
torch.save(generator.state_dict(), 'CGAN T2 generatorEPOCH5000.pth')
#torch.save(discriminator.state_dict(), 'CGAN T2 discriminatorEPOCH5000.pth')



import os
import torch
from torchvision.utils import save_image

def generate_and_save_images(generator, category, num_images, latent_dim, output_dir, device):

    os.makedirs(output_dir, exist_ok=True)
    # 設置生成器為評估模式
    generator.eval()

    noise = torch.randn(num_images, latent_dim, device=device)

    labels = torch.full((num_images, 1), category, device=device, dtype=torch.float32)  # 保持与训练时的标签维度一致
    # 生成圖片
    with torch.no_grad():
        generated_images = generator(noise, labels).detach().cpu()
    # 保存圖片
    for i, img in enumerate(generated_images):
        save_image(img, os.path.join(output_dir, 'category_{}_image_{}.png'.format(category, i)))



latent_dim = 200  

#設置圖像輸出路徑資料夾
output_dir = r"./generation4"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加載生成器模型
generator = Generator(latent_dim, condition_dim=1, img_channels=3, img_size=512).to(device)
generator.load_state_dict(torch.load('CGAN T2 generatorEPOCH5000.pth', map_location=device))

# 生成類別0的圖片
generate_and_save_images(generator, category=0, num_images=100, latent_dim=latent_dim, output_dir=output_dir, device=device)
# 生成類別1的圖片
generate_and_save_images(generator, category=1, num_images=100, latent_dim=latent_dim, output_dir=output_dir, device=device)
