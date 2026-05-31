import torch
from torch import nn
from torchvision import transforms,datasets
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary
import os

class mixed_net(nn.Module):
    def __init__(self):
        super(mixed_net,self).__init__()
        # 【第一层：卷积层1】
        # 输入通道: 3 (RGB图像), 输出通道: 16, 卷积核: 3x3, 步长: 1, 填充: 1
        # 输入尺寸: [Batch_Size, 3, 64, 64]
        # 输出尺寸: [Batch_Size, 16, 64, 64]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 【第二层：卷积层2】
        # 输入通道: 16, 输出通道: 32, 卷积核: 3x3, 步长: 1, 填充: 1
        # 输入尺寸: [Batch_Size, 16, 32, 32] (经过第一次池化后)
        # 输出尺寸: [Batch_Size, 32, 32, 32]
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # 【第三层：卷积层3】
        # 输入通道: 32, 输出通道: 64, 卷积核: 3x3, 步长: 1, 填充: 1
        # 输入尺寸: [Batch_Size, 32, 16, 16] (经过第二次池化后)
        # 输出尺寸: [Batch_Size, 64, 16, 16]

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # 最大池化层 
        # 窗口大小: 2x2, 步长: 2。每次调用都会将特征图的高和宽减半。
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 【第四层：全连接层1】
        # 输入特征数: 64 * 8 * 8 = 4096 (由第三次池化后的特征图展平得到)
        # 输出特征数: 128
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.4)
        
        # 【第五层：全连接层2】
        # 输入特征数: 128, 输出特征数: 3 (对应 blue, red, yellow 三个分类)
        self.fc2 = nn.Linear(128, 3)
    
    def forward(self, x):
        # 初始输入 x 尺寸: [Batch_Size, 3, 64, 64]

         # 第一阶段：卷积 -> 批归一化 -> 激活 -> 池化
        x = self.conv1(x)           #  [Batch_Size, 16, 64, 64] ，通道从 3 扩展到 16)
        x = self.bn1(x)             #  [Batch_Size, 16, 64, 64] ，归一化分布
        x = F.relu(x)               #  [Batch_Size, 16, 64, 64] ，激活函数
        x = self.pool(x)            #  [Batch_Size, 16, 32, 32] ，空间分辨率减半
                 
        
        # 第二阶段：卷积 -> 批归一化 -> 激活 -> 池化
        x = self.conv2(x)           #  [Batch_Size, 32, 32, 32] 通道从 16 扩展到 32
        x = self.bn2(x)             #  [Batch_Size, 32, 32, 32]
        x = F.relu(x)               #  [Batch_Size, 32, 32, 32]
        x = self.pool(x)            #  [Batch_Size, 32, 16, 16] 空间分辨率减半  
        
        # 第三阶段：卷积 -> 批归一化 -> 激活 -> 池化
        x = self.conv3(x)           #  [Batch_Size, 64, 16, 16] 通道从 32 扩展到 64
        x = self.bn3(x)             #  [Batch_Size, 64, 16, 16]
        x = F.relu(x)               #  [Batch_Size, 64, 16, 16]
        x = self.pool(x)            #  [Batch_Size, 64, 8, 8]   空间分辨率减半       
        
        # 扁平化操作，准备接入全连接层
        x = x.view(x.size(0), -1)    #  [Batch_Size, 4096]        64 * 8 * 8 = 4096
        
        # 第四阶段：全连接层 1
        x = self.fc1(x)             #  [Batch_Size, 128]
        x = F.relu(x)               #  [Batch_Size, 128]
        x = self.dropout(x)         #  [Batch_Size, 128]    随机失活，抑制过拟合  

        # 第五阶段：全连接层 2 (分类输出)
        x = self.fc2(x)             # 尺寸变化: [Batch_Size, 3]        得到3个分类的原始预测分数
        
        return x

        return x

if __name__ == "__main__":
    #图像转换
    transforms = transforms.Compose(
        [
            transforms.Resize([64, 64]),
            transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转提升泛化能力
            transforms.RandomRotation(15),          # 随机旋转，这两行为ai建议
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    
    #超参数设置
    BATCH_SIZE = 1024
    EPOCH = 200

    #加载数据
    trainset = datasets.ImageFolder(root=r'dataset/train',transform=transforms)
    testset1 = datasets.ImageFolder(root=r'dataset/test1',transform=transforms)
    testset2 = datasets.ImageFolder(root=r'dataset/test2',transform=transforms)

    print(f"训练集图片数量: {len(trainset)}")
    print(f"测试集1图片数量: {len(testset1)}")
    print(f"测试集2图片数量: {len(testset2)}")
    
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader1 = DataLoader(testset1, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader2 = DataLoader(testset2, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    #创建网络
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = mixed_net().to(device)
    
    #打印网络信息
    summary(net, input_size=(1, 3, 64, 64), device=device)
    print(f'标签对应的ID: {trainset.class_to_idx}')

    #设置优化器、损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer =optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)

    #开始训练

    print("Start")
    for epoch in range(EPOCH):
        train_loss = 0.0
        #print(epoch)
        
        for batch_id, (datas, labels) in enumerate(train_loader):
            datas, labels = datas.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(datas)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            
# 每隔 5 个 Epoch 评估一次并保存更佳的模型
            if epoch > 50 and (epoch + 1) % 10 == 0:
                os.makedirs("pth", exist_ok=True)
                PATH = "pth/modeltemp.pth"
                torch.save(net.state_dict(), PATH)
                model = mixed_net()
                model.load_state_dict(torch.load(PATH))
                model.eval()
                model.to(device)

                #限定保存条件
                max_correct = 99
                correct1 = 0
                correct2 = 0
                total1 = 0
                total2 = 0

                #分别测试两个数据集
                with torch.no_grad():
                    for i ,(datas1, labels1) in enumerate(test_loader1):
                        datas1, labels1 = datas1.to(device), labels1.to(device)
                        output_test1 = model(datas1)
                        _, predicted1 = torch.max(output_test1.data, dim=1)
                        total1 += predicted1.size(0)
                        correct1 += (predicted1 == labels1).sum()

                    for i ,(datas2, labels2) in enumerate(test_loader2):
                        datas2, labels2 = datas2.to(device), labels2.to(device)
                        output_test2 = model(datas2)
                        _, predicted2 = torch.max(output_test2.data, dim=1)
                        total2 += predicted2.size(0)
                        correct2 += (predicted2 == labels2).sum()

                    #打印消息
                    c1 = 0
                    c2 = 0
                    c2 = correct2 / total2 * 100
                    c1 = correct1 / total1 * 100
                    print(
                        f"epoch:{epoch + 1}\tbatch_id:{batch_id + 1}\taverage_loss:{(train_loss / len(train_loader.dataset)):.5f}\t"
                        f"correct1:{c1:.2f}%\tcorrect2:{c2:.2f}%"
                    )
                    if (c1 > max_correct):
                        max_correct = c1
                        MAX_PATH = f"pth/model_best_{max_correct}.pth"
                        print(f"save {MAX_PATH}")
                        torch.save(net.state_dict(),MAX_PATH)

