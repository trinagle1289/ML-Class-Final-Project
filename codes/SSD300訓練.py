file_name='PeopleCounting_202305031650.pth' # 副檔名通常以.pt或.pth儲存，建議使用.pth
import torch 
device=torch.device('cuda') # 'cuda'/'cpu'，import torch
num_classes=2 # 物件類別數+1(背景)
train_size=500
valid_size=0
batch_size=10
learning_rate=0.001
step_size=100 # Reriod of learning rate decay
threshold=0.5 # 錨框匹配為物件/背景的閥值，參考值=0.5
variances=[0.1,0.2] # 設定gHat中cx、cy與w、h間的權重
epochs=1000
TrainingImage='D:\TrainingImage/'
Annotation='D:\Annotation/'

from torchvision import transforms
transforms=transforms.Compose([transforms.Resize((300,300)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]) # ToTensor將影像像素歸一化至0~1(直接除以255)，from torchvision import transforms

# 建立dataset
from torch.utils.data import Dataset
class ImageLabel(Dataset): # from torch.utils.data import Dataset
    def __init__(self,img,bbox,cls):
        self.img=img
        self.bbox=bbox
        self.cls=cls
    def __getitem__(self,idx):
        return self.img[idx],self.bbox[idx],self.cls[idx]
    def __len__(self):
        return len(self.img)
def collate_fn(batch):
    img=list()
    bbox=list()
    cls=list()
    for data in batch:
        img.append(data[0])
        bbox.append(data[1])
        cls.append(data[2])
    img=torch.stack(img,dim=0) # import torch
    return img,bbox,cls
import os
all_image_name=os.listdir(TrainingImage) # 所有影像檔名(含.jpg)，import os
img=list()
bbox=list()
cls=list()
from PIL import Image
import xml.etree.ElementTree as ET
for image_name in all_image_name:
    chi_en=list()
    chi_hua=list()
    I=Image.open(TrainingImage+image_name,mode='r') # from PIL import Image
    I=transforms(I)
    img.append(I) # 列表長度為影像個數，列表中每個元素為一個[3,300,300]的tensor
    image_name=image_name[:-4] # 移除4個字元(.jpg)
    root=ET.parse(Annotation+image_name+'.xml').getroot() # 獲取xml文件物件的根結點，import xml.etree.ElementTree as ET
    size=root.find('size') # 獲取size子結點
    width=int(size.find('width').text) # 原始影像的寬(像素)
    height=int(size.find('height').text) # 原始影像的高(像素)
    width_scale=I.size(1)/width # 輸入影像與原始影像的寬比
    height_scale=I.size(1)/height # 輸入影像與原始影像的高比
    for object in root.iter('object'): # 遞迴查詢所有的object子結點
        bndbox=object.find('bndbox')
        chi_en.append([int(bndbox.find('xmin').text)*width_scale,int(bndbox.find('ymin').text)*height_scale,int(bndbox.find('xmax').text)*width_scale,int(bndbox.find('ymax').text)*height_scale])    
        name=object.find('name').text # 1,2,3,4,...  
        chi_hua.append(int(name))
    chi_en=torch.Tensor(chi_en) # 將chi_en轉成tensor，[該影像中的物件個數,4]，import torch
    chi_hua=torch.Tensor(chi_hua) # 將chi_hua轉成tensor，import torch
    bbox.append(chi_en) # 列表長度為影像個數，列表中每個元素為一個[該影像中的物件個數,4]的tensor
    cls.append(chi_hua) # 列表長度為影像個數，列表中每個元素為一個[該影像中的物件個數]的tensor
dataset=ImageLabel(img,bbox,cls)
train_data,valid_data=torch.utils.data.random_split(dataset,[train_size,valid_size]) # import torch
train_loader=torch.utils.data.DataLoader(train_data,batch_size,shuffle=True,collate_fn=collate_fn) # imort torch
valid_loader=torch.utils.data.DataLoader(valid_data,batch_size,shuffle=False,collate_fn=collate_fn) # imort torch

# 建立錨框
feature_scale=[38,19,10,5,3,1] # 預測用的特徵圖尺寸(以像素為單位)
sk=[0.07,0.15,0.33,0.51,0.69,0.87,1.05] # 各預測特徵圖的默認框尺度(相對於輸入影像的比例)，比預測特徵圖的個數多1
aspect_ratio=[[1,2,1/2],[1,2,3,1/2,1/3],[1,2,3,1/2,1/3],[1,2,3,1/2,1/3],[1,2,1/2],[1,2,1/2]] # 各預測特徵圖的縱橫比(須檢查loc、conf的濾波器個數)
abox=[]
import itertools
import math
for i,j in enumerate(feature_scale):
    for m,n in itertools.product(range(j),repeat=2):
        cx=(n+0.5)/j # 等同於cx相對於輸入影像的比例位置(乘以輸入影像尺寸即為cx在輸入影像的像素位置)
        cy=(m+0.5)/j # 等同於cy相對於輸入影像的比例位置(乘以輸入影像尺寸即為cy在輸入影像的像素位置)
        for ar in aspect_ratio[i]:
            abox+=[cx-sk[i]*math.sqrt(ar)/2,cy-sk[i]/math.sqrt(ar)/2,cx+sk[i]*math.sqrt(ar)/2,cy+sk[i]/math.sqrt(ar)/2] # [cxmin cymin cxmax cymax]
        abox+=[cx-math.sqrt(sk[i]*sk[i+1])/2,cy-math.sqrt(sk[i]*sk[i+1])/2,cx+math.sqrt(sk[i]*sk[i+1])/2,cy+math.sqrt(sk[i]*sk[i+1])/2] # [xmin ymin xmax ymax]
anchor=torch.Tensor(abox).view(-1,4).to(device) # [8732,4] (所有錨框的[xmin ymin xmax ymax]，皆相對於輸入影像的比例位置，乘以輸入影像尺寸即為在輸入影像的像素位置)，import torch
anchor.clamp_(max=1, min=0) # 限定最大值為1、最小值0
anchor=anchor*(img[0].size(1)) # 轉換成輸入影像尺寸

from torch import nn
class SSD(nn.Module):
    def __init__(self):
        super().__init__()

        # block_1：Conv1_1~Conv4_3+ReLU
        self.block_1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1), # [batch_size,64,300,300]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1), # [batch_size,64,300,300]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2), # [batch_size,64,150,150]
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1), # [batch_size,128,150,150]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1), # [batch_size,128,150,150]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2), # [batch_size,128,75,75]
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1), # [batch_size,256,75,75]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1), # [batch_size,256,75,75]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1), # [batch_size,256,75,75]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True), # [batch_size,256,38,38]
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1), # [batch_size,512,38,38]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1), # [batch_size,512,38,38] 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1), # [batch_size,512,38,38]
            nn.ReLU(inplace=True),
        )
        
        # Layer learns to scale the L2 normalized features from conv4_3
        self.l2norm=L2Norm(512,20) # 512為輸入的特徵圖個數，20為scale
         
        # block_2：Pool4~Conv7+ReLU
        self.block_2=nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2), # [batch_size,512,19,19]
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1), # [batch_size,512,19,19]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1), # [batch_size,512,19,19]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1), # [batch_size,512,19,19]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1), # [batch_size,512,19,19]
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,stride=1,padding=6,dilation=6), # [batch_size,1024,19,19]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=1,stride=1), # [batch_size,1024,19,19]
            nn.ReLU(inplace=True),
        )

        # block_3：Conv8_1~Conv8_2+ReLU
        self.block_3=nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=256,kernel_size=1), # [batch_size,256,19,19]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=2,padding=1), # [batch_size,512,10,10]
            nn.ReLU(inplace=True),
        )

        # block_4：Conv9_1~Conv9_2+ReLU
        self.block_4=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=128,kernel_size=1), # [batch_size,128,10,10]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1), # [batch_size,256,5,5]
            nn.ReLU(inplace=True),
        )

        # block_5：Conv10_1~Conv10_2+ReLU
        self.block_5=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=128,kernel_size=1), # [batch_size,128,5,5]
            nn.ReLU(inplace=True),                            
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3), # [batch_size,256,3,3]
            nn.ReLU(inplace=True),
        )

        # block_6：Conv11_1~Conv11_2+ReLU
        self.block_6=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=128,kernel_size=1), # [batch_size,128,3,3]                            
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3), # [batch_size,256,1,1]
            nn.ReLU(inplace=True),
        )

        # loc_1
        self.loc_1=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=4*4,kernel_size=3,stride=1,padding=1), # [batch_size,16,38,38]
        )
        # conf_1
        self.conf_1=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=4*num_classes,kernel_size=3,stride=1,padding=1), # [batch_size,(4*num_classes),38,38]
        )
        # loc_2
        self.loc_2=nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=6*4,kernel_size=3,stride=1,padding=1), # [batch_size,24,19,19]
        )
        # conf_2
        self.conf_2=nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=6*num_classes,kernel_size=3,stride=1,padding=1), # [batch_size,(6*num_classes),19,19]
        ) 
        # loc_3
        self.loc_3=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=6*4,kernel_size=3,stride=1,padding=1), # [batch_size,24,10,10]
        )
        # conf_3
        self.conf_3=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=6*num_classes,kernel_size=3,stride=1,padding=1), # [batch_size,(6*num_classes),10,10]
        ) 
        # loc_4
        self.loc_4=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=6*4,kernel_size=3,stride=1,padding=1), # [batch_size,24,5,5]
        )
        # conf_4
        self.conf_4=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=6*num_classes,kernel_size=3,stride=1,padding=1), # [batch_size,(6*num_classes),5,5]
        )       
        # loc_5
        self.loc_5=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=4*4,kernel_size=3,stride=1,padding=1), # [batch_size,16,3,3]
        )
        # conf_5
        self.conf_5=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=4*num_classes,kernel_size=3,stride=1,padding=1), # [batch_size,(4*num_classes),3,3]
        )   
        # loc_6
        self.loc_6=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=4*4,kernel_size=3,stride=1,padding=1), # [batch_size,16,1,1]
        )
        # conf_6
        self.conf_6=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=4*num_classes,kernel_size=3,stride=1,padding=1), # [batch_size,(4*num_classes),1,1]
        )   

    def forward(self,x):
        x=self.block_1(x) # [batch_size,512,38,38] (Conv4_3+ReLU輸出)
        n=self.l2norm(x)
        loc1=self.loc_1(n).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,4)
        conf1=self.conf_1(n).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,num_classes)
        x=self.block_2(x) # [batch_size,1024,19,19] (Conv7+ReLU輸出)
        loc2=self.loc_2(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,4)
        conf2=self.conf_2(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,num_classes)
        x=self.block_3(x) # [batch_size,512,10,10] (Conv8_2+ReLU輸出)
        loc3=self.loc_3(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,4)
        conf3=self.conf_3(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,num_classes)
        x=self.block_4(x) # [batch_size,256,5,5] (Conv9_2+ReLU輸出)
        loc4=self.loc_4(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,4)
        conf4=self.conf_4(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,num_classes)
        x=self.block_5(x) # [batch_size,256,3,3] (Conv10_2+ReLU輸出)
        loc5=self.loc_5(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,4)
        conf5=self.conf_5(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,num_classes)
        x=self.block_6(x) # [batch_size,256,1,1] (Conv11_2+ReLU輸出)
        loc6=self.loc_6(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,4)
        conf6=self.conf_6(x).permute(0,2,3,1).contiguous().view(batch_size,-1).view(batch_size,-1,num_classes)
        loc=torch.cat((loc1,loc2,loc3,loc4,loc5,loc6),1) # [batch_size,8732,4]，import torch
        conf=torch.cat((conf1,conf2,conf3,conf4,conf5,conf6),1) # [batch_size,8732,num_classes]，import torch
        return loc,conf

class L2Norm(nn.Module):
    def __init__(self,in_channels,scale):
        super(L2Norm,self).__init__()
        self.in_channels=in_channels
        self.gamma=scale or None
        self.eps=1e-10
        self.weight=nn.Parameter(torch.Tensor(self.in_channels)) # from torch import nn，import torch
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.constant_(self.weight,self.gamma) # from torch import nn 
    def forward(self,x):
        norm=x.pow(2).sum(dim=1,keepdim=True).sqrt()+self.eps
        x=torch.div(x,norm) # import torch
        out=self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)*x
        return out

detector=SSD().to(device)
optimizer=torch.optim.Adam(detector.parameters(),lr=learning_rate) # import torch
#optimizer=torch.optim.SGD(detecotr.parameters(),lr=learning_rate) # import torch
scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size,0.1) # import torch
num_anchor=anchor.size(0) # 8732，anchor為[8732,4]
for i in range(1,epochs+1):
    print('Running Epoch:'+str(i))
    train_loss,train_batch,valid_loss,valid_batch=0,0,0,0
    detector.train()
    for img,bbox_,cls_ in train_loader: # 一個batch的img、bbox_、cls_，img：[batch_size,3,300,300]，bbox_：列表長度為batch_size，列表中每個元素為一個[該影像中的物件個數,4]的tensor，cls_：列表長度為batch_size，列表中每個元素為一個[該影像中的物件個數]的tensor
        if img.size(0)!=batch_size: # 最後不足一個batch的訓練影像不進行訓練
            break
        img=img.to(device)
        gHat=torch.Tensor(batch_size,num_anchor,4) # import torch
        matched_class=torch.LongTensor(batch_size,num_anchor) # import torch
        pos=list()
        for j in range(batch_size):

            # 找出每個錨框匹配的物件(以比較IoU為主，但若IoU值為該物件的最大IoU值，則直接指定對應該物件。若IoU為0，則隨便匹配物件，待之後用threshold去除)
            bbox=bbox_[j].to(device) # [該影像中的物件個數,4]
            cls=cls_[j].to(device) # [該影像中的物件個數]
            num_objects=bbox.size(0) # 該影像中的物件個數，[1]。bbox：[該影像中的物件個數,4]
            min_xy=torch.max(bbox[:,:2].unsqueeze(1).expand(num_objects,num_anchor,2),anchor[:,:2].broadcast_to(num_objects,num_anchor,2)) # 每個物件與8732個錨框比較取較大的xmin與ymin，[該影像中的物件個數,8732,2]，2表示較大的xmin與ymin，import torch
            max_xy=torch.min(bbox[:,2:].unsqueeze(1).expand(num_objects,num_anchor,2),anchor[:,2:].broadcast_to(num_objects,num_anchor,2)) # 每個物件與8732個錨框比較取較小的xmax與ymax，[該影像中的物件個數,8732,2]，2表示較小xmax與ymax，import torch
            side_length=(max_xy-min_xy).clamp(min=0) # 交集面積的邊長，[該影像中的物件個數,8732,2]
            area_inter=side_length[:,:,0]*side_length[:,:,1] # 交集面積，[該影像中的物件個數,8732]
            area_bbox=((bbox[:,2]-bbox[:,0])*(bbox[:,3]-bbox[:,1])).unsqueeze(1).expand(num_objects,num_anchor)
            area_anchor=((anchor[:,2]-anchor[:,0])*(anchor[:,3]-anchor[:,1])).broadcast_to(num_objects,num_anchor)
            IoU=area_inter/(area_bbox+area_anchor-area_inter) # IOU，[該影像中的物件個數,8732]
            maxIoU_object,anchor_idx=torch.max(IoU,dim=1) # dim=1表示取每列的最大值。maxIoU_object為每個物件的最大IoU，[該影像中的物件個數]。anchor_idx為每個物件最大IoU的錨框編號，[該影像中的物件個數]。import torch
            maxIoU_anchor,object_idx=torch.max(IoU,dim=0) # dim=0表示取每行的最大值。maxIoU_anchor為每個錨框的最大IoU值，[8732]。object_idx為每個錨框最大IoU的物件編號(非類別)(0,1,2,...)，[8732]。import torch
            maxIoU_anchor.index_fill_(0,anchor_idx,2) # 修改maxIoU_anchor(每個錨框的最大IoU)，令每個物件最大IoU的錨框(即anchor_idx)的IoU為2
            pos.append(maxIoU_anchor>=threshold) # True/False，利用threshold篩選出背景，令有匹配到物件的錨框為True，背景為False(大部分為False)，pos：列表長度為batch_size，列表中每個元素為[8732]的tensor
            for k in range(num_objects): # k:0~(num_objects-1)
                object_idx[anchor_idx[k]]=k # 將每個物件IoU=2的錨框所對應的物件指定為該物件
            matched_bbox=bbox[object_idx] # 每個錨框匹配物件的邊界框，[8732,4]，[xmin ymin xmax ymax]
            matched_class[j]=cls[object_idx] # 每個錨框匹配物件的類別(1,2, ...)，matched_class[0]：[8732]，matched_class：[batch_size,8732]
            matched_class[j][maxIoU_anchor<threshold]=0 # 利用threshold決定那些錨框匹配的物件類別為背景(0)，matched_class表示每個錨框匹配物件的類別(0,1,2, ...)，[batch_size,8732]
            gHat_cx=((matched_bbox[:,0]+matched_bbox[:,2])/2-(anchor[:,0]+anchor[:,2])/2)/((anchor[:,2]-anchor[:,0])*variances[0]) # [8732]
            gHat_cy=((matched_bbox[:,1]+matched_bbox[:,3])/2-(anchor[:,1]+anchor[:,3])/2)/((anchor[:,3]-anchor[:,1])*variances[0]) # [8732]
            gHat_w=torch.log((matched_bbox[:,2]-matched_bbox[:,0])/(anchor[:,2]-anchor[:,0]))/variances[1] # [8732]，import torch
            gHat_h=torch.log((matched_bbox[:,3]-matched_bbox[:,1])/(anchor[:,3]-anchor[:,1]))/variances[1] # [8732]，import torch
            gHat[j]=torch.stack((gHat_cx,gHat_cy,gHat_w,gHat_h),1) # gHat[0]：[8732,4]，gHat：[batch_size,8732,4]，import torch
        pos=torch.stack(pos,0) # 將pos從list轉為tensor，pos：[batch_size,8732]，True/False，import torch
        pred_loc,pred_conf=detector(img) # pred_loc：[batch_size,8732,4]，pred_conf：[batch_size,8732,num_classes]
        gHat,matched_class=gHat.to(device),matched_class.to(device) # gHat：[batch_size,8732,4]，matched_class：[batch_size,8732]
        num_pos=pos.sum(dim=1,keepdim=True) # 利用threshold篩選後有匹配到物件的錨框數量，[batch_size,1]
        pos_expand=pos.unsqueeze(pos.dim()).expand_as(pred_loc) # 將[batch_size,8732]的pos擴增成[batch_size,8732,4]，Ture/False
        pos_l=pred_loc[pos_expand].view(-1,4) # 取出有匹配到物件的錨框的pred_loc，[batch內有匹配到物件的錨框總數(即num_pos內值的加總),4]
        pos_gHat=gHat[pos_expand].view(-1,4) # 取出有匹配到物件的錨框的gHat，[batch內有匹配到物件的錨框總數(即num_pos內值的加總),4]
        L_loc=torch.nn.functional.smooth_l1_loss(pos_l,pos_gHat) # 計算一個batch內有匹配到物件的錨框的L_loc，[1]，import torch
        batch_pred_conf=pred_conf.view(-1,num_classes) # 將pred_conf內的batch整合在一起，[batch_size*8732,num_classes]
        crossEntropy=torch.logsumexp(batch_pred_conf,dim=1,keepdim=True)-batch_pred_conf.gather(1,matched_class.view(-1,1)) # 計算每個錨框匹配物件(包含背景)的負logsumexp，[batch_size*8732,1]，import torch
            # matched_class.view(-1,1)：將matched_class(每個錨框匹配物件的類別(0,1,2, ...))內的batch整合在一起，[batch_size*8732,1]
            # batch_pred_conf.gather(1,matched_class.view(-1,1))：根據matched_class.view(-1,1)(每個錨框匹配物件的類別(0,1,2, ...))取出該物件類別的預測置信值(pred_conf)
        crossEntropy[pos.view(-1,1)]=0 # 利用threshold篩選後若錨框有匹配到物件，則令該錨框匹配物件的負logsumexp為0，[batch_size*8732,1]
        crossEntropy=crossEntropy.view(batch_size,-1) # 將crossEntropy從[batch_size*8732,1]轉換成[batch_size,8732]        
        _,background_idx=crossEntropy.sort(1,descending=True) # background_idx：[batch_size,8732]，將有匹配到物件的錨框的負logsumexp設為0後，依負logsumexp由大而小排列並取得錨框編號(如編號5即表示第5個錨框所匹配的背景的負logsumexp為最大)
        _,idx_rank=background_idx.sort(1) # idx_rank：[batch_size,8732]，依crossEntropy由小而大排序，如4、1、3、2表示第1個錨框在負logsumexp中排第4(愈大表示負logsumexp愈小)，第2個錨框在負logsumexp中排第1
        num_neg=torch.clamp(3*num_pos,max=pos.size(1)-num_pos) # num_neg：[batch_size,1]，定義每張影像的負樣本個數為正樣本個數的3倍，上限改為錨框個數-正樣本個數，import torch
        neg=idx_rank<num_neg.expand_as(idx_rank) # neg：True/False，將負logsumexp最大的前num_neg個設為True，[batch_size,8732]
        pos_pred_conf=pos.unsqueeze(2).expand_as(pred_conf) # True/False，將pred_conf中正樣本的部分令為True，其餘為False，[batch_size,8732,num_classes]，
        neg_pred_conf=neg.unsqueeze(2).expand_as(pred_conf) # True/False，將pred_conf中負樣本的部分令為True，其餘為False，[batch_size,8732,num_classes]
        input=pred_conf[(pos_pred_conf+neg_pred_conf).gt(0)].view(-1,num_classes) # 挑出正樣本與負樣本的pred_conf，[num_pos+num_neg,num_classes]
        target=matched_class[(pos+neg).gt(0)] # 挑出正樣本與負樣本的類別(包含背景)，[num_pos+num_neg]
        L_conf=torch.nn.functional.cross_entropy(input,target) # 計算一個batch內正樣本與負樣本的L_conf，[1]，import torch
        N=num_pos.data.sum()
        loss=(L_loc+L_conf)/N
        train_loss+=loss.item()
        train_batch+=1
        optimizer.zero_grad() # 權重梯度歸零
        loss.backward() # 計算每個權重的loss梯度
        optimizer.step() # 權重更新
    scheduler.step()
    if train_batch!=0:
        print('Training Loss='+str(train_loss/train_batch)) # 計算每一個epoch的平均訓練loss

    detector.eval()
    for img,bbox_,cls_ in valid_loader: # 一個batch的img、bbox_、cls_，img：[batch_size,3,300,300]，bbox_：列表長度為batch_size，列表中每個元素為一個[該影像中的物件個數,4]的tensor，cls_：列表長度為batch_size，列表中每個元素為一個[該影像中的物件個數]的tensor
        if img.size(0)!=batch_size: # 最後不足一個batch的驗證影像不進行驗證
            break 
        img=img.to(device)
        gHat=torch.Tensor(batch_size,num_anchor,4) # import torch
        matched_class=torch.LongTensor(batch_size,num_anchor) # import torch
        pos=list()
        for j in range(batch_size):

            # 找出每個錨框匹配的物件(以比較IoU為主，但若IoU值為該物件的最大IoU值，則直接指定對應該物件。若IoU為0，則隨便匹配物件，待之後用threshold去除)
            bbox=bbox_[j].to(device) # [該影像中的物件個數,4]
            cls=cls_[j].to(device) # [該影像中的物件個數]
            num_objects=bbox.size(0) # 該影像中的物件個數，[1]。bbox：[該影像中的物件個數,4]
            min_xy=torch.max(bbox[:,:2].unsqueeze(1).expand(num_objects,num_anchor,2),anchor[:,:2].broadcast_to(num_objects,num_anchor,2)) # 每個物件與8732個錨框比較取較大的xmin與ymin，[該影像中的物件個數,8732,2]，2表示較大的xmin與ymin，import torch
            max_xy=torch.min(bbox[:,2:].unsqueeze(1).expand(num_objects,num_anchor,2),anchor[:,2:].broadcast_to(num_objects,num_anchor,2)) # 每個物件與8732個錨框比較取較小的xmax與ymax，[該影像中的物件個數,8732,2]，2表示較小xmax與ymax，import torch
            side_length=(max_xy-min_xy).clamp(min=0) # 交集面積的邊長，[該影像中的物件個數,8732,2]
            area_inter=side_length[:,:,0]*side_length[:,:,1] # 交集面積，[該影像中的物件個數,8732]
            area_bbox=((bbox[:,2]-bbox[:,0])*(bbox[:,3]-bbox[:,1])).unsqueeze(1).expand(num_objects,num_anchor)
            area_anchor=((anchor[:,2]-anchor[:,0])*(anchor[:,3]-anchor[:,1])).broadcast_to(num_objects,num_anchor)
            IoU=area_inter/(area_bbox+area_anchor-area_inter) # IOU，[該影像中的物件個數,8732]
            maxIoU_object,anchor_idx=torch.max(IoU,dim=1) # dim=1表示取每列的最大值。maxIoU_object為每個物件的最大IoU，[該影像中的物件個數]。anchor_idx為每個物件最大IoU的錨框編號，[該影像中的物件個數]。import torch
            maxIoU_anchor,object_idx=torch.max(IoU,dim=0) # dim=0表示取每行的最大值。maxIoU_anchor為每個錨框的最大IoU值，[8732]。object_idx為每個錨框最大IoU的物件編號(非類別)(0,1,2,...)，[8732]。import torch
            maxIoU_anchor.index_fill_(0,anchor_idx,2) # 修改maxIoU_anchor(每個錨框的最大IoU)，令每個物件最大IoU的錨框(即anchor_idx)的IoU為2
            pos.append(maxIoU_anchor>=threshold) # True/False，利用threshold篩選出背景，令有匹配到物件的錨框為True，背景為False(大部分為False)，pos：列表長度為batch_size，列表中每個元素為[8732]的tensor
            for k in range(num_objects): # k:0~(num_objects-1)
                object_idx[anchor_idx[k]]=k # 將每個物件IoU=2的錨框所對應的物件指定為該物件
            matched_bbox=bbox[object_idx] # 每個錨框匹配物件的邊界框，[8732,4]，[xmin ymin xmax ymax]
            matched_class[j]=cls[object_idx] # 每個錨框匹配物件的類別(1,2, ...)，matched_class[0]：[8732]，matched_class：[batch_size,8732]
            matched_class[j][maxIoU_anchor<threshold]=0 # 利用threshold決定那些錨框匹配的物件類別為背景(0)，matched_class表示每個錨框匹配物件的類別(0,1,2, ...)，[batch_size,8732]
            gHat_cx=((matched_bbox[:,0]+matched_bbox[:,2])/2-(anchor[:,0]+anchor[:,2])/2)/((anchor[:,2]-anchor[:,0])*variances[0]) # [8732]
            gHat_cy=((matched_bbox[:,1]+matched_bbox[:,3])/2-(anchor[:,1]+anchor[:,3])/2)/((anchor[:,3]-anchor[:,1])*variances[0]) # [8732]
            gHat_w=torch.log((matched_bbox[:,2]-matched_bbox[:,0])/(anchor[:,2]-anchor[:,0]))/variances[1] # [8732]，import torch
            gHat_h=torch.log((matched_bbox[:,3]-matched_bbox[:,1])/(anchor[:,3]-anchor[:,1]))/variances[1] # [8732]，import torch
            gHat[j]=torch.stack((gHat_cx,gHat_cy,gHat_w,gHat_h),1) # gHat[0]：[8732,4]，gHat：[batch_size,8732,4]，import torch
        pos=torch.stack(pos,0) # 將pos從list轉為tensor，pos：[batch_size,8732]，True/False，import torch
        pred_loc,pred_conf=detector(img) # pred_loc：[batch_size,8732,4]，pred_conf：[batch_size,8732,num_classes]
        gHat,matched_class=gHat.to(device),matched_class.to(device) # gHat：[batch_size,8732,4]，matched_class：[batch_size,8732]
        num_pos=pos.sum(dim=1,keepdim=True) # 利用threshold篩選後有匹配到物件的錨框數量，[batch_size,1]
        pos_expand=pos.unsqueeze(pos.dim()).expand_as(pred_loc) # 將[batch_size,8732]的pos擴增成[batch_size,8732,4]，Ture/False
        pos_l=pred_loc[pos_expand].view(-1,4) # 取出有匹配到物件的錨框的pred_loc，[batch內有匹配到物件的錨框總數(即num_pos內值的加總),4]
        pos_gHat=gHat[pos_expand].view(-1,4) # 取出有匹配到物件的錨框的gHat，[batch內有匹配到物件的錨框總數(即num_pos內值的加總),4]
        L_loc=torch.nn.functional.smooth_l1_loss(pos_l,pos_gHat) # 計算一個batch內有匹配到物件的錨框的L_loc，[1]，import torch
        batch_pred_conf=pred_conf.view(-1,num_classes) # 將pred_conf內的batch整合在一起，[batch_size*8732,num_classes]
        crossEntropy=torch.logsumexp(batch_pred_conf,dim=1,keepdim=True)-batch_pred_conf.gather(1,matched_class.view(-1,1)) # 計算每個錨框匹配物件(包含背景)的負logsumexp，[batch_size*8732,1]，import torch
            # matched_class.view(-1,1)：將matched_class(每個錨框匹配物件的類別(0,1,2, ...))內的batch整合在一起，[batch_size*8732,1]
            # batch_pred_conf.gather(1,matched_class.view(-1,1))：根據matched_class.view(-1,1)(每個錨框匹配物件的類別(0,1,2, ...))取出該物件類別的預測置信值(pred_conf)
        crossEntropy[pos.view(-1,1)]=0 # 利用threshold篩選後若錨框有匹配到物件，則令該錨框匹配物件的負logsumexp為0，[batch_size*8732,1]
        crossEntropy=crossEntropy.view(batch_size,-1) # 將crossEntropy從[batch_size*8732,1]轉換成[batch_size,8732]        
        _,background_idx=crossEntropy.sort(1,descending=True) # background_idx：[batch_size,8732]，將有匹配到物件的錨框的負logsumexp設為0後，依負logsumexp由大而小排列並取得錨框編號(如編號5即表示第5個錨框所匹配的背景的負logsumexp為最大)
        _,idx_rank=background_idx.sort(1) # idx_rank：[batch_size,8732]，依crossEntropy由小而大排序，如4、1、3、2表示第1個錨框在負logsumexp中排第4(愈大表示負logsumexp愈小)，第2個錨框在負logsumexp中排第1
        num_neg=torch.clamp(3*num_pos,max=pos.size(1)-num_pos) # num_neg：[batch_size,1]，定義每張影像的負樣本個數為正樣本個數的3倍，上限改為錨框個數-正樣本個數，import torch
        neg=idx_rank<num_neg.expand_as(idx_rank) # neg：True/False，將負logsumexp最大的前num_neg個設為True，[batch_size,8732]
        pos_pred_conf=pos.unsqueeze(2).expand_as(pred_conf) # True/False，將pred_conf中正樣本的部分令為True，其餘為False，[batch_size,8732,num_classes]，
        neg_pred_conf=neg.unsqueeze(2).expand_as(pred_conf) # True/False，將pred_conf中負樣本的部分令為True，其餘為False，[batch_size,8732,num_classes]
        input=pred_conf[(pos_pred_conf+neg_pred_conf).gt(0)].view(-1,num_classes) # 挑出正樣本與負樣本的pred_conf，[num_pos+num_neg,num_classes]
        target=matched_class[(pos+neg).gt(0)] # 挑出正樣本與負樣本的類別(包含背景)，[num_pos+num_neg]
        L_conf=torch.nn.functional.cross_entropy(input,target) # 計算一個batch內正樣本與負樣本的L_conf，[1]，import torch
        N=num_pos.data.sum()
        loss=(L_loc+L_conf)/N
        valid_loss+=loss.item()
        valid_batch+=1
    if valid_batch!=0:
        print('Validation Loss='+str(valid_loss/valid_batch)) # 計算每一個epoch的平均驗證loss

torch.save(detector.state_dict(),file_name) # import torch