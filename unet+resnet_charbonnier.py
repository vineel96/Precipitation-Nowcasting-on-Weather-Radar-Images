import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import os
import glob
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import numpy as np
import cv2

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.residual = nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0)

    def forward(self, x):
      double_conv = self.double_conv(x)
      residue = self.residual(x)
      #print("double conv size:",double_conv.shape())
      #print("residue size:",residue.shape())
      out = double_conv + residue  #residual skip connection [we add instead of concatenation]
      return out

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        #print("size of zeroth x1:",x1.size())
        x1 = self.up(x1)
        #print("size of first x1:",x1.size())
        # input is CHW
        '''diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]


        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])'''
        #print("size of second x1:",x1.size())
        #print("size of x2:",x2.size())
        x = torch.cat([x2, x1], dim=1)
        #print("concatenated output size:",x.size())
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1)            #kernel_size=1 no need to pad,input size remains same
        self.conv2 = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
      out1 = self.conv1(x)
      return self.act(self.conv2(out1))

class UNet(nn.Module):
    def __init__(self, n_channels, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.outc(x)
        return output


# Data Generators for generating 4 radar images and one ground truth image(output) for training
class radardata(Dataset):
    def __init__(self, path1, path2, month1, day1, month2, day2):
        # Data Loading

        #self.files = sorted(glob.glob(path))[:251]
        #print(self.files[-1])
        self.files = []
        for i in range(len(month1)):
            days = day1[i]
            for d in days:
                if month1[i]=='02':
                    if d=='06':
                        temp = sorted(glob.glob(path1 + "/" + str(month1[i]) + "/" + str(d) + "/*"))[:93]
                    elif d=='28':
                        temp = sorted(glob.glob(path1 + "/" + str(month1[i]) + "/" + str(d) + "/*"))[:121]
                elif month1[i]=='03':
                    if d=='06':
                        temp = sorted(glob.glob(path1 + "/" + str(month1[i]) + "/" + str(d) + "/*"))[:221]
                    elif d=='10':
                        temp = sorted(glob.glob(path1 + "/" + str(month1[i]) + "/" + str(d) + "/*"))[:245]
                elif month1[i]=='05':
                    if d=='20':
                        temp = sorted(glob.glob(path1 + "/" + str(month1[i]) + "/" + str(d) + "/*"))[:193]
                    elif d=='21':
                        temp = sorted(glob.glob(path1 + "/" + str(month1[i]) + "/" + str(d) + "/*"))
                elif month1[i]=='07':
                    temp = sorted(glob.glob(path1 + "/" + str(month1[i]) + "/" + str(d) + "/*"))
                elif month1[i]=='09':
                    if d=='08':
                        temp = sorted(glob.glob(path1 + "/" + str(month1[i]) + "/" + str(d) + "/*"))[:251]
                elif month1[i]=='10':
                    if d=='11':
                        temp = sorted(glob.glob(path1 + "/" + str(month1[i]) + "/" + str(d) + "/*"))
                    elif d=='12':
                        temp = sorted(glob.glob(path1 + "/" + str(month1[i]) + "/" + str(d) + "/*"))[:132]
                for item in temp:
                    self.files.append(item)

        for i in range(len(month2)):
            days = day2[i]
            for d in days:
                temp = sorted(glob.glob(path2 + "/" + str(month2[i]) + "/" + str(d) + "/*"))
                for item in temp:
                    self.files.append(item)

        print(self.files[-1])
        print(len(self.files))

        self.change = -1
        self.new_index = -1

        '''self.files = []
        self.input_seq_len = 4
        self.inner_index, self.outer_index = 0, 0

        for i in range(len(month)):
          days = day[i]
          for d in days:
            self.files.append(sorted(glob.glob(path + "/" + str(month[i]) + "/" + str(d) + "/*"))[:251])

        self.total = sum([len(f) for f in self.files]) - 5 * len(self.files) # last 5 data values are not read from each inner list as input seq length is 6
        print(self.total,[len(f) for f in self.files])'''
        self.preprocess = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        # Indexing, data[index]

        #print("old index:", index)
        split1 = self.files[index].split("/")[-1]
        m1 = split1[4:6]
        d1 = split1[6:8]
        for i in range(1, 5):
            split2 = self.files[index + i].split("/")[-1]
            m2 = split2[4:6]
            d2 = split2[6:8]
            if m1 != m2 or d1 != d2:
                #print("m1,d1:", m1, d1)
                #print("m2,d2:", m2, d2)
                self.change = 1
                self.new_index = index + i
                break
        if self.change == 1:
            index = self.new_index
            print("new index:", index)
            self.change = -1
            self.new_index = -1

        #print("file:", self.files[index])

        input_1 = np.load(self.files[index])['arr_0']
        input_1 = cv2.resize(input_1, (512, 512), interpolation=cv2.INTER_CUBIC)

        input_2 = np.load(self.files[index + 1])['arr_0']
        input_2 = cv2.resize(input_2, (512, 512), interpolation=cv2.INTER_CUBIC)

        input_3 = np.load(self.files[index + 2])['arr_0']
        input_3 = cv2.resize(input_3, (512, 512), interpolation=cv2.INTER_CUBIC)

        input_4 = np.load(self.files[index + 3])['arr_0']
        input_4 = cv2.resize(input_4, (512, 512), interpolation=cv2.INTER_CUBIC)

        output = np.load(self.files[index + 4])['arr_0']
        output = cv2.resize(output, (512, 512), interpolation=cv2.INTER_CUBIC)
        # print("in get:",output.shape)

        '''input_1 = Image.open(self.files[index]).convert('RGB')
        input_2 = Image.open(self.files[(index+1)]).convert('RGB')
        input_3 = Image.open(self.files[(index+2)]).convert('RGB')
        input_4 = Image.open(self.files[(index+3)]).convert('RGB')
        output = Image.open(self.files[(index+4)]).convert('RGB')'''

        '''if self.inner_index <= len(self.files[self.outer_index]) - self.input_seq_len - 1:
          input_1, input_2, input_3, input_4, output = self.generate()
          self.inner_index = self.inner_index + 1
        else:
          self.inner_index = 0
          self.outer_index = self.outer_index + 1
          if self.outer_index == len(self.files):
            print("INSIDE")
            self.outer_index = self.outer_index - 1
          input_1, input_2, input_3, input_4, output = self.generate()
          self.inner_index = self.inner_index + 1'''

        input_1 = self.preprocess(input_1)
        input_2 = self.preprocess(input_2)
        input_3 = self.preprocess(input_3)
        input_4 = self.preprocess(input_4)
        # input_5 = self.preprocess(input_5)
        # input_6 = self.preprocess(input_6)
        output = self.preprocess(output)
        # print("in get2:",output.size())

        # print(input_1,output)

        return input_1, input_2, input_3, input_4, output

    def generate(self):
        # print("inside data gen")
        input_1 = np.load(self.files[self.outer_index][self.inner_index])['arr_0']
        input_1 = cv2.resize(input_1, (512, 512), interpolation=cv2.INTER_CUBIC)

        input_2 = np.load(self.files[self.outer_index][self.inner_index + 1])['arr_0']
        input_2 = cv2.resize(input_2, (512, 512), interpolation=cv2.INTER_CUBIC)

        input_3 = np.load(self.files[self.outer_index][self.inner_index + 2])['arr_0']
        input_3 = cv2.resize(input_3, (512, 512), interpolation=cv2.INTER_CUBIC)

        input_4 = np.load(self.files[self.outer_index][self.inner_index + 3])['arr_0']
        input_4 = cv2.resize(input_4, (512, 512), interpolation=cv2.INTER_CUBIC)

        input_5 = np.load(self.files[self.outer_index][self.inner_index + 4])['arr_0']  # CHECK INDEX WHILE USING...
        input_5 = cv2.resize(input_5, (512, 512), interpolation=cv2.INTER_CUBIC)

        input_6 = np.load(self.files[self.outer_index][self.inner_index + 5])['arr_0']
        input_6 = cv2.resize(input_6, (512, 512), interpolation=cv2.INTER_CUBIC)

        output = np.load(self.files[self.outer_index][self.inner_index + 6])['arr_0']
        output = cv2.resize(output, (512, 512), interpolation=cv2.INTER_CUBIC)

        return input_1, input_2, input_3, input_4, output

    def __len__(self):
        # Total Number Of Samples
        return len(self.files) - 4  # subtract with no of input radar images given to model


def charbonnier_loss(output,target, epsilon = 1e-6):
    loss = torch.mean(torch.sqrt((output - target)**2 + epsilon * epsilon))
    return loss

def logcosh_loss(output, target):
  loss = torch.sum(torch.log(torch.cosh( output - target)))
  return loss


def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
    f_path = checkpoint_dir + 'u-net_with_resnet_charbonnier_raw_new.pth'
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir + '/' + 'best_model.pt'
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    if os.path.isfile(checkpoint_fpath):
        checkpoint = torch.load(checkpoint_fpath)
        if checkpoint is not None:
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch =  checkpoint['epoch']
        else:
            start_epoch = 0
    else:
        start_epoch = 0
    return model, optimizer, start_epoch


if __name__=="__main__":
    net = UNet(4)  # input channel depth dimension is 12 i.e 4 radar images with each radar image color channel depth as 3, so 4*3=12
    net = net.cuda()

    #summary(net, input_size=(4, 512, 512),batch_size=1)

    path1 = "/media/data/mot_1/data/crop1/binary/2019"
    path2 = "/media/data/mot_1/data/crop2/binary/2019"

    month1 = ['02','03','05','07','09','10']
    day1 = [['06','28'],['06','10'],['20','21'],['03','05','18'],['08'],['11','12']]

    month2 = ['07', '08', '10', '12']
    day2 = [['03'], ['05', '14'], ['20'], ['25']]

    radar_image = radardata(path1, path2, month1, day1, month2, day2)
    dataloader = DataLoader(dataset=radar_image, batch_size=2, num_workers=0)

    '''for i, (input_1, input_2, input_3, input_4, output) in enumerate(dataloader):
        print("")
    exit()'''

    n_epochs = 200
    batch_size = 2
    n_samples = radar_image.__len__()
    n_iter = n_samples / batch_size

    print("Total samples: ", n_samples)
    print("No of iterations per epoch: ", n_iter)

    #criterion = nn.MSELoss()
    # optimizer = optim.Adadelta(net.parameters(),lr=0.001)
    #criterion.cuda()

    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60], gamma=0.1)

    save_path = '/media/data/mot_1/code/deep_learning/unet_resnet/'

    # loading saved model to resume training from that point
    ckp_path = save_path + 'u-net_with_resnet_charbonnier_raw_new.pth'
    net, optimizer, start_epoch = load_ckp(ckp_path, net, optimizer)
    #print(start_epoch)
    #exit()


    for epoch in range(start_epoch, n_epochs):
        running_loss = 0.0
        for i, (input_1, input_2, input_3, input_4, output) in enumerate(dataloader):
            # print(input_1.size(),input_2.size())
            input = torch.cat([input_1, input_2, input_3, input_4], dim=1).cuda()
            # print("input size:",input.size())
            # print("output size:",output.size())
            output = output.cuda()
            optimizer.zero_grad()
            outputs = net(input)
            # print("prediction size:",outputs.size())
            loss = charbonnier_loss(outputs, output)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss = loss.item()
            # correct = (outputs==output).float().sum()
            print(
                f'epoch {epoch + 1}/{n_epochs}, iteration step {i + 1}/{n_iter},Loss: {running_loss}')  # , Accuracy: {correct/outputs.shape[0]}')

        # saving model
        checkpoint = {
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()
            }
        save_ckp(checkpoint, False, save_path, "")
