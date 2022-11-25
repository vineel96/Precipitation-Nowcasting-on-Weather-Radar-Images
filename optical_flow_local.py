import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
import pandas as pd
from torch.utils import data
from torchsummary import summary

import glob
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from PIL import Image
import numpy as np
import cv2

import os
import shutil

def conv(input_channels, output_channels, kernel_size, stride, dropout_rate):
    layer = nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size,
                  stride = stride, padding=(kernel_size - 1) // 2),
        nn.BatchNorm2d(output_channels),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout(dropout_rate)
    )
    return layer

def deconv(input_channels, output_channels):
    layer = nn.Sequential(
        nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.1, inplace=True)
    )
    return layer


class Encoder(nn.Module):
    def __init__(self, input_channels, kernel_size, dropout_rate):
        super(Encoder, self).__init__()
        self.conv1 = conv(input_channels, 64, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate)
        self.conv2 = conv(64, 128, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate)
        self.conv3 = conv(128, 256, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate)
        self.conv4 = conv(256, 512, kernel_size=kernel_size, stride=2, dropout_rate=dropout_rate)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.002 / n)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        return out_conv1, out_conv2, out_conv3, out_conv4

def RYScaler(X):
    '''
    Scale RY data from mm (in float64) to brightness (in uint8).
    Args:
        X (numpy.ndarray): RY radar image
    Returns:
        numpy.ndarray(uint8): brightness integer values from 0 to 255
                              for corresponding input rainfall intensity
        float: c1, scaling coefficient
        float: c2, scaling coefficient
    '''
    def mmh2rfl(r, a=256., b=1.42):
        '''
        .. based on wradlib.zr.r2z function
        .. r --> z
        '''
        return a * r ** b

    def rfl2dbz(z):
        '''
        .. based on wradlib.trafo.decibel function
        .. z --> d
        '''
        return 10. * np.log10(z)

    # mm to mm/h
    #X_mmh = depth2intensity(X_mm)
    # mm/h to reflectivity
    X_rfl = mmh2rfl(X)
    # remove zero reflectivity
    # then log10(0.1) = -1 not inf (numpy warning arised)
    X_rfl[X_rfl == 0] = 0.1
    # reflectivity to dBz
    X_dbz = rfl2dbz(X_rfl)
    # remove all -inf
    X_dbz[X_dbz < 0] = 0

    # MinMaxScaling
    c1 = X_dbz.min()
    c2 = X_dbz.max()

    return ((X_dbz - c1) / (c2 - c1) * 255).astype(np.uint8) #, c1, c2


class optical_flow(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super(optical_flow, self).__init__()
        self.input_channels = input_channels

        self.encoder1 = Encoder(4, kernel_size, dropout_rate)  # input depth is no of OF concatenated

        self.deconv3 = deconv(512, 256)
        self.deconv2 = deconv(256, 128)
        self.deconv1 = deconv(128, 64)
        self.deconv0 = deconv(64, 32)
        self.output_layer = nn.Conv2d(32, output_channels, kernel_size=kernel_size,
                                      padding=(kernel_size - 1) // 2)

    def forward(self, xx):
        # input is 5 raw input images (-1, 5, 512, 512)
        # print("xx size:",xx.size())
        # calculate optical flow between 2 images of 5 images:
        '''flow = []
        for i in range(0,4):
          t1 = xx[:,i,:,:].squeeze(0).cpu().detach().numpy()
          t2 = xx[:,i+1,:,:].squeeze(0).cpu().detach().numpy()
          #print("t1:",t1.dtype)
          #print("t2:",t2.shape)
          #im0_scale = t1
          #im1_scale = t2
          im0_scale = RYScaler(t1)
          im1_scale = RYScaler(t2)
          #print("im0_scale:",im0_scale.dtype)

          # TVL1 OF
          of_instance_tvl = cv2.optflow.DualTVL1OpticalFlow_create()
          f = of_instance_tvl.calc(im0_scale, im1_scale, None)
          #print("f:",f.dtype)
          f = np.moveaxis(f,-1,0)
          f = torch.from_numpy(f)
          f = f.unsqueeze(0).cuda()
          #print("f:",f.type())
          flow.append(f)

        #concatenate optical flows
        flow_concat = flow[0]
        for i in range(1,4):
          flow_concat = torch.cat([ flow_concat, flow[i] ], dim=1)'''

        out_conv1, out_conv2, out_conv3, out_conv4 = self.encoder1(xx)

        out_deconv3 = self.deconv3(out_conv4)
        out_deconv2 = self.deconv2(out_conv3)
        out_deconv1 = self.deconv1(out_conv2)
        out_deconv0 = self.deconv0(out_conv1)

        out = self.output_layer(out_deconv0)

        return out

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
                temp = sorted(glob.glob(path1 + "/" + str(month1[i]) + "/" + str(d) + "/*"))[:251]
                for item in temp:
                    self.files.append(item)

        for i in range(len(month2)):
            days = day2[i]
            for d in days:
                if month2[i]=='03':
                    temp = sorted(glob.glob(path2 + "/" + str(month2[i]) + "/" + str(d) + "/*"))[:251]
                else:
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
        for i in range(1, 4):
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
        #input_1 = cv2.resize(input_1, (512, 512), interpolation=cv2.INTER_CUBIC)

        input_2 = np.load(self.files[index + 1])['arr_0']
        #input_2 = cv2.resize(input_2, (512, 512), interpolation=cv2.INTER_CUBIC)

        input_3 = np.load(self.files[index + 2])['arr_0']
        #input_3 = cv2.resize(input_3, (512, 512), interpolation=cv2.INTER_CUBIC)

        #input_4 = np.load(self.files[index + 3])['arr_0']
        #input_4 = cv2.resize(input_4, (512, 512), interpolation=cv2.INTER_CUBIC)

        #input_5 = np.load(self.files[index + 4])['arr_0']
        #input_5 = cv2.resize(input_5, (512, 512), interpolation=cv2.INTER_CUBIC)

        output = np.load(self.files[index + 3])['arr_0']
        #output = cv2.resize(output, (512, 512), interpolation=cv2.INTER_CUBIC)
        # print("in get:",output.shape)

        input_1 = input_1.astype(np.float64)
        input_2 = input_2.astype(np.float64)
        input_3 = input_3.astype(np.float64)
        # input_4 = input_4.astype(np.float64)
        # input_5 = input_5.astype(np.float64)
        output = output.astype(np.float64)

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

        '''input_1 = self.preprocess(input_1)
        input_2 = self.preprocess(input_2)
        input_3 = self.preprocess(input_3)
        input_4 = self.preprocess(input_4)
        input_5 = self.preprocess(input_5)'''
        # input_6 = self.preprocess(input_6)
        #output = self.preprocess(output)
        # print("in get2:",output.size())

        # print(input_1,output)

        return input_1, input_2, input_3, output

    def __len__(self):
        # Total Number Of Samples
        return len(self.files) - 3  # subtract with no of input radar images given to model

def log1p_safe(x):
  """The same as torch.log1p(x), but clamps the input to prevent NaNs."""
  x = torch.as_tensor(x)
  return torch.log1p(torch.min(x, torch.tensor(33e37).to(x)))

def expm1_safe(x):
  """The same as tf.math.expm1(x), but clamps the input to prevent NaNs."""
  x = torch.as_tensor(x)
  return torch.expm1(torch.min(x, torch.tensor(87.5).to(x)))


def field_grad(f, dim):
    # dim = 1: derivative to x direction, dim = 2: derivative to y direction
    dx = 1
    dim += 1
    N = len(f.shape)
    out = torch.zeros(f.shape).cuda()
    slice1 = [slice(None)] * N
    slice2 = [slice(None)] * N
    slice3 = [slice(None)] * N
    slice4 = [slice(None)] * N

    # 2nd order interior
    slice1[-dim] = slice(1, -1)
    slice2[-dim] = slice(None, -2)
    slice3[-dim] = slice(1, -1)
    slice4[-dim] = slice(2, None)
    out[tuple(slice1)] = (f[tuple(slice4)] - f[tuple(slice2)]) / (2 * dx)

    # 2nd order edges
    slice1[-dim] = 0
    slice2[-dim] = 0
    slice3[-dim] = 1
    slice4[-dim] = 2
    a = -1.5 / dx
    b = 2. / dx
    c = -0.5 / dx
    out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
    slice1[-dim] = -1
    slice2[-dim] = -3
    slice3[-dim] = -2
    slice4[-dim] = -1
    a = 0.5 / dx
    b = -2. / dx
    c = 1.5 / dx

    out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
    return out


def vorticity(u,v):
    return field_grad(v, 0) - field_grad(u, 1)


# robust loss function
class robust_loss(nn.Module):
    def __init__(self, alpha, scale):
        super().__init__()
        self.alpha = alpha
        self.scale = scale
        self.approximate = False
        self.epsilon = 1e-6

    def forward(self, x):

        alpha = self.alpha.cuda()
        scale = self.scale.cuda()
        approximate = self.approximate
        epsilon = self.epsilon

        assert torch.is_tensor(x)
        assert torch.is_tensor(scale)
        assert torch.is_tensor(alpha)
        assert alpha.dtype == x.dtype
        assert scale.dtype == x.dtype
        assert (scale > 0).all()
        if approximate:
            # `epsilon` must be greater than single-precision machine epsilon.
            assert epsilon > np.finfo(np.float32).eps
            # Compute an approximate form of the loss which is faster, but innacurate
            # when x and alpha are near zero.
            b = torch.abs(alpha - 2) + epsilon
            d = torch.where(alpha >= 0, alpha + epsilon, alpha - epsilon)
            loss = (b / d) * (torch.pow((x / scale) ** 2 / b + 1., 0.5 * d) - 1.)
        else:
            # Compute the exact loss.
            # This will be used repeatedly.
            squared_scaled_x = (x / scale) ** 2
            # The loss when alpha == 2.
            loss_two = 0.5 * squared_scaled_x
            # The loss when alpha == 0.
            loss_zero = log1p_safe(0.5 * squared_scaled_x)
            # The loss when alpha == -infinity.
            loss_neginf = -torch.expm1(-0.5 * squared_scaled_x)
            # The loss when alpha == +infinity.
            loss_posinf = expm1_safe(0.5 * squared_scaled_x)

            # The loss when not in one of the above special cases.
            machine_epsilon = torch.tensor(np.finfo(np.float32).eps).to(x)
            # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
            beta_safe = torch.max(machine_epsilon, torch.abs(alpha - 2.))
            # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
            alpha_safe = torch.where(alpha >= 0, torch.ones_like(alpha),
                                     -torch.ones_like(alpha)) * torch.max(
                machine_epsilon, torch.abs(alpha))
            loss_otherwise = (beta_safe / alpha_safe) * (
                    torch.pow(squared_scaled_x / beta_safe + 1., 0.5 * alpha) - 1.)

            # Select which of the cases of the loss to return.
            loss = torch.where(
                alpha == -float('inf'), loss_neginf,
                torch.where(
                    alpha == 0, loss_zero,
                    torch.where(
                        alpha == 2, loss_two,
                        torch.where(alpha == float('inf'), loss_posinf,
                                    loss_otherwise))))

        return torch.mean(loss)  # calulating mean to get scalar instead of vector


class DivergenceLoss(torch.nn.Module):
    def __init__(self, loss, delta=1):
        super(DivergenceLoss, self).__init__()
        self.delta = delta
        self.loss = loss

    def forward(self, preds, trues):
        # preds: bs*steps*2*H*W
        u = preds[:, :1]
        v = preds[:, -1:]
        u_x = field_grad(u, 0)
        v_y = field_grad(v, 1)
        div_pred = v_y + u_x

        u = trues[:, :1]
        v = trues[:, -1:]
        u_x = field_grad(u, 0)
        v_y = field_grad(v, 1)
        div_true = v_y + u_x
        #residue for robust loss function
        residue = div_pred - div_true
        return self.loss(residue)

#Vorticity loss
class VorticityLoss(torch.nn.Module):
    def __init__(self, loss):
        super(VorticityLoss, self).__init__()
        self.loss = loss

    def forward(self, preds, trues):
        u, v = trues[:, :1], trues[:, -1:]
        u_pred, v_pred = preds[:, :1], preds[:, -1:]
        #residue for robust loss function
        residue = vorticity(u, v) - vorticity(u_pred, v_pred)
        return self.loss(residue)

def charbonnier_loss(output,target, epsilon = 1e-6):
    loss = torch.mean(torch.sqrt((output - target)**2 + epsilon * epsilon))
    return loss

def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
    f_path = checkpoint_dir + 'optical_flow_local.pth'
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

    net = optical_flow(input_channels=4, output_channels=2, kernel_size=3, dropout_rate=0)
    net = net.cuda()

    path1 = "/media/data/mot_1/data/crop1/binary/2019"
    path2 = "/media/data/mot_1/data/crop2/binary/2019"

    month1 = ['09']
    day1 = [['08']]

    month2 = ['03', '07']
    day2 = [['09'], ['03']]

    radar_image = radardata(path1, path2, month1, day1, month2, day2)
    dataloader = DataLoader(dataset=radar_image, batch_size=1, num_workers=0)

    n_epochs = 200
    batch_size = 1
    n_samples = radar_image.__len__()
    n_iter = n_samples / batch_size

    print("Total samples: ", n_samples)
    print("No of iterations per epoch: ", n_iter)

    lortentzian_loss = robust_loss(alpha=torch.Tensor([0]), scale=torch.Tensor([0.01]))
    divergence_loss = DivergenceLoss(loss=lortentzian_loss)
    vorticity_loss = VorticityLoss(loss=lortentzian_loss)

    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [80], gamma=0.1)

    save_path = '/media/data/mot_1/code/deep_learning/optical_flow/'

    # loading saved model to resume training from that point
    ckp_path = save_path + 'optical_flow_local.pth'
    net, optimizer, start_epoch = load_ckp(ckp_path, net, optimizer)
    print(start_epoch)

    for epoch in range(start_epoch, n_epochs):
        running_loss = 0.0

        for iter_num, (input_1, input_2, input_3, output) in enumerate(dataloader):

            input_1 = input_1.cpu().detach().numpy()
            input_2 = input_2.cpu().detach().numpy()
            input_3 = input_3.cpu().detach().numpy()
            # input_4 = input_4.cpu().detach().numpy()
            # input_5 = input_5.cpu().detach().numpy()
            # print("input_1",input_1.shape,input_1.dtype)

            input = np.concatenate((input_1, input_2, input_3), axis=0)
            # print("input:",input.shape)

            flow = []
            for i in range(0, 2):
                t1 = input[i, :, :]
                t2 = input[i + 1, :, :]
                # print("t1:",t1.dtype)
                # print("t2:",t2.shape)
                # im0_scale = t1
                # im1_scale = t2
                im0_scale = RYScaler(t1)
                im1_scale = RYScaler(t2)
                im0_scale = cv2.resize(im0_scale, (512, 512), interpolation=cv2.INTER_CUBIC)
                im1_scale = cv2.resize(im1_scale, (512, 512), interpolation=cv2.INTER_CUBIC)
                # print("im0_scale:",im0_scale.dtype)

                # TVL1 OF
                of_instance_tvl = cv2.DualTVL1OpticalFlow_create()
                f = of_instance_tvl.calc(im0_scale, im1_scale, None)
                # print("f:",f.dtype)
                f = np.moveaxis(f, -1, 0)
                f = torch.from_numpy(f)
                f = f.unsqueeze(0).cuda()
                # print("f:",f.type())
                flow.append(f)

            # concatenate optical flows
            flow_concat = flow[0]
            for i in range(1, 2):
                flow_concat = torch.cat([flow_concat, flow[i]], dim=1)

            # print("input flow:",flow_concat.size(), flow_concat.type())

            optimizer.zero_grad()

            pred_flow = net(flow_concat)

            # print("input_5:",input_5.shape,input_5.dtype)
            t1 = np.squeeze(input_3, axis=0)  # already converted to numpy above
            t2 = np.squeeze(output, axis=0).cpu().detach().numpy()
            # print("t2:",t2.shape,t2.dtype)

            im0_scale = RYScaler(t1)
            im1_scale = RYScaler(t2)
            im0_scale = cv2.resize(im0_scale, (512, 512), interpolation=cv2.INTER_CUBIC)
            im1_scale = cv2.resize(im1_scale, (512, 512), interpolation=cv2.INTER_CUBIC)
            # TVL1 OF
            of_instance_tvl = cv2.DualTVL1OpticalFlow_create()
            flow = of_instance_tvl.calc(im0_scale, im1_scale, None)
            flow = np.moveaxis(flow, -1, 0)
            flow = torch.from_numpy(flow)
            flow = flow.unsqueeze(0).cuda()

            residue = pred_flow - flow
            residue = residue.cuda()

            loss = lortentzian_loss(residue) + divergence_loss(pred_flow, flow) + vorticity_loss(pred_flow, flow)
            # loss = charbonnier_loss(pred_flow, flow)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss = loss.item()

            # correct = (outputs==output).float().sum()
            print(
                f'epoch {epoch + 1}/{n_epochs}, iteration step {iter_num + 1}/{n_iter},Loss: {running_loss}')

        # saving model
        checkpoint = {
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()
            }
        save_ckp(checkpoint, False, save_path, "")