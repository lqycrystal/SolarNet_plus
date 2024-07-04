import torch
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, CenterCrop, ToTensor
from PIL import Image
import cv2

from skimage import io

from SolarNet import SolarNet

class generateDataset(Dataset):

        def __init__(self, dirFiles,img_size,colordim,isTrain=True):
                self.isTrain = isTrain
                self.dirFiles = dirFiles
                self.nameFiles = [name for name in os.listdir(dirFiles) if os.path.isfile(os.path.join(dirFiles, name))]
                self.numFiles = len(self.nameFiles)
                self.img_size = img_size
                self.colordim = colordim
                print('number of files : ' + str(self.numFiles))
                
        def __getitem__(self, index):
                filename = self.dirFiles + self.nameFiles[index]
                img=io.imread(filename)
                img=img[:,:,0:3]/255.0
                img = torch.from_numpy(img).float()
                img = img.transpose(0, 1).transpose(0, 2)
                imgName, imgSuf = os.path.splitext(self.nameFiles[index])
                return img, imgName
        
        def __len__(self):
                return int(self.numFiles)

def main(args):
    if args.cuda and not torch.cuda.is_available():
      raise Exception("No GPU found, please run without --cuda")
    num_class=args.num_class
    if args.id==0:
      model=SolarNet(in_channels =args.colordim,n_class=num_class)
    if args.cuda:
      model=model.cuda()
    model.load_state_dict(torch.load(args.pretrain_net))
    model.eval()
    predDataset = generateDataset(args.pre_root_dir, args.img_size, args.colordim, isTrain=False)
    predLoader = DataLoader(dataset=predDataset, batch_size=args.predictbatchsize, num_workers=args.threads)
    with torch.no_grad():
      cm_w = np.zeros((2,2))
      for batch_idx, (batch_x, batch_name) in enumerate(predLoader):
        batch_x = batch_x
        
        if args.cuda:
            batch_x = batch_x.cuda()
        out = model(batch_x)
        out2=out[1]
        out3=out[2]
        pred_prop2, pred_label2 = torch.max(out2, 1)
        pred_label_np2 = pred_label2.cpu().numpy() 
        pred_prop3, pred_label3 = torch.max(out3, 1)
        pred_label_np3 = pred_label3.cpu().numpy()    
        for id in range(len(batch_name)):
                pred_label_single2 = pred_label_np2[id, :, :]
                predLabel_filename2 = args.preDir2 +  batch_name[id] + '.png'
                cv2.imwrite(predLabel_filename2, pred_label_single2.astype(np.uint8))
                pred_label_single3 = pred_label_np3[id, :, :]
                predLabel_filename3 = args.preDir3 +  batch_name[id] + '.png'
                cv2.imwrite(predLabel_filename3, pred_label_single3.astype(np.uint8))



# Prediction settings
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default=0,type=int,
                        help="a name for identifying the model")
    parser.add_argument('--cuda', default=True,
                        help="a name for identifying the model")
    parser.add_argument('--predictbatchsize', default=1,type=int,
                        help="input batch size per gpu for prediction")
    parser.add_argument('--threads', default=1,type=int,
                        help="number of threads for data loader to use")
    parser.add_argument('--img_size', default=512,type=int,
                        help="image size of the input")
    parser.add_argument('--seed', default=123,type=int,
                        help="random seed to use")
    parser.add_argument('--colordim', default=3,type=int,
                        help="color dimension of the input image") 
    parser.add_argument('--pretrain_net', default='./checkpoint-batchsize4-learning_rate0.001-optimizersgd/best_model.pth',
                        help='path of saved pretrained model')                       
    parser.add_argument('--pre_root_dir', default='./wbf_data/test/data/',
                        help='path of input datasets for predict')
    parser.add_argument('--num_class', default=9, type=int,
                        help='number of classes of superstructures')
    parser.add_argument('--preDir2', default='./predictionroofsegment/',
                        help='path of prediction for roof orientation map')
    parser.add_argument('--preDir3', default='./predictionsuperstructure/',
                        help='path of prediction for roof superstructure map')
    args = parser.parse_args()
    if not os.path.isdir(args.preDir2):
        os.makedirs(args.preDir2)
    if not os.path.isdir(args.preDir3):
        os.makedirs(args.preDir3)
    main(args)