import numpy as np
import torch

import os
import argparse
from torch.utils.data import Dataset, DataLoader

from skimage import io
import itertools

from SolarNet import SolarNet
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main(args):
    if args.cuda and not torch.cuda.is_available():
      raise Exception("No GPU found, please run without --cuda")
    num_class=args.num_class

    model=SolarNet(in_channels =args.colordim,n_class=num_class)

    if args.cuda:
      model=model.cuda()
    model.load_state_dict(torch.load(args.pretrain_net))
    model.eval()
    batch_size = args.predictbatchsize 
    with torch.no_grad():
      print('Inferencing begin')
      img = io.imread(args.pred_rgb_file).astype('float32') / 255.0
      img = img[:,:,0:3]
      pred = np.zeros(img.shape[:2] + (6,)) # 6 is the number of roof orientation classes
      pred2 = np.zeros(img.shape[:2] + (num_class,))
      batch_total = count_sliding_window(img, step=args.step, window_size=(args.img_size,args.img_size)) // batch_size
      print('Total Batch : ' + str(batch_total))

      for batch_idx, coords in enumerate(grouper(batch_size, sliding_window(img, step=args.step, window_size=(args.img_size,args.img_size)))):

                image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2,0,1)) for x,y,w,h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = torch.from_numpy(image_patches).cuda()

                outs = model(image_patches)
                outs_np = outs[1].detach().cpu().numpy()
                outs_np2 = outs[2].detach().cpu().numpy()

                for out, (x, y, w, h) in zip(outs_np, coords):
                    out = out.transpose((1,2,0))
                    pred[x:x+w, y:y+h] += out
                for out2, (x, y, w, h) in zip(outs_np2, coords):
                    out2 = out2.transpose((1,2,0))
                    pred2[x:x+w, y:y+h] += out2
                       
      pred = np.argmax(pred, axis=-1)
      io.imsave(args.pre_result,pred.astype(np.uint8))
      pred2 = np.argmax(pred2, axis=-1)
      io.imsave(args.pre_result2,pred2.astype(np.uint8))         


def sliding_window(img, step=128, window_size=(256,256)):

    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, img.shape[0], step):
        if x + window_size[0] > img.shape[0]:
            x = img.shape[0] - window_size[0]
        for y in range(0, img.shape[1], step):
            if y + window_size[1] > img.shape[1]:
                y = img.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]

def grouper(n, iterable):

    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def count_sliding_window(img, step=128, window_size=(256, 256)):

    """ Count the number of windows in an image """
    nSW = 0
    for x in range(0, img.shape[0], step):
        if x + window_size[0] > img.shape[0]:
            x = img.shape[0] - window_size[0]
        for y in range(0, img.shape[1], step):
            if y + window_size[1] > img.shape[1]:
                y = img.shape[1] - window_size[1]
            nSW += 1

    return nSW

# Prediction settings
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default=0, type=int,
                        help="a name for identifying the model")
    parser.add_argument('--cuda', default=True,
                        help="a name for identifying the model")
    parser.add_argument('--predictbatchsize', default=1, type=int,
                        help="input batch size per gpu for prediction")
    parser.add_argument('--threads', default=1, type=int,
                        help="number of threads for data loader to use")
    parser.add_argument('--colordim', default=3, type=int,
                        help="the channels of patch")
    parser.add_argument('--img_size', default=512, type=int,
                        help="patch size of the input")
    parser.add_argument('--step', default=128, type=int,
                        help="the overlap between neigbouring patch")
    parser.add_argument('--seed', default=123, type=int,
                        help="random seed to use")    
    parser.add_argument('--pretrain_net', default='./checkpoint-batchsize4-learning_rate0.001-optimizersgd/best_model.pth',
                        help='the name of checkpoint for predict')                                 
    parser.add_argument('--pred_rgb_file', default='large_img.tif',
                        help='the name of input rgb datasets for predict')
    parser.add_argument('--num_class', default=9, type=int,
                        help='number of classes of roof superstructurs')
    parser.add_argument('--pre_result', default='large_rf.png',
                        help='the name of predicted roof orientation map')
    parser.add_argument('--pre_result2', default='large_su.png',
                        help='the name of predicted roof superstructure map')
    args = parser.parse_args()
    main(args)