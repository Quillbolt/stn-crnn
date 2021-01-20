import matplotlib.pyplot as plt
from torchvision.transforms.transforms import Resize
from torchvision.utils import make_grid
import torch
import numpy as np
from utils import *
from trainer import OCRTrainer
from models import CRNN, STNet
from loss import CustomCTCLoss
from tqdm import *
from dataset import SynthDataset, SynthCollator
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset
import queue
# from loss import CustomCTCLoss
import torchvision.transforms.functional as TF
from PIL import Image
from easyocr import Reader
import os
import torch
# path = 'Vietnamese_plate_for_Ho_Chi_Minh_City.jpeg'
# img = Image.open(path)
# img.resize((32,100))
alphabet = """*QWERTYUIOPASDFGHJKLZXCVBNM1234567890-"""
args = {
    'name': 'exp4',
    'path': 'OCR',
    'imgdir': 'train',
    'imgH': 32,
    'nChannels': 1,
    'nHidden': 256,
    'nClasses': len(alphabet)+1,
    'lr': 0.001,
    'epochs': 4,
    'batch_size': 32,
    'save_dir': 'checkpoints',
    'log_dir': 'logs',
    'resume': False,
    'cuda': False,
    'schedule': False,
    'alphabet': alphabet

}
resume_file = os.path.join(args['save_dir'], args['name'], 'best.ckpt')
print('Loading model %s' % resume_file)
checkpoint = torch.load(resume_file)
model =  torch.nn.Sequential(STNet(),
                        CRNN(args))
model.load_state_dict(checkpoint['state_dict'])
converter = OCRLabelConverter(args['alphabet'])
evaluator = Eval()
transform_list = transforms.Compose([transforms.Grayscale(1),
                  transforms.Resize((32,100)),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5,), (0.5,))])
reader = Reader(['en'],gpu=True)
path = 'images.jpeg'
img = Image.open(path)
# w, h = img.size
# crop = img.crop((0,h/2,w,h))
# crop.show()
txt = reader.readtext(path,allowlist=alphabet)
print(txt)
result = reader.detect(path)
# print(result)
text = torch.IntTensor(args['batch_size'] * 5)
length = torch.IntTensor(args['batch_size'])
ctc_loss = CustomCTCLoss()
crop = []
for i in result:
    print(i)
    for x,y,z,t in i:
    # x,y,z,t = i
        cropped_img = img.crop((x,z,y,t))
        crop.append(cropped_img)
predictions = []
for img in crop:
    imgT = transform_list(img)
    input_ = torch.unsqueeze(imgT, 0 )
    logits = model(input_).transpose(1, 0)
    
    logits = torch.nn.functional.log_softmax(logits, 2)
    pred_norm = logits.sum(axis=2)
    logits = logits.contiguous().cpu()
    T, B, H = logits.size()
    print(T,B,H)
    pred_sizes = torch.LongTensor([T for i in range(B)])
    probs, pos = logits.max(2)
    pos = pos.transpose(1, 0).contiguous().view(-1)
    sim_preds = converter.decode(pos.data, pred_sizes.data, raw=False)
    predictions.extend(sim_preds)
    print(pos.shape)
    preds_prob = logits.cpu().detach().numpy()
    pred_norm = pred_norm.cpu().detach().numpy()
    preds_prob = preds_prob/np.expand_dims(pred_norm, axis=-1)
    preds_prob = torch.from_numpy(preds_prob).float().to(device)
 
    print(pred_sizes)
    preds_max_prob, _ = preds_prob.max(dim=2)
    
    # input_lengths = torch.full(size=(1,), fill_value=T, dtype=torch.long)
    # target_lengths = torch.randint(low=1, high=T, size=(1,), dtype=torch.long)
    # pro2 = ctc_loss(logits,pred_sizes,input_lengths,pos))   
    # print(pro) 
    # print(pro2)
    # pro2 = torch.exp(probs)
    confidence_score = preds_max_prob.cumprod(dim=0)[-1]
    # confidence_score2 = pro2.cumprod(dim=0)[-1]
    print(confidence_score.item())
    # print(confidence_score2.item())
print(predictions)