import matplotlib.pyplot as plt
from torchvision.transforms.transforms import Resize
from torchvision.utils import make_grid
import torch
import numpy as np
from utils import *
from trainer import OCRTrainer
from models import CRNN
from loss import CustomCTCLoss
from tqdm import *
from dataset import SynthDataset, SynthCollator
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset
import queue
import torchvision.transforms.functional as TF


class MyDataset(IterableDataset):
    def __init__(self, image_queue, transform=None):
        self.queue = image_queue
        self.transform = transform

    def read_next_image(self):
        while self.queue.qsize() > 0:
            # you can add transform here
            yield self.queue.get()
        return None

    def __iter__(self):
        return self.read_next_image()


def safe_pil_loader(path, from_memory=False):

    try:
        if from_memory:
            img = Image.open(path)
            res = img.convert('RGB')
        else:
            with open(path, 'rb') as f:
                img = Image.open(f)
                res = img.convert('RGB')
    except:
        res = Image.new('RGB', (227, 227), color=0)
    return res


alphabet = """*QWERTYUIOPASDFGHJKLZXCVBNM1234567890-"""
args = {
    'name': 'exp3',
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
checkpoint = torch.load(resume_file, map_location='cpu')
model = CRNN(args)
model.load_state_dict(checkpoint['state_dict'])
args['model'] = model
model.eval()
converter = OCRLabelConverter(args['alphabet'])
evaluator = Eval()
# path = 'index.jpeg'
path = '2.jpeg'
# buffer = queue.Queue()
# new_input = Image.open(path)
# buffer.put(TF.to_tensor(new_input))
img = Image.open(path)
# w, h = img.size
# crop = img.crop((0,h/2,w,h))
# print(img.size)
# img = safe_pil_loader(path)
transform_list = transforms.Compose([transforms.Grayscale(1),
                  transforms.Resize((32,100)),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5,), (0.5,))])
img = transform_list(img)
# dataset = MyDataset(buffer,transform = transform_list)

# loader = torch.utils.data.DataLoader(args['data'],
#                                      batch_size=args['batch_size'],
#                                      collate_fn=args['collate_fn'])
labels, predictions, images = [], [], []
print(img.shape)
input_ = torch.unsqueeze(img, 0 )
print(1)
print(input_.shape)
# images.extend(input_.squeeze().detach())
# labels.extend(targets)
# targets, lengths = converter.encode(targets)
logits = model(input_).transpose(1, 0)
print(2)
logits = torch.nn.functional.log_softmax(logits, 2)
logits = logits.contiguous().cpu()
print(3)
T, B, H = logits.size()
pred_sizes = torch.LongTensor([T for i in range(B)])

probs, pos = logits.max(2)
# preds_prob = torch.from_numpy(probs).float().to(device)
print(4)
pos = pos.transpose(1, 0).contiguous().view(-1)
sim_preds = converter.decode(pos.data, pred_sizes.data, raw=False)
predictions.extend(sim_preds)
print(5)
# ca = np.mean(
#     (list(map(evaluator.char_accuracy, list(zip(predictions, labels))))))
# print(6)
# wa = np.mean(
#     (list(map(evaluator.word_accuracy_line, list(zip(predictions, labels))))))
# print("Character Accuracy: %.2f\nWord Accuracy: %.2f" % (ca, wa))
print(sim_preds)
print(predictions)
print(probs)
# print(preds_prob)

pro = torch.nn.functional.softmax(probs,dim=1)
confidence_score = pro.cumprod(dim=0)[-1]
print(confidence_score)