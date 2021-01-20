import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch
import numpy as np
from utils import *
from trainer import OCRTrainer
from models import CRNN,STNet
from loss import CustomCTCLoss
from tqdm import *
from dataset import SynthDataset, SynthCollator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def get_accuracy(args):
    loader = torch.utils.data.DataLoader(args['data'],
                batch_size=args['batch_size'],
                collate_fn=args['collate_fn'])
    model = args['model']
    model.eval()
    converter = OCRLabelConverter(args['alphabet'])
    evaluator = Eval()
    labels, predictions, images = [], [], []
    for iteration, batch in enumerate(tqdm(loader)):
        input_, targets = batch['img'].to(device), batch['label']
        images.extend(input_.squeeze().detach())
        labels.extend(targets)
        targets, lengths = converter.encode(targets)
        logits = model(input_).transpose(1, 0)
        logits = torch.nn.functional.log_softmax(logits, 2)
        logits = logits.contiguous().cpu()
        T, B, H = logits.size()
        pred_sizes = torch.LongTensor([T for i in range(B)])
        probs, pos = logits.max(2)
        pos = pos.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(pos.data, pred_sizes.data, raw=False)
        predictions.extend(sim_preds)
        
    # make_grid(images[:10], nrow=2)
    # fig=plt.figure(figsize=(8, 8))
    # columns = 4
    # rows = 5
    # pairs = list(zip(images, predictions))
    # indices = np.random.permutation(len(pairs))
    # for i in range(1, columns*rows +1):
    #     img = images[indices[i]]
    #     img = (img - img.min())/(img.max() - img.min())
    #     img = img.cpu()
    #     img = np.array(img * 255.0, dtype=np.uint8)
    #     fig.add_subplot(rows, columns, i)
    #     plt.title(predictions[indices[i]])
    #     plt.axis('off')
    #     plt.imshow(img)
    # plt.show()
    ca = np.mean((list(map(evaluator.char_accuracy, list(zip(predictions, labels))))))
    wa = np.mean((list(map(evaluator.word_accuracy_line, list(zip(predictions, labels))))))
    return ca, wa

if __name__ == "__main__":
    alphabet = """*QWERTYUIOPASDFGHJKLZXCVBNM1234567890-"""
    args = {
        'name': 'exp5',
        'path': 'content',
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
        'cuda': True,
        'schedule': False,
        'alphabet': alphabet
    }
    
    args['imgdir'] = 'val'
    args['data'] = SynthDataset(args)
    args['collate_fn'] = SynthCollator()
    resume_file = os.path.join(args['save_dir'], args['name'], 'best.ckpt')
    if os.path.isfile(resume_file):
        print('Loading model %s'%resume_file)
        checkpoint = torch.load(resume_file)
        model =  torch.nn.Sequential(STNet(),
                        CRNN(args))
        model.load_state_dict(checkpoint['state_dict'])
        args['model'] = model.cuda()
        ca, wa = get_accuracy(args)
        print("Character Accuracy: %.2f\nWord Accuracy: %.2f"%(ca, wa))
    else:
        print("=> no checkpoint found at '{}'".format(os.path.join(args['save_dir'], args['name'])))
        print('Exiting')
