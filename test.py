import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn


model_path = './expr/new_netCRNN_49_20.pth'
img_path = './data/1.jpg'
alphabet = '0123456789-.'

model = crnn.CRNN(32, 1, 13, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)


image = Image.open(img_path)
scale=image.size[1]/32
new_w=int(image.size[0]/scale)
image=image.resize((new_w,32)).convert('L')
image=dataset.transforms.ToTensor()(image)
image.sub_(0.5).div_(0.5)
if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)
print(image.size())
model.eval()
preds = model(image)
print(preds)
_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
