import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
from collections import OrderedDict
import models.crnn as crnn


model_path = './expr/new_netCRNN_49_20.pth'
img_path = './data/2.jpg'
alphabet = '0123456789-.'

model = crnn.CRNN(32, 1, 13, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
state_dict=torch.load(model_path)
'''
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove module.
    new_state_dict[name] = v
'''
model.load_state_dict(state_dict)


#model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((128, 32))
image = Image.open(img_path).convert('L')
image = transformer(image)
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
