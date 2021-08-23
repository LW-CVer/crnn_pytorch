import torch
import models.crnn as crnn
from collections import OrderedDict
batch_size=1
h=32
w=128
nclass=13
x=torch.randn(batch_size,*(1,h,w)).cuda()
model=crnn.CRNN(32,1,13,256)
model.load_state_dict(torch.load('./expr/new_netCRNN_49_20.pth'))
'''
state_dict=torch.load("./weight/best_model.pth.tar")["state_dict"]
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove module.
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
#print(type(model),model)
'''
model=model.cuda()
model.eval()
'''
torch.onnx.export( model,
        x,
        "./onnx/trt_crnn.onnx",
        opset_version=11,
        verbose=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["f_score,f_geomotry"])
'''
torch.onnx.export( model,
        x,
        "./onnx/trt_crnn_9.onnx",
        opset_version=9,
        verbose=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input":{0:"batch_size",3:"w"},
                      "output":{0:"batch_size",3:"w"}})
