import re
import base64
import torch
import numpy as np
import time
from libs.model import model, modelHeatmap, modelLocmap
from io import StringIO, BytesIO
from PIL import Image
from libs.model_utils import makePosList

model.load_state_dict(torch.load('./models/model_param_e3_i7800.pt'))
modelHeatmap.load_state_dict(torch.load('./models/modelHeatmap_param_e3_i7800.pt'))
modelLocmap.load_state_dict(torch.load('./models/modelLocmap_param_e3_i7800.pt'))

# switch to eval mode
# important to switch all models to eval mode for batchnorm and dropout layers
model = model.eval()
modelHeatmap = modelHeatmap.eval()
modelLocmap = modelLocmap.eval()

model = model.cuda()
modelHeatmap = modelHeatmap.cuda()
modelLocmap = modelLocmap.cuda()

# flag used to track if the server is currently processing or not, if it is, don't process new images
# convert the img to suit our model's input

img = torch.randn(1, 3, 224, 224).cuda()

start_time = time.time()
y_pred = model(img)
# h_pred is of size 21 x 224 x 224
h_pred = modelHeatmap(y_pred)
# l_pred is of size 3 x 224 x 224, the 3 representing x, y, z location maps of all 21 joints
l_pred = modelLocmap(y_pred)
print("Model speed: --- %s seconds ---" % (time.time() - start_time))
