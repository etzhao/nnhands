import re
import base64
import torch
import numpy as np
import time
from libs.model import model, modelHeatmap, modelLocmap
from io import StringIO, BytesIO
from PIL import Image
from bottle import run, post, request, response, get, route, install
from libs.model_utils import makePosList

model.load_state_dict(torch.load('./models/model_param_e3_i7800.pt', map_location={'cuda:0': 'cpu'}))
modelHeatmap.load_state_dict(torch.load('./models/modelHeatmap_param_e3_i7800.pt', map_location={'cuda:0': 'cpu'}))
modelLocmap.load_state_dict(torch.load('./models/modelLocmap_param_e3_i7800.pt', map_location={'cuda:0': 'cpu'}))

# switch to eval mode
# important to switch all models to eval mode for batchnorm and dropout layers
model.eval()
modelHeatmap.eval()
modelLocmap.eval()

# flag used to track if the server is currently processing or not, if it is, don't process new images
processing = False

class EnableCors(object):
    name = 'enable_cors'
    api = 2

    def apply(self, fn, context):
        def _enable_cors(*args, **kwargs):
            # set CORS headers
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'

            if request.method != 'OPTIONS':
                # actual request; reply with the actual response
                return fn(*args, **kwargs)

        return _enable_cors

@route('/hello')
def hello():
    return "Hello World!"

@route('/prediction', method=['OPTIONS', 'POST', 'GET'])
def prediction():
    response.content_type = 'application/json'
    global processing
    global model
    global modelHeatmap
    global modelLocamap
    if processing:
        print("Still processing...")
        return {}
    processing = True
    #print(request.json.keys())
    data = request.json
    if data is None:
        # This is a get request
        return 'This is supposed to be a POST handler'
    #print(imgD[:25])
    imgstr = re.search(r'base64,(.*)', data['img']).group(1)
    image_bytes = BytesIO(base64.b64decode(imgstr))
    im = Image.open(image_bytes)
    img = np.array(im)[:,:,:]
    img = torch.from_numpy( np.expand_dims(img.transpose((2,0,1)), axis=0) ).float()
    print(img.size())
    # convert the img to suit our model's input
    start_time = time.time()
    y_pred = model(img)
    # h_pred is of size 21 x 224 x 224
    h_pred = modelHeatmap(y_pred)
    # l_pred is of size 3 x 224 x 224, the 3 representing x, y, z location maps of all 21 joints
    l_pred = modelLocmap(y_pred)
    print("Model speed: --- %s seconds ---" % (time.time() - start_time))
    p2d, p3d = makePosList(h_pred[0], l_pred[0], {'num_joints': 21, 'image_size': 224})

    # reset flag
    processing = False
    print("...Model done...")
    return {"p2d": p2d.tolist(), "p3d": p3d.tolist()}

install(EnableCors())

run(host='localhost', port=8080, debug=True)

# from bottle import route, run

# @route('/hello')
# def hello():
#     return "Hello World!"

# run(host='localhost', port=8080, debug=True)