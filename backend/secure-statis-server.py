import sys, os
sys.path.append('..')

import securestatis as sst
import securefe as sfe
import securefunc as sfunc
import numpy as np
import participants as pt
import syft as sy
import torch

from flask import Flask, jsonify, request, Response

app = Flask(__name__)

data_owners = None
crypto_provider = None
data = None
shares = None

def init():
    global data_owners, crypro_provider,data
    
    parties_num = 8
    owner_names = []
    for i in range(0, parties_num):
        owner_names.append("workers" + str(i))

    parties = pt.Parties()
    parties.init_parties(owner_names, "crypro_provider")

    data_owners = parties.get_parties()
    crypro_provider = parties.get_cryptopvd()

init()

def file_to_tensor():
    """

    """
    file_path = 'uploads/test.txt'  
    with open(file_path, 'r') as file:
        lines = file.readlines()

    matrix = []
    for line in lines:
        row = [float(x) for x in line.split()]  
        matrix.append(row)

    return torch.tensor(matrix, dtype=torch.float32)

def divide_into_shares():
    """

    """
    global data, data_owners, crypro_provider
    
    data = file_to_tensor()
    dim = len(data.shape)
    
    global shares
    shares = []
    if dim == 1:
        shares.append(sfe.secret_share(data, data_owners, crypto_provider, False))

    elif dim == 2:
        data = np.transpose(data)
        for d in data:
            shares.append(sfe.secret_share(d, data_owners, crypto_provider, False))

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            return Response("No file part", status=400)

        file = request.files['file']
        
        print("test")

        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], "test.txt")
            file.save(filename)
            
            divide_into_shares()
            return Response("File uploaded successfully", status=200)
        else:
            return Response("Invalid file type", status=400)
    

@app.route('/api/shares', methods=['GET'])
def get_shares():
    """
    Get the shares every worker has.
    """
    def get_worker_share(worker):
        total_size = 0
        objects = []
        for obj_id in worker._objects:
            obj = worker._objects[obj_id]
            objects.append(obj)
            total_size += obj.__sizeof__()
        return objects, total_size

    ret = []
    global data_owners
    for idx, i in enumerate(data_owners):
        objects, objects_total_size = get_worker_share(i)
        ret.append([idx, objects, objects_total_size])
        # print(f"Local Worker {idx}: {objects} objects, {objects_total_size} bytes")
    
    # TODO: 让ret更加标准一点
    return jsonify({"shares:": ret })

@app.route('/api/mean', methods=['GET'])
def secure_mean_compute():
    """

    """
    # data = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7]])

    fix_prec = 3
    prec = 5
    # mean = sst.secure_mean(parties.data_owners, parties.crypto_provider, data, prec)

    global data
    dim = len(data.shape)
    
    ret = []

    global shares
    if dim == 1:
        num = data.shape[0]
        for share in shares:
            sum = share.sum()
            mean = sfunc.secure_compute(sum, num, "div", prec)
            ret = {float(mean.get())/pow(10, fix_prec + prec)}

    elif dim == 2:
        data = np.transpose(data)
        mean = []
        for share in shares:
            mean.append(sfunc.secure_compute(share.sum(), share.shape[0], "div", prec))

        for m in mean:
            # print("The mean of data:", float(m.get())/pow(10, fix_prec + prec))
            ret.append(float(m.get())/pow(10, fix_prec + prec))

    # TODO: 让ret更加标准一点
    return jsonify({"shares:": ret })
    
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)

