import sys, base64, json, os
from datetime import datetime
from data_parser import Parser
from train import Train
from predict import Predict



if len(sys.argv) > 2: 
    mode = sys.argv[1]
    input_ = sys.argv[2]
    input_ = base64.b64decode(input_).decode('utf-8')
    input_ = json.loads(input_)
else:
    print("Input parameter error.")



if mode == "parser":
    zip_name = input_["ZIP_NAME"]
    temp_path = input_["TEMP_PATH"]
    clean_path = input_["CLEAN_PATH"]
    init_path = input_["INIT_PATH"]
    data_info = input_["DATA_INFO"]
    log_path = input_["LOG_PATH"]
    

    parser = Parser()
    parser.data_clean(zip_name, temp_path, clean_path, init_path, data_info, log_path)


elif mode == "train":
    start = input_["START_DATE"]
    end = input_["END_DATE"]
    init_path = input_["INIT_PATH"]
    log_path = input_["TRAIN_LOG"]
    model_path = input_["MODEL_PATH"]


    if (start == None):
        start = "2022-05-25"
    if (end == None):
        end = datetime.today().strftime("%Y-%m-%d")


    start = datetime.strptime(start, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")


    if not os.path.isdir(model_path):
        os.makedirs(model_path)
        
    model_detail = os.path.join(model_path, "model")
    if not os.path.isdir(model_detail):
        os.makedirs(model_detail)


    train = Train(start, end, init_path, log_path, model_path, model_detail)
    train.main()


elif mode == "predict":
    input_path = input_["INPUT"]
    output_path = input_["OUTPUT"]
    model_detail = os.path.join(input_["MODEL"], "model")
    log_path = input_["LOG"]
    pred_path = os.path.dirname(input_path)


    predict = Predict(input_path, model_detail, output_path, log_path, pred_path)
    predict.main()