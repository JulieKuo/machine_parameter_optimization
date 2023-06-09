import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import pickle, warnings, json, os, log_config, sys, base64
from datetime import datetime
from traceback import format_exc
warnings.filterwarnings("ignore")



class Train():
    def __init__(self, start, end, init_path, train_log, model_path, model_detail):
        self.shuffle = True
        self.start = start
        self.end = end
        self.init_path = init_path
        self.model_path = model_path
        self.model_detail = model_detail

        self.logging = log_config.set_log(filepath = train_log, level = 2, freq = "D", interval = 30)
    

    
    def load_data(self, start, end, init_path):
        df0 = pd.DataFrame()
        for dirs1 in os.listdir(init_path):
            if (int(dirs1) < start.year) or (int(dirs1) > end.year):
                continue

            for dirs2 in os.listdir(os.path.join(init_path, dirs1)): # dir - month
                if ((int(dirs1) <= start.year) & (int(dirs2) < start.month)) or ((int(dirs1) >= end.year) & (int(dirs2) > end.month)):
                    continue
                
                for dirs3 in os.listdir(os.path.join(init_path, dirs1, dirs2)): # dir - day
                    date = datetime.strptime(dirs3[:-4], "%Y%m%d")
                    if (date < start) or (date > end):
                        continue

                    path = os.path.join(init_path, dirs1, dirs2, dirs3)
                    df1 = pd.read_csv(path)
                    df0 = pd.concat([df0, df1], ignore_index = True)

        df0["time"] = pd.to_datetime(df0["time"])
        df0 = df0.dropna()

        return df0



    def preprocess(self):
        self.logging.info("{:=^80}".format(" Data preprocessing. "))


        self.logging.info("{:-^80}".format(" Loading data. "))
        df0 = self.load_data(self.start, self.end, self.init_path)
        self.logging.debug(f"The shape of the data.csv is {df0.shape}.")


        self.logging.info("{:-^80}".format(" Select features and target. "))
        self.target = "Hourly_Production"
        self.features = df0.columns[5:].to_list()

        df = df0.copy()
        df = df.set_index("time")
        df = df[[self.target] + self.features]


        self.logging.info("{:-^80}".format(" Slide window 10 min. "))

        #補上缺失的時間段
        time_ = pd.date_range(df.index[0], df.index[-1], freq = "min").to_frame(name = "time")
        df = pd.merge(df, time_, left_index = True, right_index = True, how = "right")

        #每十分鐘取平均
        df = df.drop("time", axis = 1)
        df = df.rolling(10).mean()
        df = df.dropna()

        # 刪除十分鐘內芯層泵啟停改變的sample
        pump = [0, 2]
        df = df.query("headbox_feed_fan_pump in @pump")

        self.logging.debug(f"The shape of the new data is {df.shape}. ")


        self.logging.info("{:-^80}".format(" Target analysis. "))
        df = df[(df[self.target] > 10)] # 砍掉異常狀態
        self.logging.debug(f"The shape of the new data is {df.shape}. ")


        self.logging.info("{:-^80}".format(" Split train, test data. "))
        train, test = train_test_split(df, test_size = 0.2, shuffle = self.shuffle)
        self.logging.debug(f"train shape: {train.shape}, test shape: {test.shape}")

        
        self.logging.info("{:-^80}".format(" Outlier. "))
        for col in self.features:
            Q1   = train[col].quantile(0.25)
            Q3   = train[col].quantile(0.75)
            IQR  = Q3 - Q1
            min_ = Q1 - (1.5 * IQR)
            max_ = Q3 + (1.5 * IQR)
            
            train[col] = train[col].apply(lambda X: max_ if X > max_ else X)
            train[col] = train[col].apply(lambda X: min_ if X < min_ else X)

            test[col] = test[col].apply(lambda X: max_ if X > max_ else X)
            test[col] = test[col].apply(lambda X: min_ if X < min_ else X)
        
        train.to_csv(f"{self.model_detail}/preprocess.csv", index = False)


        self.logging.info("{:-^80}".format(" Skew. "))
        skewness = train[self.features].apply(lambda X: skew(X)).sort_values(ascending=False)
        skewness = pd.DataFrame({'Feature' : skewness.index, 'Skew' : skewness.values})
        skewness = skewness.query("(Skew > 0.75) | (Skew < -0.75)")
        self.skewness = skewness.reset_index(drop = True)

        self.pt = PowerTransformer(method = 'yeo-johnson')
        train[self.skewness["Feature"]] = self.pt.fit_transform(train[self.skewness["Feature"]])
        test[self.skewness["Feature"]] = self.pt.transform(test[self.skewness["Feature"]])


        self.logging.info("{:-^80}".format(" Scaling. "))
        self.scaler = StandardScaler()
        train[self.features] = self.scaler.fit_transform(train[self.features])
        test[self.features] = self.scaler.transform(test[self.features])


        return train, test
    


    def model_prepare(self, train, test):
        self.logging.info("{:=^80}".format(" Modeling. "))


        self.logging.info("{:-^80}".format(" Split train, validate, test data. "))
        self.train_data, self.test_data = train, test
        self.train_data, self.valid_data = train_test_split(self.train_data, test_size = 0.2, shuffle = self.shuffle)
        self.logging.debug(f"train shape: {self.train_data.shape}, validate shape: {self.valid_data.shape}, test shape: {self.test_data.shape}")


        self.logging.info("{:-^80}".format(" DataFrame transform to torch dataset. "))
        class Dataset_transform(Dataset):
            def __init__(self, df, features, target):
                self.n_samples = len(df)
                self.X = torch.Tensor(df[features].values)#.to(device)
                self.y = torch.Tensor(df[target].values.reshape(-1, 1))#.to(device)
                                                    
            def __len__(self):
                return self.n_samples

            def __getitem__(self, index):
                return self.X[index], self.y[index]
        
        train_dataset = Dataset_transform(self.train_data, self.features, self.target)
        valid_dataset = Dataset_transform(self.valid_data, self.features, self.target)
        test_dataset = Dataset_transform(self.test_data, self.features, self.target)


        self.logging.info("{:-^80}".format(" DataLoader to use for batch. "))
        self.train_dataloader = DataLoader(train_dataset, batch_size = 128, shuffle = self.shuffle)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size = len(valid_dataset))
        self.test_dataloader = DataLoader(test_dataset, batch_size = len(test_dataset))


        self.logging.info("{:-^80}".format(" Bulid model structure. "))
        class Model(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.net  = nn.Sequential(
                    nn.Linear(input_size, 32),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(32),
                    nn.Linear(32, 32),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(32),
                    nn.Linear(32, 1),
                )
            
            def forward(self, x):
                x = self.net(x)
                return x
        
        self.epochs = 500
        self.model = Model(input_size = len(self.features))
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = 1e-3)


        self.logging.info("{:-^80}".format(" Initialize weights. "))
        # 初始化權重，使其符合常態分布
        for m in self.model.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.kaiming_normal_(m.weight)



    # 計算validate、test的損失函數，以及相關分數
    def eval_score(self, dataloader, model, criterion, mode = "eval"): 
        with torch.no_grad():
            losses = 0
            pred1, y1 = torch.Tensor([]), torch.Tensor([])
            for batch, (X, y) in enumerate(dataloader):
                pred = model(X) # 預測
                loss = criterion(pred, y) # 計算損失函數
                losses += loss.item()

                pred1 = torch.concat([pred1, pred])
                y1 = torch.concat([y1, y])

            losses /= (batch + 1)

            if mode == "train":
                return losses
                

            pred1 = pred1.detach().numpy()
            y1 = y1.detach().numpy()
            MSE = mean_squared_error(y1, pred1)
            RMSE = MSE ** (1/2)
            MAPE = mean_absolute_percentage_error(y1, pred1)
            # SMAPE = (abs(y1 - pred1) / ((abs(y1) + abs(pred1)) / 2)).mean()
            R2 = r2_score(y1, pred1)
        

        return MSE, RMSE, MAPE, R2, pred1, y1
    


    def train_model(self):
        self.logging.info("{:-^80}".format(" Train model. "))


        # modeling
        best_loss = np.inf
        paitence = 30
        train_losses = []
        valid_losses = []
        for epoch in range(self.epochs):
            train_loss = 0
            valid_loss = 0
            # train model
            self.model.train() # 模型為訓練模式
            for batch, (X_train, y_train) in enumerate(self.train_dataloader):
                train_pred = self.model(X_train) #預測
                loss = self.criterion(train_pred, y_train) #計算損失函數

                self.optimizer.zero_grad() # 梯度在反向傳播前先清零
                loss.backward() # 反向傳播，計算權重對損失函數的梯度
                self.optimizer.step()  # 根據梯度更新權重
                train_loss += loss.item()
            train_loss /= (batch + 1)
            train_losses.append(train_loss)

            # validate model
            self.model.eval()# 模型為評估模式
            valid_loss = self.eval_score(self.valid_dataloader, self.model, self.criterion, mode = "train")
            valid_losses.append(valid_loss)

            self.logging.debug(f"Epoch {epoch} - train_loss: {train_loss:.4f},  valid_loss: {valid_loss:.4f}")

            # 損失函數連續30個epoches都沒下降的話就終止訓練
            if valid_loss < best_loss:
                best_loss = valid_loss
                remain_patience = paitence
            else:
                remain_patience -= 1
                if remain_patience == 0:
                    self.logging.debug('early stop!')
                    break
        self.logging.debug("Done!")


        # plot losses curve
        plt.figure(figsize=(15, 4))
        plt.plot(train_losses, label='Training loss')
        plt.plot(valid_losses, label='Validation loss')
        plt.title("Training and Validation losses for each epoch")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(f"{self.model_path}/chart.png")
    


    def test_model(self):
        self.logging.info("{:-^80}".format(" Test model. "))


        # predict
        data_name = ["Train", "Validation", "Test"]
        dataloaders = [self.train_dataloader, self.valid_dataloader, self.test_dataloader]
        score = pd.DataFrame(columns = ["MSE", "RMSE", "MAPE", "R2"])
        result = {}

        for name, dataloader in zip(data_name, dataloaders):
            MSE, RMSE, MAPE, R2, pred, true = self.eval_score(dataloader, self.model, self.criterion)
            score.loc[name] = [MSE, RMSE, MAPE, R2]
            result[name] = {
                "true": true.reshape(-1),
                "pred": pred.reshape(-1),
            }

        train_result = pd.DataFrame(result["Train"], index = self.train_data.index).sort_index()
        valid_result = pd.DataFrame(result["Validation"], index = self.valid_data.index).sort_index()
        test_result = pd.DataFrame(result["Test"], index = self.test_data.index).sort_index()
        
        score["COUNT"] = [len(self.train_data), len(self.valid_data), len(self.test_data)]
        self.score = score.round(4)


        # save result
        with open(f'{self.model_path}/train.json', 'w') as f:
            json.dump(self.score.to_dict(), f)
        

        # plot predict result
        data = [train_result, valid_result, test_result]
        fig, ax = plt.subplots(3, 1, figsize = (20, 18))
        for i in range(3):
            ax[i].plot(data[i])
            ax[i].set(ylabel = self.target, xlabel = "Sample", title = data_name[i])
            ax[i].legend(["true", "pred"], fontsize = 11)
        fig.savefig(f"{self.model_path}/pred.png")



    def save_model(self):
        self.logging.info("{:-^80}".format(" Save model. "))


        # 儲存model
        pickle.dump(self.features, open(f'{self.model_detail}/feat_order.pkl','wb'))
        skew_feat = [list(self.train_data.columns[1:]).index(self.skewness["Feature"][i]) for i in range(len(self.skewness["Feature"]))]
        pickle.dump(skew_feat, open(f'{self.model_detail}/skew_feat.pkl','wb'))
        pickle.dump(self.pt, open(f'{self.model_detail}/power_tf.pkl','wb'))
        pickle.dump(self.scaler, open(f'{self.model_detail}/scaler.pkl','wb'))
        torch.save(self.model.state_dict(), f"{self.model_detail}/nn_weights.pt") # 儲存權重


        # 抓出各產出正負5的sample的所有feature之四分位數
        range_path = f"{self.model_detail}/pred_range"
        if not os.path.isdir(range_path):
            os.makedirs(range_path)
            
        df1 = pd.read_csv(f"{self.model_detail}/preprocess.csv")

        for i in range(10, 46):
            df_range = df1[(df1[self.target] <= i+5) & (df1[self.target] >= i-5)]
            df_range = df_range.describe().T[["min", "25%", "50%", "75%", "max"]]

            vibration = [i for i in df_range.index if "Motor_Side_Vibration" in i]
            df_range.loc[vibration] = df_range.loc[vibration].applymap(lambda X: 20 if X > 20 else X) # 震動值不可大於20
            
            df_range = df_range.reset_index()
            df_range = df_range.rename(columns = {"index": "feature"})
            
            df_range.to_csv(f"{range_path}/{i}.csv", index = False)


        # 前端的預設參數
        df40_range = pd.read_csv(f"{range_path}/40.csv")
        df40_range = df40_range.set_index("feature")
        df40_range.columns = ["q0", "q1", "q2", "q3", "q4"]

        with open(f'{self.model_path}/parameter.json', 'w') as f:
            json.dump(df40_range.T.to_dict(), f)
        

        self.logging.info("{:=^80}".format(" Finished. "))



    def main(self):
        try:
            train, test = self.preprocess()
            self.model_prepare(train, test)
            self.train_model()
            self.test_model()
            self.save_model()
        except:
            self.logging.error(format_exc())
    



if __name__ == '__main__':
   
    if len(sys.argv) > 1: 
        input_ = sys.argv[1]
        input_ = base64.b64decode(input_).decode('utf-8')

        input_ = json.loads(input_)
    else:
        print("Input parameter error.")
        

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