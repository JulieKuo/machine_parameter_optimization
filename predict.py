import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time, pickle, warnings, json, os, log_config, sys, base64
from traceback import format_exc
warnings.filterwarnings("ignore")



class Predict():
    def __init__(self, input_path, model_detail, output_path, log_path, pred_path):
        self.input_path = input_path
        self.model_detail = model_detail
        self.output_path = output_path
        self.pred_path = pred_path

        self.logging = log_config.set_log(filepath = log_path, level = 2, freq = "D", interval = 30)

        self.target = "Hourly_Production"



    def load_model(self):
        self.logging.info("{:-^80}".format(" Load model. "))


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
            
            
        # 載入訓練好的模型
        self.features = pickle.load(open(f'{self.model_detail}/feat_order.pkl','rb'))
        self.skew_feat = pickle.load(open(f'{self.model_detail}/skew_feat.pkl','rb'))
        self.pt = pickle.load(open(f'{self.model_detail}/power_tf.pkl','rb'))
        self.scaler = pickle.load(open(f'{self.model_detail}/scaler.pkl','rb'))

        self.model = Model(input_size = len(self.features))
        self.model.load_state_dict(torch.load(f"{self.model_detail}/nn_weights.pt")) # 更改model權重
        self.model.eval()



    def load_input(self):
        self.logging.info("{:-^80}".format(" Load input data. "))
        with open(self.input_path, newline='') as file:
            input_data = json.load(file)

        self.target1 = input_data[self.target]["init"]

        input_data1 = pd.DataFrame(input_data)

        #乾強使用量、木漿流量、芯層泵啟停不可改變
        input_data1.loc["fixed", ["dry_strength_use", "flow_nukp", "headbox_feed_fan_pump"]] = 1

        self.input_X = input_data1.loc[["init"], self.features]
        fixed_mask = input_data1.loc[["fixed"], self.features]

        self.fixed_mask = fixed_mask.values[0]
    


    def check_distribution(self):
        self.logging.info("{:-^80}".format(" Check input data is in distribution range. "))
        

        self.init_X = self.input_X.values.copy()
        df_range = pd.read_csv(f"{self.model_detail}/pred_range/{self.target1}.csv")
        self.df_range = df_range.iloc[1:].reset_index(drop = True)

        # # 查看輸入的參數是否在25% ~ 75%之間
        # for i, x in enumerate(init_X[0]):
        #     if (x < self.df_range.iloc[i, 2]) or (x > self.df_range.iloc[i, 4]):
        #         if (self.fixed_mask[i] == 0):
        #             self.logging.debug(f"{i} : ({round(float(x), 4)}) not in {self.df_range.iloc[i, 2].round(4)} ~ {self.df_range.iloc[i, 4].round(4)}")
        #         else:
        #             self.logging.debug(f"{i} : ({round(float(x), 4)}) not in {self.df_range.iloc[i, 2].round(4)} ~ {self.df_range.iloc[i, 4].round(4)},  but parameter is fixed.")

        
        # 如果輸入的參數不在25% ~ 75%之間，就用中位數取代
        self.logging.debug(f"init X: {self.init_X}")
        for j, x in enumerate(self.init_X[0]):
            if ((x < self.df_range.iloc[j, 2]) or (x > self.df_range.iloc[j, 4])) and (self.fixed_mask[j] == 0):
                self.init_X[0][j] = self.df_range.iloc[j, 3]
        self.logging.debug(f"new X: {self.init_X}")
    


    def transform(self, X):
        X[0, self.skew_feat] = self.pt.transform(X[0, self.skew_feat].reshape(1, -1)) # 偏態轉換
        X[0] = self.scaler.transform(X[0].reshape(1, -1)) # 標準化轉換
        X = torch.Tensor(X)

        return X

    
    def check_optimized(self):
        self.logging.info("{:-^80}".format(" Check parameters can be optimized. "))


        # 如果參數調整後預測值仍無法藉於0~50，此輪的input不可用，應減少fixed mask的數量
        X = self.init_X.copy()
        X = self.transform(X)
        self.logging.debug(f"init X = {self.init_X}")
        self.logging.debug(f"transform X = {X}")


        init_pred = self.model(X).item()
        self.logging.debug(f"target = {self.target1}")
        self.logging.debug(f"predict = {init_pred}")


        if (init_pred < 20) or (init_pred > 50):
            self.logging.info("{:-^80}".format(" Unable to optimize input data. Please change the fixed setting of the parameters."))

            update = {
                "status": "fail",
                "reason": "Unable to optimize input data. Please change the fixed setting of the parameters."
            }

            with open(self.output_path, 'w') as f:
                json.dump(update, f)
            

            return False
        else:
            return True



    def optimize(self):
        self.logging.info("{:-^80}".format(" Optimizing. "))
        start = time.time()
        preds = []
        losses = []
        h = 1e-3 # 參數的變化量
        learn_rate = 1e-2 # 1e3
        best_loss = np.inf
        loss_limit = 50
        boundary_low = self.target1 - 0.1 
        boundary_high = self.target1 + 0.1 
        boundary_limit = 20
        remain_boundary = 20
        time_limit = 40
        epoch = 1
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-08
        v = np.zeros(len(self.init_X[0]))
        s = np.zeros(len(self.init_X[0]))
        X = self.init_X.copy()
        while True:
            for i in range(len(X[0])):
                # 若該參數為固定值，梯度為0，不更新參數
                if self.fixed_mask[i]:
                    continue
                
                # 計算梯度: dloss_dx = (loss(x+h) - loss(x-h)) / (2*h)
                X_up = X.copy()
                X_down = X.copy()

                X_up[0][i] += h
                X_down[0][i] -= h

                X_up = self.transform(X_up)
                X_down = self.transform(X_down)

                loss_up = (self.target1 - self.model(X_up)) ** 2
                loss_down = (self.target1 - self.model(X_down)) ** 2

                dloss_dx = (loss_up - loss_down) / (2 * h)

                # 以Adam的方式更新參數，需先計算v、s
                # v = bata1 * v + (1 - beta1) * dloss_dweight  # Momentum: 累積過去梯度，讓跟當前趨勢同方向的參數有更多的更新，即沿著動量的方向越滾越快
                # s = bata2 * s + (1 - beta2) * (dloss_dweight ⊙ dloss_dweight) # Adagrad: 累積過去梯度，以獲得參數被修正程度，修正大的參數學習率會逐漸變小
                v[i] = (beta1 * v[i]) + ((1 - beta1) * dloss_dx.item())
                s[i] = beta2 * s[i] + (1 - beta2) * np.multiply(dloss_dx.item(), dloss_dx.item())

            # 透過梯度計算新的參數
            # weight = weight - learning_rate * (1 / ((s + eps) ** (1/2))) * v  # eps: 是極小值，避免s為0時發生除以0的情況
            grad = (learn_rate * (1 / ((s + eps) ** (1/2))) * v)
            new_X = (X[0] - grad).reshape(1, -1)

            # 確認新參數是否在25%~75%的分布範圍內，並將不在分布範圍內的新參數的梯度轉為0，此次不更新該參數
            mask = [True if (new_x >= self.df_range.iloc[j, 2]) and (new_x <= self.df_range.iloc[j, 4]) else False for j, new_x in enumerate(new_X[0])]
            # mask = torch.Tensor(mask)
            grad *= mask

            # 更新參數
            X[0] -= grad

            # 查看新預測結果
            new_X1 = self.transform(new_X)
            pred = self.model(new_X1).item()
            preds.append(pred)

            loss = (self.target1 - pred) ** 2
            losses.append(loss)
            self.logging.debug(f"Epoch {epoch} - loss: {loss:.4f},  predict: {pred:.4f}")

            # 損失函數連續n個epoches都沒下降的話就終止訓練
            if loss < best_loss:
                best_loss = loss
                remain_loss = loss_limit
            else:
                remain_loss -= 1
                if remain_loss == 0:
                    self.logging.debug('early stop (unable to converge)!')
                    break

            # 預測產出達標就終止訓練
            if (pred < boundary_low) or (pred > boundary_high):
                remain_boundary = boundary_limit
            else:
                remain_boundary -= 1
                if remain_boundary == 0:
                    # 輸出時X要轉為小數點後一位，確認轉換後仍滿足條件
                    X1 = np.round(X, 2)
                    pred_round1 = self.model(self.transform(X1.copy())).item()
                    if (pred_round1 >= boundary_low) or (pred_round1 <= boundary_high):
                        self.logging.debug('early stop (reach the standard)!')
                        break
                    else: 
                        remain_boundary += 1

            # 時間到就終止訓練
            end = time.time()
            if ((end - start) > time_limit):
                self.logging.debug('Done!')
                break
            else:
                epoch += 1

        self.X = np.round(X, 2)
        self.pred_round1 = round(self.model(self.transform(self.X.copy())).item(), 2)
        self.logging.debug(f"new X: {self.X}\nnew pred: {self.pred_round1}")


        plt.figure(figsize=(20,5))
        plt.plot(losses)
        plt.title("Loss results after adjusting the input for each epoch")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig(f"{self.pred_path}/loss.png")


        plt.figure(figsize=(20,5))
        plt.plot(preds)
        plt.title("Prediction results after adjusting  the input for each epoch")
        plt.xlabel("epoch")
        plt.ylabel("pred")
        plt.savefig(f"{self.pred_path}/pred.png")



    def save_output(self):
        self.logging.info("{:-^80}".format(" Save output. "))

        
        self.X = pd.DataFrame(self.X, columns = self.features, index = ["new"])
        update = pd.concat([self.input_X, self.X]).T
        update["fixed"] = self.fixed_mask
        update["change"] = update.eval("(init != new)").astype(int)

        update = update.T.to_dict()
        update[self.target] = {"init": self.target1, "new": self.pred_round1}
        update["status"] = "success"
        update["reason"] = ""
        with open(self.output_path, 'w') as f:
            json.dump(update, f)



    def main(self):
        try:
            self.logging.info("{:=^80}".format(" Predicting."))
            self.load_model()
            self.load_input()
            self.check_distribution()
            optimize_flag = self.check_optimized()
            
            if optimize_flag:
                self.optimize()
                self.save_output()

            self.logging.info("{:=^80}".format(" Finished."))
        except:
            self.logging.error(format_exc())

            update = {
                "status": "fail",
                "reason": format_exc()
            }

            with open(self.output_path, 'w') as f:
                json.dump(update, f)
    



if __name__ == '__main__':

    if len(sys.argv) > 1: 
        input_ = sys.argv[1]
        input_ = base64.b64decode(input_).decode('utf-8')

        input_ = json.loads(input_)
    else:
        print("Input parameter error.")


    input_path = input_["INPUT"]
    output_path = input_["OUTPUT"]
    model_detail = os.path.join(input_["MODEL"], "model")
    log_path = input_["LOG"]
    pred_path = os.path.dirname(input_path)


    predict = Predict(input_path, model_detail, output_path, log_path, pred_path)
    predict.main()