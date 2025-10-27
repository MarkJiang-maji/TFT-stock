# ---------------------------------------------------#
#
#   File       : PytorchForecasting.py
#   Author     : Soham Deshpande
#   Date       : January 2022
#   Description: Assembling and training the model
#                using Pytorch
#
#
# ----------------------------------------------------#


#Imports
#############################

#General
import datetime
import time
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import warnings
#Pytorch
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
# import pytorch_lightning as pl
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
# from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
# from pytorch_lightning.loggers import TensorBoardLogger
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_forecasting.data.encoders import NaNLabelEncoder
import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

##############################


warnings.filterwarnings("ignore")  #to avoid printing out absolute paths


class FTSEDataSet:
    """
    FTSE Dataset

    Extracsts the data from the CSV file
    Runs through data loaders
    Null values are removed
    Dataset is split into training, validation and testing datasets
    Converted into an appropriate format for the TFT

    """

    def __init__(self, start=datetime.datetime(2010, 1, 1), stop=datetime.datetime.now()):
        #self.df_returns = None
        #self.stocks_file_name = "/home/soham/Documents/PycharmProjects/NEA/Code/Data/NEAFTSE2010-21.csv"
        root = Path(__file__).resolve().parents[1]      # 專案根目錄：.../Stock-TFT-main
        self.stocks_file_name = root / "Data" / "NEAFTSE2010-21.csv"
        #self.start = start
        #self.stop = stop

    def load(self, binary = True):
        # print("Reading CSV:", self.stocks_file_name)
        # if not Path(self.stocks_file_name).exists():
        #     raise FileNotFoundError(f"CSV not found: {self.stocks_file_name}")

        # #start = self.start
        # #end = self.stop

        # df0 = pd.read_csv(self.stocks_file_name, index_col=0, parse_dates=True)
        # print(df0)
        # 讀 CSV（注意 dayfirst）
        df0 = pd.read_csv(
            self.stocks_file_name,
            index_col=0, parse_dates=True, dayfirst=True
        ).sort_index()
        print(df0)

        df0.dropna(axis=1, how='all', inplace=True)
        df0.dropna(axis=0, how='all', inplace=True)
        print("Dropping columns due to nans > 50%:",
              df0.loc[:, list((100 * (df0.isnull().sum() / len(df0.index)) > 50))].columns)# changed here
        
        cols_to_drop = df0.columns[df0.isnull().mean() > 0.5]   # >50% 缺值
        print("Dropping columns due to nans > 50%:", list(cols_to_drop))
        df0 = df0.drop(columns=cols_to_drop)
        df0 = df0.ffill().bfill()

        print("Any columns still contain nans:", df0.isnull().values.any())

        df_returns = pd.DataFrame()
        print(df_returns)
        for name in df0.columns:
            df_returns[name] = np.log(df0[name]).diff()
        print(df_returns)


        # split into train and test
        df_returns.dropna(axis=0, how='any', inplace=True)
        if binary:
            df_returns.FTSE = [1 if ftse > 0 else 0 for ftse in df_returns.FTSE]
        self.df_returns = df_returns
        return df_returns

    def get_loaders(self, batch_size=16, n_test=1000, device='cpu'):
        if self.df_returns is None:
            self.load()

        features = self.df_returns.drop('Open', axis=1).values
        labels = self.df_returns.FTSE
        training_data = data_utils.TensorDataset(torch.tensor(features[:-n_test]).float().to(device),
                                                 torch.tensor(labels[:-n_test]).float().to(device))
        test_data = data_utils.TensorDataset(torch.tensor(features[n_test:]).float().to(device),
                                             torch.tensor(labels[n_test:]).float().to(device))
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        return train_dataloader, test_dataloader



class TFT:

    """
    Temporal Fusion Transformer

    Setting up the model using PyTorch lighting.
    The class determines the main key features of the model, listed below:

    Tuneable Hyperparameters:
        int prediction length
        str   features
        int   max encoder length
        int   training cutoff
        str   time index
        str   group ids
        int   min encoder length
        int   min prediction length
        str   target
        int   max epochs
        int   gpus
        int   learning rate
        int   hidden layer size
        int   drop out
        int   hidden continous size
        int   output size
        int   attention head size
        float loss function

    """

    def __init__(self, prediction_length = 2000):
        self.prediction_length = prediction_length
        self.training = None
        self.validation = None
        self.trainer = None
        self.model = None
        self.batch_size =16

    def load_data(self):
        """
        Load data using the FTSEDataSet class
        Set prediction and encoder lengths
        Set up training data using TimeSeriesDataSet function
        """

        dataset = FTSEDataSet()
        print("Dataset",dataset)
        ftse_df = dataset.load(binary=False)
        print(dataset)
        time_index = "Date"
        target = "Open"
        
        # features：先從原始資料欄位抓（這時還沒有加上 Date/Open_Prediction）
        features = ftse_df.columns.tolist()
        print("Features",features)
        features.remove(target) # -> ['High', 'Low', 'Close']

        # 轉成 time_idx（以天為單位的整數）
        ftse_df[time_index] = pd.to_datetime(ftse_df.index)
        min_date = ftse_df[time_index].min()
        ftse_df[time_index] = (ftse_df[time_index] - min_date).dt.days.astype(int)

        # 單群組（單序列）
        ftse_df["Open_Prediction"] = "Open"
        ftse_df = ftse_df.sort_values(time_index)  # 保險

        # === 依資料量自動選安全長度（關鍵修正）===
        N = len(ftse_df)
        max_prediction_length = max(7, min(60, N // 20))   # 約 5% 的長度，上限 60
        max_encoder_length    = max(30, min(256, N // 4))  # 約 25% 的長度，上限 256
        min_encoder_length    = max_encoder_length // 2

        training_cutoff = int(ftse_df[time_index].max() - max_prediction_length)

        print(f"[TFT] N={N}, max_encoder_length={max_encoder_length}, "
            f"min_encoder_length={min_encoder_length}, "
            f"max_prediction_length={max_prediction_length}, "
            f"training_cutoff={training_cutoff}")
        print('time_idx', time_index)
        print("train slice preview:",
            ftse_df.loc[ftse_df[time_index] <= training_cutoff].head(3),
            sep="\n")

        # 建 TimeSeriesDataSet
        self.training = TimeSeriesDataSet(
            ftse_df[lambda x: x[time_index] <= training_cutoff],
            time_idx=time_index,
            target=target,
            categorical_encoders={"Open_Prediction": NaNLabelEncoder().fit(ftse_df.Open_Prediction)},
            group_ids=["Open_Prediction"],
            min_encoder_length=min_encoder_length,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            time_varying_unknown_reals=features,  # ['High','Low','Close']
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )

        print(self.training.get_parameters())

        # 驗證集（predict=True：每個序列最後一段拿來當預測目標）
        self.validation = TimeSeriesDataSet.from_dataset(
            self.training, ftse_df, predict=True, stop_randomization=True
        )

        # print(self.training.create_tft_model())

        # create validation set (predict=True) which means to predict the last max_prediction_length points in time
        # for each series
        self.validation = TimeSeriesDataSet.from_dataset(self.training, ftse_df, predict=True, stop_randomization=True)

    def create_tft_model(self):
        """
        Create the model
        Define hyperparameters
        Declare input, hidden, drop out, attention head and output size
        Declare epochs


        TFT Design
            1. Variable Selection Network
            2. LSTM Encoder
            3. Normalisation
            4. GRN
            5. MutiHead Attention
            6. Normalisation
            7. GRN
            8. Normalisation
            9. Dense network
            10.Quantile outputs


        """
        # configure network and trainer
        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=1e-4, patience=10, mode="min"
        )
        lr_logger = LearningRateMonitor()  # log the learning rate
        logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

        # Lightning 2.x 寫法（移除 weights_summary；devices 而不是 device）
        self.trainer = L.Trainer(
            max_epochs=30,
            accelerator='gpu',
            devices=1,
            # precision="16-mixed",  # 想更快可打開；不穩先關
            gradient_clip_val=0.1,
            callbacks=[lr_logger, early_stop_callback],
            logger=logger,
        )

        self.model = TemporalFusionTransformer.from_dataset(
            self.training,
            learning_rate=0.02,
            hidden_size=8,
            attention_head_size=2,
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=7,  # 7 quantiles
            loss=QuantileLoss(),
            reduce_on_plateau_patience=4,
        )
        print(f"Number of parameters in network: {self.model.size() / 1e3:.1f}k")
        
    def train(self):
        # DataLoader：Windows 建議 num_workers 小幅增加；GPU 用 pin_memory 可快一點
        train_dataloader = self.training.to_dataloader(
            train=True, batch_size=self.batch_size, num_workers=2, pin_memory=True
        )
        val_dataloader = self.validation.to_dataloader(
            train=False, batch_size=self.batch_size * 10, num_workers=2, pin_memory=True
        )

        self.trainer.fit(
            self.model,
            train_dataloader,
            val_dataloader,
        )

    # def evaluate(self, number_of_examples = 15):
    #     """
    #     Evaluate the model
    #     Load the saved model from the last saved epoch
    #     Compare predictions against real values
    #     Create graphs to visualise performance
    #     """
    #     # load the best model according to the validation loss
    #     # (given that we use early stopping, this is not necessarily the last epoch)
    #     best_model_path = self.trainer.checkpoint_callback.best_model_path        
    #     print("Best checkpoint:", best_model_path)
    #     best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        
        
    #     # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
    #     val_dataloader = self.validation.to_dataloader(train=False, batch_size=self.batch_size * 10, num_workers=0)
    #     raw_predictions, x = best_tft.predict(val_dataloader, mode="raw", return_x=True)
    #     #print('raw_predictions', raw_predictions)
    #     for idx in range(number_of_examples):  # plot 10 examples
    #         best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True);

    #     predictions, x = best_tft.predict(val_dataloader, return_x=True)
    #     #print('predictions2', predictions)
    #     #print('x values', x)
    #     predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(x, predictions)
    #     #print('predictions_vs_actuals', predictions_vs_actuals)
    #     best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals);
    #     #best_tft.plot(predictions,x)
    #     # print(best_tft)

    def evaluate(self, number_of_examples=15, save_dir="outputs", show=False):
        """
        Evaluate the model (PF 1.x 相容)
        - 不再依賴 predict(..., return_x=True) 的舊行為
        - 以 batch 對齊的方式繪圖與儲存
        """
        from pathlib import Path
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes
        import torch

        def _save_figlike(obj, basepath: Path):
            saved = []

            def _save_figure(fig: Figure, suffix=""):
                out = Path(f"{basepath}{suffix}.png")
                fig.savefig(out, dpi=150, bbox_inches="tight")
                print("saved:", out)
                saved.append(out)

            if obj is None:
                return saved
            if hasattr(obj, "savefig"):              # Figure
                _save_figure(obj)
            elif isinstance(obj, Axes) or hasattr(obj, "get_figure"):  # Axes
                fig = obj.get_figure()
                if fig is not None:
                    _save_figure(fig)
            elif isinstance(obj, dict):              # dict: 逐項遞迴存
                for k, v in obj.items():
                    saved += _save_figlike(v, Path(str(basepath) + f"_{k}"))
            elif isinstance(obj, (list, tuple)):     # list/tuple: 逐一存
                for i, v in enumerate(obj):
                    saved += _save_figlike(v, Path(str(basepath) + f"_{i}"))
            else:                                     # 退而求其次：存 gcf
                fig = plt.gcf()
                _save_figure(fig, "_gcf")
            return saved

        # 1) 拿最佳 checkpoint；沒有就用當前模型
        best_model_path = ""
        try:
            from lightning.pytorch.callbacks import ModelCheckpoint
            ckpts = [cb for cb in self.trainer.callbacks if isinstance(cb, ModelCheckpoint)]
            if ckpts:
                best_model_path = ckpts[0].best_model_path
        except Exception:
            pass

        print("Best checkpoint:", best_model_path if best_model_path else "(none)")
        if best_model_path:
            best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        else:
            best_tft = self.model

        # 2) 準備輸出資料夾
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # 3) 產生驗證 dataloader（保持與訓練相同設定）
        val_loader = self.validation.to_dataloader(
            train=False, batch_size=self.batch_size * 10, num_workers=0
        )

        # 4) 取得 raw 預測（PF 1.x：回傳通常是「每個 batch 的 raw dict」清單）
        raw_list = best_tft.predict(val_loader, mode="raw")
        # 若不是 list，就包成 list 以便對齊 zip
        if not isinstance(raw_list, (list, tuple)):
            raw_list = [raw_list]

        # 5) 產出「單一視窗：預測 vs 真值」圖（PF 1.x 穩定做法）
        try:
            import torch, numpy as np, matplotlib.pyplot as plt
            # 先拿點預測（非 raw），以及它對應的 index
            preds_list, idx_df = best_tft.predict(
                self.validation.to_dataloader(train=False, batch_size=self.batch_size * 10, num_workers=0),
                return_index=True
            )

            # 統一成 list[tensor] 方便逐 batch 對齊
            if not isinstance(preds_list, (list, tuple)):
                preds_list = [preds_list]

            # 逐 batch 跟 dataloader 對齊，畫幾張你要的「預測 vs 真值」
            made = 0
            for b, (x, y) in enumerate(self.validation.to_dataloader(train=False, batch_size=self.batch_size * 10, num_workers=0)):
                if b >= len(preds_list):
                    break
                y_pred_b = preds_list[b]                          # 形狀大約 (B, horizon[, 1])
                y_true_b = y[0] if isinstance(y, (list, tuple)) else y  # 常見形狀 (B, horizon[, 1])

                # 取 batch 裡第 j 筆樣本（你要多張就 loop j）
                j = 0
                # squeeze 成 1D
                yp = torch.as_tensor(y_pred_b[j]).detach().cpu().numpy().squeeze()
                yt = torch.as_tensor(y_true_b[j]).detach().cpu().numpy().squeeze()

                steps = np.arange(len(yp))
                plt.figure()
                plt.plot(steps, yt, label="observed")
                plt.plot(steps, yp, label="predicted")
                plt.legend()
                plt.title("Single window: observed vs predicted")
                plt.xlabel("forecast step")
                plt.ylabel("target")

                out = save_path / f"pred_vs_true_b{b}_j{j}.png"
                plt.savefig(out, dpi=150, bbox_inches="tight")
                print("saved:", out)
                plt.close()

                made += 1
                if made >= number_of_examples:
                    break
        except Exception as e:
            print(f"[WARN] single-window pred/true plot failed: {e}")

        # 6)（保留原本的整體圖）
        try:
            preds = best_tft.predict(self.validation.to_dataloader(train=False, batch_size=self.batch_size * 10, num_workers=0))
            if isinstance(preds, (list, tuple)):
                preds_cat = torch.cat([p if isinstance(p, torch.Tensor) else torch.as_tensor(p) for p in preds], dim=0)
            else:
                preds_cat = preds if isinstance(preds, torch.Tensor) else torch.as_tensor(preds)

            plt.figure()
            plt.plot(preds_cat.squeeze().detach().cpu().numpy())
            plt.title("Predictions (concatenated over validation)")
            plt.xlabel("time windows (concat)")
            plt.ylabel("predicted target")
            out = save_path / "predictions_overall.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            print("saved:", out)
            plt.close()

            saved_any = True
        except Exception as e:
            print(f"[WARN] overall prediction plot failed: {e}")

        if show and saved_any:
            try:
                plt.show()
            except Exception:
                pass
        else:
            plt.close("all")



def tft():
    tft = TFT()
    tft.load_data()
    tft.create_tft_model()
    tft.train()
    #torch.save(tft,"Model.pickle")
    tft.evaluate(number_of_examples=1)
    plt.show()


if __name__ == "__main__":

    # import matplotlib
    # matplotlib.use("Agg")
    import torch
    print("cuda_is_available:", torch.cuda.is_available(),
        "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    tft()





