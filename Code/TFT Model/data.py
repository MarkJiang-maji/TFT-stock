# ---------------------------------------------------#
#
#   File       : data.py
#   Description: Data loading and preprocessing for TFT
#                (extracted from PytorchForecasting.py)
#
# ----------------------------------------------------#

import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder


warnings.filterwarnings("ignore")


class FTSEDataSet:
    """
    FTSE Dataset

    Extracts the data from the CSV file
    Runs through data loaders
    Null values are removed
    Dataset is split into training, validation and testing datasets
    Converted into an appropriate format for the TFT

    """

    def __init__(self, start=datetime.datetime(2010, 1, 1), stop=datetime.datetime.now()):
        root = Path(__file__).resolve().parents[1]
        self.stocks_file_name = root / "Data" / "NEAFTSE2010-21.csv"

    def load(self, binary: bool = True, visualize: bool = True, save_dir: str = "outputs"):
        # 讀 CSV（注意 dayfirst）
        df0 = (
            pd.read_csv(self.stocks_file_name, index_col=0, parse_dates=True, dayfirst=True)
            .sort_index()
        )
        print("[data] raw shape:", df0.shape)
        print("[data] raw df head:\n", df0.head(3))

        df0.dropna(axis=1, how="all", inplace=True)
        df0.dropna(axis=0, how="all", inplace=True)
        na_pct = (df0.isnull().sum() / len(df0.index)).sort_values(ascending=False)
        print("[data] top-NA columns (%):\n", (na_pct * 100).head(10))
        print("Dropping columns due to nans > 50%:",
              df0.loc[:, list((100 * (df0.isnull().sum() / len(df0.index)) > 50))].columns)

        cols_to_drop = df0.columns[df0.isnull().mean() > 0.5]
        print("Dropping columns due to nans > 50%:", list(cols_to_drop))
        df0 = df0.drop(columns=cols_to_drop)
        df0 = df0.ffill().bfill()

        print("Any columns still contain nans:", df0.isnull().values.any())
        print("[data] cleaned df head:\n", df0.head(3))

        df_returns = pd.DataFrame()
        
        for name in df0.columns:
            df_returns[name] = np.log(df0[name]).diff()
        print("[data] returns df head:\n", df_returns.head(3))

        # split into train and test
        df_returns.dropna(axis=0, how="any", inplace=True)
        if binary and "FTSE" in df_returns.columns:
            df_returns.FTSE = [1 if ftse > 0 else 0 for ftse in df_returns.FTSE]
        self.df_returns = df_returns

        # --- simple visualizations ---
        if visualize:
            out_dir = (Path(__file__).resolve().parent / save_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            # 1) Price lines (if available)
            try:
                cols = [c for c in ["Open", "High", "Low", "Close"] if c in df0.columns]
                if cols:
                    plt.figure(figsize=(8, 3))
                    df0[cols].plot(ax=plt.gca())
                    plt.title("Raw prices (cleaned)")
                    plt.tight_layout()
                    p = out_dir / "raw_prices.png"
                    plt.savefig(p, dpi=130, bbox_inches="tight")
                    print("saved:", p)
                    plt.close()
            except Exception as e:
                print("[viz warn] raw price plot:", e)

            # 2) Returns histograms
            try:
                cols = [c for c in ["Open", "High", "Low", "Close"] if c in df_returns.columns]
                if cols:
                    df_returns[cols].plot(kind="hist", bins=50, alpha=0.6, subplots=True, layout=(2, 2), figsize=(8, 4))
                    plt.suptitle("Log-returns histograms")
                    plt.tight_layout()
                    p = out_dir / "returns_hist.png"
                    plt.savefig(p, dpi=130, bbox_inches="tight")
                    print("saved:", p)
                    plt.close()
            except Exception as e:
                print("[viz warn] returns hist:", e)

            # 3) Correlation heatmap (returns)
            try:
                corr = df_returns.corr(numeric_only=True)
                plt.figure(figsize=(5, 4))
                im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
                plt.colorbar(im, fraction=0.046, pad=0.04)
                plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
                plt.yticks(range(len(corr.index)), corr.index)
                plt.title("Returns correlation")
                plt.tight_layout()
                p = out_dir / "returns_corr.png"
                plt.savefig(p, dpi=130, bbox_inches="tight")
                print("saved:", p)
                plt.close()
            except Exception as e:
                print("[viz warn] corr heatmap:", e)

        return df_returns


def load_data(visualize: bool = True):
    """
    Load data using the FTSEDataSet class
    Set prediction and encoder lengths
    Set up training/validation TimeSeriesDataSet

    Returns
    -------
    training: TimeSeriesDataSet
    validation: TimeSeriesDataSet
    """

    dataset = FTSEDataSet()
    print("Dataset", dataset)
    ftse_df = dataset.load(binary=False, visualize=visualize)
    print(dataset)
    time_index = "Date"
    target = "Open"

    # features: start with all, then remove target
    features = ftse_df.columns.tolist()
    print("Features", features)
    if target in features:
        features.remove(target)

    # convert time index (days since min)
    ftse_df[time_index] = pd.to_datetime(ftse_df.index)
    min_date = ftse_df[time_index].min()
    ftse_df[time_index] = (ftse_df[time_index] - min_date).dt.days.astype(int)

    # group id
    ftse_df["Open_Prediction"] = "Open"
    ftse_df = ftse_df.sort_values(time_index)

    # window lengths
    N = len(ftse_df)
    max_prediction_length = max(7, min(60, N // 20))
    max_encoder_length = max(30, min(256, N // 4))
    min_encoder_length = max_encoder_length // 2

    training_cutoff = int(ftse_df[time_index].max() - max_prediction_length)

    print(
        f"[TFT] N={N}, max_encoder_length={max_encoder_length}, "
        f"min_encoder_length={min_encoder_length}, "
        f"max_prediction_length={max_prediction_length}, "
        f"training_cutoff={training_cutoff}"
    )
    print("time_idx", time_index)
    print(
        "train slice preview:",
        ftse_df.loc[ftse_df[time_index] <= training_cutoff].head(3),
        sep="\n",
    )

    training = TimeSeriesDataSet(
        ftse_df[lambda x: x[time_index] <= training_cutoff],
        time_idx=time_index,
        target=target,
        categorical_encoders={"Open_Prediction": NaNLabelEncoder().fit(ftse_df.Open_Prediction)},
        group_ids=["Open_Prediction"],
        min_encoder_length=min_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=features,  # e.g. ['High','Low','Close']
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    print(training.get_parameters())

    validation = TimeSeriesDataSet.from_dataset(
        training, ftse_df, predict=True, stop_randomization=True
    )

    return training, validation



if __name__ == "__main__":

    # import matplotlib
    # matplotlib.use("Agg")
    import torch
    print("cuda_is_available:", torch.cuda.is_available(),
        "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    load_data()
