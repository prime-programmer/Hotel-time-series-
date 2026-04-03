import os
import shutil
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch.callbacks import EarlyStopping
import logging
from src.data_loader import get_tft_data

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

def train_tft():
    print("Loading data...")
    df = get_tft_data()

    max_prediction_length = 10
    max_encoder_length = 24
    training_cutoff = df["time_idx"].max() - max_prediction_length

    print("Configuring TimeSeriesDataSet...")
    training = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="City_Bookings",
        group_ids=["group"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["group"],
        time_varying_known_categoricals=["month"],
        time_varying_known_reals=["time_idx", "Seasonal_Baseline", "ADR_Diff"],
        time_varying_unknown_reals=["City_Bookings", "City_LeadTime", "City_CancelRate"], 
        target_normalizer=GroupNormalizer(groups=["group"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
    batch_size = 32
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    # Best seed from assignement
    pl.seed_everything(123)

    print("Initializing Temporal Fusion Transformer...")
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        output_size=7,
        loss=QuantileLoss(),
        log_interval=0,
        reduce_on_plateau_patience=4,
    )

    trainer = pl.Trainer(
        max_epochs=35,
        accelerator="auto",
        gradient_clip_val=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10)],
        enable_model_summary=False,
        enable_progress_bar=True, 
    )

    print("Training model...")
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    best_model_path = trainer.checkpoint_callback.best_model_path
    
    os.makedirs("models", exist_ok=True)
    final_model_destination = "models/tft.ckpt"
    
    shutil.copy(best_model_path, final_model_destination)
    print(f"\nSuccess! Best TFT model saved to: {final_model_destination}")

if __name__ == "__main__":
    train_tft()