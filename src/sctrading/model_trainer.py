import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import pandas as pd
from pathlib import Path
from datetime import datetime, date, timezone, timedelta
import logging
import numpy as np
import os
from typing import List, Tuple, Dict, Generator
from collections import deque
import csv
import time
from concurrent.futures import ThreadPoolExecutor

# Import your existing data pipeline components
from data_manager import DataManager
from pretraining_dataset_builder import PretrainingDatasetBuilder

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- LossLogger Class (Unchanged) ---
class LossLogger:
    def __init__(self, log_path: Path, rolling_window_size: int = 100):
        self.log_path, self.rolling_window_size = log_path, rolling_window_size
        self.losses = deque(maxlen=rolling_window_size)
        self.data_times = deque(maxlen=rolling_window_size)
        self.gpu_times = deque(maxlen=rolling_window_size)
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'batch', 'raw_loss', 'data_time_s', 'gpu_time_s', 'avg_epoch_train_loss', 'validation_loss', 'learning_rate'])
    def log_batch(self, epoch: int, batch_idx: int, loss: float, data_time: float, gpu_time: float):
        self.losses.append(loss)
        self.data_times.append(data_time)
        self.gpu_times.append(gpu_time)
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, batch_idx, loss, data_time, gpu_time, '', '', ''])
    def log_epoch(self, epoch: int, avg_train_loss: float, validation_loss: float, learning_rate: float):
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, 'N/A', '', '', '', avg_train_loss, validation_loss, learning_rate])
    def get_rolling_avg_loss(self) -> float:
        return sum(self.losses) / len(self.losses) if self.losses else 0.0
    def get_rolling_avg_data_time(self) -> float:
        return sum(self.data_times) / len(self.data_times) if self.data_times else 0.0
    def get_rolling_avg_gpu_time(self) -> float:
        return sum(self.gpu_times) / len(self.gpu_times) if self.gpu_times else 0.0

# --- PyTorch Model & Dataset Classes ---
class PretrainingLSTM(nn.Module):
    """
    --- NEW: A 3-layer LSTM model with Layer Normalization for stability. ---
    """
    def __init__(self, input_size: int, hidden_sizes: List[int], dropout: float, output_size: int = 1):
        super(PretrainingLSTM, self).__init__()
        self.layers = nn.ModuleList()
        
        current_size = input_size
        for hidden_size in hidden_sizes:
            # Each "layer" now consists of an LSTM, LayerNorm, and Dropout
            self.layers.append(nn.LSTM(current_size, hidden_size, batch_first=True))
            self.layers.append(nn.LayerNorm(hidden_size))
            self.layers.append(nn.Dropout(dropout))
            current_size = hidden_size
        
        # Final fully connected layer
        self.fc = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        h = x
        # Iterate through the layers in steps of 3 (LSTM, LayerNorm, Dropout)
        for i in range(0, len(self.layers), 3):
            lstm = self.layers[i]
            norm = self.layers[i+1]
            drop = self.layers[i+2]
            
            h, _ = lstm(h)
            h = norm(h)
            # No dropout on the last layer's output before the FC layer
            if i < len(self.layers) - 3:
                h = drop(h)
        
        # We only use the output of the last time step for prediction
        out = self.fc(h[:, -1, :])
        return out

class GeneratorDataset(IterableDataset):
    def __init__(self, generator_func, **kwargs):
        self.generator_func = generator_func
        self.kwargs = kwargs
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        return self.generator_func(worker_info=worker_info, **self.kwargs)

# --- Main Training Logic ---
def run_fixed_block_training(dataset_builder: PretrainingDatasetBuilder, model: nn.Module, 
                             training_start_date: date, training_end_date: date, 
                             validation_start_date: date, validation_end_date: date,
                             batch_size: int, num_epochs: int, models_dir: Path, device, num_workers: int,
                             debug_ticker: str = None):
    # --- CHANGE: Adjusted learning rate for smaller batch size ---
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5) 
    criterion = nn.MSELoss() 
    log_path = models_dir / "fixed_training_log.csv"
    loss_logger = LossLogger(log_path)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 5
    GRAD_CLIP_NORM = 1.0
    total_batch_counter = 0

    logger.info(f"--- Starting Training from {training_start_date} to {training_end_date} for {num_epochs} epochs ---")
    if debug_ticker:
        logger.warning(f"--- RUNNING IN DEBUG MODE ON A SINGLE TICKER: {debug_ticker} ---")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        num_epoch_batches = 0
        
        date_chunks = dataset_builder.get_date_chunks(training_start_date, training_end_date)
        
        logger.info(f"--- Epoch {epoch+1}, Pre-loading first data chunk... ---")
        current_chunk_cache = dataset_builder.load_data_for_chunk(date_chunks[0][0], date_chunks[0][1], debug_ticker)

        with ThreadPoolExecutor(max_workers=1) as executor:
            for i in range(len(date_chunks)):
                chunk_start, chunk_end = date_chunks[i]
                
                future = None
                if i + 1 < len(date_chunks):
                    next_chunk_start, next_chunk_end = date_chunks[i+1]
                    future = executor.submit(dataset_builder.load_data_for_chunk, next_chunk_start, next_chunk_end, debug_ticker)

                logger.info(f"--- Training on current chunk: {chunk_start} to {chunk_end} ---")
                if not current_chunk_cache:
                    if future: current_chunk_cache = future.result()
                    continue

                train_dataset = GeneratorDataset(dataset_builder.generate_samples_from_cache, data_cache=current_chunk_cache, chunk_start_date=chunk_start, chunk_end_date=chunk_end, seed=epoch, debug_ticker=debug_ticker)
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, prefetch_factor=2)
                
                data_load_start_time = time.time()
                for features, targets, context in train_dataloader:
                    data_load_end_time = time.time()
                    data_time = data_load_end_time - data_load_start_time
                    
                    gpu_start_time = time.time()
                    features, targets = features.to(device), targets.to(device).float()
                    target_mean, target_std = targets.mean(), targets.std()
                    targets_normalized = (targets - target_mean) / target_std if target_std > 1e-6 else targets - target_mean
                    
                    optimizer.zero_grad()
                    output = model(features)
                    loss = criterion(output, targets_normalized)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    optimizer.step()
                    gpu_end_time = time.time()
                    gpu_time = gpu_end_time - gpu_start_time
                    
                    batch_loss = loss.item()
                    epoch_loss += batch_loss
                    num_epoch_batches += 1
                    total_batch_counter += 1
                    loss_logger.log_batch(epoch + 1, total_batch_counter, batch_loss, data_time, gpu_time)
                    
                    if total_batch_counter % 100 == 0:
                        logger.info(f"  Epoch {epoch+1}/{num_epochs}, Batch {total_batch_counter}, Loss: {loss_logger.get_rolling_avg_loss():.6f}, "
                                    f"Data Time: {loss_logger.get_rolling_avg_data_time():.4f}s, GPU Time: {loss_logger.get_rolling_avg_gpu_time():.4f}s")
                    
                    data_load_start_time = time.time()

                if future:
                    current_chunk_cache = future.result()

        avg_epoch_loss = epoch_loss / num_epoch_batches if num_epoch_batches > 0 else 0
        logger.info(f"--- Epoch {epoch+1} Complete. Average Training Loss: {avg_epoch_loss:.6f} ---")

        model.eval()
        val_loss, val_batches = 0, 0
        val_date_chunks = dataset_builder.get_date_chunks(validation_start_date, validation_end_date)
        with torch.no_grad():
            for chunk_start, chunk_end in val_date_chunks:
                logger.info(f"--- Validating on chunk: {chunk_start} to {chunk_end} ---")
                val_chunk_cache = dataset_builder.load_data_for_chunk(chunk_start, chunk_end, debug_ticker)
                if not val_chunk_cache: continue
                
                val_dataset = GeneratorDataset(dataset_builder.generate_samples_from_cache, data_cache=val_chunk_cache, chunk_start_date=chunk_start, chunk_end_date=chunk_end, seed=epoch, debug_ticker=debug_ticker)
                val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

                for features, targets, context in val_dataloader:
                    features, targets = features.to(device), targets.to(device).float()
                    target_mean, target_std = targets.mean(), targets.std()
                    targets_normalized = (targets - target_mean) / target_std if target_std > 1e-6 else targets - target_mean
                    output = model(features)
                    loss = criterion(output, targets_normalized)
                    val_loss += loss.item()
                    val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        logger.info(f"--- Validation Loss for Epoch {epoch+1}: {avg_val_loss:.6f} ---")

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        logger.info(f"Current Learning Rate: {current_lr:.7f}")
        loss_logger.log_epoch(epoch + 1, avg_epoch_loss, avg_val_loss, current_lr)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), models_dir / "best_pretrained_model.pt")
            logger.info(f"New best model saved with validation loss: {best_val_loss:.6f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping after {patience} epochs with no improvement.")
                break
        
    logger.info("--- Training Complete ---")
    return model

if __name__ == "__main__":
    try:
        logging.getLogger("file_io").setLevel(logging.WARNING)
        logging.getLogger("data_manager").setLevel(logging.WARNING)

        DEBUG_MODE = False
        NUM_FEATURES = 22 

        LOCAL_CACHE_DIR = Path("/home/kyle/data/polygon_cache")
        MODELS_DIR = LOCAL_CACHE_DIR / "models"
        MODELS_DIR.mkdir(exist_ok=True)
        
        dm = DataManager(LOCAL_CACHE_DIR)
        dataset_builder = PretrainingDatasetBuilder(
            dm, 
            universe_file_path=LOCAL_CACHE_DIR / "universes" / "pretraining_universe.json",
            prediction_horizons_bars=[1, 2, 3, 6, 12, 36, 78],
            num_features=NUM_FEATURES
        )
        
        model = PretrainingLSTM(
            input_size=NUM_FEATURES, 
            hidden_sizes=[512, 512, 256], 
            dropout=0.3,
            output_size=len(dataset_builder.prediction_horizons_bars) 
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(model)
        
        debug_ticker_symbol = None
        if DEBUG_MODE:
            training_start = date(2021, 1, 4)
            training_end = date(2021, 3, 31)
            validation_start = date(2021, 4, 1)
            validation_end = date(2021, 4, 30)
            debug_ticker_symbol = 'AAIC'
        else:
            training_start = dataset_builder.start_date
            training_end = training_start + timedelta(days=365)
            validation_start = training_end + timedelta(days=7)
            validation_end = validation_start + timedelta(days=30)
        
        run_fixed_block_training(
            dataset_builder=dataset_builder, model=model,
            training_start_date=training_start, training_end_date=training_end,
            validation_start_date=validation_start, validation_end_date=validation_end,
            # --- CHANGE: Reduced batch size ---
            batch_size=512, 
            num_epochs=20, models_dir=MODELS_DIR,
            device=model.fc.weight.device, num_workers=(os.cpu_count() or 4) // 2,
            debug_ticker=debug_ticker_symbol
        )
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}", exc_info=True)
