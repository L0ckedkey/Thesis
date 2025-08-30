import torch
import csv
import os
import time
import matplotlib.pyplot as plt
import numpy as np


def get_general_model_size(file_path):
    """Check model size in MB and save to a log file in the same folder"""
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    msg = f"📦 Model size: {size_mb:.2f} MB"
    print(msg)

    # Simpan ke file di folder yang sama
    folder = os.path.dirname(file_path)
    log_path = os.path.join(folder, "model_size.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

    return size_mb

def get_model_size(model, tmp_path="temp_model.pth"):
    torch.save(model.state_dict(), tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    os.remove(tmp_path)
    return size_mb

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_model(model, dataloader, criterion, device):
    model.to(device)
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    inference_times = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            start_time = time.time()
            outputs = model(X_batch)
            inference_times.append(time.time() - start_time)

            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == y_batch).sum().item()
            total_samples += y_batch.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    avg_inf_time_batch = np.mean(inference_times)
    avg_inf_time_sample = avg_inf_time_batch / dataloader.batch_size

    return avg_loss, accuracy, avg_inf_time_batch, avg_inf_time_sample

def train_model(model, criterion, optimizer, train_loader, val_loader, device, num_epochs, patience, log_path, result_dir):
    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    # 📌 Hitung ukuran & jumlah parameter model
    model_size = get_model_size(model)
    num_params = count_parameters(model)
    print(f"Model size: {model_size:.2f} MB | Parameters: {num_params:,}")

    with open(log_path, mode="w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
            "model_size_MB", "num_params", "avg_infer_time_ms", "throughput_samples_per_s"
        ])

    for epoch in range(num_epochs):
        # ----------------- TRAIN -----------------
        model.train()
        running_loss, correct, total = 0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # ----------------- VALIDATION -----------------
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        infer_times = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                start_time = time.perf_counter()
                outputs = model(X_batch)
                end_time = time.perf_counter()

                infer_times.append((end_time - start_time) * 1000)  # ms

                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        avg_infer_time = sum(infer_times) / len(infer_times)  # ms per batch
        throughput = (val_total / (sum(infer_times) / 1000))  # samples/sec

        # Simpan loss & acc
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Logging ke CSV
        with open(log_path, mode="a", newline="") as f:
            csv.writer(f).writerow([
                epoch+1, train_loss, train_acc, val_loss, val_acc,
                model_size, num_params, avg_infer_time, throughput
            ])

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"Infer Time: {avg_infer_time:.2f} ms/batch | "
              f"Throughput: {throughput:.2f} samples/s")

        # ----------------- EARLY STOPPING -----------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(result_dir, "best_model.pth"))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # ----------------- PLOT -----------------
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.legend()
    plt.savefig(os.path.join(result_dir, "loss_acc_plot.png"))
    plt.close()

    # Load best model
    model.load_state_dict(torch.load(os.path.join(result_dir, "best_model.pth")))
    return model
