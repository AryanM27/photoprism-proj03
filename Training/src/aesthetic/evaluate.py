import torch
import torch.nn.functional as F

def evaluate_model(model, dataloader, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    preds = []
    targets = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            scores = batch["score"].to(device)

            outputs = model(images).squeeze()

            loss = F.mse_loss(outputs, scores, reduction="sum")

            total_loss += loss.item()
            total_samples += scores.size(0)

            preds.append(outputs.detach().cpu())
            targets.append(scores.detach().cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)

    # Metrics
    mse = F.mse_loss(preds, targets).item()
    rmse = mse ** 0.5
    mae = torch.mean(torch.abs(preds - targets)).item()

    avg_loss = total_loss / total_samples

    return {
        "val_loss": avg_loss,
        "val_mae": mae,
        "val_rmse": rmse
    }