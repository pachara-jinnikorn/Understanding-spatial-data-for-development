import torch
from tqdm import tqdm
import numpy as np

def format_logs(logs):
    formatted_logs = []
    for k, v in logs.items():
        if "CELoss" in k:  # Keep CELoss as is (not a percentage)
            formatted_logs.append("{}={:.2f}".format(k, v))
        else:  # Convert mIoU and other ratios to percentage
            formatted_logs.append("{}={:.2f}%".format(k, v * 100))
    return ", ".join(formatted_logs)



def train_epoch(model, optimizer, criterion, metric, dataloader, device="cpu"):
    loss_meter = 0
    score_meter = 0
    iou_per_class = None
    count = 0
    
    model.to(device).train()
    with tqdm(dataloader, desc="Train") as iterator:
        for sample in iterator:
            x = sample["x"].to(device)
            y = sample["y"].to(device)
            n = x.shape[0]
            
            optimizer.zero_grad()
            outputs = model.forward(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            # ✅ Fix: Unpack metric output before calling .cpu()
            mean_iou, iou_scores = metric(outputs, y)  
            mean_iou = mean_iou.item()  # Convert tensor to scalar

            if isinstance(iou_scores, np.ndarray):
                if iou_per_class is None:
                    iou_per_class = np.zeros_like(iou_scores)
                iou_per_class += iou_scores
            
            loss_meter += loss.cpu().detach().numpy() * n
            score_meter += mean_iou * n
            count += n
            
            logs = {"CELoss": loss_meter / count, "mIoU": score_meter / count}
            iterator.set_postfix_str(format_logs(logs))
    
    iou_per_class /= count if iou_per_class is not None else np.zeros(8)
    return logs, iou_per_class

def valid_epoch(model, criterion, metric, dataloader, device="cpu"):
    loss_meter = 0
    score_meter = 0
    iou_per_class = None
    count = 0
    
    model.to(device).eval()
    with tqdm(dataloader, desc="Valid") as iterator:
        for sample in iterator:
            x = sample["x"].to(device)
            y = sample["y"].to(device)
            n = x.shape[0]
            
            with torch.no_grad():
                outputs = model.forward(x)
                loss = criterion(outputs, y)
                
            # ✅ Fix: Unpack metric output before calling .cpu()
            mean_iou, iou_scores = metric(outputs, y)  
            mean_iou = mean_iou.item()  # Convert tensor to scalar

            if isinstance(iou_scores, np.ndarray):
                if iou_per_class is None:
                    iou_per_class = np.zeros_like(iou_scores)
                iou_per_class += iou_scores
            
            loss_meter += loss.cpu().detach().numpy() * n
            score_meter += mean_iou * n
            count += n
            
            logs = {"CELoss": loss_meter / count, "mIoU": score_meter / count}
            iterator.set_postfix_str(format_logs(logs))
    
    iou_per_class /= count if iou_per_class is not None else np.zeros(8)
    return logs, iou_per_class
