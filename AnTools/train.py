import os
import torch
import torch.nn as nn
import numpy as np
import loss # Make sure you have this loss.py file
from torch.utils.data import DataLoader

# REMOVED: collater function is no longer needed, 
# as the DataLoader is created externally with custom_collate_fn

class TrainModule(object):
    # CHANGED: Now accepts the DataLoader instance directly
    def __init__(self, train_loader, model, down_ratio=4):
        torch.manual_seed(317)
        self.train_loader = train_loader # Store the passed-in loader
        self.down_ratio = down_ratio
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = model

    def save_model(self, path, epoch, model, optimizer):
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict()
        }, path)

    def load_model(self, model, optimizer, resume, strict=True):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print(f'Loaded weights from {resume}, epoch {checkpoint["epoch"]}')
        state_dict_ = checkpoint['model_state_dict']
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = model.state_dict()
        if not strict:
            for k in state_dict:
                if k in model_state_dict:
                    if state_dict[k].shape != model_state_dict[k].shape:
                        print(f'Skip loading parameter {k}')
                        state_dict[k] = model_state_dict[k]
                else:
                    print(f'Drop parameter {k}')
            for k in model_state_dict:
                if k not in state_dict:
                    print(f'No param {k}')
                    state_dict[k] = model_state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        epoch = checkpoint['epoch']
        return model, optimizer, epoch

    def train_network(self, args):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.init_lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.96)
        
        # NOTE: You might need to update this path logic if args.dataset is not available
        # For now, let's create a generic 'weights' folder
        save_path = f'weights' 
        start_epoch = 1

        if args.resume:
            self.model, self.optimizer, start_epoch = self.load_model(
                self.model, self.optimizer, args.resume, strict=True
            )

        os.makedirs(save_path, exist_ok=True)

        # Assuming args.ngpus is part of your args
        if hasattr(args, 'ngpus') and args.ngpus > 1 and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)
        
        # Assuming loss.py has a LossAll class
        try:
            criterion = loss.LossAll()
        except AttributeError:
            print("Error: 'loss.py' does not seem to have a 'LossAll' class.")
            print("Using a placeholder loss (MSE). Please fix your loss.py.")
            criterion = nn.MSELoss() # Placeholder

        # REMOVED: The internal dataset and DataLoader creation
        
        # CHANGED: Use the train_loader passed during __init__
        loader = self.train_loader

        print("Starting training...")
        train_loss = []

        for epoch in range(start_epoch, args.num_epoch + 1):
            print("-" * 10)
            print(f"Epoch: {epoch}/{args.num_epoch}")
            
            # Pass the loader and criterion to the epoch runner
            epoch_loss = self.run_epoch(loader, criterion)
            train_loss.append(epoch_loss)
            
            # Note: scheduler.step() should be called after optimizer.step()
            # Calling it here steps it once per epoch.
            self.scheduler.step() # Was scheduler.step(epoch) - check your scheduler type

            np.savetxt(os.path.join(save_path, "train_loss.txt"), train_loss, fmt="%.6f")

            if epoch % 1 == 0 or epoch > 20:
                self.save_model(
                    os.path.join(save_path, f"model_{epoch}.pth"),
                    epoch,
                    self.model,
                    self.optimizer
                )

            self.save_model(
                os.path.join(save_path, "model_last.pth"),
                epoch,
                self.model,
                self.optimizer
            )

    def run_epoch(self, loader, criterion):
        self.model.train()
        running_loss = 0.0

        # CHANGED: Updated loop to match the output of custom_collate_fn
        # It now expects (query_tensor, frame_tensor, targets_list)
        for query_tensor, frame_tensor, targets in loader:
            
            # Move data to the device
            query_tensor = query_tensor.to(self.device, non_blocking=True)
            frame_tensor = frame_tensor.to(self.device, non_blocking=True)
            
            # Targets is a list of dictionaries, move each one
            targets_gpu = []
            for t in targets:
                targets_gpu.append({
                    'boxes': t['boxes'].to(self.device, non_blocking=True),
                    'labels': t['labels'].to(self.device, non_blocking=True)
                })
            targets = targets_gpu # Use the list of tensors on GPU

            # Zero gradients
            self.optimizer.zero_grad()
            
            with torch.enable_grad():
                # CHANGED: Model call now uses both query and frame tensors.
                # This assumes your CombinedModelV3's forward pass is: forward(self, query, frame)
                preds = self.model(query_tensor, frame_tensor)
                
                # --- REMOVED: All the gt_tensor creation logic ---

                # CHANGED: Loss call now passes preds, targets, AND frame_tensor
                loss = criterion(preds, targets, frame_tensor)
                
                if isinstance(loss, dict):
                    # Handle if criterion returns a dict of losses
                    loss_val = loss["total"]
                else:
                    loss_val = loss
                    
                loss_val.backward()
                self.optimizer.step()

            running_loss += loss_val.item()

        epoch_loss = running_loss / len(loader)
        print(f"Train loss: {epoch_loss:.6f}")
        return epoch_loss


