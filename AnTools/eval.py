import os
import torch
from tqdm import tqdm

def write_results(model, dataset, device, decoder, result_path, print_ps=False, batch_size=1):
    model.eval()
    os.makedirs(result_path, exist_ok=True)
    out_file = os.path.join(result_path, "results.txt")

    print(f"Saving results to {out_file}")

    # Create DataLoader for batching
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: list(zip(*x))  # simple collate to unpack tuples
    )

    with open(out_file, "w") as f:
        for batch_idx, (query_tensors, frame_images, targets) in enumerate(
            tqdm(loader, desc="Evaluating", total=len(loader))
        ):
            # Stack and move to device
            query_batch = torch.stack(query_tensors).to(device)   # [B, 3, C, H, W]
            frame_batch = torch.stack(frame_images).to(device)    # [B, C, H, W]

            with torch.no_grad():
                preds = model(query_batch, frame_batch)
                boxes_list, scores_list = decoder(preds)

            # Process outputs for each sample in batch
            for i in range(len(targets)):
                boxes = boxes_list[i].cpu().numpy()
                scores = scores_list[i].cpu().numpy()
                if len(boxes) == 0:
                    continue

                query_names = [f"q{j}" for j in range(3)]
                frame_name = f"frame_{batch_idx * batch_size + i}.jpg"
                cls_id = int(targets[i]['labels'][0].item())

                for box in boxes:
                    cx, cy, w, h = box
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    line = f"{query_names[0]} {query_names[1]} {query_names[2]} {frame_name} {cls_id} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n"
                    f.write(line)

    if print_ps:
        print(f"Results written to {out_file}")

class EvalModule:
    def __init__(self, model, decoder, device, batch_size=8):
        torch.manual_seed(317)
        self.device = device
        self.model = model
        self.decoder = decoder
        self.batch_size = batch_size

    def load_model(self, resume_path):
        checkpoint = torch.load(resume_path, map_location="cpu")
        print(f"Loaded weights from {resume_path}")
        
        # Handle both raw state_dict and dict-style checkpoints
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict_ = checkpoint["model_state_dict"]
        else:
            state_dict_ = checkpoint

        self.model.load_state_dict(state_dict_, strict=False)
        self.model.to(self.device)
        self.model.eval()


    def evaluate(self, dataset, result_dir="results", resume_path=None):
        if resume_path:
            self.load_model(resume_path)

        os.makedirs(result_dir, exist_ok=True)
        write_results(self.model, dataset, self.device, self.decoder, result_dir, print_ps=True, batch_size=self.batch_size)