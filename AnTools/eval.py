import os
import torch
from tqdm import tqdm

def write_results(model, dataset, device, decoder, result_path, print_ps=False, batch_size=1):
    model.eval()
    os.makedirs(result_path, exist_ok=True)
    out_file = os.path.join(result_path, "results.txt")

    print(f"Saving results to {out_file}")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: list(zip(*x))
    )

    with open(out_file, "w") as f:
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating", total=len(loader))):
            # --- Handle phase differences ---
            if dataset.phase == 'train':
                query_tensors, frame_images, targets = batch
                video_names = [f"frame_{batch_idx * batch_size + i}" for i in range(len(frame_images))]
            else:  # test phase
                video_names, query_tensors, frame_images = batch
                targets = [None] * len(video_names)

            # Move data to device
            query_batch = torch.stack(query_tensors).to(device)
            frame_batch = torch.stack(frame_images).to(device)

            with torch.no_grad():
                model = model.to(device).to(torch.float32)
                query_batch = query_batch.to(device, dtype=torch.float32)
                frame_batch = frame_batch.to(device, dtype=torch.float32)
                preds = model(query_batch, frame_batch)
                boxes_list, scores_list = decoder(preds)

            # Write predictions
            for i in range(len(video_names)):
                boxes = boxes_list[i].cpu().numpy()
                scores = scores_list[i].cpu().numpy()
                if len(boxes) == 0:
                    continue

                video_name = video_names[i]
                query_names = [f"q{j}" for j in range(3)]
                cls_id = int(targets[i]['labels'][0].item()) if targets[i] else -1  # -1 if test phase

                for box in boxes:
                    cx, cy, w, h = box
                    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                    line = f"{video_name} {query_names[0]} {query_names[1]} {query_names[2]} {cls_id} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n"
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