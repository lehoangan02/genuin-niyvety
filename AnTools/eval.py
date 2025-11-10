import os
import torch
from tqdm import tqdm
from typing import List

def write_results(model, dataset, device, decoder, result_path, print_ps=False, batch_size=1, num_workers=4):
    model.eval()
    os.makedirs(result_path, exist_ok=True)
    out_file = os.path.join(result_path, "results.txt")

    print(f"Saving results to {out_file}")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: list(zip(*x)),
        num_workers=num_workers
    )

    with open(out_file, "w") as f:
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating", total=len(loader))):
            # --- Handle phase differences ---
            if dataset.phase == 'train':
                query_tensors, frame_images, targets = batch
                video_names = [f"frame_{batch_idx * batch_size + i}" for i in range(len(frame_images))]
            else:  # test phase
                video_names, query_names, query_tensors, frame_names, frame_images = batch
                targets = [None] * len(video_names)

            # Move data to device
            query_batch = torch.stack(query_tensors).to(device)
            frame_batch = torch.stack(frame_images).to(device)

            with torch.no_grad():
                model = model.to(device).to(torch.float32)
                query_batch = query_batch.to(device, dtype=torch.float32)
                frame_batch = frame_batch.to(device, dtype=torch.float32)
                preds = model(query_batch, frame_batch)
                # print("pred ", preds)
                boxes_list, scores_list = decoder(preds)

            # Write predictions
            for i in range(len(video_names)):
                boxes = boxes_list[i].cpu().numpy()
                scores = scores_list[i].cpu().numpy()
                if len(boxes) == 0:
                    continue

                video_name = video_names[i]
                query_list = query_names[i]
                frame_name = frame_names[i]
                
                for box, score in zip(boxes, scores):
                    x1, y1, x2, y2 = box
                    line = " ".join([video_name, *query_list, frame_name, str(score), str(int(x1)), str(int(y1)), str(int(x2)), str(int(y2))]) + "\n"
                    f.write(line)

    if print_ps:
        print(f"Results written to {out_file}")

class EvalModule:
    def __init__(self, model, decoder, device, batch_size=8, num_workers=4):
        torch.manual_seed(317)
        self.device = device
        self.model = model
        self.decoder = decoder
        self.batch_size = batch_size
        self.num_workers = num_workers

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
        write_results(self.model, dataset, self.device, self.decoder, result_dir, print_ps=True, batch_size=self.batch_size, num_workers=self.num_workers)