import argparse
from model.combined_model import CombinedModel

def parse_args():
    parser = argparse.ArgumentParser(description='banpath model')
    parser.add_argument('--num_epoch', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--init_lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--resume_train', type=str, default=None, help='Path to resume training from a checkpoint')
    parser.add_argument('--phase', type=str, default='test', help='Phase choice= {train, test, eval}')
    parser.add_argument('--data_dir', type=str, default='./../DATA', help='Path to dataset root directory')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    