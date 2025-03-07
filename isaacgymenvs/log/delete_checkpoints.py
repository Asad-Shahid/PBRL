import os
import re
from pathlib import Path


def delete_old_checkpoints(root_dir):
    pattern = re.compile(r"ckpt_(\d+)\.pt")
    
    for folder_path in Path(root_dir).rglob('agent_*'):
        checkpoints = []
        for file_path in folder_path.glob("ckpt_*.pt"):
            match = pattern.search(file_path.name)
            if match:
                checkpoints.append((file_path, int(match.group(1))))
        
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: x[1])
            for file_path, _ in checkpoints:
                if file_path != latest_checkpoint[0]:
                    print(f"Deleting: {file_path}")
                    file_path.unlink()  # Use unlink() to delete the file
            print(f"Keeping latest: {latest_checkpoint[0]}")


if __name__ == "__main__":
    root_directory = "."  # Change to your root directory
    delete_old_checkpoints(root_directory)
