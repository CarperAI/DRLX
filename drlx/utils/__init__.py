import os
import glob

def get_latest_checkpoint(root_dir):
    """
    Assume folder root_dir stores checkpoints for model, all named numerically (in terms of training steps associated with said checkpoints).
    This function returns the path to the latest checkpoint, aka the subfolder with largest numerical name. Returns none if the root dir is empty
    """
    subdirs = glob.glob(os.path.join(checkpoint_root, '*'))
    if not subdirs:
        return None
    
    # Filter out any paths that are not directories or are not numeric
    subdirs = [s for s in subdirs if os.path.isdir(s) and os.path.basename(s).isdigit()]
    # Find the maximum directory number (assuming all subdirectories are numeric)
    latest_checkpoint = max(subdirs, key=lambda s: int(os.path.basename(s)))
    return latest_checkpoint
