import skvideo.io as skv
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import os
import numpy as np 

RUN_NAME = "new_samples"
SAMPLE_INDEX = 0 # Which sample prompt to animate?
FPS = 15

def animate(run_name, sample_index, fps, add_timestamps = False):
    """
    Takes a specific run, a specific sample prompt and animates all the images resulting from sampling with that prompt during training.
    Useful to visualize how a models generation given a single prompt changed over time. Note this script may get permission errors 
    when trying to save videos using FFMPEG if it is not setup properly.

    :param run_name: Name of run to draw samples from
    :param sample_index: Index of sample to animate
    :param fps: FPS for resulting video
    :param add_timesteps: Add timestamps? Note: needs fonts to be installed, otherwise will raise error
    """
    root_path = f"./samples/{run_name}/"

    paths = os.listdir(root_path)
    paths = list(sorted(paths, key=lambda x : int(x)))
    paths = [os.path.join(root_path, path, str(sample_index)+".png") for path in paths]

    print("Loading as Images")
    imgs = [Image.open(path) for path in tqdm(paths)]

    if add_timestamps:
        print("Adding timestamps")
        for i in tqdm(range(len(imgs))):
            draw = ImageDraw.Draw(imgs[i])
            font = ImageFont.truetype("arial", 20)
            draw.text((10, 10), str(i), fill="white", font=font)
    
    print("Saving as video")
    frames = np.stack([np.asarray(img) for img in imgs])

    skv.vwrite(f"{run_name}/{sample_index}.mp4", frames, outputdict={"-r": f"{fps}"})


if __name__ == "__main__":
    animate(RUN_NAME,SAMPLE_INDEX,FPS)


