import os
import random
import numpy as np
from pathlib import Path
from torchvision.utils import save_image

def visualize_dataset(dataset, num_samples, split, name='zebra'):
    # outdir1 = str(Path(__file__).parent / 'visualizations' / split / f'{name}_Original')
    outdir2 = str(Path(__file__).parent / 'visualizations' / split / f'{name}_Loaded')
    for outdir in [outdir2]:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    
    samples = [random.randint(0, len(dataset)) for _ in range(num_samples)]
    for idx in samples:
        image, seg_map = dataset[idx]['pixel_values'], dataset[idx]['labels']
        seg_map[seg_map==255] = 0
        seg_map = seg_map.type('torch.FloatTensor')

        save_image(image, outdir2 + f'/{idx}_img.png')
        save_image(seg_map, outdir2 + f'/{idx}_label.png')