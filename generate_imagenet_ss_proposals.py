import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import PIL.Image as Image

import selectivesearch.selectivesearch as selectivesearch

imagenet_root = '/D_data/Self/imagenet_root'
imagenet_root_proposals = '/D_data/Self/imagenet_root_proposals'

split = 'train'
scale = 300
min_size = 100

selective_proposals = {}

obj_path = f'imagenet_{split}_{scale}_{min_size}.pkl'

if os.path.exists(obj_path):
    with open(obj_path, 'rb') as f:
        selective_proposals = pickle.load(f)
else:
    sub_dirs = sorted(os.listdir(os.path.join(imagenet_root, split)))
    for sub in sub_dirs:
        filenames = sorted(os.listdir(os.path.join(imagenet_root, split, sub)))
        os.makedirs(os.path.join(imagenet_root_proposals, split, sub))
        for filename in filenames:
            img_path = os.path.join(imagenet_root, split, sub, filename)
            img = np.array(Image.open(img_path).convert('RGB'))

            img_with_lbl, regions = selectivesearch.selective_search(img, scale=scale, sigma=0.9, min_size=min_size)

            region_label = img_with_lbl[:, :, 3]
            cur_img_proposal = {}
            cur_img_proposal['label'] = region_label
            cur_img_proposal['regions'] = regions
            # selective_proposals[filename] = cur_img_proposal
            cur_img_pro_path = os.path.join(imagenet_root_proposals, split, sub, filename+'.pkl')

            with open(cur_img_pro_path, 'wb') as f:
                pickle.dump(cur_img_proposal, f)
            
    # with open(obj_path, 'wb') as f:
    #     pickle.dump(selective_proposals, f)
