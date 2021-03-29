import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from skimage.io import imread, imsave
import selectivesearch.selectivesearch as selectivesearch

image_path = '/D_data/Self/imagenet_root/train/n01440764/n01440764_9981.JPEG'

img = imread(image_path)
img_pil = Image.open(image_path)
imsave("visual_results/n01440764_9981.JPEG", img)

img_with_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=100)
print("number of regions", len(regions))

region_label = img_with_lbl[:, :, 3]
print("label", region_label.shape)
print("img", img.shape)
print("img_pil", img_pil.size)
regions_top = regions[:5]

fig, ax = plt.subplots()

ax.imshow(img)  # h, w, 3
# rect = patches.Rectangle((10, 10), 40, 30, linewidth=1, edgecolor='r', facecolor='none'),  # w, h
# for region in regions:
bbox = regions[2]["rect"]
print("bbox", bbox)
rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
# Add the patch to the Axes
ax.add_patch(rect)

plt.savefig("visual_results/n01440764_9981_visualize.png")
