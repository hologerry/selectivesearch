import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.io import imread, imsave
import selectivesearch.selectivesearch as selectivesearch

image_path = '/D_data/Self/data/coco/train2017/000000000009.jpg'

img = imread(image_path)
imsave("000000000009.jpg", img)

img_with_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=100)
print("number of regions", len(regions))

region_label = img_with_lbl[:, :, 3]

regions_top = regions[:100]

fig, ax = plt.subplots()

ax.imshow(img)  # h, w, 3
# rect = patches.Rectangle((10, 10), 40, 30, linewidth=1, edgecolor='r', facecolor='none'),  # w, h
for region in regions:
    bbox = region["rect"]
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)

plt.savefig("000000000009_visualize.png")
