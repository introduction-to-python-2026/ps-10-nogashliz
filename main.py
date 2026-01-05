mport numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
from image_utils import load_image, edge_detection

image_path = 'your_image.jpg' 
image = load_image(image_path)
clean_image = median(image, ball(3))
edgeMAG = edge_detection(clean_image)
threshold = 50 
edge_binary = edgeMAG > threshold

plt.figure(figsize=(10, 5))
plt.imshow(edge_binary, cmap='gray')
plt.title(f'Binary Edges (Threshold={threshold})')
plt.axis('off')
plt.show()

edge_image = Image.fromarray((edge_binary * 255).astype(np.uint8))
edge_image.save('my_edges.png')
