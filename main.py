from PIL import Image
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.morphology import ball
from image_utils import load_image , edge_detection
 
original_image = load_image('/content/mada.jpeg')
clean_image = median(original_image, ball(3))
edge_mag = edge_detection(clean_image)
threshold_value = 100 
binary_edges = edge_mag > threshold_value

edge_image_to_save = Image.fromarray((binary_edges * 255).astype(np.uint8))
edge_image_to_save.save('my_edges.png')

