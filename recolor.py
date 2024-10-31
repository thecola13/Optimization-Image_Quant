import sys
from PIL import Image
import random
import time

# Clustering class 
class Clustering:

    """
    A class to perform k-means clustering on color data from an image.

    Attributes:
        k (int): Number of centroids (clusters).
        data (list): List of color data from the image.
        weights (list): List of weights corresponding to each color in data.
        centroids (list): List of initial centroids.
        assignment (list): List to track the centroid assignment for each data point.
    """

    def __init__(self, n_centroids, data):

        """
        Initialize Clustering instance.

        Args:
            n_centroids (int): The number of centroids to create.
            data (dict): A dictionary with color data as keys and weights as values.
        """

        self.k = n_centroids
        self.data = list(data.keys())
        self.weights = list(data.values())
        self.centroids = self.initialize_centroids(self.k)
        self.assignment = [-1 for i in range(len(self.data))]

    def initialize_centroids(self, k):

        """
        Initialize centroids with random RGB colors.

        Args:
            k (int): The number of centroids to initialize.

        Returns:
            list: A list of randomly generated centroids as RGB tuples.
        """

        centroids = []
        for i in range(k):
            centroids.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

        return centroids

    def do_step(self):

        """
        Assign each data point to the closest centroid based on distance.
        """

        for i in range(len(self.data)):
            d_min = float('inf')
            sel_centroid = None
            for j in range(len(self.centroids)):
                d = calc_dist(self.data[i], self.centroids[j]) * self.weights[i]
                if d < d_min:
                    d_min = d
                    sel_centroid = j
            
            self.assignment[i] = sel_centroid

    def calc_loss(self):
        """""
        Calculate the total loss for the current centroid assignment.

        Returns:
            float: The total loss value.
        """

        l = 0
        for i in range(len(self.data)):
            l += calc_dist(self.data[i], self.centroids[self.assignment[i]]) * self.weights[i]
        return l
    
    def calc_centroid(self):

        """
        Update centroid positions based on the weighted mean of the assigned points.
        """

        for j in range(len(self.centroids)):
            sum_r = 0
            sum_g = 0
            sum_b = 0
            total_weight = 0

            # Calculate weighted sum of colors assigned to centroid j
            for i in range(len(self.data)):
                if self.assignment[i] == j:
                    sum_r += self.data[i][0] * self.weights[i]
                    sum_g += self.data[i][1] * self.weights[i]
                    sum_b += self.data[i][2] * self.weights[i]
                    total_weight += self.weights[i]

            # Avoid division by zero in case no points are assigned to this centroid
            if total_weight > 0:
                mean_r = sum_r // total_weight
                mean_g = sum_g // total_weight
                mean_b = sum_b // total_weight
                self.centroids[j] = (mean_r, mean_g, mean_b)


    def fit(self, max_iter = 200, step = 1):

        """
        Perform k-means clustering.

        Args:
            max_iter (int): The maximum number of iterations.
            step (int): The step interval at which to print loss information.
        """

        past_loss = float('inf')
        for i in range(max_iter):
            self.do_step()
            loss = self.calc_loss()

            if i % step == 0:
                print(f'Iteration n°{i + 1}, loss = {int(loss):,}'.replace(',', '_'))

            if loss == past_loss:
                break
            else:
                past_loss = loss

            self.calc_centroid()
    
    def gen_palette(self):

        """
        Generate a color palette based on the centroids.

        Returns:
            dict: A dictionary containing hex color codes as keys and RGB tuples as values.
        """

        colors = {}

        for c in self.centroids:
            colors[rgb_to_hex(c)] = c

        print(colors.keys())
        return colors



# Utility functions
def rgb_to_hex(rgb):

    """
    Convert an RGB tuple to a hex color string.

    Args:
        rgb (tuple): A tuple containing three integers representing an RGB color.

    Returns:
        str: The hex color code as a string.
    """

    return '#' + ('{:02x}{:02x}{:02x}').format(rgb[0], rgb[1], rgb[2])

def calc_dist(col1, col2):

    """
    Calculate the squared Euclidean distance between two RGB colors.

    Args:
        col1 (tuple): A tuple representing the first RGB color.
        col2 (tuple): A tuple representing the second RGB color.

    Returns:
        int: The squared distance between the two colors.
    """

    return ((col1[0] - col2[0]) * (col1[0] - col2[0]) + 
            (col1[1] - col2[1]) * (col1[1] - col2[1]) + 
            (col1[2] - col2[2]) * (col1[2] - col2[2]))

def gen_new_image(img, im_size, color_palette):

    """
    Generate a new recolored image using a color palette.

    Args:
        img (PIL.Image.Image): The original image.
        im_size (tuple): The dimensions of the image.
        color_palette (dict): A dictionary mapping hex color codes to RGB tuples.

    Returns:
        PIL.Image.Image: The recolored image.
    """

    new_img = Image.new('RGB', im_size)
    px_old = img.load()
    px_new = new_img.load()

    px_lookup = {}

    for i in range(im_size[0]):
        for j in range(im_size[1]):
            rgb = px_old[i, j]
            hex = rgb_to_hex(rgb)

            if hex in px_lookup:
                px_new[i, j] = color_palette[px_lookup[hex]]

            else:
                d_min = float('inf')
                sel_color = None

                for col_hex, col_rgb in color_palette.items():
                    d = calc_dist(rgb, col_rgb)
                    if d < d_min:
                        d_min = d
                        sel_color = col_hex

                px_lookup[hex] = sel_color
                px_new[i, j] = color_palette[sel_color]

    return new_img

def gen_data(img):
    
    """
    Extract color data from an image.

    Args:
        img (PIL.Image.Image): The input image.

    Returns:
        dict: A dictionary containing colors as keys and their occurrence counts as values.
    """

    allvalues = img.getdata()
    col_dict = {}
    for val in allvalues:
        if val not in col_dict:
            col_dict[val] = 1
        else:
            col_dict[val] += 1

    return col_dict

def main():
    if len(sys.argv) != 4:
        print("Correct usage: python3 recolor.py input-image output-image k")
        sys.exit(1)
    
    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]
    k = int(sys.argv[3])

    # print(color_palette)
    
    try:
        with Image.open(input_image_path) as im:
            dim = im.size
            im = im.convert('RGB')

            data = gen_data(im)
            
            cluster = Clustering(k, data)
            cluster.fit()

            color_palette = cluster.gen_palette()

            new_img = gen_new_image(im, dim, color_palette)

            new_img.save(output_image_path, 'PNG')

    except OSError:
        print(f'Error in opening image {input_image_path}')

        sys.exit(1)
    except FileNotFoundError:
        print(f'Image {input_image_path} was not found')
        sys.exit(1)

if __name__ == "__main__":
    # random.seed(1)  
    random.seed(time.time())
    main()
