import sys
from PIL import Image

def find_col_weights(img):
    """
    Calculate the weight of each color in the given image.

    Args:
        img (PIL.Image.Image): The input image.

    Returns:
        dict: A dictionary containing hex color codes as keys and their respective counts as values.
    """

    col_dict = {}
    px = img.getdata()

    for val in px:
        hex = rgb_to_hex(val)
        if hex not in col_dict:
            col_dict[hex] = 1
        else:
            col_dict[hex] += 1

    return col_dict

def rgb_to_hex(rgb):
    """
    Convert an RGB tuple to a hex color string.

    Args:
        rgb (tuple): A tuple containing three integers representing an RGB color (R, G, B).

    Returns:
        str: The hex color code as a string.
    """
    return '#' + ('{:02x}{:02x}{:02x}').format(rgb[0], rgb[1], rgb[2])

def hex_to_rgb(hex):
    """
    Convert a hex color string to an RGB tuple.

    Args:
        hex (str): The hex color code as a string.

    Returns:
        tuple: A tuple containing three integers representing an RGB color (R, G, B).
    """
    rgb = []
    for i in (1, 3, 5):
        decimal = int(hex[i:i+2], 16)
        rgb.append(decimal)

    return tuple(rgb)

def calc_dist(col1, col2):
    """
    Calculate the Euclidean distance between two RGB colors.

    Args:
        col1 (tuple): A tuple containing three integers representing the first RGB color (R, G, B).
        col2 (tuple): A tuple containing three integers representing the second RGB color (R, G, B).

    Returns:
        int: The squared distance between the two colors.
    """
    return ((col1[0] - col2[0]) * (col1[0] - col2[0]) + 
            (col1[1] - col2[1]) * (col1[1] - col2[1]) + 
            (col1[2] - col2[2]) * (col1[2] - col2[2]))

def calc_loss_dist(centr, assignment, col_dict):
    """
    Calculate the loss based on the distance between centroids and assigned colors.

    Args:
        centr (list): A list of centroid RGB tuples.
        assignment (list): A list of indices representing the centroid assigned to each color.
        col_dict (dict): A dictionary with hex color codes as keys and their respective counts as values.

    Returns:
        float: The calculated loss.
    """

    l = 0
    d = []
    col_values = [hex_to_rgb(i) for i in col_dict.keys()]
    col_weights = [i for i in col_dict.values()]

    for i in range(len(assignment)):
        dist = calc_dist(centr, col_values[i])
        d.append(dist)
        if assignment[i]:
            l += dist * col_weights[i]
    
    return l, d

def generate_dat(output, k, cols, assignments, losses):
    """
    Generate a .dat file containing parameters for optimization.

    Args:
        output (str): The path where the .dat file will be saved.
        k (int): The number of clusters.
        cols (dict): A dictionary containing color information, with hex color codes as keys.
        assignments (list): A list of lists, where each sublist represents the assignment of colors to centroids.
        losses (list): A list of losses for each centroid.

    Writes:
        A .dat file at the specified output location.
    """
    
    with open(output, 'w') as data:

        n = len(cols)
        P = len(losses)
        
        data.write(f'param n := {n};\n')
        data.write(f'param P := {P};\n')
        data.write(f'param k := {k};\n\n\n')

        data.write(f'param c :=\n')
        for i in range(1, P + 1):
            data.write(f' {i} {losses[i-1]}\n')
        data.write(';\n\n\n')

        data.write(f'param C :')
        for i in range(1, n + 1):
            data.write(f'  {i}')
        data.write(f' :=\n')

        for i in range(1, P + 1):
            data.write(f'  {i} ')
            for j in range(1, n + 1):
                data.write(f' {assignments[i - 1][j - 1]}')
            data.write(f'\n')
        data.write(';\n\n end;\n')
        data.close()

    print(f'.dat file correctly generated at {output}!')

def generate_bitcomb(num): # Fast generation of clusters
    """
    Creates an iterable generating sequentially bit combinations of a given length.

    Args:
        num (int): The length of the bit combination.

    Returns:
        An iterator outputting sequentially the bit combinations, starting from the least significant bit.
    """

    for i in range(1, 1 << num):
        yield tuple((i >> j) & 1 for j in range(num - 1, -1, -1))

def main():
    if len(sys.argv) != 5:
        print("Correct usage: python3 datagen.py input-image-path output-path k threshold")
        sys.exit(1)
    
    try:
        input_image_path = sys.argv[1]
        output_path = sys.argv[2]
        k = int(sys.argv[3])
        threshold = int(sys.argv[4])
    except:
        print('Incorrect usage of variables in script call')
    
    try:
        with Image.open(input_image_path) as im:
            im = im.convert('RGB') # Convert the image as RGB to avoid problems

            threshold = threshold # Set a threshold for the loss

            unique_col = find_col_weights(im) # Find the unique colors in the image and associated weghts (# of pixels)
            chosen_ass = []
            chosen_losses = []

            for cluster in generate_bitcomb(len(unique_col)): # Takes a possible cluster of the data points

                col_values = list(unique_col.keys())
                col_weights = list(unique_col.values())

                sel_cols = []
                sel_weights = []
                centr_coords = [0, 0, 0]

                for i in range(len(cluster)):
                    if cluster[i]: # If the data point belongs to the cluster, use it for centroid calculation
                        rgb = hex_to_rgb(col_values[i])
                        w = col_weights[i]

                        sel_cols.append(rgb) # Store weights and values of data points assigned to a certain cluster
                        sel_weights.append(w)

                        centr_coords[0] += rgb[0] * w # Add the data point component to the centroid
                        centr_coords[1] += rgb[1] * w
                        centr_coords[2] += rgb[2] * w
                    
                centr_coords[0] = centr_coords[0] // sum(sel_weights) # Normalize by the sum of all weights
                centr_coords[1] = centr_coords[1] // sum(sel_weights)
                centr_coords[2] = centr_coords[2] // sum(sel_weights)

                loss, _ = calc_loss_dist(tuple(centr_coords), cluster, unique_col) # Return the loss associated to the cluster

                if loss <= threshold: # If the loss is under the threshold, store the possible configuration
                    print(f'Cluster: {cluster}, center: {centr_coords}, loss: {loss:0,}, # colors: {sum(cluster)}')
                    chosen_ass.append(cluster)
                    chosen_losses.append(loss)

            # print(max(chosen_losses))
            # print(len(chosen_losses))
            generate_dat(output_path, k, unique_col, chosen_ass, chosen_losses) # Generate the .dat file with all possible clusters


    except OSError:
        print(f'Error in opening image {input_image_path}')
        sys.exit(1)

    except FileNotFoundError:
        print(f'Image {input_image_path} was not found')
        sys.exit(1)

if __name__ == "__main__":
    main()
