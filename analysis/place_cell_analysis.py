import math

def pair(x, y):
    z = (((x + y + 1) * (x + y)) / 2) + y
    return z

def invert(z):
    w = math.floor(((math.sqrt(8*z + 1) - 1) / 2))
    t = (w**2 + w) / 2
    y = z - t
    x = w - y
    return x, y

def apply_cantor_pairing(x_coords, y_coords):
    """Reduce dimensionality from 2d to 3d.

    Args:
        x_coords: list

        y:coords: list

    Returns:
        coords: list
    """
    if len(x_coords) != len(y_coords):
        raise ValueError("The length of x_coords must equal the length of y_coords!")

    z_coords = []
    for x, y in zip(x_coords, y_coords):
        z = pair(x, y)
        z_coords.append(z)

    return z_coords


