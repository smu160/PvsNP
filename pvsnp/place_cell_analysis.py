#
# PvsNP: toolbox for reproducible analysis & visualization of neurophysiological data.
# Copyright (C) 2019
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
"""
This module contains functions for carrying out place cell analysis.
Specifically, it contains functions for dimensionality reduction, binning,
and filtering.
"""

__author__ = "Saveliy Yusufov"
__date__ = "1 March 2019"
__license__ = "GPL"
__maintainer__ = "Saveliy Yusufov"
__email__ = "sy2685@columbia.edu"

import math


def pair(x, y):
    r"""Uniquely encode two natural numbers into a single natural number

    The Cantor pairing function is a primitive recursive pairing function
    \pi: \mathbb{N} \times \mathbb{N} \rightarrow \mathbb(N)

    defined by:

    \pi(x, y) := \frac{1}{2}(x + y)(x + y + 1) + y

    Source: https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function

    Args:
        x: int
            One of the natural numbers to encode into a single natural number.

        y: int
            One of the natural numbers to encode into a single natural number.

    Returns:
        z: int
            The single natural number uniquely encoded from the the provided
            natural numbers, x and y.
    """
    if not isinstance(x, int) or not isinstance(y, int):
        raise TypeError("x and y must be members of the natural numbers!")
    if x < 0 or y < 0:
        raise ValueError("x and y cannot be less than 0!")

    z = (((x + y + 1) * (x + y)) / 2) + y
    return z


def invert(z):
    """Invert z into a unique pair of values in the natural numbers

    Source: https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function

    Args:
        z: int
            A natural number that is comprised of two unique natural numbers.

    Returns:
        x, y: tuple
            The two unique natural numbers, x, y, that comprise the unqique
            encoding of z.
    """
    if not isinstance(z, int):
        raise TypeError("z must be a member of the natural numbers!")
    if z < 0:
        raise ValueError("z cannot be less than 0!")

    w = math.floor(((math.sqrt(8*z + 1) - 1) / 2))
    t = (w**2 + w) / 2
    y = z - t
    x = w - y
    return x, y


def apply_cantor_pairing(x_coords, y_coords):
    """Reduce dimensionality from 2d to 1d.

    Args:
        x_coords: list
            A list of natural numbers, such that the value at each index,
            corresponds to the value at each index in y_coords.

        y:coords: list
            A list of natural numbers, such that the value at each index,
            corresponds to the value at each index in x_coords.

    Returns:
        z_coords: list
            The resulting list from applying the cantor pairing function to each
            corresponding pair of natural numbers, i.e., (x_i, y_i).

    """
    if len(x_coords) != len(y_coords):
        raise ValueError("x_coords and y_coords must be of equal length!")

    z_coords = [pair(x, y) for x, y in zip(x_coords, y_coords)]
    return z_coords


# TODO: Finish typing up function documentation
def bin_coordinates(mouse, bin_size=5, x_col="X_center", y_col="Y_center"):
    """Bins the x and y coordinates for place cell analysis.

    Args:
        mouse: Mouse
            The mouse object whose spikes_and_beh dataframe contains an `X` and
            a `Y` column to bin by the provided bin size.

        bin_size: int, optional, default: 5
            (i.e., bin_factor=5 ==> 5 x 5 bins)

        x_col: str, optional, default: 'X_center'
            The name of the column that contains the `X` coordinates in the
            spikes_and_beh dataframe.

        y_col: str, optional, default: 'Y_center'
            The name of the column that contains the `Y` coordinates in the
            spikes_and_beh dataframe.

    Returns:
        x_coords, y_coords: tuple
            The binned x coordinates and the binned y coordinates in a tuple.
    """
    # Extract the X and Y coordinate column vectors
    x_coords = mouse.spikes_and_beh[x_col].astype(float)
    y_coords = mouse.spikes_and_beh[y_col].astype(float)

    # Subtract the mean value of the coordinate to ensure the binning centers
    # around the main track
    x_coords -= x_coords.mean()
    y_coords -= y_coords.mean()

    # Bin X and Y coordinates by the specified bin size
    x_coords = (x_coords / bin_size).astype(int)
    y_coords = (y_coords / bin_size).astype(int)

    return x_coords, y_coords


# TODO: Finish typing up function documentation
def remove_immobile(mouse):
    """Removes immobile time bins from mouse.

    Args:
        mouse: Mouse object

    Return:
        mobile_s:

        mobile_c:

        mobile_beh:
    """
    mobile_s = mouse.spikes[mouse.behavior.immobile == 0]
    mobile_c = mouse.cell_transients[mouse.behavior.immobile == 0]
    mobile_beh = mouse.behavior[mouse.behavior.immobile == 0]

    return mobile_s, mobile_c, mobile_beh


# TODO: Cleanup & finish typing up function documentation
def remove_low_occupancy(mouse, x_bin, y_bin, min_occupancy=1):
    """Removes spatial bins that had low occupancy

    Args:
        mouse: Mouse

        x_bin: pandas Series
            Binned x coordinates

        y_bin: pandas Series
            Binned y coordinates

        min_occupancy:
            minimum number of time bins for a spatial bin to be included

    Returns:
        filtered_binned_s:

        filtered_binned_c:

        filtered_binned_beh:
            Low-occupancy spatial bins removed, and the new columns
    """
    raise NotImplementedError("Major refactoring in progress.")
