# -*- coding: utf-8 -*-
"""

"""

import os
import sys
import isx

def preprocess(file_path, output_directory):
    if not os.path.isfile(file_path):
        raise FileNotFoundError("Wrong file or file path")

    rec_files = [file_path]

    # Preprocess the recordings by spatially downsampling by a factor of 2.
    pp_files = isx.make_output_file_paths(rec_files, output_directory, "PP")
    isx.preprocess(rec_files, pp_files, spatial_downsample_factor=2)

    # Perform spatial bandpass filtering with defaults.
    bp_files = isx.make_output_file_paths(pp_files, output_directory, "BP")
    isx.spatial_filter(pp_files, bp_files, low_cutoff=0.005, high_cutoff=0.500)

def main():
    if len(sys.argv) != 3:
        print("Usage: {} file_path output_directory".format(sys.argv[0]), file=sys.stderr)
        sys.exit(1)

    file_path = sys.argv[1] # "H:/Mossy Cell/Drd87/nVista/Drd87_Day4/recording_20171209_105147.xml"
    if os.path.isfile(file_path):
        print("{} is a file".format(file_path))
    else:
        print("{} is not a file!".format(file_path))
        sys.exit(1)
    
    output_directory = sys.argv[2]
    if os.path.isdir(output_directory):
        print("{} is a directory".format(output_directory))
    else:
        print("{} is not a directory!".format(output_directory))
        sys.exit(1)

    preprocess(file_path, output_directory)
    print("Preprocessing complete!")

if __name__ == "__main__":
    main()