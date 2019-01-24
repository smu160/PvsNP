import pandas as pd
import numpy as np
import cv2


def get_neuron_positions(coordinates_df):
    """Create a dictionary of all the neuron cartestian coordinates

    """
    pos = [(int(coordinates_df.loc[neuron, :]["x"]), int(coordinates_df.loc[neuron, :]["y"])) for neuron in coordinates_df.index]
    return pos

if __name__ == "__main__":
    coordinates = pd.read_csv("~/Jack_Berry_Repo/Hen_Lab/Mice/OLD_DRD87/EPM_NO_OFT_POPP_centroids.csv", header=None)
    coordinates.columns = ['x', 'y']

    # Reset the index so that it starts from 1, since it is more natural to enumerate neurons as 1, 2, ..., n
    coordinates.index = pd.RangeIndex(1, len(coordinates.index)+1)

    positions = get_neuron_positions(coordinates)

    cap = cv2.VideoCapture("/Users/saveliyyusufov/Desktop/Drd87_EPM_bgremoved.avi")

    while cap.isOpened():
        ret, frame = cap.read()

        # TODO: convert coordinates to their respective coordinates on the frame
        for pos in positions:
            cv2.circle(frame, pos, 10, (0, 0, 255), 0)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
