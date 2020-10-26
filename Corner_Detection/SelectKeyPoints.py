import numpy as np

def select_key_points(scores, num, radius):

    # Selects the num best scores as keypoints and performs a non-max
    # supression of a (2r + 1)*(2r + 1) patch around the maximum

    keypoints = np.zeros((2, num), dtype='int')
    temp_scores = np.pad(scores, radius, 'constant')
    for i in range(num):
        kp = np.where(temp_scores == np.amax(temp_scores))
        keypoints[:, i] = kp[:]
        keypoints[:, i] = keypoints[:, i] - radius
        temp_scores[(kp[0] - radius).item(): (kp[0] + radius + 1).item(), (kp[1] - radius).item(): (kp[1] + radius + 1).item()] = 0

    return keypoints