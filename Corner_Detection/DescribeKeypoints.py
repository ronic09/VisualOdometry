import numpy as np

def describe_keypoints(img, keypoints, radius):

    num = keypoints.shape[1]
    kp = np.empty((2), dtype='int')
    pad_img = np.pad(img, radius, 'constant')

    # Get first descriptor
    kp[:] = keypoints[:, 0] + radius
    descriptors = pad_img[(kp[0] - radius).item(): (kp[0] + radius + 1).item(),
                                (kp[1] - radius).item(): (kp[1] + radius + 1).item()]

    # Stack 200 descriptors
    for i in range(1, num):
        kp[:] = keypoints[:, i] + radius
        descriptor_i = pad_img[(kp[0] - radius).item(): (kp[0] + radius + 1).item(),
                                (kp[1] - radius).item(): (kp[1] + radius + 1).item()]
        descriptors = np.dstack((descriptors, descriptor_i))

    return descriptors