from scipy.spatial.distance import cdist
import numpy as np



def match_descriptors(db_descriptors, query_descriptors, lambda_mult):

    # Returns a 1xQ matrix where the i-th coefficient is the index of the
    # database descriptor which matches to the i-th query descriptor.
    # The descriptor vectors are MxQ and MxD where M is the descriptor
    # dimension and Q and D the amount of query and database descriptors
    # respectively. Matches(i) will be zero if there is no database descriptor
    # with an SSD < lambda * min(SSD). No two non-zero elements of matches will
    # be equal.

    num_desc = db_descriptors.shape[2]
    patch_size = db_descriptors.shape[0]**2
    db_descriptors_2d = db_descriptors.transpose(2, 0, 1).reshape(num_desc, patch_size)
    query_descriptors_2d = query_descriptors.transpose(2, 0, 1).reshape(num_desc, patch_size)
    distance = cdist(db_descriptors_2d, query_descriptors_2d, 'euclidean')
    min_dist = np.amin(distance, 0)
    min_dist_index = np.argmin(distance, 0)
    min_nonzero_dist = np.amin(min_dist[np.nonzero(min_dist)])

    # Set matches to -1 where minimum distance is exceeded
    matches = min_dist_index
    matches[np.argwhere(min_dist > lambda_mult * min_nonzero_dist)] = -1

    # Remove double matches
    matches_unique = np.full(200, -1, dtype='int')
    unique_values, matches_unique_index = np.unique(matches, return_index='True')
    matches_unique[matches_unique_index] = matches[matches_unique_index]
    matches = matches_unique

    return matches
