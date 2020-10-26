from scipy.spatial.distance import cdist
import numpy as np



def match_descriptors(db_descriptors, query_descriptors, adapt_threshold):

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
    ssd = cdist(db_descriptors_2d, query_descriptors_2d, 'euclidean')


    return
