

def distort_points(normalized_vec, dist_vec_d):

    # Applies lens distortion D(2x1) to 2d points x(2xN) on the image plane

    k1 = dist_vec_d[0]
    k2 = dist_vec_d[1]

    x_norm = normalized_vec[0]
    y_norm = normalized_vec[1]

    r2 = x_norm**2 + y_norm**2

    x_norm_dist = x_norm * (1 + k1 * r2 + k2 * r2**2)
    y_norm_dist = y_norm * (1 + k1 * r2 + k2 * r2**2)

    dist_vec = [x_norm_dist, y_norm_dist]
    return dist_vec

