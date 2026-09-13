def k_means_assignment(points: list, centroids: list) -> list:
    """
    Returns the nearest-centroid index for every point.
    """
    import numpy as np
    points = np.asarray(points)
    centroids = np.asarray(centroids)
    closest_indices = []
    for point in points:
      distances = np.sum((point - centroids)**2, axis = 1)
      closest_centroids = np.argmin(distances)
      closest_indices.append(int(closest_centroids))
    return closest_indices
    