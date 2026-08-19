def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    import numpy as np
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    # Cosine similarity
    dot_product = np.sum(x1 * x2)

    x1_norm = np.linalg.norm(x1)
    x2_norm = np.linalg.norm(x2)

    cosine_similarity = dot_product / (x1_norm * x2_norm + 1e-8)

    # Calculate loss
    if label == 1:
        loss = 1 - cosine_similarity

    elif label == -1:
        loss = max(0, cosine_similarity - margin)

    else:
        raise ValueError("label must be either 1 or -1")

    return float(loss)