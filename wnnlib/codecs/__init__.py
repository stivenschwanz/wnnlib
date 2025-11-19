from wnnlib.codecs import KDTree
from wnnlib.codecs import globals


def create(codec_id, max_depth, learning_rate, min_splitting_volume, min_bounds, max_bounds, sparse_vectors_file):
    """
    Create a new sparse encoder/decoder.

    Parameters:
        codec_id (int): Unique identifier of the encoder/decoder.
        max_depth (int): Maximum depth of the kd-tree.
        learning_rate (double): Learning rate.
        min_splitting_volume (double): Minimum splitting volume.
        min_bounds (double[]): Minimum bounds.
        max_bounds (double[]): Maximum bounds.
        sparse_vectors_file (string): File containing the pre-computed sparse vectors.

    Returns:
        (boolean): Ownership flag indicating whether the API caller owns the codec or not.
    """
    # Check whether the codec already exists or not. In the former case, the existing codec already has an owner.
    if codec_id in globals.g_sparse_codecs:
        return False

    # Creates the kd-tree codec
    globals.g_sparse_codecs[codec_id] = KDTree.KDTree(max_depth=int(max_depth), learning_rate=learning_rate,
                                                      min_splitting_volume=min_splitting_volume,
                                                      min_bounds=min_bounds, max_bounds=max_bounds,
                                                      sparse_vectors_file=sparse_vectors_file)

    # The caller is the owner of the created codec. Thus, the caller is supposed to destroy the codec.
    return True


def delete(codec_id):
    """
    Delete an existing sparse encoder/decoder.

    Parameters:
        codec_id (int): Unique identifier of the encoder/decoder.
    """
    # Check whether the codec already exists or not. In the former case, the owner can destroy the codec.
    if codec_id in globals.g_sparse_codecs:
        globals.g_sparse_codecs.pop(codec_id)


def get_attr(codec_id, attr_name):
    """
    Get the value of the attribute of an existing sparse encoder/decoder.

    Parameters:
        codec_id (int): Unique identifier of the encoder/decoder.
        attr_name (string): Attribute name.

    Returns:
        (object): Attribute value.
    """
    if hasattr(globals.g_sparse_codecs[codec_id], attr_name):
        return getattr(globals.g_sparse_codecs[codec_id], attr_name)
    else:
        return None


def get_ids():
    """
    Get the identifiers of the existing sparse encoders/decoders.

    Returns:
        (int []): Identifiers of the existing encoders/decoders.
    """
    globals.g_sparse_codecs.keys()


def clear_ids():
    """
    Delete all existing sparse encoders/decoders.
    """
    globals.g_sparse_codecs.clear()


def debug(codec_id, style):
    """
    Debug the sparse encoder/decoder structure.

    Parameters:
        codec_id (int): Unique identifier of the encoder/decoder.
        style (int): Cartesian (0) or polar (1) coordinates
    """
    globals.g_sparse_codecs[codec_id].debug(style, 0, 1)


def encode(codec_id, point):
    """
    Encode a point using an existing sparse encoder/decoder.

    Parameters:
        codec_id (int): Unique identifier of the encoder/decoder.
        point (double[]): Point to encode.

    Returns:
        (int[]): Sparse representation of the given point.
    """
    return globals.g_sparse_codecs[codec_id].encode(point)


def one_hot_encode(codec_id, code):
    """
    Encode a code using an existing sparse encoder/decoder.

    Parameters:
        codec_id (int): Unique identifier of the encoder/decoder.
        code (int[]): Sparse representation.

    Returns:
        (int[]): One hot encoding of the given point.
    """
    return globals.g_sparse_codecs[codec_id].one_hot_encode(code)


def decode(codec_id, code):
    """
    Decode a sparse representation using existing sparse encoder/decoder.

    Parameters:
        codec_id (int): Unique identifier of the encoder/decoder.
        code (int[]): Sparse representation.

    Returns:
        (double[]): Point corresponding to the sparse representation.
    """
    return globals.g_sparse_codecs[codec_id].decode(code)


def one_hot_decode(codec_id, one_hot_vector):
    """
    Decode a one-hot vector using an existing sparse encoder/decoder.

    Parameters:
        codec_id (int): Unique identifier of the encoder/decoder.
        one_hot_vector (int[]): One-hot vector representation.

    Returns:
        (int[]): Sparse vector corresponding to the given one-hot-vector.
    """
    return globals.g_sparse_codecs[codec_id].one_hot_decode(one_hot_vector)
