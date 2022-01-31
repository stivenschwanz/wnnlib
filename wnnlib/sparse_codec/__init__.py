from wnnlib.sparse_codec import KDTree
from wnnlib.sparse_codec import globals


def create(codec_id, output_dims, learning_rate, min_splitting_volume, min_bounds, max_bounds):
    """
    Create a new sparse encoder/decoder.

    Parameters:
        codec_id (int): Unique identifier of the encoder/decoder.
        output_dims (int): Number of output dimensions.
        learning_rate (double): Learning rate.
        min_splitting_volume (double): Minimum splitting volume.
        min_bounds (double[]): Minimum bounds.
        max_bounds (double[]): Maximum bounds.
    """
    globals.g_sparse_codecs[codec_id] = KDTree.KDTree(max_depth=int(output_dims), learning_rate=learning_rate,
                                                      min_splitting_volume=min_splitting_volume,
                                                      min_bounds=min_bounds, max_bounds=max_bounds)


def delete(codec_id):
    """
    Delete an existing sparse encoder/decoder.

    Parameters:
        codec_id (int): Unique identifier of the encoder/decoder.
    """
    globals.g_sparse_codecs.pop(codec_id)


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


def debug(codec_id, point, style, marker):
    """
    Debug the sparse encoder/decoder structure.

    Parameters:
        codec_id (int): Unique identifier of the encoder/decoder.
        point (double[]): Point to draw.
        style (int): Cartesian (0) or polar (1) coordinates
        marker (string): Point marker.
    """
    globals.g_sparse_codecs[codec_id].debug(point, style, 0, 1, marker)


def encode(codec_id, point):
    """
    Encode a point using an existing sparse encoder/decoder.

    Parameters:
        codec_id (int): Unique identifier of the encoder/decoder.
        point (double[]): Point to draw.

    Returns:
        (int[]): Sparse representation of the given point.
    """
    return globals.g_sparse_codecs[codec_id].encode(point)


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
