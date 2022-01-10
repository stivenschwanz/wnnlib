import KDTree
import VGRAMLayer
import globals


def add_vgram_layer(layer_id, output_dims, pattern_length, min_mem_size, max_mem_size, min_dist, max_dist):
    """
    Add a new VGRAM layer.

    Parameters:
        layer_id (int): Unique identifier of the VGRAM layer.
        output_dims (int): Number of output dimensions.
        pattern_length (int): Length of the stored patterns.
        min_mem_size (int): Minimum memory size.
        max_mem_size (int): Maximum memory size.
        min_dist (int): Minimum Hamming distance.
        max_dist (int): Maximum Hamming distance.
    """
    globals.g_vgram_layers[layer_id] = VGRAMLayer.VGRAMLayer(output_dims=output_dims, pattern_length=pattern_length,
                                                             min_mem_size=min_mem_size, max_mem_size=max_mem_size,
                                                             min_dist=min_dist, max_dist=max_dist)


def del_vgram_layer(layer_id):
    """
    Delete an existing VGRAM layer.

    Parameters:
        layer_id (int): Unique identifier of the VGRAM layer.
    """
    globals.g_vgram_layers.pop(layer_id)


def get_vgram_layer_ids():
    """
    Get the identifiers of the existing VGRAM layers.

    Returns:
        (int []): Identifiers of the existing VGRAM layers.
    """
    globals.g_vgram_layers.keys()


def clear_vgram_layers():
    """
    Delete all existing VGRAM layers.
    """
    globals.g_vgram_layers.clear()


def draw_vgram_layer(layer_id):
    """
    Draw the VGRAM layer outputs.

    Parameters:
        layer_id (int): Unique identifier of the VGRAM layer.
    """
    globals.g_vgram_layers[layer_id].debug()


def recall_vgram_layer(layer_id, input_pattern):
    """
    Recall the VGRAM layer for an input pattern.

    Parameters:
        layer_id (int): Unique identifier of the VGRAM layer.
        input_pattern (bool[]): Input pattern.

    Returns:
        (int[]): Output values.
    """
    return globals.g_vgram_layers[layer_id].recall(input_pattern)


def learn_vgram_layer(layer_id, input_pattern, output_steps):
    """
    Decode a sparse representation.

    Parameters:
        layer_id (int): Unique identifier of the VGRAM layer.
        input_pattern (bool[]): Input pattern.
        output_steps (int[]): Output steps.
    """
    return globals.g_vgram_layers[layer_id].learn(input_pattern, output_steps)


def add_sparse_codec(codec_id, output_dims, learning_rate, min_splitting_volume, min_bounds, max_bounds):
    """
    Add a new sparse encoder/decoder.

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


def del_sparse_codec(codec_id):
    """
    Delete an existing sparse encoder/decoder.

    Parameters:
        codec_id (int): Unique identifier of the encoder/decoder.
    """
    globals.g_sparse_codecs.pop(codec_id)


def get_sparse_codec_ids():
    """
    Get the identifiers of the existing sparse encoders/decoders.

    Returns:
        (int []): Identifiers of the existing encoders/decoders.
    """
    globals.g_sparse_codecs.keys()


def clear_sparse_codecs():
    """
    Delete all existing sparse encoders/decoders.
    """
    globals.g_sparse_codecs.clear()


def draw_sparse_codec(codec_id, point, min_bounds, max_bounds, marker):
    """
    Draw the sparse encoder/decoder structure.

    Parameters:
        codec_id (int): Unique identifier of the encoder/decoder.
        point (double[]): Point to draw.
        min_bounds (double[]): Minimum bounds.
        max_bounds (double[]): Maximum bounds.
        marker (string): Point marker.
    """
    globals.g_sparse_codecs[codec_id].draw_tree(point, min_bounds, max_bounds, 0, 1, marker)


def encode_sparse_codec(codec_id, point):
    """
    Encode a point using an existing sparse encoder/decoder.

    Parameters:
        codec_id (int): Unique identifier of the encoder/decoder.
        point (double[]): Point to draw.

    Returns:
        (int[]): Sparse representation of the given point.
    """
    return globals.g_sparse_codecs[codec_id].encode(point)


def decode_sparse_codec(codec_id, code):
    """
    Decode a sparse representation using existing sparse encoder/decoder.

    Parameters:
        codec_id (int): Unique identifier of the encoder/decoder.
        code (int[]): Sparse representation.

    Returns:
        (double[]): Point corresponding to the sparse representation.
    """
    return globals.g_sparse_codecs[codec_id].decode(code)
