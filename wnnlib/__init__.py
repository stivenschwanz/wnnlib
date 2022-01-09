import KDTree
import VGRAMNode
import globals


def add_vgram_layer(layer_id, output_dims):
    return 0


def add_encoder(encoder_id, output_dims, learning_rate, min_splitting_volume, min_bounds, max_bounds):
    """
    Add a new sparse encoder/decoder.

    Parameters:
        encoder_id (int): Unique identifier of the encoder/decoder.
        output_dims (int): Number of output dimensions.
        learning_rate (double): Learning rate.
        min_splitting_volume (double): Minimum splitting volume.
        min_bounds (double[]): Minimum bounds.
        max_bounds (double[]): Maximum bounds.
    """
    globals.g_encoders[encoder_id] = KDTree(int(output_dims), learning_rate, min_splitting_volume, min_bounds, max_bounds)


def del_encoder(encoder_id):
    """
       Delete an existing sparse encoder/decoder.

       Parameters:
           encoder_id (int): Unique identifier of the encoder/decoder.
    """
    globals.g_encoders.pop(encoder_id)


def get_encoder_ids():
    """
       Get the identifiers of the existing sparse encoders/decoders.

       Returns:
           (int []): Identifiers of the existing encoders/decoders.
    """
    globals.g_encoders.keys()


def clear_encoders():
    """
        Delete all existing encoders/decoders.
    """
    globals.g_encoders.clear()


def draw_encoder(encoder_id, point, min_bounds, max_bounds, marker):
    """
        Draw the encoder/decoder structure.

        Parameters:
            encoder_id (int): Unique identifier of the encoder/decoder.
            point (double[]): Point to draw.
            min_bounds (double[]): Minimum bounds.
            max_bounds (double[]): Maximum bounds.
            marker (string): Point marker.
    """
    globals.g_encoders[encoder_id].draw_tree(point, min_bounds, max_bounds, 0, 1, marker)


def encode(encoder_id, point):
    """
        Encode a point.

        Parameters:
            encoder_id (int): Unique identifier of the encoder/decoder.
            point (double[]): Point to draw.

        Returns:
            (int[]): Sparse representation of the given point.
    """
    return globals.g_encoders[encoder_id].encode(point)


def decode(encoder_id, code):
    """
        Decode a sparse representation.

        Parameters:
            encoder_id (int): Unique identifier of the encoder/decoder.
            code (int[]): Sparse representation.

        Returns:
            (double[]): Point corresponding to the sparse representation.
    """
    return globals.g_encoders[encoder_id].decode(code)
