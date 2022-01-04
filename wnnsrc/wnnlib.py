from wnntypes import KDTree
import wnnglobals


def add_encoder(encoder_id, max_depth, learning_rate, min_splitting_volume, min_bounds, max_bounds):
    wnnglobals.g_encoders[encoder_id] = KDTree(int(max_depth), learning_rate, min_splitting_volume, min_bounds, max_bounds)


def del_encoder(encoder_id):
    wnnglobals.g_encoders.pop(encoder_id)


def get_encoders():
    wnnglobals.g_encoders.keys()


def clear_encoders():
    wnnglobals.g_encoders.clear()


def encode(encoder_id, point):
    return wnnglobals.g_encoders[encoder_id].encode(point)


def decode(encoder_id, code):
    return wnnglobals.g_encoders[encoder_id].decode(code)
