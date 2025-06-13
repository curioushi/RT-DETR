import numpy as np
from itertools import permutations

def quad_to_obd(quad):
    """
    Encode a quadrangle to orderless box discretization.
    Input:
        quad: (4, 2) array or list of 8 values -> [x1, y1, x2, y2, x3, y3, x4, y4]
    Output:
        x_key_edges: list of 4 sorted x values
        y_key_edges: list of 4 sorted y values 
        mtl_type_index: int ∈ [0, 23], indicates best matching type
    """
    pts = np.array(quad, dtype=np.float32).reshape(4, 2)
    
    x_indices = np.argsort(pts[:, 0])
    x_key_edges = pts[x_indices, 0]
    y_key_edges = np.sort(pts[:, 1])
    ord_pts = pts[x_indices]
    
    # generate all 24 permutations
    perm = np.array(list(permutations(range(4))))
    matched_pts = np.empty((24, 4, 2), dtype=pts.dtype)
    matched_pts[:, :, 0] = x_key_edges
    matched_pts[:, :, 1] = y_key_edges[perm]

    # find best match
    diff = matched_pts - ord_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2).sum(axis=1)
    best_match_index = int(np.argmin(dist))

    return x_key_edges, y_key_edges, best_match_index

def odb_to_quad(x_ke, y_ke, mtl_type):
    """
    Decode an orderless box discretization to a quadrangle.
    Input:
        x_ke: list of 4 sorted x values
        y_ke: list of 4 sorted y values 
        mtl_type_index: int ∈ [0, 23], indicates best matching type
    Output:
        quad: (4, 2) array
    """
    perm = np.array(list(permutations(range(4))))
    y_ke_perm = y_ke[perm[mtl_type]]
    quad = np.stack([x_ke, y_ke_perm], axis=1)
    return quad

bbox = [100, 50, 150, 50, 150, 100, 100, 100]
print(bbox)
x_ke, y_ke, mtl_type = quad_to_obd(bbox)
print("X Key Edges:", x_ke)
print("Y Key Edges:", y_ke)
print("MTL Type Index:", mtl_type)

quad = odb_to_quad(x_ke, y_ke, mtl_type)
print("Quad:", quad)
