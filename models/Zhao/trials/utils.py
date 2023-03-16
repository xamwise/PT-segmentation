from numba import cuda


def swap(x, y):
    tmp = x
    x = y
    y = tmp
    return x, y


def reheap(dist, idx, k):
    root = 0
    child = root * 2 + 1
    while child < k:
        if child + 1 < k and dist[child+1] > dist[child]:
            child += 1
        if dist[root] > dist[child]:
            return
        dist[root], dist[child] = swap(dist[root], dist[child])
        idx[root], idx[child] = swap(idx[root], idx[child])
        root = child
        child = root * 2 + 1
        
    return dist, idx


def heap_sort(dist, idx, k):
    for i in range(k - 1, 0, -1):
        dist[0], dist[i] = swap(dist[0], dist[i])
        idx[0], idx[i] = swap(idx[0], idx[i])
        dist, idx = reheap(dist[:i], idx[:i], i)
        
    return dist, idx


def get_bt_idx(idx, offset):
    i = 0
    while True:
        if idx < offset[i]:
            break
        else:
            i += 1
    return i


@cuda.jit
def knn_query_cuda(m, nsample, xyz, new_xyz, offset, new_offset, idx, dist2):
    # input: xyz (n, 3) new_xyz (m, 3)
    # output: idx (m, nsample) dist2 (m, nsample)
    pt_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if pt_idx >= m:
        return

    new_xyz += pt_idx * 3
    idx += pt_idx * nsample
    dist2 += pt_idx * nsample

    bt_idx = get_bt_idx(pt_idx, new_offset)
    if bt_idx == 0:
        start = 0
    else:
        start = offset[bt_idx - 1]
    end = offset[bt_idx]

    new_x, new_y, new_z = new_xyz[0], new_xyz[1], new_xyz[2]

    best_dist = cuda.local.array(128, dtype=float)
    best_idx = cuda.local.array(128, dtype=int)
    for i in range(nsample):
        best_dist[i] = 1e10
        best_idx[i] = -1
    for i in range(start, end):
        x, y, z = xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2]
        d2 = (new_x - x) ** 2 + (new_y - y) ** 2 + (new_z - z) ** 2
        if d2 < best_dist[0]:
            best_dist[0] = d2
            best_idx[0] = i
            best_dist, best_idx = reheap(best_dist, best_idx, nsample)
    best_dist, best_idx = heap_sort(best_dist, best_idx, nsample)
    for i in range(nsample):
        idx[i] = best_idx[i]
        dist2[i] = best_dist[i]
    
    return idx, dist2


