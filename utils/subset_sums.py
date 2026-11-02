import numpy as np


def covering_radius(sums):
    s = np.sort(sums)
    # one neighbour on each side
    left = max(np.searchsorted(s, -1, side="right") - 1, 0)
    right = min(np.searchsorted(s, 1, side="left"), s.size - 1)
    window = s[left : right + 1]
    radius = max(window[0] + 1, 1 - window[-1], 0.0)
    if window.size > 1:
        a = window[:-1]
        b = window[1:]
        mid = np.clip((a + b) / 2, -1, 1)
        radius = max(radius, np.minimum(mid - a, b - mid).max())
    return float(radius)


def lattice_radii(weights, scale, margin=0.25, floor=2.0**-16):
    # weights are integers, the real weight is w / scale, so the bitset is exact
    n = len(weights)
    radii = np.full(n, np.inf)
    pos = np.cumsum(np.clip(weights, 0, None)[::-1])[::-1]
    neg = np.cumsum(np.clip(-weights, 0, None)[::-1])[::-1]
    edge = scale + int(margin * scale)
    lo, bits = 0, 1
    for k, x in enumerate(int(w) for w in weights):
        if x >= 0:
            bits |= bits << x
        else:
            bits = (bits << -x) | bits
            lo += x
        rest_pos = int(pos[k + 1]) if k + 1 < n else 0
        rest_neg = int(neg[k + 1]) if k + 1 < n else 0
        low, high = -edge - rest_pos, edge + rest_neg
        if lo < low:
            bits >>= low - lo
            lo = low
        bits &= (1 << (high - lo + 1)) - 1
        shift = -edge - lo
        window = bits >> shift if shift >= 0 else bits << -shift
        window &= (1 << (2 * edge + 1)) - 1
        raw = np.frombuffer(window.to_bytes((2 * edge + 8) // 8, "little"), np.uint8)
        sums = (np.flatnonzero(np.unpackbits(raw, bitorder="little")) - edge) / scale
        if sums.size:
            radius = covering_radius(sums)
            if radius <= margin:
                radii[k] = radius
        if radii[k] <= floor:
            radii[k:] = radii[k]
            break
    return radii
