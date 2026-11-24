import gzip
import struct

import pytest

np = pytest.importorskip("numpy")
routes = pytest.importorskip("app.api.object3d_routes")
SH_C0, to_spz = routes.SH_C0, routes.to_spz

PROPS = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
         "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def _ply(rows: np.ndarray) -> bytes:
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {len(rows)}\n"
    header += "".join(f"property float {p}\n" for p in PROPS) + "end_header\n"
    return header.encode() + rows.astype(np.float32).tobytes()


def test_spz_header_layout_and_values_decode_back():
    rows = np.zeros((2, len(PROPS)), dtype=np.float32)
    rows[0, :3] = [0.5, -0.25, 1.0]
    rows[0, 6:9] = [(0.8 - 0.5) / SH_C0, (0.5 - 0.5) / SH_C0, (0.2 - 0.5) / SH_C0]
    rows[0, 9] = 0.0
    rows[0, 10:13] = [np.log(0.01)] * 3
    rows[0, 13:17] = [1, 0, 0, 0]
    rows[1, 13:17] = [0, 0, 0, -1]

    raw = gzip.decompress(to_spz(_ply(rows)))
    magic, version, n = struct.unpack("<III", raw[:12])
    assert (magic, version, n) == (0x5053474E, 3, 2)
    assert raw[12:16] == bytes([0, 12, 1, 0])
    assert len(raw) == 16 + n * (9 + 1 + 3 + 3 + 4)

    x = int.from_bytes(raw[16:19], "little", signed=True)
    assert x == round(0.5 * 4096)
    assert raw[16 + 9 * n] == 128
    rgb = raw[16 + 10 * n:16 + 10 * n + 3]
    assert [round(((v / 255 - 0.5) * (SH_C0 / 0.15)) + 0.5, 1) for v in rgb] == [0.8, 0.5, 0.2]
    assert abs(raw[16 + 13 * n] / 16 - 10 - np.log(0.01)) < 1 / 16

    rot = raw[16 + 16 * n:16 + 20 * n]
    assert int.from_bytes(rot[0:4], "little") >> 30 == 3
    assert int.from_bytes(rot[4:8], "little") >> 30 == 2
