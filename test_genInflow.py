import numpy as np
from genInflow import extractPALM, getRSTVectors, ainf

class MockVariable:
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        return self.data[idx]

class MockDataset:
    def __init__(self, x, y, z, nt):
        # build synthetic fields: U=x, V=y, W=z (linear so pchip interpolation is exact)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        # PALM stores as [t, z, y, x] so transpose accordingly
        u = np.stack([xx.T]*nt, axis=0)  # shape (nt, nz, ny, nx)
        v = np.stack([yy.T]*nt, axis=0)
        w = np.stack([zz.T]*nt, axis=0)
        self.variables = {
            'u': MockVariable(u),
            'v': MockVariable(v),
            'w': MockVariable(w),
            'x': MockVariable(x),
            'yv': MockVariable(y),
            'zu_3d': MockVariable(z),
            'time': MockVariable(np.linspace(0, 20, nt)),
        }

# module-level mock dataset shared across tests
x = np.linspace(0, 100, 50)
y = np.linspace(0, 100, 50)
z = np.linspace(0, 50, 25)
file_id = MockDataset(x, y, z, nt=1)

# -------------------------------------------------------
# extractPALM tests
# -------------------------------------------------------

def test_extractPALM_known_field():
    """interpolated value at interior point should match U=x, V=y, W=z"""
    xyz = np.array([[[[40.0, 60.0, 25.0]]]])  # shape (1,1,1,3)
    U, V, W = extractPALM(file_id, xyz, 0, x, y, z)
    assert np.isclose(U[0,0,0], 40.0, atol=0.1), f"U={U[0,0,0]}, expected 40.0"
    assert np.isclose(V[0,0,0], 60.0, atol=0.1), f"V={V[0,0,0]}, expected 60.0"
    assert np.isclose(W[0,0,0], 25.0, atol=0.1), f"W={W[0,0,0]}, expected 25.0"

def test_extractPALM_out_of_domain_returns_zero():
    """points outside PALM domain should return 0 (still air in ground frame)"""
    xyz = np.array([[[[999.0, 999.0, 999.0]]]])  # way outside
    U, V, W = extractPALM(file_id, xyz, 0, x, y, z)
    assert U[0,0,0] == 0.0, f"U={U[0,0,0]}, expected 0.0"
    assert V[0,0,0] == 0.0, f"V={V[0,0,0]}, expected 0.0"
    assert W[0,0,0] == 0.0, f"W={W[0,0,0]}, expected 0.0"

def test_extractPALM_multiple_points():
    """check that multiple query points all return correct values"""
    pts = np.array([[[[10.0, 20.0, 5.0],
                      [50.0, 50.0, 25.0],
                      [90.0, 80.0, 45.0]]]])  # shape (1,1,3,3)
    U, V, W = extractPALM(file_id, pts, 0, x, y, z)
    assert np.isclose(U[0,0,0], 10.0, atol=0.1)
    assert np.isclose(U[0,0,1], 50.0, atol=0.1)
    assert np.isclose(U[0,0,2], 90.0, atol=0.1)
    assert np.isclose(V[0,0,0], 20.0, atol=0.1)
    assert np.isclose(W[0,0,2], 45.0, atol=0.1)

# -------------------------------------------------------
# getRSTVectors tests
# -------------------------------------------------------

def test_getRSTVectors_orthonormal():
    """RST basis vectors must be unit length and mutually orthogonal"""
    ftime = np.array([0.0, 1.0, 2.0])
    vh_s = np.array([[10.0, 10.0, 10.0],
                     [8.0,  8.0,  9.0],
                     [6.0,  6.0,  8.0]])
    r_hat, s_hat, t_hat = getRSTVectors(ftime, vh_s, 0, 1)

    assert np.isclose(np.linalg.norm(r_hat), 1.0), "r_hat not unit length"
    assert np.isclose(np.linalg.norm(s_hat), 1.0), "s_hat not unit length"
    assert np.isclose(np.linalg.norm(t_hat), 1.0), "t_hat not unit length"

    assert np.isclose(np.dot(r_hat, s_hat), 0.0, atol=1e-10), "r_hat and s_hat not orthogonal"
    assert np.isclose(np.dot(r_hat, t_hat), 0.0, atol=1e-10), "r_hat and t_hat not orthogonal"
    assert np.isclose(np.dot(s_hat, t_hat), 0.0, atol=1e-10), "s_hat and t_hat not orthogonal"

def test_getRSTVectors_r_hat_points_aft():
    """for vehicle moving in +x, r_hat should point in -x (aft)"""
    ftime = np.array([0.0, 1.0])
    vh_s = np.array([[0.0, 0.0, 0.0],
                     [1.0, 0.0, 0.0]])
    r_hat, s_hat, t_hat = getRSTVectors(ftime, vh_s, 0, 1)

    assert r_hat[0] < 0, "r_hat should point aft (negative x)"
    assert np.isclose(r_hat, [-1, 0, 0], atol=1e-10).all(), f"r_hat={r_hat}, expected [-1,0,0]"

def test_getRSTVectors_t_hat_is_vertical():
    """t_hat should always be world-up [0,0,1] regardless of trajectory"""
    ftime = np.array([0.0, 1.0])
    vh_s = np.array([[0.0, 0.0, 0.0],
                     [1.0, 1.0, 0.5]])  # diagonal trajectory
    r_hat, s_hat, t_hat = getRSTVectors(ftime, vh_s, 0, 1)
    assert np.allclose(t_hat, [0, 0, 1]), f"t_hat={t_hat}, expected [0,0,1]"

def test_getRSTVectors_right_handed():
    """RST frame should form a right-handed coordinate system"""
    ftime = np.array([0.0, 1.0])
    vh_s = np.array([[0.0, 0.0, 0.0],
                     [1.0, 0.5, 0.2]])
    r_hat, s_hat, t_hat = getRSTVectors(ftime, vh_s, 0, 1)
    assert np.allclose(np.cross(r_hat, s_hat), t_hat, atol=1e-10), \
        "RST frame is not right-handed"

# -------------------------------------------------------
# Velocity frame transformation tests
# -------------------------------------------------------

def test_rst_velocity_roundtrip():
    """projecting XYZ velocity into RST and back should recover original"""
    r_hat = np.array([-1.0, 0.0, 0.0])
    s_hat = np.array([0.0,  1.0, 0.0])
    t_hat = np.array([0.0,  0.0, 1.0])

    vec_xyz = np.array([3.0, -2.0, 1.5])
    vr = np.dot(vec_xyz, r_hat)
    vs = np.dot(vec_xyz, s_hat)
    vt = np.dot(vec_xyz, t_hat)

    reconstructed = vr*r_hat + vs*s_hat + vt*t_hat
    assert np.allclose(reconstructed, vec_xyz), \
        f"Roundtrip failed: got {reconstructed}, expected {vec_xyz}"

def test_still_air_gives_vehicle_velocity():
    """if PALM returns zero (still air), RST velocity should equal vh_v projected onto RST"""
    r_hat  = np.array([-1.0, 0.0, 0.0])
    s_hat  = np.array([0.0,  1.0, 0.0])
    t_hat  = np.array([0.0,  0.0, 1.0])
    vh_v_k = np.array([10.0, 5.0, -1.0])

    U = np.zeros((1,1,1))
    V = np.zeros((1,1,1))
    W = np.zeros((1,1,1))
    vec = np.stack([U, V, W], axis=-1) + vh_v_k  # shape (1,1,1,3)

    vr = (vec @ r_hat)[0,0,0] / ainf
    vs = (vec @ s_hat)[0,0,0] / ainf
    vt = (vec @ t_hat)[0,0,0] / ainf

    assert np.isclose(vr, np.dot(vh_v_k, r_hat) / ainf), f"vr={vr}"
    assert np.isclose(vs, np.dot(vh_v_k, s_hat) / ainf), f"vs={vs}"
    assert np.isclose(vt, np.dot(vh_v_k, t_hat) / ainf), f"vt={vt}"

# -------------------------------------------------------
# XYZ grid construction test
# -------------------------------------------------------

def test_xyz_grid_centered_on_vehicle():
    """center point of RST grid (r=s=t=0) should equal vehicle position"""
    r_hat = np.array([-1.0, 0.0, 0.0])
    s_hat = np.array([0.0,  1.0, 0.0])
    t_hat = np.array([0.0,  0.0, 1.0])
    vh_pos = np.array([50.0, 60.0, 70.0])

    vecr = np.linspace(-10, 10, 3)  # index 1 = 0
    vecs = np.linspace(-10, 10, 3)
    vect = np.linspace(-10, 10, 3)
    rgrids, sgrids, tgrids = np.meshgrid(vecr, vecs, vect, indexing='ij')
    xyz = (vh_pos
           + rgrids[..., np.newaxis] * r_hat
           + sgrids[..., np.newaxis] * s_hat
           + tgrids[..., np.newaxis] * t_hat)

    assert np.allclose(xyz[1,1,1,:], vh_pos), \
        f"Center={xyz[1,1,1,:]}, expected {vh_pos}"
