import numpy as np
from wban_opt.objective import ObjectiveContext, energy_and_penalty_from_x
from wban_opt.energy_model import EnergyParams

def test_objective_calculation():
    # Mock data
    points = np.array([[0,0], [1,0], [2,0], [0,1]]) # M=4
    gw = np.array([[0,0]])
    ep = EnergyParams(50e-9, 10e-12, 0.0013e-12, 5e-9, 4000, 1.0)
    
    ctx = ObjectiveContext(
        points_pool=points,
        N=2, K=1, gw_xy=gw,
        energy_params=ep,
        d_max_sn_ch=10.0, # Duży zasięg -> brak kary
        penalty_weight=1e6
    )
    
    # x wybiera p0, p1, p2 (D=3 <= M=4)
    # indices: [0, 1, 2] -> SN=[0,1], CH=[2]
    # SN at (0,0), (1,0). CH at (2,0).
    # Assigment: SN0->CH(dist 2), SN1->CH(dist 1)
    
    # Dummy x that maps to [0, 1, 2] roughly
    x = np.array([0.1, 0.3, 0.6]) 
    
    E, P, feas = energy_and_penalty_from_x(x, ctx)
    
    assert P == 0.0
    assert feas is True
    assert E > 0.0