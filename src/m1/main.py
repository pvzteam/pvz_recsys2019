# -*- coding: utf-8 -*-

import sys
import data 
import feat
import model
import resource


# resource.DEBUG = True


def run_sim():
    data.t_sim.execute()
    

def run_feat():
    feat.f_top30.execute()
    feat.f_top100.execute()
    feat.f_si_sim.execute()
    feat.f_si_cmp.execute()
    feat.f_si_win.execute()

    
def run_model():
    model.m_20190622.execute()
    model.m_20190624.execute()
    model.m_20190626.execute()
    

if __name__ == '__main__':
    module = sys.argv[1]
    assert module in ['sim', 'feat', 'model']
    
    if module == 'sim':
        run_sim()
    elif module == 'feat':
        run_feat()
    else:
        run_model()

