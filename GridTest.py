import pandapower as pp
import pandapower.networks as pn
from numba import jit
import pandas as pd
import numpy as np
import copy
import random as rnd

#action = [rnd.uniform(-1,1) for i in range(9)]

net = pn.case5()



class Grid(object):
    def __init__(self, net,max_loading=1, Vbusmax=1.03, Vbusmin=0.98, max_shedding =0.2):
        """
        Initialize the environment
        :param net: Power grid (for example IEEE 24-bus)
        :param max_loading: maximum line thermal limit
        :param Vbusmax: Maximum upper bound for Bus magnitude voltage
        :param Vbusmin : Minimum lower bound for Bus magnitude voltage
        """
        self.net_origin= net
        self.net = copy.deepcopy(net)
        self.max_loading = max_loading
        self.Vbusmax = Vbusmax
        self.Vbusmin = Vbusmin
        self.max_shedding = max_shedding

    def LftoState(self,grid):
        # pp.runpp(grid, lumba=False)
        pij = np.zeros((grid.bus.shape[0], grid.bus.shape[0]))
        qij = np.zeros((grid.bus.shape[0], grid.bus.shape[0]))
        for l, res in zip(grid.line.iterrows(), grid.res_line.iterrows()):
            pij[l[1].from_bus, l[1].to_bus] = res[1].p_from_mw
            pij[l[1].to_bus, l[1].from_bus] = -res[1].p_to_mw
            qij[l[1].from_bus, l[1].to_bus] = res[1].q_from_mvar
            qij[l[1].to_bus, l[1].from_bus] = -res[1].q_to_mvar
        return np.concatenate((grid.res_bus.values, pij, qij), axis=1)

    def Attack(self,ind=-1):
        if ind not in np.arange(self.net.line.shape[0]):
            print('You choose wrong number ({}) for line under attack'.format(ind))
            ind = np.random.choice(np.arange(self.net.line.shape[0]))
        print('Line with index -->{}<-- between bus {} and {} is attacked '.
              format(ind,self.net.line.from_bus[ind],self.net.line.to_bus[ind]))
        self.net.line.in_service[ind] = False

    def assessment(self, grid,line_limit, v_upper, v_lower):
        free_cap = list(map(lambda x: line_limit-x if x < line_limit else 0 , grid.res_line.loading_percent.values/100))
        overload = list(map(lambda x : x-line_limit if x > line_limit else 0, grid.res_line.loading_percent.values/100))
        overvoltage = list(map(lambda x : x-v_upper if x > v_upper else 0, grid.res_bus.vm_pu.values))
        undervoltage = list(map(lambda x : v_lower-x if x < v_lower else 0, grid.res_bus.vm_pu.values))
        return free_cap, overload , overvoltage , undervoltage

    def InitState(self):
        pp.runpp(self.net,algorithm='nr', numba= False)
        return self.LftoState(self.net)

    def StateFeatures(self):
        return [self.net.bus.shape[0],4+2*self.net.bus.shape[0]]

    def ActionFeature(self):
        return 2*self.net.gen.shape[0] + self.net.load.shape[0]

    # def attack(self,n):
    #     self.net.line.in_service[n] = False

    def take_action(self,action):
        Ng= self.net.res_gen.vm_pu.values.shape[0]
        self.net.gen.vm_pu= self.net.res_gen.vm_pu.values + np.array([action[i]*(self.Vbusmax-self.net.res_gen.vm_pu.values[i])
                                                             if action[i]>0 else action[i]*(self.net.res_gen.vm_pu.values[i]-self.Vbusmin)
                                                             for i in range(Ng)])

        # self.net.gen.vm_pu = self.net.res_gen.vm_pu + \
        #                      action[0:self.net.gen.shape[0]]*(self.Vbusmax-self.net.res_gen.vm_pu)
        print('Network new Voltages are :----> \n')
        print(self.net.gen.vm_pu)

        # self.net.gen.p_mw = self.net.res_gen.p_mw + \
        #                     action[self.net.gen.shape[0]:2*self.net.gen.shape[0]]*(self.net.gen.max_p_mw-self.net.res_gen.p_mw)

        self.net.gen.p_mw = self.net.res_gen.p_mw.values +np.array([action[i+Ng]*(self.net.gen.max_p_mw.values[i]-self.net.res_gen.p_mw[i])
                                                            if action[i+Ng] > 0 else
                                                            action[i+Ng] * (self.net.res_gen.p_mw[i]-self.net.gen.min_p_mw.values[i])
                                                            for i in range(Ng)])
        print('Network new Active Power are :----> \n')
        print(self.net.gen.p_mw)

        self.net.load.scaling = self.net.load.scaling.values - np.array([action[2*Ng+i]*self.max_shedding if self.net.load.scaling.values[i]-action[2*Ng+i]*self.max_shedding>0 else 0
         for i in range(Ng)])
        # self.net.load.scaling = self.net.load.scaling.values - np.array(action[2*Ng:self.ActionFeature()])*self.max_shedding
        # self.net.load.q_mvar = action[2*self.net.gen.shape[0]:self.ActionFeature()]
        pp.runpp(self.net,'nr',lumba=False)
        State = self.LftoState(self.net)
        free_cap, overload, overvoltage, undervoltage = self.assessment(self.net, self.max_loading, self.Vbusmax, self.Vbusmin)
        conditions = sum(overload) + sum(overvoltage) + sum(undervoltage)
        done = False
        if conditions == 0:
            done = True
            tmp = sum(free_cap)+sum(action[2*self.net.gen.shape[0]:self.ActionFeature()])
            reward= 0.8*(sum(free_cap)/tmp) - 0.2*(sum(action[2*Ng:self.ActionFeature()])/tmp)
        else:
            reward = -conditions
        return State, reward, done

    def reset(self):
        self.net = copy.deepcopy(self.net_origin)





a=1