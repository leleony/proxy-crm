import plotly.graph_objects as go
import networkx as nx
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import scienceplots
import proxy_crm

class interwell_visual:
  def __init__(self, well_name: list, coor_x: NDArray, coor_y: NDArray, coor_z: NDArray, lambda_ip : NDArray = None):
    if well_name is None:
      msg = 'Well names are needed.'
      raise ValueError(msg)
    if coor_x is None:
      msg = 'x value is not inserted.'
      raise ValueError(msg)
    if coor_y is None:
      msg = 'y value is not inserted.'
      raise ValueError(msg)
    if lambda_ip is None:
      self.lambda_ip = proxy_crm.lambda_ip

    self.coor_x = coor_x
    self.coor_y = coor_y
    self.coor_z = coor_z
    self.lambda_ip = lambda_ip
    self.well_name = well_name

    self.opt_prod = {'node_color':'blue', 'with_labels':True, 'node_size':600}
    self.opt_inj = {'node_color':'orange', 'with_labels':True, 'node_size':600}
    
  def interwell_2d(self):
    coor_x = self.coor_x
    coor_y = self.coor_y
    well_name = self.well_name
    g_2d = nx.Graph()

    for i in range(coor_x):
      g_2d.add_node(well_name[i],pos=(coor_x[i],coor_y[i]))

    nx.draw(g_2d, nx.get_node_attributes(g_2d, 'pos'), **self.opt_prod)
    
    return