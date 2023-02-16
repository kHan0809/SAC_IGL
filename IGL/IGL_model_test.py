from model.IGL import IGL
import numpy as np
import os
import argparse
from IGL_utils import get_args


args  = get_args()

# print(args)
x = np.load(os.getcwd()+'/IGL_data/IGL_x_'+args.env+'.npy')
y = np.load(os.getcwd()+'/IGL_data/IGL_y_'+args.env+'.npy')
igl = IGL(x.shape[1],y.shape[1],args)
igl.dataset_split(x,y)
igl.load_model(os.getcwd()+'/IGL_model/', args.env)
igl.test()