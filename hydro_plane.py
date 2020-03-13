from imports import *

import numpy as np
import pandas as pd 
import os

class Ellipse(object):
    """
    Ellipse located at the bottom of the tray in a 2D plane. 
    """

    def __init__(self, lb, ub, names = ["bottom", "lowercentre", "uppercentre", "top", "lip"], \
                        y = np.array([-14.9375, -10.875, -6.0, -2.34375, 0.8375]), \
                        p1 = np.array([0, 6.0625, 0])):
        """
        lb (float): lower boundary on the radius (we used 7.5 before)
        ub (float): upper boundary on the raidus (we used 17 before)
        filename (str): name of the glyph file
        preamble (str): preamble in the line
        ids (list of str): identifier for the values to change in glyph file
        multipliers (numpy array): sign of the ids
        postamble (str): end of the line
        """
        self.lb = lb
        self.ub = ub
        self.names = names
        self.y = y
        self.p1 = p1
        self.p2 = np.zeros(5)
        self.c = np.zeros(5)
        self.a = np.zeros(5)
        self.b = np.zeros(5)
        self.theta = np.zeros(5)
        self.mesh_path = './data/HeadCell/meshes/'

    def get_ellipse_points(self, p1, p2, y):
        '''
        Calculate the centre of an ellipse along a straight line 
        
        Parameters.
        -----------
        p1 (np array): vector of fixed centre coordinates.
        p2 (np array): vector of fixed centre coordinates.
        y (float): vertical distance at centre.
        
        returns np array of (x, z) coordinates of centre.
        '''
        
        x1=p1[0]
        y1=p1[1]
        z1=p1[2]
        x2=p2[0]
        y2=p2[1]
        z2=p2[2]

        new_x = x1 + (x2 - x1)*(y - y1)/(y2 - y1)
        new_z = z1 + (z2 - z1)*(y - y1)/(y2 - y1)
        new_p = np.array([new_x, new_z])
        
        return new_p    
        
    def get_ellipse_angles(self, angle_max, y):
        '''
        Calculate the angle of an ellipse along a straight line - angle is zero at 1.3125
        
        Parameters.
        -----------
        angle_max (float): vector of fixed centre coordinates.
        y (float): vertical distance at centre.
        
        returns (float) angle.
        '''
        
        angle = (angle_max/(-14.9375-1.3125))*(y-1.3125)
        
        return angle    
        
    def get_ellipse_radii(self, radius_max, radius_min, y):
        '''
        Calculate the angle of an ellipse along a straight line 
        
        Parameters.
        -----------
        angle_max (float): vector of fixed centre coordinates.
        y (float): vertical distance at centre.
        
        returns (float) angle.
        '''
        
        radius = ((radius_max-radius_min)/(1.3125+14.9375))*(y-1.3125) + radius_max
        
        return radius
        
    def generate_ellipse(self, c, a, b, angles, theta=0, n=0):
        '''
        Produce the points on the perimeter of an ellipse 
        defined by parameters, c, a, b and theta. 
        
        Parameters.
        -----------
        c (np array): vector of centre coordinates.
        a (float): semi major axis length.
        b (float): semi minor axis length.
        theta (float): rotation in radians. 
        n (int): number of points to generate on the perimeter. 
        
        returns array of coordinates on the perimeter of a parametric ellipse.
        '''
        #angles = np.linspace(start=0, stop=2*np.pi, num=n, endpoint=False)
        x = [c[0] + a * np.cos(t) for t in angles]
        y = [c[1] + b * np.sin(t) for t in angles]
        coords = np.array([x, y]).T
        rot = np.array([[np.cos(theta), -np.sin(theta)], \
                        [np.sin(theta), np.cos(theta)]])
        return np.dot(coords - c, rot.T) + c    
        
    def generate_tray_layer(self, c, a, b, y, rot_angle, name, x):
        '''
        Generate the connectors (cb*.dat) and points for splines (*.glf) for one layer
        
        Parameters.
        -----------
        c (np array): vector of centre coordinates.
        a (float): semi major axis length.
        b (float): semi minor axis length.
        rot_angle (float): rotation in radians. 
        name (string): name of the layer. 
        
        returns cb*.dat and *.glf files.
        '''
        dir_name = "_"
        for j in range(len(x)):
            dir_name += "{0:.4f}_".format(x[j])
            
        ntotal = 5000
        os.remove(self.mesh_path + dir_name + "/" + name + ".glf")

        for i in range(1,5):
            newangles = np.linspace(start=0.5*(i-1)*np.pi, stop=0.5*i*np.pi, num=ntotal, endpoint=True)
            corners = np.linspace(start=0.5*i*np.pi, stop=0.5*(i-1)*np.pi, num=1, endpoint=True)
            coords = self.generate_ellipse(c, a, b, angles = newangles, theta=rot_angle+np.pi, n=ntotal)
            cornercoords = self.generate_ellipse(c, a, b, angles = corners, theta=rot_angle+np.pi, n=4)
            firstline = [ntotal, '', '']
            yaxis = np.full((ntotal, 1), y)
            ycolumn = np.hstack((coords,yaxis))
            altogether = np.vstack((firstline,ycolumn))
            df = pd.DataFrame(altogether)
            df = df[[0,2,1]]
            file_name="cb"+name+str(i)+".dat" #need to add 4 for 
            corner_entry = "set"+' '+"xp"+name+str(i)+' '+str(cornercoords[0,0])+";"
            corner_entry2 = "set"+' '+"zp"+name+str(i)+' '+str(cornercoords[0,1])+";"
            
            with open(self.mesh_path + dir_name + "/" + name + ".glf", "a") as myfile:
                myfile.write(corner_entry+ '\n')
                myfile.write(corner_entry2+ '\n')
            df.to_csv(self.mesh_path + dir_name + "/" + file_name, index=False, sep=' ', header=False)        

    def generate_grit_pot(self, a, b, name, true_centre_x, true_centre_z, true_angle, x):

        y = -19.1875
        c = [0.0, 0.0]
        name = "pot"
        rot_angle = 0.0
        ntotal = 5000

        dir_name = "_"
        for j in range(len(x)):
            dir_name += "{0:.4f}_".format(x[j])

        for i in range(1,5):
            newangles = np.linspace(start=0.5*(i-1)*np.pi, stop=0.5*i*np.pi, num=ntotal, endpoint=True)
            #corners = np.linspace(start=0.5*i*np.pi, stop=0.5*(i-1)*np.pi, num=1, endpoint=True)
            coords = self.generate_ellipse(c, a, b, angles = newangles, theta=rot_angle+np.pi, n=ntotal)
            #cornercoords = self.generate_ellipse(c, a, b, angles = corners, theta=rot_angle+np.pi, n=4)
            firstline = [ntotal, '', '']
            yaxis = np.full((ntotal, 1), y)
            ycolumn = np.hstack((coords,yaxis))
            altogether = np.vstack((firstline,ycolumn))
            df = pd.DataFrame(altogether)
            df = df[[0,2,1]]
            file_name="cb"+name+str(i)+".dat" #need to add 4 for 
            df.to_csv(self.mesh_path + dir_name + "/" + file_name, index=False, sep=' ', header=False)

        os.remove(self.mesh_path + dir_name + "/" + name + ".glf")

        myfile = open(self.mesh_path + dir_name + "/" + name + ".glf", "w")   # The file is newly created where foo.py is 

        d = np.zeros(3)
        d[0] = -1.0*(b - 3.12)
        d[1] = -1.0*(b - 1.73)
        d[2] = -1.0*(b - 0.32)

        for i in range(3):    
            entry = "set"+' '+"zp"+name+str(i+1)+' '+str(d[i])+";"
            myfile.write(entry + '\n')
            
        entry = "set"+' '+"x"+name+' '+str(true_centre_x)+";"
        myfile.write(entry + '\n')
        entry = "set"+' '+"z"+name+' '+str(true_centre_z)+";"
        myfile.write(entry + '\n')
        entry = "set"+' '+"angle"+name+' '+str(true_angle)+";"
        myfile.write(entry + '\n')
            
        myfile.close() 





    def update(self, x):
        '''
        Generate the connectors (cb*.dat) and points for splines (*.glf) for all layers
        
        Parameters.
        -----------
        x (np array): decision vector of five variables: [bottom_angle, x_centre, z_centre, major_radius, minor_radius]
        c (np array): vector of centre coordinates.
        a (float): semi major axis length.
        b (float): semi minor axis length.
        theta (float): rotation in radians. 
        name (string): name of the layer. 
        
        returns cb*.dat and *.glf files.
        '''
        #constants
        y = self.y
        p1 = self.p1
        names = self.names
        
        bottom_x_centre = x[0]
        bottom_z_centre = x[1]
        bottom_angle = x[2]
        bottom_major_radius = x[3]
        bottom_minor_radius = x[4]

        p2 = np.array([bottom_x_centre, y[0], bottom_z_centre])
        self.p2 = p2
        
        c0 = self.get_ellipse_points(p1, p2, -14.9375)                      # bottom of tray
        c1 = self.get_ellipse_points(p1, p2, -10.875)                       # lower centre of tray
        c2 = self.get_ellipse_points(p1, p2, -6.0)                          # upper centre of tray
        c3 = self.get_ellipse_points(p1, p2, -2.34375)                      # top of tray
        c4 = self.get_ellipse_points(p1, p2, 0.8375)                        # lip
        c = np.array([c0, c1, c2, c3, c4])
        self.c = c
        
        # major radii
        a0 = bottom_major_radius
        a1 = self.get_ellipse_radii(21.25, bottom_major_radius, -10.875)   # lower centre of tray
        a2 = self.get_ellipse_radii(21.25, bottom_major_radius, -6.0)      # upper centre of tray
        a3 = self.get_ellipse_radii(21.25, bottom_major_radius, -2.34375)  # top of tray
        a4 = a3
        a = np.array([a0, a1, a2, a3, a4])
        self.a = a
        
        #minor radii
        b0 = bottom_minor_radius
        b1 = self.get_ellipse_radii(21.25, bottom_minor_radius, -10.875)   # lower centre of tray
        b2 = self.get_ellipse_radii(21.25, bottom_minor_radius, -6.0)      # upper centre of tray
        b3 = self.get_ellipse_radii(21.25, bottom_minor_radius, -2.34375)  # top of tray
        b4 = b3
        b = np.array([b0, b1, b2, b3, b4])
        self.b = b
        
        #angles
        theta0 = self.get_ellipse_angles(bottom_angle, -14.9375)    # bottom of tray
        theta1 = self.get_ellipse_angles(bottom_angle, -10.875)     # lower centre of tray
        theta2 = self.get_ellipse_angles(bottom_angle, -6.0)        # upper centre of tray
        theta3 = self.get_ellipse_angles(bottom_angle, -2.34375)    # top of tray
        theta4 = theta3
        theta = (np.pi/180.0)*np.array([theta0, theta1, theta2, theta3, theta4])
        self.theta = theta

        #grit pot
        self.generate_grit_pot(bottom_major_radius, bottom_minor_radius, "pot", bottom_x_centre, bottom_z_centre, bottom_angle, x)
        
        #rot_angle = x[4,:]
        #names = x[5,:]

        for i in range(0,5):
            self.generate_tray_layer(c[i], a[i], b[i], y[i], theta[i], names[i], x)
     
        
#if __name__ == "__main__":
#        
#    import headcell
#    
#    prob = HeadCell({})
#    
#    lb, ub = prob.get_decision_boundary()
#    
#    layout = Ellipse(lb, ub)
#    
#    d = np.random.random(len(lb)) * (ub - lb) + lb
#    
#    layout.update(d)