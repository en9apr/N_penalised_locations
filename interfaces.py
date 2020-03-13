import numpy as np
try:
    from base_class import Interface
except:
    from .base_class import Interface
try:
    from data.support import Chebyshev, MonotonicBetaCDF, PipeRow
except:
    from .data.support import Chebyshev, MonotonicBetaCDF, PipeRow

class PipeInterface(Interface):

    def __init__(self, vert_origin, vert_positions, xlb, xub, rlb, rub, nlb, nub,\
                        n_coeffs_radii, n_coeffs_num, n_betas,\
                        clb=-1, cub=1, domain=[0,1], window=[-1,1]):
        self.vert_origin = vert_origin # where is the vertical origin
        self.vert_positions =  vert_positions # coordinates of the vertical axis
        self.xlb = xlb # horizontal limits
        self.xub = xub
        self.rlb = rlb # radius limits
        self.rub = rub
        self.nlb = nlb # number of pipes limits
        self.nub = nub
        self.n_coeffs_radii = n_coeffs_radii # coefficient for radii
        self.n_coeffs_num = n_coeffs_num # coefficients for number of pipes
        self.n_betas = n_betas # number of beta functions -- used for
        self.n_pipes_count = Chebyshev(self.n_coeffs_num, clb=clb, cub=cub,\
                                        domain=domain, window=window)
        self.n_pipes = np.rint(self.n_pipes_count.evaluate(len(self.vert_positions), self.nlb, self.nub)).astype('int')
        rows = []
        for i in range(len(self.vert_positions)):
            rows.append(PipeRow(vert_origin, vert_positions[i], self.n_pipes[i], xlb, xub,\
                            rlb, rub, n_coeffs_radii[i], n_betas[i]))

        self.rows = rows
        self.lb, self.ub = self.get_decision_boundary()

    def convert_shape_to_decision(self):
        d = [self.n_pipes_count.coeffs]
        for row in self.rows:
            d.append(row.centers.alphas)
            d.append(row.centers.betas)
            d.append(row.centers.omegas)
            d.append(row.radii.coeffs)
        d = np.concatenate(d)
        assert d.shape[0] == self.lb.shape[0] == self.ub.shape[0]
        return d

    def get_decision_boundary(self):
        lb = [self.n_pipes_count.clb]*len(self.n_pipes_count.coeffs)
        ub = [self.n_pipes_count.cub]*len(self.n_pipes_count.coeffs)
        for row in self.rows:
            lb.extend([row.centers.alb]*len(row.centers.alphas))
            ub.extend([row.centers.aub]*len(row.centers.alphas))
            lb.extend([row.centers.blb]*len(row.centers.betas))
            ub.extend([row.centers.bub]*len(row.centers.betas))
            lb.extend([row.centers.wlb]*len(row.centers.omegas))
            ub.extend([row.centers.wub]*len(row.centers.omegas))
            lb.extend([row.radii.clb]*len(row.radii.coeffs))
            ub.extend([row.radii.cub]*len(row.radii.coeffs))
        return np.array(lb), np.array(ub)

    def update_layout(self, d):
        """
        d = decision variables
        """
        init = 0
        next_ind = self.n_pipes_count.coeffs.shape[0]
        #print("number of pipes coeffcients: ", self.n_pipes_c.coeffs, d[init:next_ind])
        self.n_pipes_count.coeffs = d[init:next_ind]
        self.n_pipes_count.update_function()
        #print(self.n_pipes_count.evaluate(len(self.vert_positions), self.nlb, self.nub), self.nlb, self.nub)
        self.n_pipes = np.rint(self.n_pipes_count.evaluate(len(self.vert_positions), self.nlb, self.nub)).astype('int')
        for i in range(len(self.rows)):
            self.rows[i].n_pipes = self.n_pipes[i]
            #print("row: ", i)
            init += next_ind
            next_ind = self.rows[i].centers.alphas.shape[0]
            #print("number of center alphas: ", self.rows[i].centers.alphas, d[init:init+next_ind])
            self.rows[i].centers.alphas = d[init:init+next_ind]
            init += next_ind
            next_ind = self.rows[i].centers.betas.shape[0]
            #print("number of centter betas: ", self.rows[i].centers.betas, d[init:init+next_ind])
            self.rows[i].centers.betas = d[init:init+next_ind]
            init += next_ind
            next_ind = self.rows[i].centers.omegas.shape[0]
            #self.rows[i].centers.omegas = d[init:init+next_ind]
            omegas = d[init:init+next_ind]
            #print("number of omegas: ", self.rows[i].centers.omegas, omegas)
            self.rows[i].centers.set_omegas(omegas)
            init += next_ind
            next_ind = self.rows[i].radii.coeffs.shape[0]
            #print("number of radii coeffcients: ", self.rows[i].radii.coeffs, d[init:init+next_ind])
            self.rows[i].radii.coeffs = d[init:init+next_ind]
            self.rows[i].radii.update_function()
            self.rows[i].evaluate()

    def constraint(self, d):
        self.update_layout(d)
        for row in self.rows:
            if not row.check_constraints():
                return False
        nrows = len(self.rows)
        for  i in range(nrows):
            for j in range(i+1, nrows, 1):
                if not self.rows[i].check_constraints(other=self.rows[j]):
                    return False
        return True

    def convert_decision_to_shape(self, d):
        self.update_layout(d)
        xs, ys, rs = [], [], []
        for row in self.rows:
            ys.extend( [row.y] * row.n_pipes)
            xs.extend( list(row.x))
            rs.extend( list(row.r))
            if row.repeat:
                ys.extend([(2*row.y0) - row.y]*row.n_pipes)
                xs.extend(xs)
                rs.extend(rs)
        return xs, ys, rs

    def plot(self, fignum=None):
        import matplotlib.pyplot as plt
        plt.ion()
        try:
            plt.figure(fignum)
        except:
            plt.figure()
        ax = plt.gca()
        ax.cla()
        min_x, min_y, max_x, max_y, max_r = 0, 0, 0, 0, 0
        for row in self.rows:
            ys = [row.y] * row.n_pipes
            xs = list(row.x)
            rs = list(row.r)
            if row.repeat:
                ys.extend([(2*row.y0) - row.y]*row.n_pipes)
                xs.extend(xs)
                rs.extend(rs)
            for i in range(len(xs)):
                circle = plt.Circle((xs[i], ys[i]), rs[i], facecolor="blue", edgecolor="black", alpha=0.25)
                ax.add_artist(circle)
            min_x = np.min(np.concatenate([[min_x], xs]))
            max_x = np.max(np.concatenate([[max_x], xs]))
            min_y = np.min(np.concatenate([[min_y], ys]))
            max_y = np.max(np.concatenate([[max_y], ys]))
            max_r = np.max(np.concatenate([[max_r], rs]))
        plt.xlim(min_x - max_r, max_x + max_r)
        plt.ylim(min_y - max_r, max_y + max_r)
        plt.draw()


class EllipseInterface(Interface):

    def __init__(self, xlb, xub, zlb, zub, anglelb, angleub, \
                 majorlb, majorub, minorlb, minorub):#,\
                        #n_coeffs_radii, n_coeffs_num, n_betas,\
                       # clb=-1, cub=1, domain=[0,1], window=[-1,1]):
        self.xlb = xlb # horizontal limits
        self.xub = xub
        self.zlb = zlb # vertical limits
        self.zub = zub
        self.anglelb = anglelb
        self.angleub = angleub
        self.majorlb = majorlb
        self.majorub = majorub
        self.minorlb = minorlb
        self.minorub = minorub
#        self.nlb = nlb # number of pipes limits
#        self.nub = nub
#        self.n_coeffs_radii = n_coeffs_radii # coefficient for radii
#        self.n_coeffs_num = n_coeffs_num # coefficients for number of pipes
#        self.n_betas = n_betas # number of beta functions -- used for
#        self.n_pipes_count = Chebyshev(self.n_coeffs_num, clb=clb, cub=cub,\
#                                        domain=domain, window=window)
#        self.n_pipes = np.rint(self.n_pipes_count.evaluate(len(self.vert_positions), self.nlb, self.nub)).astype('int')
#        rows = []
#        for i in range(len(self.vert_positions)):
#            rows.append(PipeRow(vert_origin, vert_positions[i], self.n_pipes[i], xlb, xub,\
#                            rlb, rub, n_coeffs_radii[i], n_betas[i]))
#
#        self.rows = rows
        self.lb, self.ub = self.get_decision_boundary()

    def convert_shape_to_decision(self):
        d = [self.n_pipes_count.coeffs]
        for row in self.rows:
            d.append(row.centers.alphas)
            d.append(row.centers.betas)
            d.append(row.centers.omegas)
            d.append(row.radii.coeffs)
        d = np.concatenate(d)
        assert d.shape[0] == self.lb.shape[0] == self.ub.shape[0]
        return d

    def get_decision_boundary(self):
        lb = [self.xlb, self.zlb, self.anglelb, self.majorlb, self.minorlb]
        ub = [self.xub, self.zub, self.angleub, self.majorub, self.minorub]
#        for row in self.rows:
#            lb.extend([row.centers.alb]*len(row.centers.alphas))
#            ub.extend([row.centers.aub]*len(row.centers.alphas))
#            lb.extend([row.centers.blb]*len(row.centers.betas))
#            ub.extend([row.centers.bub]*len(row.centers.betas))
#            lb.extend([row.centers.wlb]*len(row.centers.omegas))
#            ub.extend([row.centers.wub]*len(row.centers.omegas))
#            lb.extend([row.radii.clb]*len(row.radii.coeffs))
#            ub.extend([row.radii.cub]*len(row.radii.coeffs))
        return np.array(lb), np.array(ub)

    def update_layout(self, d):
        """
        d = decision variables
        """
        init = 0
        next_ind = self.n_pipes_count.coeffs.shape[0]
        #print("number of pipes coeffcients: ", self.n_pipes_c.coeffs, d[init:next_ind])
        self.n_pipes_count.coeffs = d[init:next_ind]
        self.n_pipes_count.update_function()
        #print(self.n_pipes_count.evaluate(len(self.vert_positions), self.nlb, self.nub), self.nlb, self.nub)
        self.n_pipes = np.rint(self.n_pipes_count.evaluate(len(self.vert_positions), self.nlb, self.nub)).astype('int')
        for i in range(len(self.rows)):
            self.rows[i].n_pipes = self.n_pipes[i]
            #print("row: ", i)
            init += next_ind
            next_ind = self.rows[i].centers.alphas.shape[0]
            #print("number of center alphas: ", self.rows[i].centers.alphas, d[init:init+next_ind])
            self.rows[i].centers.alphas = d[init:init+next_ind]
            init += next_ind
            next_ind = self.rows[i].centers.betas.shape[0]
            #print("number of centter betas: ", self.rows[i].centers.betas, d[init:init+next_ind])
            self.rows[i].centers.betas = d[init:init+next_ind]
            init += next_ind
            next_ind = self.rows[i].centers.omegas.shape[0]
            #self.rows[i].centers.omegas = d[init:init+next_ind]
            omegas = d[init:init+next_ind]
            #print("number of omegas: ", self.rows[i].centers.omegas, omegas)
            self.rows[i].centers.set_omegas(omegas)
            init += next_ind
            next_ind = self.rows[i].radii.coeffs.shape[0]
            #print("number of radii coeffcients: ", self.rows[i].radii.coeffs, d[init:init+next_ind])
            self.rows[i].radii.coeffs = d[init:init+next_ind]
            self.rows[i].radii.update_function()
            self.rows[i].evaluate()

    def constraint(self, d):
        self.update_layout(d)
        for row in self.rows:
            if not row.check_constraints():
                return False
        nrows = len(self.rows)
        for  i in range(nrows):
            for j in range(i+1, nrows, 1):
                if not self.rows[i].check_constraints(other=self.rows[j]):
                    return False
        return True

    def convert_decision_to_shape(self, d):
        self.update_layout(d)
        xs, ys, rs = [], [], []
        for row in self.rows:
            ys.extend( [row.y] * row.n_pipes)
            xs.extend( list(row.x))
            rs.extend( list(row.r))
            if row.repeat:
                ys.extend([(2*row.y0) - row.y]*row.n_pipes)
                xs.extend(xs)
                rs.extend(rs)
        return xs, ys, rs

    def plot(self, fignum=None):
        import matplotlib.pyplot as plt
        plt.ion()
        try:
            plt.figure(fignum)
        except:
            plt.figure()
        ax = plt.gca()
        ax.cla()
        min_x, min_y, max_x, max_y, max_r = 0, 0, 0, 0, 0
        for row in self.rows:
            ys = [row.y] * row.n_pipes
            xs = list(row.x)
            rs = list(row.r)
            if row.repeat:
                ys.extend([(2*row.y0) - row.y]*row.n_pipes)
                xs.extend(xs)
                rs.extend(rs)
            for i in range(len(xs)):
                circle = plt.Circle((xs[i], ys[i]), rs[i], facecolor="blue", edgecolor="black", alpha=0.25)
                ax.add_artist(circle)
            min_x = np.min(np.concatenate([[min_x], xs]))
            max_x = np.max(np.concatenate([[max_x], xs]))
            min_y = np.min(np.concatenate([[min_y], ys]))
            max_y = np.max(np.concatenate([[max_y], ys]))
            max_r = np.max(np.concatenate([[max_r], rs]))
        plt.xlim(min_x - max_r, max_x + max_r)
        plt.ylim(min_y - max_r, max_y + max_r)
        plt.draw()





################################################################################
#tests
################################################################################

def test_pipes():
    import numpy as np
    D = 0.2
    vert_origin = 0
    n_rows = 3
    vert_positions = np.array([-D, 0, D])
    
    xlb, xub = -D, 3.25*D
    rlb, rub = 0.005, 0.5*D
    nlb, nub = 1, 5
    n_coeffs_radii = [3]*n_rows
    n_coeffs_num = 4
    n_betas = [3]*n_rows
    pipes = PipeInterface(vert_origin, vert_positions, xlb, xub, rlb, rub, \
                            nlb, nub, n_coeffs_radii, n_coeffs_num, n_betas)
    lb, ub = pipes.lb, pipes.ub
    import numpy as np
    for i in range(100):
        d = np.random.random(len(lb)) * (ub - lb) + lb
        d[0] = -1
        d[1:4] = 1
        if pipes.constraint(d):
            print(d)
            pipes.plot(1)
        else:
            print('invalid')
        input()
            
        
        
        
def test_radius():
    import numpy as np
    D = 5
    vert_origin = 0
    n_rows = 4
    vert_positions = np.array([-14.9375, -10.875, -6, -2.34375])
    xlb, xub = -D, 3.25*D
    rlb, rub = 0.005, 0.5*D
    nlb, nub = 1, 5
    n_coeffs_radii = [3]*n_rows
    n_coeffs_num = 4
    n_betas = [3]*n_rows
    pipes = PipeInterface(vert_origin, vert_positions, xlb, xub, rlb, rub, \
                            nlb, nub, n_coeffs_radii, n_coeffs_num, n_betas)
    lb, ub = pipes.lb, pipes.ub
    import numpy as np
    for i in range(100):
        d = np.random.random(len(lb)) * (ub - lb) + lb
        d[0] = -1
        d[1:4] = 1
        if pipes.constraint(d):
            print(d)
            pipes.plot(1)
        else:
            print('invalid')
        input()        
        
        
