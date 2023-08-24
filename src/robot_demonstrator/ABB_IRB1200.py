# Imports
from math import *
from scipy.optimize import minimize
from robot_demonstrator.transformations import *
from robot_demonstrator.interpolator import *
from robot_demonstrator.ABB_IRB1200_Driver import *
from robot_demonstrator.plot import *


class ABB_IRB1200:
    
    def __init__(self, ip=None):
        
        # Attributes for kinematics
        self.dofs = 0
        self.links = []
        self.base_frame = np.identity(4)
        self.tool_frame = np.identity(4)
        
        # Attributes for limits
        self.q_max = []
        self.q_min = []
        self.dq_max = []
        self.ddq_max = []

        # Create robot chain (millimeters, radians)
        self.add_revolute_joint(a=0.0, alpha=-1.5707963, d=399.1, offset=-pi/2, qlim=(radians(-170), radians(170)), dqlim=radians(288))
        self.add_revolute_joint(a=350, alpha=0.0, d=0.0, qlim=(radians(-100), radians(135)), dqlim=radians(240))
        self.add_revolute_joint(a=42, alpha=-1.5707963, d=0.0, qlim=(radians(-200), radians(70)), dqlim=radians(300))
        self.add_revolute_joint(a=0.0, alpha=1.5707963, d=351, qlim=(radians(-270), radians(270)), dqlim=radians(400))
        self.add_revolute_joint(a=0.0, alpha=-1.5707963, d=0.0, qlim=(radians(-130), radians(130)), dqlim=radians(405))
        self.add_revolute_joint(a=0.0, alpha=0.0, d=82, qlim=(radians(-360), radians(360)), dqlim=radians(600))

        # Set tool frame
        #self.set_tool_frame(t_from_xyz_r(-20*math.sqrt(2), 20*math.sqrt(2), 170, np.eye(3)))

        # Attribute for tcp-ip connection
        if not ip is None: self.con = ABB_IRB1200_Driver(ip, port_motion=5000)
    
    def __iter__(self):
        return (each for each in self.links)
     
    def set_base_frame(self, base_frame):
        self.base_frame = np.array(base_frame)
    
    def set_tool_frame(self, tool_frame):
        self.tool_frame = np.array(tool_frame)
    
    def add_revolute_joint(self, a, alpha, d, kind='', offset=0.0, qlim=(-2 * pi, 2 * pi), dqlim=0, ddqlim=0):
        self.links.append(Link(a=a, alpha=alpha, d=d, kind='', offset=offset, qlim=qlim, dqlim=dqlim, ddqlim=ddqlim))
        self.dofs += 1
        self.q_min.append(qlim[0])
        self.q_max.append(qlim[1])
        self.dq_max.append(dqlim)
        self.dq_max.append(ddqlim)
    
    def fkine(self, q, segment_nr=-1):
        assert type(q) == np.ndarray and len(q) == self.dofs
        if segment_nr == -1:
            t = self.base_frame
            for i in range(self.dofs):
                t = np.dot(t, self.links[i].A(q[i]))
            t = np.dot(t, self.tool_frame)
        else:
            t = np.identity(4)
            for i in range(segment_nr - 1):
                t = np.dot(t, self.links[i].A(q[i]))
        return np.array(t)
    
    def fkine_multiple(self, q):
        assert type(q) == np.ndarray and len(q) == self.dofs
        t = self.fkine(q, 2)
        t_base_to_ji = [t]
        for i in range(1, self.dofs):
            t = self.fkine(q, i + 2)
            t_base_to_ji.append(t)
        t_base_to_ji.append(self.fkine(q))
        t_jmin1_to_ji = [t_base_to_ji[0]]
        for i in range(1, self.dofs):
            t_jmin1_to_ji.append(np.dot(np.linalg.inv(t_base_to_ji[i - 1]), t_base_to_ji[i]))
        t_world_to_ji = []
        for i in range(0, self.dofs):
            t_world_to_ji.append(np.dot(self.base_frame, t_base_to_ji[i]))
        return np.array(t_jmin1_to_ji), np.array(t_base_to_ji), np.array(t_world_to_ji)
    

    def ikine(self, t, q0=None, flip='up'): 
        
        # Assertions
        assert type(t) is np.ndarray and t.shape == (4, 4)
        if type(q0) == np.ndarray: assert type(q0) == np.ndarray and len(q0) == self.dofs

        # Define goal frame
        t = np.array(t, dtype='float')

        # Set bounds
        bounds = [(self.q_min[i], self.q_max[i]) for i in range(self.dofs)]
        #if flip == 'up': bounds[1][1] = radians(110) # Get elbow up solution

        # Set reach
        reach = 0
        for link in self.links: reach += abs(link.a) + abs(link.d)
        omega = np.diag([1, 1, 1, 3 / reach])

        # Set initial values
        if q0 is None: q0 = np.asmatrix(np.zeros((1, self.dofs)))

        # Define objective
        def objective(x):
            best_solution = np.linalg.lstsq(t, self.fkine(x), rcond=None)[0] # Linear least-squares
            return (np.square((best_solution - np.asmatrix(np.eye(4, 4))) * omega)).sum()

        # Minimize
        sol = minimize(objective, x0=q0)

        # Process solution
        if len(sol.x) == self.dofs:

            # Normalize angles in interval [-pi, pi]
            q_out = normalize_angles(sol.x)
            
            # Output q_out as array
            q_out = [q for q in q_out]
            
            # Return rounded values
            return np.round(q_out, 5)

        else:
            raise Exception("No solution found")

    
    def check_limits(self, time, q):
    
        # Assertions
        assert type(q) == np.ndarray and len(q) == self.dofs
        
        # Init trajectory
        dq = np.zeros(np.shape(q))
        ddq = np.zeros(np.shape(q))
        
        # Init joint limits
        q_limits_exceeded = False
        dq_limits_exceeded = False
        ddq_limits_exceeded = False

        # Populate joint values
        for i in range(np.shape(q)[0]):
            for j in range(np.shape(q)[1] - 1):
                dq[i, j] = (q[i, j + 1] - q[i, j]) / (time[j + 1] - time[j])
                ddq[i, j] = (dq[i, j + 1] - dq[i, j]) / (time[j + 1] - time[j])

        # Check limit violation
        for i in range(np.shape(q)[0]):

            # Get joint values
            q_max = np.max(q[i, :])
            q_min = np.min(q[i, :])
            dq_max = np.max(np.abs(dq[i, :]))
            ddq_max = np.max(np.abs(dq[i, :]))

            # Check limit violation
            if self.q_max[i] < q_max:
                q_limits_exceeded = True
            if self.q_min[i] > q_min:
                q_limits_exceeded = True
            if self.dq_max[i] < dq_max:
                dq_limits_exceeded = True
            if self.ddq_max[i] < ddq_max:
                ddq_limits_exceeded = True
                
        return {'q_lim_violation': q_limits_exceeded, 'dq_lim_violation': dq_limits_exceeded, 'ddq_lim_violation': ddq_limits_exceeded}


    def plot(self, ax, q):

        # Assertions
        assert type(q) == np.ndarray and len(q) == self.dofs
        
        # Plot world frame
        plot_frame_t(t_from_xyz_r(0, 0, 0, r_from_rpy(0, 0, 0)), ax, 'w')
        
        # Plot base frame
        plot_frame_t(self.base_frame, ax, '0')
        
        # Plot joint frames and links
        t_jmin1_to_ji, t_base_to_ji, t_world_to_ji = self.fkine_multiple(q)
        plot_frame_t(t_world_to_ji[0], ax, 'j1')
        plot_transf_p(self.base_frame, t_base_to_ji[0], ax)
        for i in range(1, self.dofs):
            plot_frame_t(t_world_to_ji[i], ax, 'j' + str(i+1))
            plot_transf_p(t_world_to_ji[i - 1], t_jmin1_to_ji[i], ax)
        plot_frame_t(t_world_to_ji[-1], ax, 'ee')

        # Set axes
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_zlim3d(0, 1000)
        set_axes_equal(ax)

class Link:
    
    def __init__(self, a, alpha, d, kind='', offset=0.0, qlim=None, dqlim=None, ddqlim=None):
        self.a = a
        self.alpha = alpha
        self.d = d
        self.kind = kind
        self.offset = offset
        self.qlim = qlim
        self.dqlim = dqlim
        self.ddqlim = ddqlim
    
    def A(self, q):
        frame = t_from_dh(self.a, self.alpha, self.d, q)
        frame = np.dot(frame, t_from_xyz_r(0, 0, 0, r_from_rpy(0, 0, self.offset)))
        return frame
    
    def to_string(self):
        '[a: ' + str(self.a) + ', ' + 'alpha: ' + str(self.alpha) + ', ' + 'd: ' + str(self.d) + ']'
