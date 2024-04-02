import numpy as np
import numpy.linalg as la 

from helpers import tilde

class MRP:
    
    def __init__(self, s1, s2, s3):
        self.vector = np.array([s1,s2,s3])
        self._dcm = None
        self._b_matrix = None
        self.norm = la.norm(self.vector)
        
    @property
    def dcm(self):
        if self._dcm is None:
            s_tilde = tilde(self.vector)
            norm_2 = self.norm**2
            self._dcm = np.identity(3) + (8*s_tilde @ s_tilde - 4*(1-norm_2)*s_tilde) / (1+norm_2)**2
        return self._dcm
    

    @property
    def b_matrix(self):
        if self._b_matrix is None:
            norm_2 = self.norm**2
            s1,s2,s3 = self.vector
            self._b_matrix = np.array([
                [1-norm_2+2*s1**2, 2*(s1*s2-s3), 2*(s1*s3+s2)],
                [2*(s2*s1+s3), 1-norm_2+2*s2**2, 2*(s2*s3-s1)],
                [2*(s3*s1-s2), 2*(s3*s2+s1), 1-norm_2+s3**2]
            ])
        return self._b_matrix
    
    def as_short_rotation(self):
        if self.norm >= 1:
            self.vector = -self.vector/self.norm**2
    
    def add(self, mrp):
        
        mrp1_2 = mrp.norm**2
        mrp2_2 = self.norm**2
        
        numerator = (1-mrp1_2)*self.vector + (1-mrp2_2)*mrp.vector - 2*np.cross(self.vector, mrp.vector)
        denom = 1 + mrp1_2*mrp2_2 - 2*np.dot(self.vector, mrp.vector)
        return MRP(*(numerator/denom))
    
    def subtract(self, mrp):
        
        mrp2_2 = mrp.norm**2
        mrp1_2 = self.norm**2
        
        numerator = (1-mrp1_2)*self.vector - (1-mrp2_2)*mrp.vector + 2*np.cross(self.vector, mrp.vector)
        denom = 1 + mrp1_2*mrp2_2 + 2*np.dot(self.vector, mrp.vector)
        return MRP(*(numerator/denom))
    
    @classmethod
    def from_dcm(cls, dcm):
        if np.isclose(np.trace(dcm), -1.0):
            raise ValueError("Can not compute modified Rodrigues parametters from a " \
                             "DCM that represent a 180º principal rotation.")
        trace_term = (np.trace(dcm) + 1)**0.5
        denom = (trace_term*(trace_term+2))
        num = np.array([
            dcm[1][2] - dcm[2][1],
            dcm[2][0] - dcm[0][2],
            dcm[0][1] - dcm[1][0]
        ])
        return cls(*(num/denom))




    def __repr__(self):
        return f"MRP<{self.vector}>"