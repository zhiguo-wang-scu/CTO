import numpy as np


class opt_weight1:
    def __init__(self, P_list,num_s,num_classifier,num_label,aa,Ep,mu=1):
        self.P = P_list
        self.num_c = num_classifier
        self.num_label = num_label
        self.num_s = num_s
        self.Ep = Ep
        self.th = 1/(1+3*np.exp((Ep-aa)*20))
        self.w = (1-self.th)/(num_classifier-1)*np.ones((num_classifier))
        self.w[-1] = self.th
        self.u =mu
        self.alpha = 0.5

    def project_simplex(self,v, z=1.0, axis=-1):
        """
        Implements the algorithm in Figure 1 of
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, Tushar Chandra,
        "Efficient Projections onto the l1-Ball for Learning in High Dimensions", ICML 2008.
        https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf

        This algorithm project vectors v onto the simplex w >= 0, \sum w_i = z.

        :param v: A numpy array, will be interpreted as a collection of vectors.
        :param z: Vectors will be projected onto the z-Simplex: \sum w_i = z.
        :param axis: Indicates the axis of v, which defines the vectors to be projected.
        :return: w: result of the projection
        """

        def _project_simplex_2d(v, z):
            """
            Helper function, assuming that all vectors are arranged in rows of v.

            :param v: NxD numpy array; Duchi et al. algorithm is applied to each row in vecotrized form
            :param z: Vectors will be projected onto the z-Simplex: \sum w_i = z.
            :return: w: result of the projection
            """
            shape = v.shape
            if shape[1] == 1:
                w = np.array(v)
                w[:] = z
                return w

            mu = np.sort(v, axis=1)
            mu = np.flip(mu, axis=1)
            cum_sum = np.cumsum(mu, axis=1)
            j = np.expand_dims(np.arange(1, shape[1] + 1), 0)
            rho = np.sum(mu * j - cum_sum + z > 0.0, axis=1, keepdims=True) - 1
            max_nn = cum_sum[np.arange(shape[0]), rho[:, 0]]
            theta = (np.expand_dims(max_nn, -1) - z) / (rho + 1)
            w = (v - theta).clip(min=0.0)
            return w

        shape = v.shape

        if len(shape) == 0:
            return np.array(1.0, dtype=v.dtype)
        elif len(shape) == 1:
            return _project_simplex_2d(np.expand_dims(v, 0), z)[0, :]
        else:
            axis = axis % len(shape)
            t_shape = tuple(range(axis)) + tuple(range(axis + 1, len(shape))) + (axis,)
            tt_shape = tuple(range(axis)) + (len(shape) - 1,) + tuple(range(axis, len(shape) - 1))
            v_t = np.transpose(v, t_shape)
            v_t_shape = v_t.shape
            v_t_unroll = np.reshape(v_t, (-1, v_t_shape[-1]))

            w_t = _project_simplex_2d(v_t_unroll, z)

            w_t_reroll = np.reshape(w_t, v_t_shape)
            return np.transpose(w_t_reroll, tt_shape)
    def grad_obj_function(self,P,w):
        obj = self.u*w
        obj0 = np.zeros((self.num_c))
        PP = np.zeros((self.num_s,self.num_label))
        for i in range(self.num_c):
            PP = PP+w[i]*P[i]
        for k in range(self.num_c):
            wa = np.zeros((self.num_s))
            for j in range(self.num_label):
                indx = np.nonzero(PP[:,j])
                wa[indx] = wa[indx] + P[k][indx,j]*np.log(PP[indx,j])+P[k][indx,j]
                #ff = filter(lambda x: x > 0, PP[:, j])
            obj0[k] = np.sum(wa)/self.num_s
        obj = obj - obj0


        return  obj
    def grad_dec(self,P,w,num=10):
        for i in range(num):
            w = w - self.alpha*self.grad_obj_function(P,w)
            w1 = self.project_simplex(w[:-1], z=(1-self.th))
            w[:-1] = w1
            w[-1] = self.th

        return  w
    def grad_dec1(self,P,w,num=10):
        for i in range(num):
            w = w - self.alpha*self.grad_obj_function(P,w)
            w = self.project_simplex(w,z=1)


        return  w
    def get_w(self,case='opt'):
        if case=='opt':
            self.w = self.grad_dec(self.P, self.w)
            return self.w
        if case=='ave':
            self.w = 1/self.num_c*np.ones((self.num_c))
            return self.w
        if case=='no_prior':
            self.w = self.grad_dec1(self.P, self.w)
            return self.w





