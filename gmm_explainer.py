import numpy as np
from scipy.linalg import inv
from scipy.optimize import root

class CFE_explainer:
	def __init__(self, m_source, m_target, S_sourse, S_target, prior_s, prior_t, covariance_type='full'):
		"""
		Initialization method.

		Parameters:
		- m_source (np.ndarray): The center of the source cluster.
		- m_target (np.ndarray): The center of the target cluster.
		- S_sourse (np.ndarray): The covariance matrix of the source cluster.
		- S_target (np.ndarray): The covariance matrix of the target cluster.
		- prior_s: Prior for source cluster.
		- prior_t: Prior for target cluster.
		- covariance_type (String): The type of the covariance matrix. Options 'full', 'tied', 'diag', 'spherical'
		"""

		self.m_source = np.asarray(m_source, dtype=np.float64)
		self.m_target = np.asarray(m_target, dtype=np.float64)
		self.S_sourse = np.asarray(S_sourse, dtype=np.float64)
		self.S_target = np.asarray(S_target, dtype=np.float64)
		self.prior_s = prior_s
		self.prior_t = prior_t
		self.I = np.eye(S_sourse.shape[0])

		if not (m_source.shape == m_target.shape):
			raise ValueError("m_source and m_target must have the same shape.")

		if not (S_sourse.shape == S_target.shape):
			raise ValueError("S_sourse and S_target must have the same shape.")

		if covariance_type == 'full':
			self.S_sourse_inv = np.linalg.inv(S_sourse)
			self.S_target_inv = np.linalg.inv(S_target)
			self.S_sourse_det = np.linalg.det(S_sourse)
			self.S_target_det = np.linalg.det(S_target)


	def compute_counterfactual(self, y, M, epsilon):
		"""
		Compute actionable counterfactuals for Gaussian clusters with full covariance,
		considering cluster priors and plausibility factor epsilon.

		Args:
		- y: Factual vector.
		- M: Mask vector (1 for free indices, 0 for fixed indices).
		- epsilon: Plausibility factor (>= 0).

		Returns:
		- z: Counterfactual vector.
		"""
		# Partition indices
		F = np.where(M == 1)[0]  # Free indices
		G = np.where(M == 0)[0]  # Fixed indices

		# Partition mean and covariance
		m_s_F, m_s_G = self.m_source[F], self.m_source[G]
		m_t_F, m_t_G = self.m_target[F], self.m_target[G]

		y_F, y_G = y[F], y[G]

		S_s_inv = inv(self.S_sourse)
		S_t_inv = inv(self.S_target)

		# Partition inverse covariance matrices
		S_s_FF = S_s_inv[np.ix_(F, F)]
		S_s_FG = S_s_inv[np.ix_(F, G)]
		S_s_GF = S_s_inv[np.ix_(G, F)]
		S_s_GG = S_s_inv[np.ix_(G, G)]

		S_t_FF = S_t_inv[np.ix_(F, F)]
		S_t_FG = S_t_inv[np.ix_(F, G)]
		S_t_GF = S_t_inv[np.ix_(G, F)]
		S_t_GG = S_t_inv[np.ix_(G, G)]

		# Compute constants
		c_alpha = (np.log(np.linalg.det(self.S_target) / np.linalg.det(self.S_sourse))
				- 2 * np.log(self.prior_t / self.prior_s)
				+ 2 * np.log(1 + epsilon))

		# Compute D and d
		D = S_t_FF - S_s_FF
		d = (S_t_FF @ m_t_F - S_s_FF @ m_s_F -
			(S_t_FG @ (y_G - m_t_G) - S_s_FG @ (y_G - m_s_G)))

		e = S_t_FF @ m_t_F - S_s_FF @ m_s_F
		f = (m_t_F.T @ S_t_FF @ m_t_F) - (m_s_F.T @ S_s_FF @ m_s_F)

		C = ((y_G - m_t_G).T @ S_t_GG @ (y_G - m_t_G) -
			(y_G - m_s_G).T @ S_s_GG @ (y_G - m_s_G))

		# Solve for lambda using numerical root-finding
		lambda_star = root(self.__equation_to_solve, args=(F, D, y_F, d, e, f, C, c_alpha), x0=0.0).x[0]

		# Compute z_F
		B = np.eye(len(F)) - lambda_star * D
		z_F = inv(B) @ (y_F - lambda_star * d)

		# Combine free and fixed variables to get the full counterfactual
		z = np.copy(y)
		z[F] = z_F
		z[G] = y_G

		return z
	
	def __equation_to_solve(self, lambda_, F, D, y_F, d, e, f, C, c_alpha):
		B = np.eye(len(F)) - lambda_ * D
		B_inv = inv(B)
		c = y_F - lambda_ * d
		
		term1 = c.T @ B_inv.T @ D @ B_inv @ c
		term2 = -2 * c.T @ B_inv.T @ e
		
		return term1 + term2 + f + C + c_alpha
