import numpy as np
from scipy.signal import find_peaks
from sympy import mobius, totient

class PeriodEstimator:
    def __init__(self, fs, num_components, f_min = 0.5, f_max = 2.0, window_size = 4.0):
        self.fs = fs
        self.f_min = f_min
        self.f_max = f_max
        self.num_components = num_components
        self.window_size = window_size

    #Section 1: Cases when time series has 1 or 2 components
    
    def find_prominent_peaks(self, magnitude_spectrum, threshold, prominence):
        peaks, _ = find_peaks(magnitude_spectrum, height=threshold, prominence=prominence)
        sorted_peaks = sorted(peaks, key=lambda p: -magnitude_spectrum[p])
        cumulative_sum = np.cumsum(magnitude_spectrum[sorted_peaks])
        total_sum = np.sum(magnitude_spectrum[sorted_peaks])
        n_peaks = np.searchsorted(cumulative_sum, 0.7 * total_sum, side='right')
        return sorted_peaks[:n_peaks]

    def band_cutoff(self, frequencies):

        p_min = 1 / self.f_max
        p_max = 1 / self.f_min
        
        epsilon = 1e-10  
        frequencies = np.where(frequencies == 0, epsilon, frequencies)
        
        periods = np.abs(1/frequencies)
        band = (periods >= p_min) & (periods <= p_max)
        idx = np.where(band)
        return idx

    def UNIVARIATE(self, F):

        try:
            dft = np.fft.fft(F)
            magnitude_spectrum = np.abs(dft)
            frequencies = np.fft.fftfreq(len(F), 1/self.fs)

            pos_mask = frequencies > 0

            positive_frequencies = frequencies[pos_mask]
            positive_magnitude_spectrum = magnitude_spectrum[pos_mask]
            # Apply band cutoff
            band_mask = self.band_cutoff(positive_frequencies)   # should return boolean mask or indices

            band_limited_spectrum = np.zeros_like(positive_magnitude_spectrum)
            band_limited_spectrum[band_mask] = positive_magnitude_spectrum[band_mask]
            
            pk = self.find_prominent_peaks(band_limited_spectrum, 0, 0) 

            if pk is None or len(pk) == 0:
                pk, _ = find_peaks(band_limited_spectrum)
                pk = [pk[np.argmax(band_limited_spectrum[pk])]] if len(pk) > 0 else []
           
            peak_center = positive_frequencies[pk][0]
            period = np.abs((1/peak_center))

        except:
            period = np.nan
            
        return period

    def BIVARIATE(self, X):
        
        F = X[:,0] + 1j * X[:,1]
        
        try:
            dft = np.fft.fft(F)
            magnitude_spectrum = np.abs(dft)
            frequencies = np.fft.fftfreq(len(F), 1/self.fs)

            band_mask = self.band_cutoff(frequencies)   # should return boolean mask or indices
            
            band_limited_spectrum = np.zeros_like(magnitude_spectrum)
            band_limited_spectrum[band_mask] = magnitude_spectrum[band_mask]
            
            pk = self.find_prominent_peaks(band_limited_spectrum, 0, 0) 
            if pk is None or len(pk) == 0:
                pk, _ = find_peaks(band_limited_spectrum)
                pk = [pk[np.argmax(band_limited_spectrum[pk])]] if len(pk) > 0 else []
                
            peak_center = frequencies[pk][0]
            period = np.abs((1/peak_center))

        except:
            period = np.nan
            
        return period
    

    #Section 2: Cases when time series has >= 3 components

    def _soft_threshold(self, X, tau):
        return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)
        
    def _pos_part(self, M):
        return np.maximum(M, 0.0)
        
    def _neg_part(self, M):
        return np.maximum(-M, 0.0)

    def _relative_change(self, a, b):
        from numpy.linalg import norm
        n = norm(a - b)
        d = max(1e-12, norm(b))
        return n / d

    # Ramanujan Dictionary Construction

    def _Pg(self, g):
        ks = np.arange(g, dtype=int)
        q1 = g // np.gcd(ks, g)
        mu = np.array([mobius(int(q)) for q in q1], dtype=float)
        phi_g = float(totient(g))
        phi_q1 = np.array([float(totient(int(q))) for q in q1], dtype=float)
        c = mu * (phi_g / phi_q1)                          
        C = np.vstack([np.roll(c, i) for i in range(g)])     
        return C[:, :int(totient(g))]                        

    def _extend(self, atom_g, T):
        reps = int(np.ceil(T / atom_g.shape[0]))
        return np.tile(atom_g, reps)[:T]

    def _build_dictionary(self, T, periods):
        cols, colp = [], []
        for p in periods:
            Pg = self._Pg(p)
            for j in range(Pg.shape[1]):
                cols.append(self._extend(Pg[:, j], T)[:, None])
                colp.append(p)
        R = np.hstack(cols) if cols else np.zeros((T, 0))
        Hinv = np.diag([1.0 / (p ** 2) for p in colp])                 
        D = R @ Hinv
    
        # Aggregation A (rows=periods, cols=atoms)
        idx = {p: i for i, p in enumerate(periods)}
        A = np.zeros((len(periods), D.shape[1]))
        for j, p in enumerate(colp):
            A[idx[p], j] = 1.0
        return D, A, np.array(colp, dtype=int)


    def _lapis(self, 
            Y,
            Kmax_periods=8, 
            lam1=5e-3, 
            lam2=5e-2, 
            lam3=5e-2, 
            rho1 = 1.0,
            rho2 = 1.0,
            mu0 = 1.0,
            mu_growth = 1.5,
            max_outer=10, 
            max_U=20, 
            max_X=20, 
            tol_outer = 1e-4,
            tol_inner = 1e-5):
        
        from scipy.linalg import cho_factor, cho_solve, svd
        from scipy.signal import resample_poly

        Y  = resample_poly(Y, up=2, down=19, axis=0)  #Resampling to ~10Hz 
        fs_down = self.fs * (2/19)

        Y = np.asarray(Y, float)
        T, N = Y.shape
        W = (~np.isnan(Y)).astype(float)
        Yf = np.nan_to_num(Y, copy=True)

        pmin = max(2, int(np.ceil(fs_down / float(self.f_max))))
        pmax = int(np.floor(fs_down / float(self.f_min)))
        periods = list(range(pmin, pmax + 1))

        # Dictionary
        D, A, colp = self._build_dictionary(T, periods)
        L = D.shape[1]
        P = len(periods)
        DtD = D.T @ D
    
        # Init
        X = Yf.copy()
        U = np.zeros((L, N))
        s = np.ones(L)
    
        # ADMM aux
        P1 = np.zeros_like(U); Theta1 = np.zeros_like(U)
        P2 = np.zeros((P, N)); Theta2 = np.zeros_like(P2)
        E = np.zeros_like(Yf); M = np.zeros_like(Yf); mu = mu0

        def make_M_lhs(svec, rho):
            return (svec[:, None] * DtD) * svec[None, :] + rho * np.eye(L)

        for _ in range(max_outer):
            U_prev, X_prev, s_prev = U.copy(), X.copy(), s.copy()

            # ---- U-step (ADMM) ----
            for _ in range(max_U):
                
                M_lhs = make_M_lhs(s, rho1)
                c, lower = cho_factor(M_lhs, check_finite=False, overwrite_a=False)
                DtX = D.T @ X
                RHS = (s[:, None] * DtX) + rho1 * (P1 - Theta1 / rho1)

                 # === PATCH to couple nuclear-norm term into U-step ===
                Z = P2 - Theta2 / rho2
                G = A.T @ (A @ np.abs(U_prev) - Z)   # L×N
                S = np.sign(U_prev)
                RHS += - rho2 * (S * G)
                
                U = cho_solve((c, lower), RHS, check_finite=False, overwrite_b=False)
    
                # Proxes
                F1 = U - Theta1 / rho1
                P1 = self._soft_threshold(F1, lam1 / rho1)
    
                Uabs = np.abs(U)
                F2 = (A @ Uabs) - (Theta2 / rho2)
                Uu, svals, Vt = svd(F2, full_matrices=False, check_finite=False)
                s_thr = np.maximum(svals - lam2 / rho2, 0.0)
                P2 = (Uu * s_thr) @ Vt
    
                Theta1 += rho1 * (P1 - U)
                Theta2 += rho2 * (P2 - (A @ Uabs))
    
                if self._relative_change(U, U_prev) < tol_inner:
                    break
                U_prev = U.copy()

            # ---- X-step (ADMM) ----
            for _ in range(max_X):
                DSU = D @ (s[:, None] * U)                     
                H = E + Yf + (M / mu)
                X = (DSU + mu * H) / (1.0 + mu)                  # Eq. (16)
    
                Tmat = (X - Yf) - (M / mu)
                E = self._soft_threshold(Tmat, (lam3 * W) / mu)        # Eq. (17)
    
                M += mu * (E - (X - Yf))                        
                mu *= mu_growth
    
                if self._relative_change(X, X_prev) < tol_inner:
                    break
                X_prev = X.copy()

            # ---- s-step (diag S) multiplicative update (Eq. 22) ----
            Hm = U @ U.T
            Gm = DtD
            Lm = D.T @ X @ U.T
            
            Hp, Hn = self._pos_part(Hm), self._neg_part(Hm)
            Gp, Gn = self._pos_part(Gm), self._neg_part(Gm)
            Lp, Ln = self._pos_part(Lm), self._neg_part(Lm)
            
            SGp = (s[:, None] * Gp)
            SGn = (s[:, None] * Gn)
            num = np.diag(Hn @ SGp) + np.diag(Hp @ SGn) + np.diag(Lp)
            den = np.diag(Hp @ SGp) + np.diag(Hn @ SGn) + np.diag(Ln)
            den = np.maximum(den, 1e-12)
            s *= np.sqrt(np.maximum(num, 0.0) / den)
            s = np.maximum(s, 0.0)
            
            if max(self._relative_change(U, U_prev), self._relative_change(X, X_prev), self._relative_change(s, s_prev)) < tol_outer:
                break

        AabsU = A @ np.abs(U)
        period_scores = np.linalg.norm(AabsU, axis=1)
        order = np.argsort(-period_scores)
        periods_sorted = np.array([periods[i] for i in order], dtype=int)
        scores_sorted = period_scores[order]
        if Kmax_periods is not None:
            periods_sorted = periods_sorted[:Kmax_periods]
            scores_sorted = scores_sorted[:Kmax_periods]

        fundamental_period = periods_sorted[0] / fs_down
            
        return fundamental_period

    #Section 3: Wrapper that automatically decide which to use based on num_components
        
    def estimate_period(self, X):

        if self.num_components == 1:
            period = self.UNIVARIATE(X)
        elif self.num_components == 2:
            period = self.BIVARIATE(X)
        else:
            period = self._lapis(X)
        return period
