
stan_code="""
functions{
    matrix cov_cond_rng(vector u, real[] z, real[] x, real sigvar, real lengthscale, real sigma, real jitter){
        int M = size(z);
        int N = size(x);
        vector[N] f2;
        matrix[N, N] cov_f2;
        {
            matrix[N, N] L_K;
            matrix[N, N] Lu;
            vector[N] K_div_u;
            matrix[N, M] Kfu;
            matrix[N, M] v_pred;

            matrix[M, M] Kuu;
            matrix[N, N] Kff;
            matrix[N, N] Qff;
            
            matrix[N, N] diag_delta;
            
            vector[N] f_u_mu;
            Kff = cov_exp_quad(x, sigvar, lengthscale);
            
        #    K = cov_exp_quad_ARD(x1, alpha, rho);
        
            for (i in 1:N){
                Kff[i,i] = Kff[i,i] + square(sigma);            
            }
            
            Kuu = cov_exp_quad(z, sigvar, lengthscale);
            Lu = cholesky_decompose(Kuu);

            # solving a triangular system here: Ax = b where A is a symmetric positive definite matrix.
            # LL'x = y1
            # Lc = y1
            # c = y1/L
            # c = L'x

            K_div_u = mdivide_left_tri_low(Lu, u);
            K_div_u = mdivide_right_tri_low(K_div_u', Lu)';
            Kfu = cov_exp_quad(x, z, sigvar, lengthscale);
      #      K_x1_x2 = cov_exp_quad_x1_x2_ARD(x1, x2, alpha, rho);
            f_u_mu = (Kfu * K_div_u);
            v_pred = mdivide_left_tri_low(Lu, Kfu);
            Qff = v_pred'*v_pred;
            cov_f2 = Kff - Qff;
            diag_delta = diag_matrix(rep_vector(jitter, N));  
            
            # sample f from multivariate normal with mean=f2mu and covariance=cov_f2
            f2 = multi_normal_rng(f_u_mu, cov_f2 + diag_delta);
        }
        return f2;
    }
    
    vector mean_cond_mat(vector u, real[] z, real[] x, real sigvar, real lengthscale, real sigma, real jitter){
        int M = size(z);
        int N = size(x);
        vector[N] f2;
        vector[N] f_u_mu;

        
        {
            matrix[N, N] L_K;
            matrix[N, N] Lu;
            vector[N] K_div_u;
            matrix[N, M] Kfu;
            matrix[N, M] v_pred;

            matrix[M, M] Kuu;
            matrix[N, N] Kff;
            matrix[N, N] Qff;
            matrix[N, N] cov_f2;
            matrix[N, N] diag_delta;
            
            Kff = cov_exp_quad(x, sigvar, lengthscale);
            
        #    K = cov_exp_quad_ARD(x1, alpha, rho);
        
            for (i in 1:N){
                Kff[i,i] = Kff[i,i] + square(sigma);            
            }
            
            Kuu = cov_exp_quad(z, sigvar, lengthscale);
            Lu = cholesky_decompose(Kuu);

            # solving a triangular system here: Ax = b where A is a symmetric positive definite matrix.
            # LL'x = y1
            # Lc = y1
            # c = y1/L
            # c = L'x

            K_div_u = mdivide_left_tri_low(Lu, u);
            K_div_u = mdivide_right_tri_low(K_div_u', Lu)';
            Kfu = cov_exp_quad(x, z, sigvar, lengthscale);
      #      K_x1_x2 = cov_exp_quad_x1_x2_ARD(x1, x2, alpha, rho);
            f_u_mu = (Kfu * K_div_u);
            v_pred = mdivide_left_tri_low(Lu, Kfu);
            Qff = v_pred'*v_pred;
            cov_f2 = Kff - Qff;
            diag_delta = diag_matrix(rep_vector(jitter, N));  
            
            # sample f from multivariate normal with mean=f2mu and covariance=cov_f2
            #             f2 = multi_normal_rng(f_u_mu, cov_f2 + diag_delta);
        }
        return f_u_mu;
    }
}

data{
    int<lower=1> N1;
    real x1[N1];
    vector[N1] y1;
    int<lower=1> N2;
    int<lower=1> M;
    real x2[N2];
    real z[M];
}

transformed data{
    int<lower=1>N = N1+N2;
    real x[N];
    vector[N1] f1_mu = rep_vector(0, N1);
    vector[M] u_mu= rep_vector(0, M);
    real jitter = 1e-7;
}

parameters{
    real<lower=0> lengthscale;
    real<lower=0> sigvar;
    real sigma;
    vector[N1] eta;
}

transformed parameters{
#     real<lower=0> rho1 = rho[0];
    vector[N1] f1;
    vector[M] u;
    matrix[N, N] Lu;
    matrix[N, N] Kuu;
    matrix[N, N] cov_cond;
    vector[N] mean_cond;

    {
        real utemp[M];
        real ftemp[N1];
        Kuu= cov_exp_quad(z, sigvar, lengthscale);
        for (i in 1:M){
            Kuu[i,i] = Kuu[i,i] + jitter;
        }
        Lu = cholesky_decompose(Kuu);
        u = u_mu + Lu * eta;
        
        cov_cond = cov_cond_mat(u, z, x, sigvar, lengthscale, sigma, jitter);
        mean_cond = mean_cond_mat(u, z, x, sigvar, lengthscale, sigma, jitter);
        
        f1 = cov_cond_rng(mean_cond, cov_cond);
        matrix Kff[N1,N1] = cov_exp_quad(x1, sigvar, lengthscale, jitter);
        matrix Kfu[N1,M] = cov_exp_quad(x1, z, sigvar, lengthscale);
    }
}

model{
    rho ~ inv_gamma(5,5);
    alpha ~ normal(0,1);
    sigma ~ normal(0,1);
    eta ~ normal(0, 1);
    u ~ normal(0, Kuu);
    y1 ~ normal(f1, sigma);

}

generated quantities {
  real[N1] f1;
  real[N2] f2;

  f1 = cov_cond_rng(mean_cond, cov_cond);
#   f2 = gp_pred_rng(x2, y1, x1, alpha, rho, sigma, jitter);
  for (n2 in 1:N2)
    y2[n2] = normal_rng(f2[n2], sigma);

}

"""