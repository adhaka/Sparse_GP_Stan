//data {
//    int<lower=1> N;
//    int<lower=1> M;
//    int<lower=1> N_star;
//    vector[N] x;
//    vector[N] y;
//    vector[N_star] x_star;    
//}

//transformed data {
//    real jitter = 1e-6;
//    real scale;
//    vector[N] f_mean = rep_vector(0, N);
//    vector[M] omega; 

//    scale = sqrt(2.0/M);

//    for (i in 1:M){
//        omega[i] = normal_rng(0,1);
//    }
//
//}

// spectral density for all the matern kernels.

functions{
    real matern_12_spectral_log(real omega, real lam, real sigma){

    }

}

parameters{
    real<lower=1e-6> rho;
    real<lower=1e-6> sigma;
    real omega;
}

transformed parameters{
    real lambda = 1./ rho;
    real lambda = sqrt(3.)/rho;
//    real lambda = sqrt(5.)/rho;
}

model {
    rho ~ normal(0, 1.0);
    sigma ~ normal(0, 1.0);
    // for matern 1/2 kernel
    target +=  log(2*square(sigma)*lambda/(square(lambda) + square(omega))) ;

    // for matern 3/2 kernel
    target += log(4*square(sigma)*lambda^(3)/square(square(lambda) + square(omega)));

    // for matern 5/2 kernel
    target += log(16*square(sigma)*lambda^(5) / (3*(square(lambda) + square(omega))^3));
}

