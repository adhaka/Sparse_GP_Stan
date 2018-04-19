data {
    int<lower=1> N;
    int<lower=1> M;
    vector [N]x;
    vector[N] y;
}

transformed data {
    real delta = 1e-6;
    vector[N] f_mean = rep_vector(0, N);
    
}

parameters{
    real <lower=1e-4> sigvar;
    real<lower=1e-6> sigma;
    
    vector[M] omega;

}

transformed parameters {
    vector[N] f;
    vector[M] b;
    
    matrix[N, N] K;
    
    matrix[N, M] features;
    matrix[N, M] cosfeatures;
    matrix[N, M] sinfeatures;
    matrix[N, 2*M] fullfeatures;
    matrix[N, M] b_mat ;
    
    {
        features = x * omega';

         for(i in 1:N){
             for(j in 1:M){
                 cosfeatures[i,j] = cos(features[i,j] + b[j]);
                 sinfeatures[i,j] = sin(features[i,j] + b[j]);
                 fullfeatures[i,j] = cosfeatures[i,j];
                 fullfeatures[i,M+j] = sinfeatures[i,j];
             }
         }

        K = cosfeatures*cosfeatures'*sqrt(2*square(sigvar)/M) ;
    }
}


model {

    sigvar ~ normal(0.85, 0.25);
    sigma ~ normal(0, 0.35);
    b ~ uniform(0, 2*pi());
    omega ~ normal(0, 1);
    
    f ~ multi_normal(f_mean, K);
    y ~ normal(f, sigma);
    
}
