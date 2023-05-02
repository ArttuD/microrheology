data {
  int<lower=0> N;
  int<lower=0> N_pred;
  vector[N] y;
  vector[N+N_pred] x;
  int train[N];
  int test[N_pred];
}
parameters {
  real beta;
  real<lower=0> sigma;
}

transformed parameters {
   real beta_scaled = beta*1e3;
   real sigma_scaled = sigma*1e-8;
}

model {
  beta ~ normal(0,5);
  sigma ~ inv_gamma(0.5, 1);
  y ~ normal(beta_scaled * x[train],
                      sigma_scaled);
}

generated quantities {
  real r2;
  real y_pred[N];
  real y_hat[N_pred];
  vector[N] log_lik;
  for (i in 1:N) {
    log_lik[i] = normal_lpdf(y[i] | beta_scaled*x[i],sigma_scaled);
  }
  y_pred = normal_rng(beta_scaled*x[train],sigma_scaled);
  y_hat = normal_rng(beta_scaled*x[test],sigma_scaled);
  r2 = variance(beta_scaled*x[train])/(variance(beta_scaled*x[train])+square(sigma_scaled));
  //r2 = mean((to_vector(y_pred)-mean(y_pred))^2)/(mean((to_vector(y_pred)-mean(y_pred))^2)+mean(((y-to_vector(y_pred))-mean(y-to_vector(y_pred)))^2));
}