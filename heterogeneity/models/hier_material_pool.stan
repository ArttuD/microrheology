data {
  int N; // number of datapoints
  int N_train;
  int N_samples;
  int N_holders;
  int N_locations;
  int N_materials;
  int N_ids;
  int material_ids[N_samples];
  int sample_ids[N_holders];
  int holder_ids[N_locations];
  int location_ids[N_ids];
  int track_ids[N];
  real y[N];
  int train_ids[N_train];
}

parameters {

  vector[N_materials] mu;
  vector<lower=0>[N_materials] sigma;

  vector[N_ids] z;

  real mu_sigma;
  vector<lower=0>[N_materials] sigma_materials;
  vector<lower=0>[N_ids] sigma_common;
}

transformed parameters {
  vector[N_ids] mu_ids = mu[material_ids][sample_ids][holder_ids][location_ids] + sigma[material_ids][sample_ids][holder_ids][location_ids].*z;
}

model {
  mu ~ std_normal();
  sigma ~ std_normal();
  z ~ std_normal();

  mu_sigma ~ normal(0,1);
  sigma_materials ~ normal(0,1);
  sigma_common ~ inv_gamma(mu_sigma,sigma_materials[material_ids][sample_ids][holder_ids][location_ids]);
  y[train_ids] ~ normal(mu_ids[track_ids][train_ids],sigma_common[track_ids][train_ids]);
}

generated quantities {
 vector[N] log_likelihood;
 real y_hat[N];
 real y_hat2[N_ids];
 for (i in 1:N) {
    log_likelihood[i] = normal_lpdf(y[i] | mu_ids[track_ids[i]],sigma_common[track_ids[i]]);
 } 
 y_hat = normal_rng(mu_ids[track_ids],sigma_common[track_ids]);
 y_hat2 = normal_rng(mu_ids,sigma_common);
}