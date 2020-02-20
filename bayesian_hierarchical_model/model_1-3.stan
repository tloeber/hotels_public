data {
  int<lower=0> N;
  int<lower=0> K;
  int<lower=0> n_J; //Number of grouping variables
  int<lower=0> J_msa;  //Number of groups in each grouping variable
  int<lower=0> J_tract;
  int<lower=0> J_category;
  int<lower=0> J_saleaffiliation;
  int<lower=0> J_year_factor;
  
  matrix [N, K] X;
  vector [N] log_saleprice;
  
  int<lower=0, upper=J_msa> msa [N]; 
  int<lower=0, upper=J_tract> tract [N]; 
  int<lower=0, upper=J_category> category [N]; 
  int<lower=0, upper=J_saleaffiliation> saleaffiliation [N]; 
  int<lower=0, upper=J_year_factor> year_factor [N]; 
  
  real mu_y; //Moments of target variable for prior for intercept 
  real<lower=0> sd_y;
}
parameters{
  real intercept;
  vector [K] b;
  
  vector[J_msa] a_msa;  // surprisingly, indexing seems to be 1-based
  vector[J_tract] a_tract;
  vector[J_category] a_category;
  vector[J_saleaffiliation] a_affiliation;
  vector[J_year_factor] a_year;
  
  real<lower=0,upper=sd_y> sigma_y;
  real<lower=0,upper=sd_y> sigma_msa;
  real<lower=0,upper=sd_y> sigma_tract;
  real<lower=0,upper=sd_y> sigma_category;
  real<lower=0,upper=sd_y> sigma_affiliation;
  real<lower=0,upper=sd_y> sigma_year;
}
transformed parameters {
  vector[N] y_hat;
  for (n in 1:N)
    y_hat[n] = intercept + a_msa[msa[n]] + a_tract[tract[n]] + 
      a_category[category[n]] + a_affiliation[saleaffiliation[n]]  +
      a_year[year_factor[n]] + X[n, ] * b;
}  
model {
  // random intercepts
  a_msa ~ normal(0, sigma_msa);
  a_tract ~ normal(0, sigma_tract);
  a_category ~ normal(0, sigma_category);
  a_affiliation ~ normal(0, sigma_affiliation);
  a_year ~ normal(0, sigma_year);
  
  // Priors
  sigma_y ~ cauchy(0.3, 1);
  b ~ normal(0, .5*sd_y);
  intercept ~ normal(mu_y, 0.1*sd_y);
  
  // Hyper-priors
  sigma_msa ~ cauchy(0, .3*sd_y);
  sigma_tract ~ cauchy(0, .3*sd_y);
  sigma_category ~ cauchy(0, .25*sd_y);
  sigma_affiliation ~ cauchy(0, .2*sd_y);
  sigma_year ~ cauchy(0, .25*sd_y);

  // likelihood
  log_saleprice ~ normal(y_hat, sigma_y);
}
generated quantities {
  vector[N] ll;   // log likelihood for LOOCV
  vector[N] residuals;  // save residuals
  for (n in 1: N)
    ll[n] = normal_lpdf(log_saleprice [n] | y_hat[n], sigma_y);
    residuals = log_saleprice - y_hat;
}
