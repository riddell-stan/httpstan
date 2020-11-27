"""Simple script to save draws to disk."""
# assumes httpstan is running on port 8080

import time
import requests

schools_code = """
data {
  int<lower=0> J;         // number of schools
  real y[J];              // estimated treatment effects
  real<lower=0> sigma[J]; // standard error of effect estimates
}
parameters {
  real mu;                // population treatment effect
  real<lower=0> tau;      // standard deviation in treatment effects
  vector[J] eta;          // unscaled deviation from mu by school
}
transformed parameters {
  vector[J] theta = mu + tau * eta;        // school treatment effects
}
model {
  target += normal_lpdf(eta | 0, 1);       // prior log-density
  target += normal_lpdf(y | theta, sigma); // log-likelihood
}
"""

schools_data = {
    "J": 8 * 100,
    "y": [28, 8, -3, 7, -1, 1, 18, 12] * 100,
    "sigma": [15, 10, 16, 11, 9, 11, 10, 18] * 100,
}


model_name = requests.post("http://localhost:8080/v1/models", json={"program_code": schools_code}).json()["name"]

fit_payload = {
    "function": "stan::services::sample::hmc_nuts_diag_e_adapt",
    "data": schools_data,
    "num_warmup": 1000,
    "num_samples": 1000,
}
fit_response = requests.post(f"http://localhost:8080/v1/{model_name}/fits", json=fit_payload)
operation_name = fit_response.json()["name"]
fit_name = fit_response.json()["metadata"]["fit"]["name"]

while True:
    if requests.get(f"http://localhost:8080/v1/{operation_name}").json()["done"]:
        break
    time.sleep(0.01)

with open("/tmp/schools.bin", "wb") as fh:
    fh.write(requests.get(f"http://localhost:8080/v1/{fit_name}").content)
