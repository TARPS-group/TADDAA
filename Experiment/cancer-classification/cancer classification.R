install.packages("rstan", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))


library(rstan)

dat <- read.csv(file = 'leukemia.csv')
d <- NCOL(dat)
n <- NROW(dat)
y <- dat[, 1]
x <- scale(dat[,-1])

# compile the model
stanmodel <- stan_model('glm_bernoulli_rhs.stan')

scale_icept=10
slab_scale=5
slab_df=4

# data and prior
tau0 <- 1/(d-1) * 2/sqrt(n) # should be a reasonable scale for tau (see Piironen&Vehtari 2017, EJS paper)
scale_global=tau0
data <- list(n=n, d=d, x=x, y=as.vector(y), scale_icept=10, scale_global=tau0,
             slab_scale=5, slab_df=4)

# NUTS solution (increase the number of iterations, here only 100 iterations to make this script run relatively fast)
fit_nuts <- sampling(stanmodel, data=data, iter=3000, control=list(adapt_delta=0.9))
