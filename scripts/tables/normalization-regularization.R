library(glmnet)

n <- 100000

x1 <- rnorm(n, 0, 2)
x2 <- rnorm(n, 0, 1)

x <- cbind(x1, x2)

s <- apply(x, 2, sd)

beta <- c(1 / 2, 1)

y <- x %*% beta + rnorm(n)

lasso_fit <- glmnet(x, y, standardize = FALSE, lambda = 0.5)

coef(lasso_fit)
coef(lasso_fit)[2:3] * s

ridge_fit <- glmnet(x, y, standardize = FALSE, lambda = 0.5, alpha = 0)

coef(ridge_fit)
coef(ridge_fit)[2:3] * s
