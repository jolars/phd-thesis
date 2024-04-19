library(lars)

data(diabetes)

fit <- lars(diabetes$x, diabetes$y, type = "lasso")
