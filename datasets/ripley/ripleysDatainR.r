#install.packages("MASS")
library(MASS)

ripleys_train = synth.tr
ripleys_test = synth.te


write.csv(ripleys_train,'ripleys_train.csv')
write.csv(ripleys_test,'ripleys_test.csv')
