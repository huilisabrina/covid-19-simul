#!/usr/bin/R

#-------------------------------------------------------
# CS 205 - Final Project
# Estimate R value

# Hui Li 4/19/2020
#-------------------------------------------------------

rm(list=ls())
gc()

library(EpiEstim)
(currentTime <- Sys.time())
(globalTime <- format(currentTime,c('%Y_%m_%d')))
local.path <- "C:/Users/LiHui/OneDrive - Harvard University/Grad_School/Courses/CS 205/project/Estimate_R/"
local.path <- paste0(local.path, "covidtracking_pull_", globalTime)

# Load in national level data
us_data <- read.table(paste0(cleaned.path, "covidtracking_US.csv"), sep=",", header=TRUE)

# Extract date information
year <- substr(us_data$date,1,4)
month <- substr(us_data$date,6,7)
day <- substr(us_data$date,9,10)
dates <- paste0(year,"-",month,"-",day)
dates <- as.Date(dates)

# Extract case counts
positive <- us_data[,c("positiveIncrease")]

# Collect data for estimation (filter down to non-NA obs)
covid_positive <- data.frame(dates=dates,I=positive)
covid_positive <- covid_positive[dim(covid_positive)[1]:1,]
covid_positive <- covid_positive[!is.na(covid_positive$I),]
covid_positive <- covid_positive[covid_positive$I>-1,]

# Make sure order is chronological
covid_positive <- covid_positive[order(covid_positive$date), ]

## Weekly sliding window
sliding_length <- 7
t_start <- seq(2, nrow(covid_positive)-(sliding_length-1))
t_end <- t_start + (sliding_length-1)

######################################
# Li et.al (NEJM 2020)
######################################
res1 <- estimate_R(covid_positive, method = "parametric_si",
                   config = make_config(list(
                     mean_si = 7.5, std_si = 3.6, t_start=t_start, t_end=t_end)))

# Save figure
png(paste0(local.path, "/figures", "/R_US_Li2020.png"), 
    width=16, height=9, units = 'in', res = 300) 
plot(res1)
dev.off()

#################################
# Nishiura et al. (2020)
#################################
res2 <- estimate_R(covid_positive, method = "parametric_si",
                   config = make_config(list(
                     mean_si = 4.7, std_si = 2.9, t_start=t_start, t_end=t_end)))

# Save figure
png(paste0(local.path, "/figures", "/R_US_Nishiura2020.png"), 
    width=16, height=9, units = 'in', res = 300) 
plot(res2)
dev.off()


#################################
# Zhao et al. (2020)
#################################
res3 <- estimate_R(covid_positive, method = "parametric_si",
                   config = make_config(list(
                     mean_si = 4.4, std_si = 3.0, t_start=t_start, t_end=t_end)))

# Save figure
png(paste0(local.path, "/figures", "/R_US_Zhao2020.png"), 
    width=16, height=9, units = 'in', res = 300) 
plot(res3)
dev.off()

#################################
# Du et al. 2020
#################################
res4 <- estimate_R(covid_positive, method = "parametric_si",
                   config = make_config(list(
                     mean_si = 3.96, std_si = 4.75, t_start=t_start, t_end=t_end)))

# Save figure
png(paste0(local.path, "/figures", "/R_US_Du2020.png"), 
    width=16, height=9, units = 'in', res = 300) 
plot(res4)
dev.off()


### save R
R_value <- data.frame(dates <- covid_positive$dates[(sliding_length+1):length(covid_positive$dates)],
                      positive <- covid_positive$I[(sliding_length+1):length(covid_positive$dates)], 
                      model1 <- res1$R[,3], 
                      model2 <- res2$R[,3], 
                      model3 <- res3$R[,3], 
                      model4 <- res4$R[,3])
colnames(R_value) <- c("dates","positive","model1","model2","model3","model4")

dir.create(file.path(getwd(), "results"), showWarnings = TRUE)
results.path <- paste0(getwd(), "/results/")

write.csv(R_value,paste0(results.path, "R_value_US_4models",".csv"))

###
png(paste0(figures.path, "R_value_US_4models.png"), width=8, height=6, units = 'in', res = 300) 
estimate_R_plots(list(res1, res2, res3, res4), what = "R",
                 options_R = list(col = c("red", "orange", "blue", "green")), 
                 legend = TRUE)
dev.off()

png(paste0(figures.path, "R_value_US_4models_April.png"), width=6, height=6, units = 'in', res = 300) 
estimate_R_plots(list(res1, res2, res3, res4), what = "R",
                 options_R = list(col = c("red", "orange", "blue", "green"),
                            xlim=c(as.Date("2020-04-01"),as.Date("2020-04-18")),
                            ylim=c(0.5,2.5)),
                 legend = TRUE)
dev.off()


