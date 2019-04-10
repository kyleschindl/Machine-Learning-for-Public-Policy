library(dplyr)
library(stargazer)
panel_data <- read_csv("panel_data.csv")
panel_data <- panel_data %>% filter(party != "0" & !is.na(clean_region))
fixed_effects <- lm(polling_locations ~ black_alone + hispanic + asian_alone
                    + median_income + democrat + factor(clean_region)
                    + factor(year), data=panel_data)
stargazer(fixed_effects, type='html', out='fixed_effects.html')