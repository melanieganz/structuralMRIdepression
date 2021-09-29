library(dplyr)
library(tidyr)
library(gtsummary)

# Load data
df.meta = read.csv('MR_NP1_HC_DBproject_baseline.csv')
df.outcome = read.csv('MR_NP1_HC_DBproject_outcome.csv')
df = left_join(df.meta, df.outcome, by=colnames(df.meta)[1])

# Rename
df = df %>% rename(
        age='Age.at.MR',
        sex='Gender.x',
        single.recurrent='Single.or.recurrent.MDD.episode..x',
        hamd6.base='HAMD.6.score...Baseline.x',
        hamd6.change.week4='Percent.change.in.HAMD.6.at.week.4.compared.to.baseline',
        hamd6.change.week8='NP1.secondary.outcome...Percent.change.in.HAMD.6.at.week.8.compared.to.baseline',
        outcome.week4='Categorical.early..week.4..treatment.response',
        outcome.week8='NP1.primary.outcome...Categorical.treatment.response..week.8.'
      )
df$hamd6.week8 = df$hamd6.base + df$hamd6.base * df$hamd6.change.week8/100

# Exclude subject 21 or younger and no reported HAMD change at week 8
df = df %>% filter(age > 21) %>% drop_na(hamd6.change.week8)

# Assign and group by responder status
df$responder = as.factor(df$hamd6.change.week8 > -50) %>%
                dplyr::recode(`TRUE`='Non-Responder', `FALSE`='Responder')

# Final rename for table
df = df %>%
  select(responder, age, sex, hamd6.base, hamd6.week8, single.recurrent) %>%
  rename(
    Age='age',
    Sex='sex',
    `Single/Recurrent`='single.recurrent',
    `HAMD-6 - Baseline`='hamd6.base',
    `HAMD-6 - Week 8`=hamd6.week8
  )

# Plot table
tab = df %>%
        tbl_summary(by=responder) %>%
        add_p() %>%
        as_flex_table()
tab
