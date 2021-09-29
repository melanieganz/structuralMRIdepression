library(tidyr)
library(dplyr)
library(broom)
library(flextable)

# DEBUG
# out_type = 'lr'
# meas = '.subcort.'

df.list = list()

lm.stats.table = function(out_type, meas, all.digits=F) { 

  # Load and format data
  df_ = read.csv(paste0('data_', out_type, '.csv'))
  other.vars = c('age', 'sex', 'group', 'ICV', 'hamd_base', 'single_recurrent')
  df_ = df_ %>% select(other.vars, contains(meas))
  
  # Recode variables
  df_$group = as.factor(df_$group)
  df_$group = df_$group %>% dplyr::recode(`0`='Non-Responder', `1`='Responder')
  df_$sex = as.factor(df_$sex)
  df_$sex = df_$sex %>% dplyr::recode(`0`='male', `1`='female')
  
  # Transform to long format
  regions = colnames(df_)[!(colnames(df_) %in% c('age', 'sex', 'group', 'ICV', 'hamd_base', 'single_recurrent'))]
  df = df_ %>%
    pivot_longer(regions, values_to = 'volume', names_to = 'region') %>%
    group_by(group, volume)
  df$region = as.factor(df$region)

  # Perform regression
  if (meas == '.cort.') {
    lm.stats = df %>% group_by(region) %>%
      do(tidy(lm(volume ~ group + age + sex, data = .)))
  } else if (meas == '.subcort.') {
    lm.stats = df %>% group_by(region) %>%
      do(tidy(lm(volume ~ group + age + sex + ICV, data = .)))
  }

  # Keep only the results for the "group" coefficient
  lm.stats = lm.stats %>% filter(term == 'groupResponder') %>% select(-term)
      
  # Compute Cohen's d according to No Alterations of Brain Structural Asymmetry in Major Depressive Disorder: An ENIGMA Consortium Analysis
  # Cohen's d: t*sqrt(1/n1+1/n2)
  n1 = sum(df$group == 'Responder')
  n2 = sum(df$group == 'Non-Responder')
  lm.stats$`Cohen's d` = lm.stats$statistic*sqrt(1/n1+1/n2)
  lm.stats = lm.stats %>% relocate(`Cohen's d`, .after=region)
  
  # Correct p-values for the number of comparisons
  lm.stats$`FDR p-value` = p.adjust(lm.stats$p.value, method='fdr')
  
  # Clean up region names
  if (out_type == 'lr') {
    lm.stats$prefix = as.factor(unlist(
      lapply(lm.stats$region, function(x) substr(x, 1, 2))))
    lm.stats$prefix = lm.stats$prefix %>% recode('lh' = 'Left ', 'rh' = 'Right ')
    lm.stats$region = sub(paste0('lh', meas), '', lm.stats$region)
    lm.stats$region = sub(paste0('rh', meas), '', lm.stats$region)
    lm.stats = lm.stats %>% arrange(region, prefix)
  } else if (out_type == 'min.max') {
    lm.stats$prefix = as.factor(unlist(
      lapply(lm.stats$region, function(x) substr(x, 1, 3))))
    lm.stats$prefix = lm.stats$prefix %>% recode('min' = 'Min ', 'max' = 'Max ')
    lm.stats$region = sub(paste0('min', meas), '', lm.stats$region)
    lm.stats$region = sub(paste0('max', meas), '', lm.stats$region)
    lm.stats = lm.stats %>% arrange(region, prefix)
  } else if (out_type == 'mean') {
    prefix = paste0(out_type, meas)
    lm.stats$region = sub(prefix, '', lm.stats$region)
  }
  
  clean.names = list(
    'accumbens.area' = 'Accumbens area',
    'amygdala' = 'Amygdala',
    'bankssts' = 'Banks of the superior temporal sulcus',
    'caudalanteriorcingulate' = 'Caudal anterior cingulate',
    'caudalmiddlefrontal' = 'Caudal middle frontal',
    'caudate' = 'Caudate',
    'cerebellum.cortex' = 'Cerebellum cortex',
    'hippocampus' = 'Hippocampus',
    'cuneus' = 'Cuneus',
    'entorhinal' = 'Entorhinal',       
    'frontalpole' = 'Frontal pole',
    'fusiform' = 'Fusiform',
    'inferiorparietal' = 'Inferior parietal',
    'inferiortemporal' = 'Inferior temporal',
    'insula' = 'Insula',
    'isthmuscingulate' = 'Isthmus of cingulate',
    'lateraloccipital' = 'Lateral occipital',
    'lateralorbitofrontal' = 'Lateral orbitofrontal',
    'lateral.ventricle' = 'Lateral ventricle',
    'lingual' = 'Lingual',
    'mean_thick' = 'Mean Thickness',
    'medialorbitofrontal' = 'Medial orbitofrontal',
    'middletemporal' = 'Middle temporal',
    'pallidum' = 'Pallidum',
    'paracentral' = 'Paracentral',
    'parahippocampal' = 'Parahippocampal',
    'parsopercularis' = 'Pars opercularis', 
    'parsorbitalis' = 'Pars orbitalis',
    'parstriangularis' = 'Pars triangularis',
    'pericalcarine' = 'Pericalcarine',
    'postcentral' = 'Postcentral',
    'posteriorcingulate' = 'Posterior cingulate',
    'precentral' = 'Precentral',
    'precuneus' = 'Precuneus',
    'putamen' = 'Putamen',
    'rostralanteriorcingulate' = 'Rostral anterior cingulate',
    'rostralmiddlefrontal' = 'Rostral middle frontal',
    'superiorfrontal' = 'Superior frontal',
    'superiorparietal' = 'Superior parietal',
    'superiortemporal' = 'Superior temporal',
    'supramarginal' = 'Supramarginal',
    'temporalpole' = 'Temporal pole',
    'thalamus' = 'Thalamus',
    'transversetemporal' = 'Tranverse temporal'
    )
  
  lm.stats$region = unlist(clean.names[lm.stats$region], use.names=F)
  
  # Re-append prefix
  if (out_type %in% c('lr', 'min.max')) {
    lm.stats$region = paste0(lm.stats$prefix, lm.stats$region)
    lm.stats = lm.stats %>% select(-prefix)
  }
  
  # Rename columns and remove unwanted
  lm.stats = lm.stats %>%
    rename(Region=region, `p-value`=p.value) %>%
    select(-c(estimate, std.error, statistic))
  
  # Create table
  if (all.digits) {
    tab = lm.stats %>% flextable()
  } else {
    tab = lm.stats %>% flextable() %>%
      colformat_double(j = colnames(lm.stats)[2], digits = 3) %>%
      colformat_double(j = colnames(lm.stats)[3:ncol(lm.stats)], digits = 2)
  }

  tab

}

# Cannot plot table in for loop for display in MS Word document
# Have to do this trick instead..
tab = lm.stats.table('mean', '.cort.')
tab
tab = lm.stats.table('mean', '.subcort.')
tab
tab = lm.stats.table('lr', '.cort.')
tab
tab = lm.stats.table('lr', '.subcort.')
tab
tab = lm.stats.table('min.max', '.cort.')
tab
tab = lm.stats.table('min.max', '.subcort.')
tab
