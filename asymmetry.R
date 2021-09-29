library(tidyr)
library(dplyr)
library(broom)
library(flextable)
library(rstatix)

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

# Load and format data
df = read.csv(paste0('data_lr.csv'))
df$group = as.factor(df$group)
df$group = df$group %>% dplyr::recode(`0`='Non-Responder', `1`='Responder')

# Get region names
regions = colnames(df)[
  unlist(lapply(colnames(df), function(x) startsWith(x, 'lh')))]
regions = unlist(lapply(regions, function(x) sub('lh', '', x)))

# Compute asymmetry index
df.assym = data.frame(group = df$group)
region = '.subcort.amygdala'
for (region in regions) {
  L = df[paste0('lh', region)]
  R = df[paste0('rh', region)]
  clean.region = unlist(clean.names[
    sub('.subcort.', '', sub('.cort.', '', region, fixed = T))], use.names=F)
  df.assym[clean.region] = (L - R)/(L + R)
}

# Pivot and group
clean.regions = colnames(df.assym)[2:ncol(df.assym)]
df.assym = df.assym %>%
  pivot_longer(clean.regions, values_to = 'AI', names_to = 'region') %>%
  group_by(region)

# Perform comparison
wt =df.assym %>%
  wilcox_test(AI ~ group) %>%
  adjust_pvalue(method = 'fdr') %>%
  select(-c(.y., group1, group2, n1, n2))

# Compute effect size
wt$effsize = df.assym %>%
  wilcox_effsize(AI ~ group) %>%
  pull(effsize)

# Clean up data frame
wt = wt %>% select(region, effsize, p, p.adj) %>%
      rename(Region=region,
             `Effect size`=effsize,
             `p-value`=p,
             `FDR p-value`=p.adj) 

flextable(wt)%>%
  colformat_double(j = 'Effect size', digits = 3) %>%
  colformat_double(j = c('p-value', 'FDR p-value'), digits = 2)

