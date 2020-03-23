library(ggplot2)
library(lubridate)
library(dplyr)
library(scales)

table_stats = read.csv('/Users/besnik/Documents/L3S/Maria_Koutraki/bionlp2019/data/table_stats.tsv', sep = '\t')
table_stats <- table_stats[order(-table_stats$num_entities),]

ggplot(data=table_stats, aes(x=factor(column, levels = column), y=num_entities)) +
  geom_bar(stat="identity", fill="steelblue") +
  theme(axis.text.x = element_blank(), axis.ticks = element_blank()) + 
  xlab('Columns') + ylab('Occurrence') 


sub <- subset(table_stats, num_entities > 1)
sub_II <- subset(table_stats, num_entities == 1)
sub[1:10, c('column', 'num_entities')]
