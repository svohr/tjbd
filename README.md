# tjbd

## Overview
This repository contains scripts for inferring Identity By Descent (IBD)
segments between a high-coverage genome from a present-day individual and
low-coverage shotgun sequencing data from a recent ancestor (<10 generations
removed).

## Hidden Markov Model

```
                    +--------+       1-(1-D)^g        +--------+
Hidden         +----+        |----------------------->|        +----+
       (1-D)^g |    |  TJBD  |                        | NoTJBD |    | (1-D)^g
               +--->|        |<-----------------------|        |<---+
                    +-+----+-+       1-(1-D)^g        +-+----+-+
                      |     \                          /     |
                      |      \                        /      |
                      |       \                      /       |
           (1+f_tj)/2 |        \         f_tj       /        | f_tj
                      |    +----\------------------+         |
                      |    |     +----------------------+    |
                      |    |           f_tj/2           |    |
                      v    v                            v    v
                    +--------+                        +--------+
Observed            |        |                        |        |
                    |  match |                        | !match |
                    |        |                        |        |
                    +--------+                        +--------+

f_tj = Frequency of observed historical base in population
   D = Probability of recombination per generation (genetic distance in Morgans)
   g = Number of generations between historical and present-day individuals
```

## Inputs
1. A bam file containing reads mapped to the reference human genome from
   the low-coverage sequencing of the historical individual.
2. A VCF file containing genotypes of the present-day descendant.
3. Population allele frequencies or counts (e.g., 1000 Genomes Project data)
4. A genetic/recombination map

## Output
The posterior probablilities of the two individuals sharing at least one
chromosome segment of IBD and not IBD at each observed position.

