# Simulations and real data analysis

This folder contains the scripts and code for the simulations and real data analysis in the paper.

The folder structure is as follows:

- `scripts` contains the julia functions to simulate protein alignments and to fit our NN model on the Zika data:
   - `simulations`:
      - `functions-zou2019.jl`: all necessary julia functions
      - `simulate-zou2019.jl`: simulating script for one specific scenario (see Usage below)
      - `MCaa.dat`: PAML control file needed for the simulation of protein sequences
      - `SeedUsed`: seed used in PAML
      - `ancestral.txt`: ancestral sequences generated during PAML simulation
      - `evolver`: PAML executable to simulate sequences
      - `evolver.out`: empty file produced by PAML
   - `real_data`: 
      - `Real_Data_Test.ipynb`: python notebook with the code to predict the quartet for a given 4-taxon sequence dataset
      - `resultsaved_TrainOptLSTM_trainoptlstm_lr_0.001_batch_16_lba_best.pth`: .pth file with the best path of the model created by the OptLSTM training process
- `notebook.md` contains the reproducible script with all detailed description of steps for the simulation of data, comparison to standard phylogenetic methods and real data analysis. For a fast summary of the scripts to run yourself, check out Usage below.


## Usage

### Simulating data

1. Download PAML following instructions [here](http://abacus.gene.ucl.ac.uk/software/pamlDOC.pdf), and downloading `paml4.8a.macosx.tgz` from [here](http://abacus.gene.ucl.ac.uk/software/paml.html)

2. In `paml4.8/bin` there is the `evolver` executable. Keep the `evolver` executable in the same folder (`scripts`) as the julia scripts. The main simulating script `simulate-zou2019.jl` has 5 arguments: seed, number of samples, b, a, c (corresponding to branch lengths defined in Figure 3 in the paper):

```
## b=0.1, a=2b, c=0.01b
julia simulate-zou2019.jl 4738282 8000 0.1 2 0.01
```

This script will produce two files: `labels4738282-0.1-2-0.01.in` with the quartet labels and `sequences4738282-0.1-2-0.01.in` with the protein sequences. For this case, the labels file will have 8000 rows (1 column) and the sequences file will have 8000*4=32000 rows and L=1550 columns. The sequence length can be changed inside the `simulate-zou2019.jl` script.

### Comparison to standard phylogenetic methods

#### 1. Extracting sample data
We will grab one 4-taxon dataset from a randomly chosen file: `sequences45435-1.0-40.0-0.1.in` and `labels45435-1.0-40.0-0.1.in`.

In the sequence file, we have a 400,000 x 1550 matrix in which we have 100,000 4-taxon datasets. We will grab the last 4 rows which correspond to one 4-taxon dataset.

```shell
sed -n '399997,400000 p' sequences45435-1.0-40.0-0.1.in > test.in
```

The order of the sequences are always S1,S2,S3,S4.

#### 2. Convert sample data to Fasta file

Most phylogenetic methods will need a fasta file as input data. We will create this in julia:

```julia
datafile = "test.in"
lines = readlines(datafile)
fastafile = "test.fasta"
io = open(fastafile, "w")

n = length(lines)
l = length(lines[1])

write(io,"$n $l \n")
for i in 1:n
   write(io, string(">",i,"\n"))
   write(io, lines[i])
   write(io, "\n")
end

close(io)
```

#### 3. Fitting maximum parsimony (MP) and neighbor-joining (NJ) in R

The easiest phylogenetic methods to fit are MP and NJ, both in R.

To install the necessary packages in R:
```r
install.packages("ape", dep=TRUE)
install.packages("phangorn", dep=TRUE)
install.packages("adegenet", dep=TRUE) ##I get a warning message
install.packages("seqinr", dep=TRUE)
```

To fit the NJ model:
```r
library(ape)
library(phangorn)
library(adegenet)

## Reading fasta file
aa = read.aa(file="test.fasta", format="fasta")

## Estimating evolutionary distances using same model as in simulations
D = dist.ml(aa, model="Dayhoff") 

## Estimating tree
tree = nj(D)

## Saving estimated tree to text file
write.tree(tree,file="nj-tree.txt")
```

Note that we cannot fit the MP model on aminoacid sequences in R (we need DNA sequences).

#### 4. Fitting maximum likelihood (ML) with RAxML

1. Download `raxml-ng` from [here](https://github.com/amkozlov/raxml-ng). You get a zipped folder: `raxml-ng_v1.0.2_macos_x86_64` which I placed in my `software` folder

2. Checking the version
```shell
cd Dropbox/software/raxml-ng_v1.0.2_macos_x86_64/
./raxml-ng -v
```

3. Infer the ML tree using the same model as in the simulations
```shell
$ raxml-ng --msa test.fasta --model Dayhoff --prefix T3 --threads 2 --seed 616

Final LogLikelihood: -14728.557112

AIC score: 29467.114224 / AICc score: 29467.153084 / BIC score: 29493.844275
Free parameters (model + branch lengths): 5

Best ML tree saved to: /Users/Clauberry/Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/results/T3.raxml.bestTree
All ML trees saved to: /Users/Clauberry/Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/results/T3.raxml.mlTrees
Optimized model saved to: /Users/Clauberry/Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/results/T3.raxml.bestModel

Execution log saved to: /Users/Clauberry/Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/results/T3.raxml.log

Analysis started: 23-Mar-2021 18:18:12 / finished: 23-Mar-2021 18:18:13

Elapsed time: 0.921 seconds
```

Best tree saved in `T3.raxml.bestTree`. We used the `T3` prefix for the files because of the raxml tutorial, but we can choose any prefix.
Warning: Note that the tree has the `>` as part of the taxon names.

We will get rid of the `>` in the shell to avoid problems later:
```shell
sed -i '' -e $'s/>//g' T3.raxml.bestTree
```

#### 5. Fitting bayesian inference (BI) with MrBayes

1. Download MrBayes from [here](http://nbisweden.github.io/MrBayes/). In mac:
```shell
brew tap brewsci/bio
brew install mrbayes --with-open-mpi

$ which mb
/usr/local/bin/mb
```

Had to troubleshoot a lot!
```shell
brew reinstall mrbayes
sudo chown -R $(whoami) /usr/local/Cellar/open-mpi/4.1.0
brew reinstall mrbayes
```

2. MrBayes needs nexus files. We will do this in R:
```r
library(ape)
library(phangorn)
library(adegenet)

## Reading fasta file
aa = read.aa(file="test.fasta", format="fasta")

## Write as nexus file
write.nexus.data(aa,file="test.nexus",format="protein")
```

3. Add the mrbayes block to the nexus file. MrBayes requires that you write a text block at the end of the nexus file. We will write this block in a text file called `mb-block.txt` and we can use the same block for all runs.
```
begin mrbayes;
set nowarnings=yes;
set autoclose=yes;
prset aamodel=fixed(dayhoff);
mcmcp ngen=100000 burninfrac=.25 samplefreq=50 printfreq=10000 [increase these for real]
diagnfreq=10000 nruns=2 nchains=2 temp=0.40 swapfreq=10;       [increase for real analysis]
mcmc;
sumt;
end;
```
This block specifies the length of the MCMC chain and the aminoacid model (Dayhoff which is the same used in simulations).

We will add the mrbayes block in the shell:
```shell
cat test.nexus mb-block.txt > test-mb.nexus
```

4. Run MrBayes:
```shell
mb test-mb.nexus
```

The estimated tree is in `test-mb.nexus.con.tre`.

#### 6. Comparing the estimated trees to the true tree

The true tree is in the file `labels45435-1.0-40.0-0.1.in`.
We will use R to compare the trees.

```r
## read true trees
d = read.table("labels45435-1.0-40.0-0.1.in", header=FALSE)
n = length(d$V1)
## labels from the simulating script:
quartets = c("((1,2),(3,4));", "((1,3),(2,4));", "((1,4),(2,3));")
## to which quartet the label corresponds to:
truetree = read.tree(text=quartets[d[n,]])

library(ape)
## read the NJ tree:
njtree = read.tree(file="nj-tree.txt")
## read the ML tree:
mltree = read.tree(file="T3.raxml.bestTree")
## read the BI tree
bitree = read.nexus(file="test-mb.nexus.con.tre")

## Calculating the Robinson-Foulds distance with true tree:
library(phangorn)
njdist = RF.dist(truetree,njtree, rooted=FALSE)
mldist = RF.dist(truetree,mltree, rooted=FALSE)
bidist = RF.dist(truetree,bitree, rooted=FALSE)
```
If the distance is equal to zero, then the method reconstructed the correct tree. For example, `njdist==0` implies that NJ estimated the correct tree.


### Real data analysis: Zika virus

- Using [NCBI](https://www.ncbi.nlm.nih.gov/genomes/VirusVariation/Database/nph-select.cgi) for Zika virus
- Searching for Human, Mammal, Primate samples; any genome region
- [Query link](https://www.ncbi.nlm.nih.gov/genomes/VirusVariation/Database/nph-select.cgi?cmd=show_builder&country=any&download-select=fP&genregion=any&go=database&host=Primate&isolation=isolation_blood&query_1_count=1124&query_1_count_genome_sets=0&query_1_country=any&query_1_genregion=any&query_1_host=Human&query_1_isolation=any&query_1_line=on&query_1_line_num=1&query_1_query_key=1&query_1_searchin=sequence&query_1_sequence=P&query_1_srcfilter_labs=include&query_1_taxid=64320&query_2_count=101&query_2_count_genome_sets=0&query_2_country=any&query_2_genregion=any&query_2_host=Primate&query_2_isolation=any&query_2_line=on&query_2_line_num=2&query_2_query_key=1&query_2_searchin=sequence&query_2_sequence=P&query_2_srcfilter_labs=include&query_2_taxid=64320&query_3_count=4&query_3_count_genome_sets=0&query_3_country=any&query_3_genregion=any&query_3_host=Mammal&query_3_isolation=any&query_3_line=on&query_3_line_num=3&query_3_query_key=1&query_3_searchin=sequence&query_3_sequence=P&query_3_srcfilter_labs=include&query_3_taxid=64320&query_4_count=0&query_4_count_genome_sets=0&query_4_country=any&query_4_genregion=any&query_4_host=Mammal&query_4_isolation=isolation_blood&query_4_line_num=4&query_4_query_key=1&query_4_searchin=sequence&query_4_sequence=P&query_4_srcfilter_labs=include&query_4_taxid=64320&query_5_count=357&query_5_count_genome_sets=0&query_5_country=any&query_5_genregion=any&query_5_host=Human&query_5_isolation=isolation_blood&query_5_line_num=5&query_5_query_key=1&query_5_searchin=sequence&query_5_sequence=P&query_5_srcfilter_labs=include&query_5_taxid=64320&query_6_count=0&query_6_count_genome_sets=0&query_6_country=any&query_6_genregion=any&query_6_host=Primate&query_6_isolation=isolation_blood&query_6_line_num=6&query_6_query_key=1&query_6_searchin=sequence&query_6_sequence=P&query_6_srcfilter_labs=include&query_6_taxid=64320&searchin=sequence&sequence=P&srcfilter_labs=include&taxid=64320)
- Manually selected (accession, length, host, country, collection year):
       - BBA85762, 3423, Homo sapiens, Japan, 2016
       - QIH53581, 3423, Homo sapiens, Brazil, 2017
       - BAP47441, 3423, Simiiformes, Uganda, 1947
       - ANG09399, 3423, Homo sapiens, Honduras, 2016
       - AXF50052, 3423, Mus Musculus, Colombia, 2016
       - AWW21402, 3423, Simiiformes, Cambodia, 2016
       - AYI50274, 3423, Macaca mulatta, xxxxx, 2015
- Downloaded as `FASTA.fa`. All sequences have the same length, so no need to align.
- The website creates a tree which is downloaded as `tree.nwk`. This tree is strange because it puts Macaca mulatta right in the middle of homo sapiens.

#### 1. Creating files with 4 taxa

Our dataset has 7 species, so we need to create subsets of 4 to fit in our NN.

```julia
data = readlines("FASTA.fa")

taxa = []
seqs = []

seq = ""
for l in data
   if occursin(">",l)
      push!(taxa,l)
      push!(seqs,seq) ##push previous seq
      seq = ""
   else
      seq *= l
   end
end
push!(seqs,seq) ##push last seq

## by the way it is constructed, we have an extra empty seq in seqs:
deleteat!(seqs, 1)
```

Now we have two vectors: `taxa` with the taxon names and `seqs` with the sequences.

First, we create a translate table with taxon names:
```julia
using DataFrames, CSV
df = DataFrame(id=1:length(taxa), name=taxa)
CSV.write("fasta-table.csv",df)
```

Now, we create one datafile for each combination:
```r
> choose(7,4)
[1] 35
```

```julia
using Combinatorics
comb = collect(combinations(1:length(taxa),4))
## 35-element Vector{Vector{Int64}}:

i = 1
for c in comb
   io = open(string("zika-fasta",i,".fa"), "w")
   for j in c
      write(io, string(">",j))
      write(io, "\n")
      write(io, seqs[j])
      write(io, "\n")
   end
   close(io)
   i += 1
end   
```

#### 2. Removing 7th taxon

The 7th taxon (see `FASTA.fa`) is "BBA85762, 3423, Homo sapiens, Japan, 2016" which contains missing sites X. Because our NN model was not trained with missing sites, we will remove this taxon from the dataset.


#### 3. Quartet puzzling

After NN model has been run (see the `real_data` folder), we can do the quartet puzzling step with Quartet Max Cut. It needs an input file that is one line with each quartet separated by a space in the form of a split: "1,2|3,4".
See `../plots/final_plots.Rmd` for the quartet stitching and the plot of the tree.
