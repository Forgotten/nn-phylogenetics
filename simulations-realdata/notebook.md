# Reproducible script for Accurate Phylogenetic Inference with a Symmetry-preserving Neural Network Model

- Claudia Solis-Lemus
- Shengwen Yang
- Leonardo Zepeda-Nunez

This script contains:
- Simulations non-LBA cases (initial work, not part of the manuscript)
- Understanding Zou2019 permutations (initial work, not part of the manuscript)
- LBA simulations (cases included in the manuscript)
- Comparison to standard phylogenetic methods (included in the manuscript)
- Real data analysis of Zika virus (included in the manuscript)


# Simulating data to replicate Zou2019

- Training data: 100,000 quartets with varying branch lengths (not explicit how)
- Testing on 2000 quartets generated in the same manner as the training
- Further testing: 6 datasets of 1000 20-taxon trees with branch lengths on the intervals: [0.02, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0), and [1.0, 2.0)
- Further testing on 5 1000-tree datasets with sequence length ranging: [100, 200) to [3,000, 10,000) amino acids
- Raw input data: 4 aligned aminoacid sequences of length L => Matrix 4xL; this matrix is converted to tensor due to one-hot encoding: 4x20xL (because there are 20 aminoacids). The tensor is later vectorized as 80xL matrix
- Parameters for simulation: number of taxa M (later pruned to 4), branch lengths B, number of aminoacid sites N_{aa}, exchangeability matrix S, shape parameter \alpha of the gamma distribution of the relative rate r (on all sites), probability p_h with which a branch is heterogeneous, proportion of sites subject to rate shuffling f, number of profile swap operations for each site (Table S2)
- From Table S2 (first row):
    - M~Uniform(5,105)
    - (internal) B~Uniform(0.02,1.02)
    - (external) B~Uniform(0.02,1.02)
    - N_{aa}~Uniform(100,3000)
    - S random
    - \alpha~Uniform(0.05,1)
    - p_h=0.9
    - f~Uniform(0,1)
    - n~Uniform(10,20)


There code to simulate data is not easy to follow/understand, so I could simulate data with SeqGen as I always do. In the Zou2019 paper, they reference PAML (from Ziheng Yang), which also simulates sequences, see [here](http://abacus.gene.ucl.ac.uk/software/pamlDOC.pdf):
```
evolver. This program can be used to simulate sequences under nucleotide, codon and amino acid substitution models. It also has some other options such as generating random trees, and calculating the partition distances (Robinson and Foulds 1981) between trees. 
```

1. Download PAML following instructions [here](http://abacus.gene.ucl.ac.uk/software/pamlDOC.pdf), and downloading `paml4.8a.macosx.tgz` from [here](http://abacus.gene.ucl.ac.uk/software/paml.html)

2. In `paml4.8/bin` there is the `evolver` executable, which can be run as an executable directly:
```
$ cd Dropbox/software/PAML4/paml4.8/bin/
$ ./evolver
EVOLVER in paml version 4.8a, August 2014
Results for options 1-4 & 8 go into evolver.out

	(1) Get random UNROOTED trees?
	(2) Get random ROOTED trees?
	(3) List all UNROOTED trees?
	(4) List all ROOTED trees?
	(5) Simulate nucleotide data sets (use MCbase.dat)?
	(6) Simulate codon data sets      (use MCcodon.dat)?
	(7) Simulate amino acid data sets (use MCaa.dat)?
	(8) Calculate identical bi-partitions between trees?
	(9) Calculate clade support values (evolver 9 treefile mastertreefile <pick1tree>)?
	(11) Label clades?
	(0) Quit?
```
Or by choosing a specfic control file: `./evolver 7 MCaa.dat`

3. We need to write the control file. I copy here the example `MCaa.dat`:
```
 0        * 0: paml format (mc.paml); 1:paup format (mc.nex)
13147       * random number seed (odd number)

5 10000 5   * <# seqs>  <# sites>  <# replicates>

-1         * <tree length, use -1 if tree below has absolute branch lengths>

(((Human:0.06135, Chimpanzee:0.07636):0.03287, Gorilla:0.08197):0.11219, Orangutan:0.28339, Gibbon:0.42389);

.5 8        * <alpha; see notes below>  <#categories for discrete gamma>
2 mtmam.dat * <model> [aa substitution rate file, need only if model=2 or 3]

0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 
0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 

 A R N D C Q E G H I
 L K M F P S T W Y V

// end of file

=============================================================================
Notes for using the option in evolver to simulate amino acid sequences. 
Change values of parameters, but do not delete them.  It is o.k. to add 
empty lines, but do not break down the same line into two or more lines.

  model = 0 (poisson), 1 (proportional), 2 (empirical), 3 (empirical_F)
  Use 0 for alpha to have the same rate for all sites.
  Use 0 for <#categories for discrete gamma> to use the continuous gamma
  <aa substitution rate file> can be dayhoff.dat, jones.dat, and so on.
  <aa frequencies> have to be in the right order, as indicated.
=================!! Check screen output carefully!! =====================
```
Note from documentation: "If you use –1 for the tree length, the program will use the branch lengths given in the tree without the re-scaling.", but if you give a number, the program will re-scale the branch lengths so that the sum of bls is equal to tree length.

**Important** All files need to be in the same path of the executable!

Example: `./evolver 7 MCaa.dat` produces:
- `mc.paml` with the sequences in PHYLIP format
- `ancestral.txt` with the simulated ancestral sequences
- `sites.txt` with the rates for each site
- `evolver.out` which is empty

So, we have all the pieces, now we only need a pipeline for our simulations.

### Simulations pipeline
Fixed parameters:
- global random seed = 03011058
- L = 1550 (average of Uniform(100,3000))
- 1 class partition, with \alpha~Uniform(0.05,1)
- model=3 (empirical) with randomly chosen dat file for rates
- nrep = 100,000
```r
set.seed(03011058)
runif(1,0.05,1) ##alpha
0.1580288
floor(runif(1,1,18)) ##dat file (18 total)
4 ##dayhoff.dat
```

For rep i=1,...nrep:
1. Choose a random tree with 4 taxa
2. Choose random branch lengths: Uniform(0.02,1.02) as in Zou2019
3. Create the control file `rep-i.dat`:
```
 0        * 0: paml format (mc.paml); 1:paup format (mc.nex)
<seed>       * random number seed (odd number)

4 1550 1   * <# seqs>  <# sites>  <# replicates>

-1         * <tree length, use -1 if tree below has absolute branch lengths>

<tree with bl>

0.1580288 1        * <alpha; see notes below>  <#categories for discrete gamma>
3 dayhoff.dat * <model> [aa substitution rate file, need only if model=2 or 3]
```
4. Run `./evolver 7 rep-i.dat` (all files in same path as executable)
5. Rename necessary files and move into folders (to avoid overwriting): `mv mc.paml mc-i.paml`
6. Read `mc-i.paml` as matrix, and convert to 4x20x1550 tensor, and then vectorize as 80x1550 matrix
7. Output two lists one for the "labels" (which quartet) and the input matrices

I created the folder `simulations-zou2019` to put all simulated data there. Copied inside the executable (`evolver`) and the model dat file.
Julia script file: `simulate-zou2019.jl` and `functions-zou2019.jl`

### Simulations on batches:

#### 20,000 replicates on 4 threads (5000 each):
Different folders (because files are overwritten and have same names for PAML): `simulations-zou2019-?`, each is running nrep=5000 (I tried 25000 but the computer crashed twice). Process started 3/2 7:49pm, finished at 10:52pm.
Process re-started after changing the root: 5/8 10pm on laptop, but process died in the night after ~4000.
Re-started on desktop 5/9 830am, finished 12pm
- Each replicate produces a label (tree) and input matrix 80xL
- For nrep replicates, we get two files:
  - `labels.h5` with a vector of dimension nrep
  - `matrices.h5` with a matrix 80*nrep by L, that has all matrices stacked

  I had to check that the sequences were simulated in the correct order in the *.dat files. The S1,S2,S3,S4 in the paml file correspond to the order in the tree in the dat file. Files look ok!

Actually, no, it seems that the numbers match what we would expect:
```
Model tree & branch lengths:
((S2: 0.100000, S1: 0.200000): 0.000500, (S3: 0.100000, S4: 0.200000): 0.000500);
((2: 0.100000, 1: 0.200000): 0.000500, (3: 0.100000, 4: 0.200000): 0.000500);
```
So, Si corresponds to taxon i.


We summarize the files:
```shell
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019

## First thread:
cd simulations-zou2019
tar -czvf simulations-zou2019-1.tar.gz rep-*
rm rep-*
mv labels.h5 labels-1.h5
mv matrices.h5 matrices-1.h5
##cp simulate-zou2019.jl simulate-zou2019-1.jl ## to keep the script ran (not used in the re-run because we had it)
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019-1.jl ../results

## Second thread
cd simulations-zou2019-2
tar -czvf simulations-zou2019-2.tar.gz rep-*
rm rep-*
mv labels.h5 labels-2.h5
mv matrices.h5 matrices-2.h5
##cp simulate-zou2019.jl simulate-zou2019-2.jl ## to keep the script ran
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019-2.jl ../results

## Third thread
cd simulations-zou2019-3
tar -czvf simulations-zou2019-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-3.h5
mv matrices.h5 matrices-3.h5
##cp simulate-zou2019.jl simulate-zou2019-3.jl ## to keep the script ran
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019-3.jl ../results

## Fourth thread
cd simulations-zou2019-4
tar -czvf simulations-zou2019-4.tar.gz rep-*
rm rep-*
mv labels.h5 labels-4.h5
mv matrices.h5 matrices-4.h5
##cp simulate-zou2019.jl simulate-zou2019-4.jl ## to keep the script ran
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019-4.jl ../results
```

#### 80,000 replicates on 10 cores (8000 each)
Different folders (because files are overwritten and have same names for PAML): `simulations-zou2019-?`, each is running nrep=8000.
Process started 3/6 5pm, finished at 3/7 2am

We rerun in Mac desktop. We start only with 5-9 (to make sure we don't run out of memory).
Process 5-9 started 5/17 1pm, finished 9:30pm
Process 10-14 started 5/17 10pm, finished 5/18 7am

We summarize the files:
```shell
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019

## 5th thread:
cd simulations-zou2019-5
mkdir tmp 
mv rep-1* tmp
mv rep-2* tmp
mv rep-* tmp
tar -czvf simulations-zou2019-5.tar.gz tmp
rm -r tmp
mv labels.h5 labels-5.h5
mv matrices.h5 matrices-5.h5
##cp simulate-zou2019.jl simulate-zou2019-5.jl ## to keep the script ran
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019-5.jl ../results

## 6th thread:
cd simulations-zou2019-6
mkdir tmp 
mv rep-1* tmp
mv rep-2* tmp
mv rep-* tmp
tar -czvf simulations-zou2019-6.tar.gz tmp
rm -r tmp
mv labels.h5 labels-6.h5
mv matrices.h5 matrices-6.h5
##cp simulate-zou2019.jl simulate-zou2019-6.jl ## to keep the script ran
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019-6.jl ../results

## 7th thread:
cd simulations-zou2019-7
mkdir tmp 
mv rep-1* tmp
mv rep-2* tmp
mv rep-* tmp
tar -czvf simulations-zou2019-7.tar.gz tmp
rm -r tmp
mv labels.h5 labels-7.h5
mv matrices.h5 matrices-7.h5
##cp simulate-zou2019.jl simulate-zou2019-7.jl ## to keep the script ran
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019-7.jl ../results

## 8th thread:
cd simulations-zou2019-8
mkdir tmp 
mv rep-1* tmp
mv rep-2* tmp
mv rep-* tmp
tar -czvf simulations-zou2019-8.tar.gz tmp
rm -r tmp
mv labels.h5 labels-8.h5
mv matrices.h5 matrices-8.h5
##cp simulate-zou2019.jl simulate-zou2019-8.jl ## to keep the script ran
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019-8.jl ../results


## 9th thread:
cd simulations-zou2019-9
mkdir tmp 
mv rep-1* tmp
mv rep-2* tmp
mv rep-* tmp
tar -czvf simulations-zou2019-9.tar.gz tmp
rm -r tmp
mv labels.h5 labels-9.h5
mv matrices.h5 matrices-9.h5
##cp simulate-zou2019.jl simulate-zou2019-9.jl ## to keep the script ran
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019-9.jl ../results

## 10th thread:
cd simulations-zou2019-10
mkdir tmp 
mv rep-1* tmp
mv rep-2* tmp
mv rep-* tmp
tar -czvf simulations-zou2019-10.tar.gz tmp
rm -r tmp
mv labels.h5 labels-10.h5
mv matrices.h5 matrices-10.h5
##cp simulate-zou2019.jl simulate-zou2019-10.jl ## to keep the script ran
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019-10.jl ../results

## 11th thread:
cd simulations-zou2019-11
mkdir tmp 
mv rep-1* tmp
mv rep-2* tmp
mv rep-* tmp
tar -czvf simulations-zou2019-11.tar.gz tmp
rm -r tmp
mv labels.h5 labels-11.h5
mv matrices.h5 matrices-11.h5
##cp simulate-zou2019.jl simulate-zou2019-11.jl ## to keep the script ran
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019-11.jl ../results

## 12th thread:
cd simulations-zou2019-12
mkdir tmp 
mv rep-1* tmp
mv rep-2* tmp
mv rep-* tmp
tar -czvf simulations-zou2019-12.tar.gz tmp
rm -r tmp
mv labels.h5 labels-12.h5
mv matrices.h5 matrices-12.h5
##cp simulate-zou2019.jl simulate-zou2019-12.jl ## to keep the script ran
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019-12.jl ../results

## 13th thread:
cd simulations-zou2019-13
mkdir tmp 
mv rep-1* tmp
mv rep-2* tmp
mv rep-* tmp
tar -czvf simulations-zou2019-13.tar.gz tmp
rm -r tmp
mv labels.h5 labels-13.h5
mv matrices.h5 matrices-13.h5
##cp simulate-zou2019.jl simulate-zou2019-13.jl ## to keep the script ran
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019-13.jl ../results

## 14th thread:
cd simulations-zou2019-14
mkdir tmp 
mv rep-1* tmp
mv rep-2* tmp
mv rep-* tmp
tar -czvf simulations-zou2019-14.tar.gz tmp
rm -r tmp
mv labels.h5 labels-14.h5
mv matrices.h5 matrices-14.h5
##cp simulate-zou2019.jl simulate-zou2019-14.jl ## to keep the script ran
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019-14.jl ../results
```

We have all scripts and results in `simulations-zou2019-results`, so we will remove the folders used to run things in parallel:
```shell
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/
rm -rf simulations-zou2019-*
```

Then, in `simulations-zou2019`, there are two folders:
- `scripts`: with julia scripts and needed executables
- `results`:
  - `labels-i.h5` (i=1,...,14): n-dimensional vector with labels for n replicates; files i=1,2,3,4 have 5000 replicates each (20,000 total) and files i=5,...,14 have 8000 replicates each (80,000 total) => 20k+80k=100k as in Zou2019
  - `matrices-i.h5` (i=1,...,14): 80x1550 input matrix per replicate; 
  files i=1,2,3,4 have 5000 replicates and matrices are stacked on top of each other 
  => (80 * 5000)x1550 matrix; 
  files i=5,...,14 have 8000 replicates each 
  => (80 * 8000)x1550 matrix
  - `simulate-zou2019-i.jl` (i=1,...,14): julia script with random seeds to simulate batch i
  - `simulations-zou2019-1.tar.gz` (i=1,...,14): tar intermediate files per replicate like protein sequences, and paml control file

I will put the labels and matrices files in a shared drive to share with Leo.

Deleting the h5 files locally because they are heavy, and they are in google drive now.


# Understanding Zou2019 permutations
From the main text:
- We generated random trees with more than four taxa and simulated amino acid sequences of varying lengths according to the trees
- After the generation of each tree and associated sequence data, we pruned the tree so that only four taxa remain, hence creating a quartet tree sample ready for training, validation, or testing of the residual network predictor. 
- To ensure that the training process involved diverse and challenging learning materials, we pruned a proportion of trees to four randomly chosen taxa (normal trees), and the other trees to four taxa with high LBA susceptibility
- Training consisted of multiple iterative epochs, based on a total training pool of 100,000 quartets containing 85% normal trees and 15% LBA trees **note:** 100,000 quartets, each with different (24) permutations as explained below

From the "Materials and Methods" section:
- The raw input data, as in conventional phylogenetic inference software, are four aligned amino acid sequences of length L (denoted as taxon0, taxon1, taxon2, and taxon3, hence dimension 4 x L). This is then one-hot-encoded, expanding each amino acid position into a dimension with twenty 0/1 codes indicating which amino acid is in this position. The 4 x 20 x L tensor is transformed to an 80 x L matrix and fed into the residual network
- The output of the network includes three numbers representing the likelihood that taxon0 is a sister of taxon1, taxon2, and taxon3, respectively
- During the training process, the four taxa in each quartet data set were permutated to create 4!=24 different orders, and each serves as an independent training sample, to ensure that the order of taxa in the data set does not influence the phylogenetic inference
- Sequences on a tree were simulated from more ancient to more recent nodes, starting at the root. 

Process:
1. Simulate large tree, then prune to quartet
2. Simulate sequences on quartet
3. Permutate all 4 taxa in the quartet to get 24 permutations for that quartet


In [data.py](https://gitlab.com/ztzou/phydl/-/blob/master/evosimz/data.py) has the function shuffle on line 225. We want to understand why (if?) all 24 permutations make sense for a given tree:
```python
class _QuartetMixin:
    ## this gives us the list of all 24 permutations (see code below)
    _ORDERS = numpy.asarray(list(itertools.permutations(range(4))))

def _shuffle(cls, tree, random_order=False):
        ## here we create a vector of size 24 with repeated tree
        ## note that cls._ORDERS.shape=(24,4)
        trees = [tree] * cls._ORDERS.shape[0] 
        leaves = tree.get_leaves()
        if random_order:
            random.shuffle(leaves)
        ## leaf.sequence is an array of length L, leaves are 4:(tx0,tx1,tx2,tx3)
        sequences = numpy.asarray([leaf.sequence for leaf in leaves])
        ## to understand view('S1') see below
        sequences = sequences.view('S1').reshape(len(leaves), -1)
        sequences = sequences[cls._ORDERS, :]
        ## the following command change the order of the leaves to match the permutations:
        leaf_list = [[leaves[i] for i in order] for order in cls._ORDERS]
        # print(len(trees), sequences.shape, cls._ORDERS.shape, len(leaf_list), sep='\n')
        # 24, (24, 4, 869), (24, 4), 24
        return trees, sequences, cls._ORDERS, leaf_list
```

Trying to understand the commands in python:
```python
$ python
Python 2.7.16 (default, Oct 16 2019, 00:34:56) 
[GCC 4.2.1 Compatible Apple LLVM 10.0.1 (clang-1001.0.37.14)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import bisect
>>> import itertools
>>> import pickle
>>> import random
>>> import numpy
>>> numpy.asarray(list(itertools.permutations(range(4))))
array([[0, 1, 2, 3],
       [0, 1, 3, 2],
       [0, 2, 1, 3],
       [0, 2, 3, 1],
       [0, 3, 1, 2],
       [0, 3, 2, 1],
       [1, 0, 2, 3],
       [1, 0, 3, 2],
       [1, 2, 0, 3],
       [1, 2, 3, 0],
       [1, 3, 0, 2],
       [1, 3, 2, 0],
       [2, 0, 1, 3],
       [2, 0, 3, 1],
       [2, 1, 0, 3],
       [2, 1, 3, 0],
       [2, 3, 0, 1],
       [2, 3, 1, 0],
       [3, 0, 1, 2],
       [3, 0, 2, 1],
       [3, 1, 0, 2],
       [3, 1, 2, 0],
       [3, 2, 0, 1],
       [3, 2, 1, 0]])
>>> strarray = numpy.array([[b"123456"], [b"654321"]])
>>> strarray
array([['123456'],
       ['654321']], 
      dtype='|S6')
>>> strarray.view('S1')
array([['1', '2', '3', '4', '5', '6'],
       ['6', '5', '4', '3', '2', '1']], 
      dtype='|S1')
>>> strarray.view('S1').reshape(2,-1)
array([['1', '2', '3', '4', '5', '6'],
       ['6', '5', '4', '3', '2', '1']], 
      dtype='|S1')
>>> strarray.view('S1').reshape(3,-1)
array([['1', '2', '3', '4'],
       ['5', '6', '6', '5'],
       ['4', '3', '2', '1']], 
      dtype='|S1')
>>> strarray[0]
array(['123456'], 
      dtype='|S6')
>>> strarray[[0,1]]
array([['123456'],
       ['654321']], 
      dtype='|S6')
>>> strarray[[1,0]]
array([['654321'],
       ['123456']], 
      dtype='|S6')
```

Conclusion after talking to colleague: A student in her group presented this paper in lab meeting. He has been trying to use the trained network on a dataset (and failing miserably). It seems that the permutations are on the rows of the matrix only, not on the labels. If you have a matrix 4xL with the sequences, they permute the rows (all 24 permutations), but they keep the labels. This is only to prevent the row order from mattering.


## Notes after studying the code

- The `shuffle` function takes a tree as input (one quartet), which repeats as a vector of dimension 24: `trees`
- The function also takes the sequence simulated on this tree as input, say:
```
0....
1....
2....
3....
```
- The function then creates all 4!=24 permutations of the 4 indices (`cls.ORDER`):
```
array([[0, 1, 2, 3],
       [0, 1, 3, 2],
       [0, 2, 1, 3],
       [0, 2, 3, 1],
       [0, 3, 1, 2],
       [0, 3, 2, 1],
       [1, 0, 2, 3],
       [1, 0, 3, 2],
       [1, 2, 0, 3],
       [1, 2, 3, 0],
       [1, 3, 0, 2],
       [1, 3, 2, 0],
       [2, 0, 1, 3],
       [2, 0, 3, 1],
       [2, 1, 0, 3],
       [2, 1, 3, 0],
       [2, 3, 0, 1],
       [2, 3, 1, 0],
       [3, 0, 1, 2],
       [3, 0, 2, 1],
       [3, 1, 0, 2],
       [3, 1, 2, 0],
       [3, 2, 0, 1],
       [3, 2, 1, 0]])
```
- Then, they adjust the `leaf_list` to the specific order. That is, if the `leaf_list` was `(tx1,tx2,tx3,tx4)`, they will get a 24-dim vector will all the permutations on `cls.ORDER`:
```
array([[tx1, tx2, tx3, tx4],
       [tx1, tx2, tx4, tx3],
       [tx1, tx3, tx2, tx4],
.
.
.
```

After `shuffle`, they call the function `_generate_class_label`, which for every 4-taxon array, change the quartet class (response label) if it was changed. This is done so that we do not need to keep the labels.
That is, the quartet 1 is 01|23. [0, 1, 2, 3] corresponds to this same quartet, so as [1, 0, 2, 3], but this one is not: [3, 1, 2, 0], this corresponds to 02|13.

### Permutation map

Now, we want to do the map of permutation to quartet class for us.
Note that `seqgen` puts the sequences in the order that we expect. That is, for the following tree:
```
((1: 0.636349, 4: 0.324226): 0.549904, (2: 0.389060, 3: 0.153263): 0.433517);
```
`seqgen` converts it to:
```
((S1: 0.636349, S4: 0.324226): 0.549904, (S2: 0.389060, S3: 0.153263): 0.433517);
```
and puts the sequences in order:
```
S1
S2
S3
S4
```
Thus, our quartet specification (12|34, 13|24, 14|23) matches the quartet specification in Zou2019 (quartet1: taxon1 and taxon2 are sisters => S1,S2 sisters for us).

#### Quartet 1 (12|34)
Indices: 1->0, 2->1, 3->2, 4->3
```
[0, 1, 2, 3] => 12|34
[0, 1, 3, 2] => 12|34
[0, 2, 1, 3] => 13|24
[0, 2, 3, 1] => 14|23
[0, 3, 1, 2] => 13|24
[0, 3, 2, 1] => 14|23
[1, 0, 2, 3] => 12|34
[1, 0, 3, 2] => 12|34
[1, 2, 0, 3] => 13|24
[1, 2, 3, 0] => 14|23
[1, 3, 0, 2] => 13|24
[1, 3, 2, 0] => 14|23
[2, 0, 1, 3] => 14|23
[2, 0, 3, 1] => 13|24
[2, 1, 0, 3] => 14|23
[2, 1, 3, 0] => 13|24
[2, 3, 0, 1] => 12|34
[2, 3, 1, 0] => 12|34
[3, 0, 1, 2] => 14|23
[3, 0, 2, 1] => 13|24
[3, 1, 0, 2] => 14|23
[3, 1, 2, 0] => 13|24
[3, 2, 0, 1] => 12|34
[3, 2, 1, 0] => 12|34
```


#### Quartet 2 (13|24)
Indices: 1->0, 3->1, 2->2, 4->3
```
[0, 1, 2, 3] => 13|24
[0, 1, 3, 2] => 14|23
[0, 2, 1, 3] => 12|34
[0, 2, 3, 1] => 12|34
[0, 3, 1, 2] => 14|23
[0, 3, 2, 1] => 13|24
[1, 0, 2, 3] => 14|23
[1, 0, 3, 2] => 13|24
[1, 2, 0, 3] => 14|23
[1, 2, 3, 0] => 13|24
[1, 3, 0, 2] => 12|34
[1, 3, 2, 0] => 12|34
[2, 0, 1, 3] => 12|34
[2, 0, 3, 1] => 12|34
[2, 1, 0, 3] => 13|24
[2, 1, 3, 0] => 14|23
[2, 3, 0, 1] => 13|24
[2, 3, 1, 0] => 14|23
[3, 0, 1, 2] => 13|24
[3, 0, 2, 1] => 14|23
[3, 1, 0, 2] => 12|34
[3, 1, 2, 0] => 12|34
[3, 2, 0, 1] => 14|23
[3, 2, 1, 0] => 13|24
```


#### Quartet 3 (14|23)
Indices: 1->0, 4->1, 2->2, 3->3
```
[0, 1, 2, 3] => 14|23
[0, 1, 3, 2] => 13|24
[0, 2, 1, 3] => 14|23
[0, 2, 3, 1] => 13|24
[0, 3, 1, 2] => 12|24
[0, 3, 2, 1] => 12|34
[1, 0, 2, 3] => 13|24
[1, 0, 3, 2] => 14|23
[1, 2, 0, 3] => 12|34
[1, 2, 3, 0] => 12|34
[1, 3, 0, 2] => 14|23
[1, 3, 2, 0] => 13|24
[2, 0, 1, 3] => 13|24
[2, 0, 3, 1] => 14|23
[2, 1, 0, 3] => 12|34
[2, 1, 3, 0] => 12|34
[2, 3, 0, 1] => 14|23
[2, 3, 1, 0] => 13|24
[3, 0, 1, 2] => 12|34
[3, 0, 2, 1] => 12|34
[3, 1, 0, 2] => 13|24
[3, 1, 2, 0] => 14|23
[3, 2, 0, 1] => 13|24
[3, 2, 1, 0] => 14|23
```


**NOTE** It will not be as straight-forward to implement this permutation strategy due to how PAML works.
In our simulating pipeline, PAML is already permuting the rows of the sequence.
That is, for `rep-1.dat`, PAML is simulating sequences from the tree:
```
(4:0.5499040314743673,(1:0.3242263809077284,(2:0.15326301295458997,3:0.4335172886379941):0.38905991077599955):0.3366654232508077);
```
The order of the taxa will be order read, so S1=4, S2=1, S3=2, S4=3.
So, we might need to simulate the data again to force the order of taxa.

Actually, no, it seems that the numbers match what we would expect:
```
Model tree & branch lengths:
((S2: 0.100000, S1: 0.200000): 0.000500, (S3: 0.100000, S4: 0.200000): 0.000500);
((2: 0.100000, 1: 0.200000): 0.000500, (3: 0.100000, 4: 0.200000): 0.000500);
```
So, Si corresponds to taxon i.

# LBA simulations

We repeat the quartet simulations, but now with long branch attraction branches (Figure 3a Zou2020): "two short external branches have lengths b ranging from 0.1 to 1.0, the two long branches have lengths a ranging from 2b to 40b, and the internal branch has a length c ranging from 0.01b to b".
Added the option `lba = true` to the simulations script.

We will only do 10,000 replicates now.

#### 10,000 replicates on 2 threads (5000 each):
Different folders (because files are overwritten and have same names for PAML): `simulations-zou2019-?`, each is running nrep=5000 (I tried 25000 but the computer crashed twice). We copy the `scripts` folder as two folder: `simulations-zou2019-lba` and `simulations-zou2019-lba-2`
```shell
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-zou2019-lba
julia simulate-zou2019.jl

cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-zou2019-lba-2
julia simulate-zou2019.jl
```
Process started 6/17 330pm, ~finish 830pm

- Each replicate produces a label (tree) and input matrix 80xL
- For nrep replicates, we get two files:
  - `labels.h5` with a vector of dimension nrep
  - `matrices.h5` with a matrix 80*nrep by L, that has all matrices stacked

We summarize the files:
```shell
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019

## First thread:
cd simulations-zou2019-lba
tar -czvf simulations-zou2019-lba-1.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-1.h5
mv matrices.h5 matrices-lba-1.h5
##cp simulate-zou2019.jl simulate-zou2019-1.jl ## to keep the script ran (not used in the re-run because we had it)
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-1.jl
mv simulate-zou2019-lba-1.jl ../results

## Second thread
cd simulations-zou2019-lba-2
tar -czvf simulations-zou2019-lba-2.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-2.h5
mv matrices.h5 matrices-lba-2.h5
##cp simulate-zou2019.jl simulate-zou2019-2.jl ## to keep the script ran
## move to results folder:
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-2.jl
mv simulate-zou2019-lba-2.jl ../results
```

It turns out that these simulations are wrong, because in Zou2020 they do no sample b,a,c from a distribution. Instead, they simply fix them.

So, I will delete the lba files, and change the scripts.

In total, there are the following 120 cases:
- b=0.1, 0.2, 0.5, 1 (4)
- a= 2b, 5b, 10b, 20b, 40b (5)
- c=0.01b, 0.02b, 0.05b, 0.1b, 0.2b, 0.5b, b (6)

We will only do the following 27 cases:
- b=0.1, 0.5, 1
- a=2b, 10b, 40b
- c=0.01b, 0.1b, b

First, I need to create all the folders so that they can run in parallel: `simulations-lba-?`.

We will run nrep=8000 and 5 cores in mac desktop (which is the limit that it can run simultaneously without running out of memory)
```shell
## b=0.1, a=2b, c=0.01b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-1
julia simulate-zou2019.jl 4738282 8000 0.1 2 0.01

## b=0.1, a=2b, c=0.1b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-2
julia simulate-zou2019.jl 68113228 8000 0.1 2 0.1

## b=0.1, a=2b, c=b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-3
julia simulate-zou2019.jl 68163228 8000 0.1 2 1.0

## b=0.1, a=10b, c=0.01b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-4
julia simulate-zou2019.jl 113683228 8000 0.1 10 0.01

## b=0.1, a=10b, c=0.1b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-5
julia simulate-zou2019.jl 68326728 8000 0.1 10 0.1
```
Started 6/20 9pm, 6am

We summarize the files:
```shell
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019

## b=0.1, a=2b, c=0.01b
cd simulations-lba-1
tar -czvf simulations-lba-1-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-1-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-1-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-1.h5
mv matrices.h5 matrices-lba-1.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-1.jl
mv simulate-zou2019-lba-1.jl ../results

## b=0.1, a=2b, c=0.1b
cd simulations-lba-2
tar -czvf simulations-lba-2-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-2-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-2-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-2.h5
mv matrices.h5 matrices-lba-2.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-2.jl
mv simulate-zou2019-lba-2.jl ../results

## b=0.1, a=2b, c=b
cd simulations-lba-3
tar -czvf simulations-lba-3-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-3-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-3-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-3.h5
mv matrices.h5 matrices-lba-3.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-3.jl
mv simulate-zou2019-lba-3.jl ../results

## b=0.1, a=10b, c=0.01b
cd simulations-lba-4
tar -czvf simulations-lba-4-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-4-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-4-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-4.h5
mv matrices.h5 matrices-lba-4.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-4.jl
mv simulate-zou2019-lba-4.jl ../results

## b=0.1, a=10b, c=0.1b
cd simulations-lba-5
tar -czvf simulations-lba-5-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-5-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-5-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-5.h5
mv matrices.h5 matrices-lba-5.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-5.jl
mv simulate-zou2019-lba-5.jl ../results
```

```shell
## b=0.1, a=10b, c=b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-6
julia simulate-zou2019.jl 18683228 8000 0.1 10 1.0

## b=0.1, a=40b, c=0.01b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-7
julia simulate-zou2019.jl 976683228 8000 0.1 40 0.01

## b=0.1, a=40b, c=0.1b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-8
julia simulate-zou2019.jl 2325654 8000 0.1 40 0.1

## b=0.1, a=40b, c=b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-9
julia simulate-zou2019.jl 372783 8000 0.1 40 1.0

## b=0.5, a=2b, c=0.01b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-10
julia simulate-zou2019.jl 58583625 8000 0.5 2 0.01
```
Started 6/21 10:30am, 7:40pm

We summarize the files:
```shell
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019

## b=0.1, a=10b, c=b
cd simulations-lba-6
tar -czvf simulations-lba-6-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-6-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-6-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-6.h5
mv matrices.h5 matrices-lba-6.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-6.jl
mv simulate-zou2019-lba-6.jl ../results

## b=0.1, a=40b, c=0.01b
cd simulations-lba-7
tar -czvf simulations-lba-7-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-7-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-7-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-7.h5
mv matrices.h5 matrices-lba-7.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-7.jl
mv simulate-zou2019-lba-7.jl ../results

## b=0.1, a=40b, c=0.1b
cd simulations-lba-8
tar -czvf simulations-lba-8-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-8-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-8-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-8.h5
mv matrices.h5 matrices-lba-8.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-8.jl
mv simulate-zou2019-lba-8.jl ../results

## b=0.1, a=40b, c=b
cd simulations-lba-9
tar -czvf simulations-lba-9-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-9-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-9-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-9.h5
mv matrices.h5 matrices-lba-9.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-9.jl
mv simulate-zou2019-lba-9.jl ../results

## b=0.5, a=2b, c=0.01b
cd simulations-lba-10
tar -czvf simulations-lba-10-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-10-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-10-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-10.h5
mv matrices.h5 matrices-lba-10.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-10.jl
mv simulate-zou2019-lba-10.jl ../results
```


```shell
## b=0.5, a=2b, c=0.1b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-11
julia simulate-zou2019.jl 5722724 8000 0.5 2 0.1

## b=0.5, a=2b, c=b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-12
julia simulate-zou2019.jl 4919173 8000 0.5 2 1.0

## b=0.5, a=10b, c=0.01b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-13
julia simulate-zou2019.jl 4728283 8000 0.5 10 0.01

## b=0.5, a=10b, c=0.1b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-14
julia simulate-zou2019.jl 4473421 8000 0.5 10 0.1

## b=0.5, a=10b, c=b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-15
julia simulate-zou2019.jl 976422 8000 0.5 10 1.0
```
Started 6/21 9pm, finished 6:30am

We summarize the files:
```shell
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019

## b=0.5, a=2b, c=0.1b
cd simulations-lba-11
tar -czvf simulations-lba-11-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-11-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-11-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-11.h5
mv matrices.h5 matrices-lba-11.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-11.jl
mv simulate-zou2019-lba-11.jl ../results

## b=0.5, a=2b, c=b
cd simulations-lba-12
tar -czvf simulations-lba-12-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-12-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-12-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-12.h5
mv matrices.h5 matrices-lba-12.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-12.jl
mv simulate-zou2019-lba-12.jl ../results

## b=0.5, a=10b, c=0.01b
cd simulations-lba-13
tar -czvf simulations-lba-13-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-13-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-13-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-13.h5
mv matrices.h5 matrices-lba-13.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-13.jl
mv simulate-zou2019-lba-13.jl ../results

## b=0.5, a=10b, c=0.1b
cd simulations-lba-14
tar -czvf simulations-lba-14-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-14-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-14-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-14.h5
mv matrices.h5 matrices-lba-14.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-14.jl
mv simulate-zou2019-lba-14.jl ../results

## b=0.5, a=10b, c=b
cd simulations-lba-15
tar -czvf simulations-lba-15-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-15-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-15-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-15.h5
mv matrices.h5 matrices-lba-15.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-15.jl
mv simulate-zou2019-lba-15.jl ../results
```

```shell
## b=0.5, a=40b, c=0.01b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-16
julia simulate-zou2019.jl 416173 8000 0.5 40 0.01

## b=0.5, a=40b, c=0.1b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-17
julia simulate-zou2019.jl 3615253 8000 0.5 40 0.1

## b=0.5, a=40b, c=b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-18
julia simulate-zou2019.jl 467733 8000 0.5 40 1.0

## b=1, a=2b, c=0.01b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-19
julia simulate-zou2019.jl 675223 8000 1 2 0.01

## b=1, a=2b, c=0.1b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-20
julia simulate-zou2019.jl 7842344 8000 1 2 0.1
```
Started 6/22 830am, finish 6:30pm

We summarize the files:
```shell
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019

## b=0.5, a=40b, c=0.01b
cd simulations-lba-16
tar -czvf simulations-lba-16-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-16-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-16-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-16.h5
mv matrices.h5 matrices-lba-16.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-16.jl
mv simulate-zou2019-lba-16.jl ../results

## b=0.5, a=40b, c=0.1b
cd simulations-lba-17
tar -czvf simulations-lba-17-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-17-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-17-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-17.h5
mv matrices.h5 matrices-lba-17.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-17.jl
mv simulate-zou2019-lba-17.jl ../results

## b=0.5, a=40b, c=b
cd simulations-lba-18
tar -czvf simulations-lba-18-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-18-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-18-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-18.h5
mv matrices.h5 matrices-lba-18.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-18.jl
mv simulate-zou2019-lba-18.jl ../results

## b=1, a=2b, c=0.01b
cd simulations-lba-19
tar -czvf simulations-lba-19-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-19-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-19-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-19.h5
mv matrices.h5 matrices-lba-19.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-19.jl
mv simulate-zou2019-lba-19.jl ../results

## b=1, a=2b, c=0.1b
cd simulations-lba-20
tar -czvf simulations-lba-20-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-20-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-20-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-20.h5
mv matrices.h5 matrices-lba-20.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-20.jl
mv simulate-zou2019-lba-20.jl ../results
```


```shell
## b=1, a=2b, c=b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-21
julia simulate-zou2019.jl 88422 8000 1 2 1.0

## b=1, a=10b, c=0.01b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-22
julia simulate-zou2019.jl 1346243 8000 1 10 0.01

## b=1, a=10b, c=0.1b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-23
julia simulate-zou2019.jl 3363123 8000 1 10 0.1

## b=1, a=10b, c=b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-24
julia simulate-zou2019.jl 114134 8000 1 10 1.0

## b=1, a=40b, c=0.01b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-25
julia simulate-zou2019.jl 3245235 8000 1 40 0.01
```
Started 6/22 630pm, finished 4:30am

We summarize the files:
```shell
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019

## b=1, a=2b, c=b
cd simulations-lba-21
tar -czvf simulations-lba-21-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-21-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-21-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-21.h5
mv matrices.h5 matrices-lba-21.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-21.jl
mv simulate-zou2019-lba-21.jl ../results

## b=1, a=10b, c=0.01b
cd simulations-lba-22
tar -czvf simulations-lba-22-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-22-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-22-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-22.h5
mv matrices.h5 matrices-lba-22.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-22.jl
mv simulate-zou2019-lba-22.jl ../results

## b=1, a=10b, c=0.1b
cd simulations-lba-23
tar -czvf simulations-lba-23-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-23-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-23-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-23.h5
mv matrices.h5 matrices-lba-23.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-23.jl
mv simulate-zou2019-lba-23.jl ../results

## b=1, a=10b, c=b
cd simulations-lba-24
tar -czvf simulations-lba-24-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-24-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-24-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-24.h5
mv matrices.h5 matrices-lba-24.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-24.jl
mv simulate-zou2019-lba-24.jl ../results

## b=1, a=40b, c=0.01b
cd simulations-lba-25
tar -czvf simulations-lba-25-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-25-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-25-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-25.h5
mv matrices.h5 matrices-lba-25.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-25.jl
mv simulate-zou2019-lba-25.jl ../results
```


```shell
## b=1, a=40b, c=0.1b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-26
julia simulate-zou2019.jl 45435 8000 1 40 0.1

## b=1, a=40b, c=b
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-27
julia simulate-zou2019.jl 346266 8000 1 40 1.0
```
Started 6/23 9am, finished 4pm.

We summarize the files:
```shell
cd Dropbox/Sharing/projects/leo-nn/nn-phylogenetics/simulations-zou2019

## b=1, a=40b, c=0.1b
cd simulations-lba-26
tar -czvf simulations-lba-26-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-26-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-26-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-26.h5
mv matrices.h5 matrices-lba-26.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-26.jl
mv simulate-zou2019-lba-26.jl ../results

## b=1, a=40b, c=b
cd simulations-lba-27
tar -czvf simulations-lba-27-1.tar.gz rep-1*
rm rep-1*
tar -czvf simulations-lba-27-2.tar.gz rep-2*
rm rep-2*
tar -czvf simulations-lba-27-3.tar.gz rep-*
rm rep-*
mv labels.h5 labels-lba-27.h5
mv matrices.h5 matrices-lba-27.h5
mv *.h5 ../results
mv *.tar* ../results
mv simulate-zou2019.jl simulate-zou2019-lba-27.jl
mv simulate-zou2019-lba-27.jl ../results
```

We move all the results to `v3`:
```shell
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/results
mv *.h5 v3
mv *.jl v3
mv *.gz v3
```

### Adding more simulations to LBA cases

We noticed that we do not have the same performance as in the Zou2019 paper. It could be that we are using a much smaller sample size (8000 vs 100k in Zou2019).

In total, there are the following 120 cases:
- b=0.1, 0.2, 0.5, 1 (4)
- a= 2b, 5b, 10b, 20b, 40b (5)
- c=0.01b, 0.02b, 0.05b, 0.1b, 0.2b, 0.5b, b (6)

We will only do the following 27 cases:
- b=0.1, 0.5, 1
- a=2b, 10b, 40b
- c=0.01b, 0.1b, b

First, I need to create all the folders so that they can run in parallel: `simulations-lba-?`:
```shell
for i in {1..27}
do
mkdir simulations-lba-$i
done

for i in {1..27}
do
cp scripts/* simulations-lba-$i/
done
```

We will run nrep=100000 and 10 cores in mac desktop (since we are not saving the onehot matrices, I think we can use more cores):
```shell
## b=0.1, a=2b, c=0.01b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-1
julia simulate-zou2019.jl 4738282 100000 0.1 2 0.01 0

## b=0.1, a=2b, c=0.1b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-2
julia simulate-zou2019.jl 68113228 100000 0.1 2 0.1 0

## b=0.1, a=2b, c=b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-3
julia simulate-zou2019.jl 68163228 100000 0.1 2 1.0 0

## b=0.1, a=10b, c=0.01b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-4
julia simulate-zou2019.jl 113683228 100000 0.1 10 0.01 0

## b=0.1, a=10b, c=0.1b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-5
julia simulate-zou2019.jl 68326728 100000 0.1 10 0.1 0

## b=0.1, a=10b, c=b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-6
julia simulate-zou2019.jl 18683228 100000 0.1 10 1.0 0

## b=0.1, a=40b, c=0.01b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-7
julia simulate-zou2019.jl 976683228 100000 0.1 40 0.01 0

## b=0.1, a=40b, c=0.1b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-8
julia simulate-zou2019.jl 2325654 100000 0.1 40 0.1 0

## b=0.1, a=40b, c=b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-9
julia simulate-zou2019.jl 372783 100000 0.1 40 1.0 0

## b=0.5, a=2b, c=0.01b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-10
julia simulate-zou2019.jl 58583625 100000 0.5 2 0.01 0
```
Started 8/2 9pm, finished 1am.


```shell
## b=0.5, a=2b, c=0.1b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-11
julia simulate-zou2019.jl 5722724 100000 0.5 2 0.1 0

## b=0.5, a=2b, c=b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-12
julia simulate-zou2019.jl 4919173 100000 0.5 2 1.0 0

## b=0.5, a=10b, c=0.01b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-13
julia simulate-zou2019.jl 4728283 100000 0.5 10 0.01 0

## b=0.5, a=10b, c=0.1b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-14
julia simulate-zou2019.jl 4473421 100000 0.5 10 0.1 0

## b=0.5, a=10b, c=b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-15
julia simulate-zou2019.jl 976422 100000 0.5 10 1.0 0

## b=0.5, a=40b, c=0.01b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-16
julia simulate-zou2019.jl 416173 100000 0.5 40 0.01 0

## b=0.5, a=40b, c=0.1b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-17
julia simulate-zou2019.jl 3615253 100000 0.5 40 0.1 0

## b=0.5, a=40b, c=b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-18
julia simulate-zou2019.jl 467733 100000 0.5 40 1.0 0

## b=1, a=2b, c=0.01b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-19
julia simulate-zou2019.jl 675223 100000 1 2 0.01 0

## b=1, a=2b, c=0.1b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-20
julia simulate-zou2019.jl 7842344 100000 1 2 0.1 0

## b=1, a=2b, c=b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-21
julia simulate-zou2019.jl 88422 100000 1 2 1.0 0

## b=1, a=10b, c=0.01b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-22
julia simulate-zou2019.jl 1346243 100000 1 10 0.01 0
```
Started 8/3 12pm, finished 4pm.


```shell
## b=1, a=10b, c=0.1b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-23
julia simulate-zou2019.jl 3363123 100000 1 10 0.1 0

## b=1, a=10b, c=b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-24
julia simulate-zou2019.jl 114134 100000 1 10 1.0 0

## b=1, a=40b, c=0.01b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-25
julia simulate-zou2019.jl 3245235 100000 1 40 0.01 0

## b=1, a=40b, c=0.1b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-26
julia simulate-zou2019.jl 45435 100000 1 40 0.1 0

## b=1, a=40b, c=b
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/simulations-lba-27
julia simulate-zou2019.jl 346266 100000 1 40 1.0 0
```
Started 8/3 830am, finish 1230pm

```shell
for i in {1..27}
do
cp simulations-lba-$i/*.in results
done

cd results
ls *.in | wc -l  ##54
```

Copy the *.in files to shared google drive.
Folders stored in `v4` for now.

Dimensions of files:
```shell
(master) $ awk '{ print length }' sequences976422-0.5-10.0-1.0.in | head
1550
1550
1550
1550
1550
1550
1550
1550
1550
1550
(master) $ wc -l sequences976422-0.5-10.0-1.0.in 
  400000 sequences976422-0.5-10.0-1.0.in
```


# Notes from new Leo code (meeting 2/18)

- Code is for 4 taxa only and it follows more closely the notation on the overleaf doc
- In the old code, we had `DescriptorModule` as \phi and `MergeModule` as \Phi so that the pipeline would be from a 1550x20x4 tensor to 1550x20 to 250x20 then into MergeModule to 50x20 and then into MergeModule2 to a score
- The new code had the pipeline of 1550x20 to a vector of 128 to another vector of 128 and then to a score
       - `NonlinearEmbedding`: \phi
       - `NonlinearMergeEmbedding`: \Phi
       - `NonlinearScoreEmbedding`: \Psi
- This new code does not work and Leo suspects it is because we are losing some of the geometry of the sequence
- An alternative version goes from 1550x20 to Matrix mxc (via \phi) then to Matrix mxc again (via \Phi) and finally to score via \Psi
- This new version works better as it seems to keep the geometry of the input
- The files `gen_json_files.py` and `gen_sh_files.py` provide automatic means for tests on slurm
- We want to start exploring extension to 5 or 6 taxa for the paper


# Comparing NN performance to standard phylogenetics inference

## 1. Extracting sample data
We will grab one 4-taxon dataset from a randomly chosen file: `sequences45435-1.0-40.0-0.1.in` and `labels45435-1.0-40.0-0.1.in`.

In the sequence file, we have a 400,000 x 1550 matrix in which we have 100,000 4-taxon datasets. We will grab the last 4 rows which correspond to one 4-taxon dataset.

```shell
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/results
wc -l sequences45435-1.0-40.0-0.1.in  ## 400000
sed -n '399997,400000 p' sequences45435-1.0-40.0-0.1.in > test.in
```

The order of the sequences are always S1,S2,S3,S4.

## 2. Convert sample data to Fasta file

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

## 3. Fitting maximum parsimony (MP) and neighbor-joining (NJ) in R

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

## 4. Fitting maximum likelihood (ML) with RAxML

1. Download `raxml-ng` from [here](https://github.com/amkozlov/raxml-ng). You get a zipped folder: `raxml-ng_v1.0.2_macos_x86_64` which I placed in my `software` folder

2. Checking the version
```shell
cd Dropbox/software/raxml-ng_v1.0.2_macos_x86_64/
./raxml-ng -v
```

3. Infer the ML tree using the same model as in the simulations
```shell
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/results
~/Dropbox/software/raxml-ng_v1.0.2_macos_x86_64/raxml-ng --msa test.fasta --model Dayhoff --prefix T3 --threads 2 --seed 616

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

## 5. Fitting bayesian inference (BI) with MrBayes

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
cd Dropbox/Sharing/projects/present/leo-nn/nn-phylogenetics/simulations-zou2019/results
mb test-mb.nexus
```

The estimated tree is in `test-mb.nexus.con.tre`.

## 6. Comparing the estimated trees to the true tree

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


# Real data analysis: Zika virus

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

## Creating files with 4 taxa

Our dataset has 7 species, so we need to create subsets of 4 to fit in our NN.

```julia
data = readlines("data/FASTA.fa")

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
CSV.write("data/fasta-table.csv",df)
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
   io = open(string("data/zika-fasta",i,".fa"), "w")
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

### Removing 7th taxon

The 7th taxon (see `FASTA.fa`) is "BBA85762, 3423, Homo sapiens, Japan, 2016" which contains missing sites X. Because our NN model was not trained with missing sites, we will remove this taxon from the dataset.


## Quartet puzzling

We can do the quartet puzzling step with Quartet Max Cut. It needs an input file that is one line with each quartet separated by a space in the form of a split: "1,2|3,4".

See `final_plots.Rmd` for the plot of the tree.