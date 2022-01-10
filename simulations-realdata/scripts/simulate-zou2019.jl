## Simulate quartets and aminoacid sequences as in Zou2019
## Using PAML, see notebook.md
## Note that this simulations is hard-coded for quartets only
## (cannot be extended to any size of tree)
## Claudia March 2020

using PhyloNetworks, Random, Distributions, Flux, HDF5
include("functions-zou2019.jl")

## Input
rseed = 03011058
L = 1550 ## sequence length
ratealpha = 0.1580288 ## see notebook.md
model = 3 ##for control file PAML
modeldatfile = "dayhoff.dat"
blL = 0.02 ##lower bound unif for BL
blU = 1.02 ##upper bound unif for BL
##long branch attraction case -------------------
lba = true
b = 0.1 ##[0.1,1.0]
rab = 2 ##a=[2b,40b]
rcb = 0.01 ##c=[0.01b,b]
## ----------------------------------------------
nrep = 5000
onehot = false ## convert seqs to one-hot encoding?

if length(ARGS) > 0
    rseed = parse(Int,ARGS[1])
    nrep = parse(Int,ARGS[2])
    if length(ARGS) > 2
        b = parse(Float64,ARGS[3])
        rab = parse(Float64,ARGS[4])
        rcb = parse(Float64,ARGS[5])
    end
    if length(ARGS) > 5
        onehot = convert(Bool,parse(Int,ARGS[6])) ## ARGS[6] must be 0/1
    end
end

Random.seed!(rseed);
seeds = sample(1:5555555555,nrep)
makeOdd!(seeds) ## we need odd seed for PAML

if onehot
    labels = zeros(nrep)
    matrices = zeros(L)
end

outfilelab = string("labels",rseed,".in")
outfileseq = string("sequences",rseed,".in")
if lba
    outfilelab = string("labels",rseed,"-",b,"-",rab,"-",rcb,".in")
    outfileseq = string("sequences",rseed,"-",b,"-",rab,"-",rcb,".in")
end
f = open(outfilelab,"w")


for i in 1:nrep
    println("=====================================================")
    @show i
    app = i == 1 ? false : true
    tree,ind = sampleRootedMetricQuartet(blL,blU, seeds[i], lba=lba, b=b, rab=rab, rcb=rcb)

    if onehot
        labels[i] = ind
    else
        global f
        write(f,string(ind))
        write(f,"\n")
    end
    namectl = string("rep-",i,".dat")
    createCtlFile(namectl, tree, seeds[i], L, ratealpha, model, modeldatfile)
    run(`./evolver 7 MCaa.dat`)
    if onehot
        run(`cp mc.paml rep-$i.paml`)
        mat = convert2onehot("mc.paml",L)
        global matrices
        matrices = hcat(matrices,mat)
    else
        global app
        writeSequence2File("mc.paml",L,outfileseq,append=app)
        run(`rm mc.paml`)
        run(`rm rep-$i.dat`)
    end
end
close(f)

if onehot
    matrices = matrices'
    matrices = matrices[2:end,:] ## 80*nrep by L: each replicate has a 80xL matrix, all stacked

    h5write("labels.h5","labels",labels)
    h5write("matrices.h5","matrices",matrices)
end

