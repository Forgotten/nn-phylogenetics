## to make seeds odd because PAML needs odd seeds
function makeOdd!(seeds::Vector{Int64})
    mod = seeds .% 2
    seeds[mod.==0]=seeds[mod.==0] .+ 1
end

## create the control file for PAML
## to simulate aminoacid sequences on a given tree
## note: tree is assumed to be a quartet
## name: name of the output control file (note that the control file
##       needs to be called MCaa.dat, but we will save also as name
##       to avoid overwriting
## tree: tree (quartet, rooted, with bl)
## s: random seed
## L: sequence length
## alpha: rate for gamma
## model: see notebook.md for model options for PAML
## modelfile: we chose one file from the ones provided by PAML with the S matrix
## n= number of leaves
function createCtlFile(name::String, tree::String, s::Integer, L::Integer, alpha::Float64,
                       model::Integer, modelfile::String; n=4::Integer)
    str = """0        * 0: paml format (mc.paml); 1:paup format (mc.nex)
$s       * random number seed (odd number)

$n $L 1   * <# seqs>  <# sites>  <# replicates>

-1         * <tree length, use -1 if tree below has absolute branch lengths>

$tree

$alpha 1        * <alpha; see notes below>  <#categories for discrete gamma>
$model $modelfile * <model> [aa substitution rate file, need only if model=2 or 3]

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
"""
    outfile = open(name,"w")
    write(outfile, str)
    close(outfile)

    ## we need to name the file MCaa.dat
    outfile2 = open("MCaa.dat","w")
    write(outfile2, str)
    close(outfile2)
end


### simulate a quartet with branch lengths uniformly dist
### U(l,u)
## ss: seed
## lba=false by default: long branch attraction case, based on Figure 3 Zou2020:
## two short external branches have lengths b ranging from 0.1 to 1.0,
## the two long branches have lengths a ranging from 2b to 40b,
## and the internal branch has a length c ranging from 0.01b to b
## Root chosen randomly
## the random root was removed to implement the permutation
## strategy in Zou2019
function sampleRootedMetricQuartet(l::Number,u::Number, ss::Integer;
                                   lba=false::Bool, b=0.01::Float64, rab=2.0::Float64, rcb=0.01::Float64)
    Random.seed!(ss)
    quartets = ["((1,2),(3,4));", "((1,3),(2,4));", "((1,4),(2,3));"] ##only thinking of unrooted
    ind = sample(1:3,1)[1]
    q = quartets[ind]
    quartet = lba ? readTopologyLevel1(q) : readTopology(q)
    ## setting branch lenghts
    if lba
        c = rcb*b
        a = rab*b
        ## internal branch:
        intbl = findall([e.istIdentifiable for e in quartet.edge])[1]
        setLength!(quartet.edge[intbl],c)
        edges = [quartet.edge[intbl]]
        ## external edges:
        extbls = setdiff(1:length(quartet.edge),intbl)
        ## first external edge, set BL=b:
        setLength!(quartet.edge[extbls[1]],b)
        push!(edges, quartet.edge[extbls[1]])
        adje = adjacentedges(quartet.edge[extbls[1]])
        ## we find the sister external edge, set BL=a:
        for e in adje
            if e!= quartet.edge[extbls[1]] && !e.istIdentifiable
                e.length = a
                push!(edges,e)
            end
        end
        misse = setdiff(quartet.edge, edges)
        ## second external edge, set BL=b:
        setLength!(misse[1],b)
        misse2 = setdiff(misse, [misse[1]])
        ## sister set BL=a
        misse2[1].length = a
        ## rooting tree to match other simulations:
        rootonedge!(quartet,quartet.edge[intbl])
    else
        for e in quartet.edge
            setLength!(e,rand(Uniform(l,u),1)[1])
        end
    end
    return writeTopology(quartet),ind
end


## converts the file of sequences into one hot encoded tensor
## e.g. if file has 4 sequences of aminoacid (20 letters) of length L
## this function should return a tensor of 4x20xL where each letter is represented
## by its one-hot code: l1->000...01
## BUT in Zou2019 they then vectorize the tensor into matrix (e.g. 80xL),
## so we will return the matrix directly
## we need the length (L) to initialize the matrix
## note: it assumes phylip format so that the first row is "n L"
function convert2onehot(name::String, L::Integer)
    mc = readlines(name)
    mc = mc[mc .!= ""]
    mat = zeros(L,1) ##output matrix
    ## assumes phylip format so that first row is "n L"
    ## we do i=2 first to have the same alll vector
    i=2
    mm = split(mc[i])
    @show mm[1]
    seq = join(mm[2:end])
    alll = unique(seq)  ##e.g. the 20 aminoacids
    m = zeros(1,length(alll)) ## output matrix
    for s in seq
        m = [m;Flux.onehot(s,alll)']
    end
    m = m[2:end,:]
    mat = hcat(mat,m)

    for i in 3:length(mc)
        mm = split(mc[i])
        @show mm[1]
        seq = join(mm[2:end])
        m = zeros(1,length(alll)) ## output matrix
        for s in seq
            m = [m;Flux.onehot(s,alll)']
        end
        m = m[2:end,:]
        mat = hcat(mat,m)
    end
    mat = mat[:,2:end]
    return mat
end



### simulate a quintet with branch lengths uniformly dist
### U(l,u)
## ss: seed
### Rooted chosen randomly
## Note that we do not simulate the tree with coalescent (as rcoal)
## to make it easier to keep the "index" of the tree
function sampleRootedMetricQuintet(l::Number,u::Number, ss::Integer)
    Random.seed!(ss)
    perm = validPermutations()
    quintet = readTopology("(((t1,t2),(t3,t4)),t5);")
    ind = sample(1:length(perm),1)[1]
    renameTips!(quintet, string.(perm[ind]))
    ## choose root randomly (or leave as is=balanced tree)
    r = rand(Uniform(0,1),1)[1]
    if(r<0.2)
        rootatnode!(quintet,"1")
    elseif(r<0.4)
        rootatnode!(quintet,"2")
    elseif(r<0.6)
        rootatnode!(quintet,"3")
    elseif(r<0.8)
        rootatnode!(quintet,"4")
    end
    ## setting branch lenghts
    for e in quintet.edge
        setLength!(e,rand(Uniform(l,u),1)[1])
    end
    return writeTopology(quintet),ind
end



## rename one tip in HybridNetworks object
function renameTip!(net::HybridNetwork, oldtip::String, newtip::String)
    ind = findall([n.name for n in net.node] .== oldtip)
    length(ind) > 0 || error("$oldtip not found in network")
    length(ind) < 2 || error("$oldtip found in multiple leaves in network")
    net.node[ind[1]].name = newtip
end


## rename tips in HybridNetworks object
## oldtip: vector with old tips to change in the order of newtip vector
function renameTips!(net::HybridNetwork, oldtip::Vector{String}, newtip::Vector{String})
    for i in 1:length(oldtip)
        renameTip!(net, oldtip[i], newtip[i])
    end
end


## rename tips in HybridNetworks object
## in this version, you do not give the vector of old names
## you simply change the tip names in the order that the names appear in net
renameTips!(net::HybridNetwork, newtip::Vector{String}) = renameTips!(net, net.names, newtip)


## manually created list of valid permutations
## this only works for quintets
function validPermutations()
    perm = [
    [1,2,3,4,5],
[1,2,3,5,4],
[1,2,4,5,3],
[1,3,2,4,5],
[1,3,2,5,4],
[1,3,4,5,2],
[1,4,2,3,5],
[1,4,2,5,3],
[1,4,3,5,2],
[1,5,2,3,4],
[1,5,2,4,3],
[1,5,3,4,2],
[2,3,4,5,1],
[2,4,3,5,1],
        [2,5,3,4,1]]
    return perm
end

## from master PhyloNetworks but can't update the package now
function adjacentedges(centeredge::PhyloNetworks.Edge)
    n = centeredge.node
    length(n) == 2 || error("center edge is connected to $(length(n)) nodes")
    @inbounds edges = copy(n[1].edge) # shallow copy, to avoid modifying the first node
    @inbounds for ei in n[2].edge
        ei === centeredge && continue # don't add the center edge again
        # a second edge between nodes n[1] and n[2] would appear twice
        push!(edges, ei)
    end
    return edges
end


## function to read the simulated sequences from PAML
## and write without taxon names for the NN
## sequences saved to outfile: 4 x L
## append=true means that the file already exists and we append the
## sequences to the existing file
function writeSequence2File(name::String, L::Integer, outfile::String ; append=true::Bool)
    if append
        f = open(outfile, "a")
    else
        f = open(outfile, "w")
    end
    mc = readlines(name)
    mc = mc[mc .!= ""]
    ## assumes phylip format so that first row is "n L"
    ## we do i=2 first to have the same alll vector
    for i in 2:length(mc)
        mm = split(mc[i])
        @show mm[1]
        seq = join(mm[2:end])
        write(f,seq)
        write(f,"\n")
    end
    close(f)
end



