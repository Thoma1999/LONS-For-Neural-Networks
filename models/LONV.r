# Check if required packages are installed or not. Install if required
packages <- c("igraph", "rgl")
if (length(setdiff(packages, rownames(installed.packages()))) > 0) {
  install.packages(setdiff(packages, rownames(installed.packages())))  
}
# Load required packages
library(igraph)  # Network analysis and visualisation
library(rgl)     # 3D plots
# Location of data files. Add here the path to your data files
path <- "/Users/larye/Desktop/grandfinal/Local-optima-networks-for-NNs/models"  # insert the path to your files here
setwd(path)  # set working directory


instance <- "Schwefel26-8D"




# Default colours for node and edges 
# Node coloring for global and local sinks and other optima
colgs <- "#4f0203"       # Color of global sinks (red)
colls <- "#003f8c"       # Color of local (i.e non-global) sinks (blue)
collo <- "#64acd9"       # Color for all other optima (gray)
colgc <- "#f06e70"
coled <- "#1f2021"


# Functions
Neutrality<-function(nodecount, neutralcount) {
  return((nodecount-neutralcount)/nodecount)
}

# ----------------------------------------------------------------------------------------
# Width of edges based on their weight
edgeWidth<-function(N, minw, maxw) {
  ewidth <- 1
  if (ecount(N) > 0) {
    ewidth <- (maxw * E(N)$weight)/max(E(N)$weight)
    ewidth = ifelse(ewidth > minw, ewidth, minw)
    ewidth = ifelse(ewidth < maxw, ewidth, maxw)
  }
  return (ewidth)
}

#----------------------------------------------------------------------------------------------------------------------------
# Plot Network in 2D. Either in the screen or as pdf file (if bpdf is True) 
# N:      Graph object
# tit:    String describing network, instance and type of LON
# ewidth: edge  width 
# asize:  arrow size for plots
# ecurv:  curvature of the edges (0 = non, 1 = max)
# mylay:  graph layout as a parameter
# bpdf:   boolean TRUE for generating a pdf

plotNet <-function(N, tit, nsize, ewidth, asize, ecurv, mylay, bpdf) {
  inst <-strsplit(instance, "\\_")[[1]][1]  # simplify instance name taking first part of name
  if (bpdf)  { # PDF for saving the 2D plot
    ofname <- paste0(inst, tit, '.pdf')
    pdf(ofname) 
    print(ofname)
  }
  vframecol <- ifelse(V(N)$fitness == best, "black", "gray50") # darker outline for global optima

  title <- paste(inst, tit,'Nodes:',vcount(N), 'Edges:',ecount(N))
  #tkplot(N, layout = mylay)
  plot(N, layout = mylay, main = title, vertex.label = NA, vertex.size = nsize, vertex.frame.color = vframecol, edge.width = ewidth, edge.arrow.size = asize, edge.curved = ecurv)
  if (bpdf)
    dev.off()
}

#-----------------------------------------------------------------------------
# Plot network in 3D 
# N = Network 
# z: the z coordinate, normally fitness, but can have some scaling.
# ewidth: vector with edge widths
# asize: arrow size for plots
# mylayout: layout as a paremter so I can use the same for 2D and 3D

plotNet3D <-function(N, z, ewidth, asize, mylay) {
  mylayout3D <- cbind(mylay, z) # append z coordinate to the 2D layout
  rgl.open()      # open new window
  bg3d("white")   # set background to white 
  rglplot(N, layout = mylayout3D, edge.width = ewidth, edge.arrow.size = asize, vertex.label = NA)
}

#------------------------------------------------------------------------
# Creates a sub-network with nodes with and below a given fitness level 
# Keeping those nodes with a fitness below a percentile
# the higher the percentile the more nodes are kept.
# This function can be used to visualise large networks

pruneNodesFit <- function(N, perc) {
  # subset network nodes below fvalue fitness
  Top <- induced.subgraph(N, V(N)$fitness <= quantile(V(N)$fitness,perc))
  return (simplify(Top))
}

## Main -----------------------------------------------------------------------------------

## Read data from zip file and construct the network models
print(instance)
zipname <- paste0(instance, ".zip")
nodename <- paste0(instance, ".nodes")
edgename <- paste0(instance, ".edges")
edges <- read.table(unz(zipname, edgename), header = F, colClasses = c("integer", "integer", "integer"))
colnames(edges) <- c("start", "end", "weight")
nodes <- read.table(unz(zipname, nodename), header = F, colClasses = c("integer", "numeric", "integer"))
colnames(nodes) <- c("id", "fitness", "basin.size")

## Create LON from dataset of nodes and edges
MLON <- graph_from_data_frame(d = edges, directed = T, vertices = nodes)
MLON <- simplify(MLON, remove.multiple=TRUE, remove.loops=TRUE)  # Remove self loops
#MLON <- pruneNodesFit(MLON,0.9)
V(MLON)$fitness <- round(V(MLON)$fitness,2)
# since we are minimising, and is full enumeration best is the minimum fitness

## Network Models
# MLON: LON with keeping only monotonic sequences (non-deteriorating edges)
# CMLON: MLON with compressed meta-plateaus used to identify sinks

# get the list of edges and fitness values in order to filter 
el <- as_edgelist(MLON)
fits <- V(MLON)$fitness
names <- V(MLON)$name
best <- round(min(nodes$fitness),2)
cat("Global Optimum Value:", best,"\n")
global_opt <- V(MLON)[fitness == best]


# get the fitness values at each endpoint of an edge
f1 <- fits[match(el[,1], names)]
f2 <- fits[match(el[,2], names)]



# Coloring edges according to type
# Coloring nodes
V(MLON)$color <- collo  # default local optima 
V(MLON)$color[which(degree(MLON, mode="out") == 0)] <- colls # local sinks

V(MLON)$color[V(MLON) %in% subcomponent(MLON, global_opt, mode = "in")] <- colgc # color of optima connected to the global sink
V(MLON)$color[V(MLON)$fitness == best] <- colgs   # global sinks

E(MLON)$color <- coled # edge color

# Size of nodes in LON model is propotional to their incoming strength
V(MLON)$size <-  V(MLON)$basin.size
sinks_ids <-which(degree(MLON, mode = "out")==0)
start_ids <-which(degree(MLON, mode = "in")==0)
sinks_fit <- vertex_attr(MLON, "fitness")[sinks_ids]
# Compute funnel metrics: number of local and global sinks  
nsinks <- length(sinks_ids)     # Number of sinks 
nglobals <- length(global_opt)  # Number of global sinks

# More funnel metrics
# Incoming strength of global optima sinks / normalised by the total incoming strength of sinks
igs<-sinks_ids[sinks_fit==best]  # index of global sinks
ils<-sinks_ids[sinks_fit>best]   # index of local sinks -- might be empty 

sing <- sum(strength(graph = MLON, vids = igs, mode = "in", loops = F), na.rm = T)  # global sinks
sinl <- sum(strength(graph = MLON, vids = ils, mode = "in", loops = F), na.rm = T)  # local sinks

gstrength <- sing/(sing+sinl)   # normalised incoming strength of global sinks
lstrength <- sinl/(sing+sinl)   # normalised incoming strength of local sinks

#distMatrix <- distances(MLON, v=V(MLON)[sinks_ids], to=V(MLON)[global_opt[0]])
#print(V(MLON)[sinks_ids])
#istMatrix <- distance_table(MLON, directed = FALSE)

distances = sinks_fit-best
#print(distances)
Deviation <- sum(distances)/length(distances)

shortest_9to4 <- betweenness(MLON, v = global_opt[0])
print(shortest_9to4)


cluster_coeffecient <- transitivity(MLON,"global")

Page_ranks <- page_rank(MLON, "prpack", vids = global_opt, directed = TRUE, damping = 0.85, personalized = NULL,weights = NULL, options = NULL )



## Constructing the CMLON. Contract meta-neutral-networks (pleteaus) to single nodes
mlon_size <- V(MLON)$size  # Keep original size as it will be modified to construct the CMLON
V(MLON)$size<-1            # required to construct the CMLON, size will be agregaterd
# check connectivity within meta-plateaus, only merge minima that are connected
gnn <- subgraph.edges(MLON, which(f1 == f2), delete.vertices=FALSE)
# get the components that are connected at the same fitness level
nn_memb <- components(gnn, mode="weak")$membership
# contract neutral connected components saving cardinality into node size
CMLON <- contract.vertices(MLON, mapping = nn_memb, vertex.attr.comb = list(fitness = "first", size = "sum", "ignore"))
# The size of nodes is the aggregation/sum of nodes in the plateau
# remove self-loops and contract multiple edges 
CMLON <- simplify(CMLON, edge.attr.comb = list(weight="sum","ignore"), remove.multiple=TRUE, remove.loops=TRUE)

# identify sinks i.e nodes without outgoing edges 
sinks_ids <-which(degree(CMLON, mode = "out")==0)
sinks_fit <- vertex_attr(CMLON, "fitness")[sinks_ids]
best <- min(V(CMLON)$fitness)
global_opt <- V(CMLON)[fitness == best]

netural <- Neutrality(length(V(MLON)),length(V(CMLON)))


# Compute funnel metrics: number of local and global sinks  
nsinks <- length(sinks_ids)     # Number of sinks 
nglobals <- length(global_opt)  # Number of global sinks
# More funnel metrics
# Incoming strength of global optima sinks / normalised by the total incoming strength of sinks
igs<-sinks_ids[sinks_fit==best]  # index of global sinks
ils<-sinks_ids[sinks_fit>best]   # index of local sinks -- might be empty 
sing <- sum(strength(graph = CMLON, vids = igs, mode = "in", loops = F), na.rm = T)  # global sinks
sinl <- sum(strength(graph = CMLON, vids = ils, mode = "in", loops = F), na.rm = T)  # local sinks
gstrength <- sing/(sing+sinl)   # normalised incoming strength of global sinks
lstrength <- sinl/(sing+sinl)   # normalised incoming strength of local sinks


## Add colour to CMLON nodes and edges. 
V(CMLON)$color <- collo  # default colour for local optima  
V(CMLON)$color[V(CMLON) %in% sinks_ids] <- colls   # Colour of suboptimal sinks
for (g in global_opt) {
  V(CMLON)$color[V(CMLON) %in% subcomponent(CMLON, g, mode = "in")] <- colgc
}

V(CMLON)$color[V(CMLON)$fitness == best] <- colgs   # global sinks

E(CMLON)$color <- coled # edges are all improving in CMLON, so improving color used.

# Restoure MLON size, as it was modified to construct CMLON
V(MLON)$size <- mlon_size 
#Page_ranks <- page_rank(CMLON, "prpack", vids = global_opt, directed = TRUE, damping = 0.85, personalized = NULL,weights = NULL, options = NULL )



####  METRICS  #### && subcomponent(MLON, V(MLON)[fitness == best], mode = "out")
# Many metrics can be computed for networks, here we report some basic LON metrics 
# and funnel metrics (CMLON model) known to relate to search difficulty
print("Metrics of the MLON model")
cat("Number of global optima:", length(V(MLON)[fitness == best]), "\n")
cat("Clustering coeffeceient:", cluster_coeffecient,"\n")
sumofRanks <- Reduce("+",Reduce("+", Page_ranks[1]))
cat("page rank:", sumofRanks,"\n")

print("Metrics of the CMLON model")
cat("Number of nodes (local optima):", vcount(CMLON), "\n")
cat("Global optima:", best, "\n")
print("Number of Edges:")
cat("Total:", ecount(CMLON), "\n")

print("Funnel Metrics ")    
cat("Total number of sinks (funnels):", nsinks, "\n")
cat("Number of global sinks (funnels):", nglobals, "\n")
cat("Fitness of sinks:", sinks_fit, "\n")
cat("Normalised incoming strength of global sinks:", gstrength, "\n")
cat("Neutrality: ", netural, '\n')
cat("Deviation: ", Deviation, '\n')



#### VISUALISATION ####
nSS<-function(N, minsize, maxsize) {
  vsize <- maxsize
  if (ecount(N) > 0) {
    vsize =  2*graph.strength(N, mode="in")
    vsize =  ifelse(vsize > minsize, vsize, minsize)
    vsize =  ifelse(vsize < maxsize, vsize, maxsize)
  }
  return (vsize)
}



# MLON MODEL
mlonlay <- layout_nicely(MLON)
mlonlayKK <- layout_with_kk(MLON)
mlonlayDRL <- layout_with_drl(MLON)
mlonlayFR <- layout_with_fr(MLON)
ew <- edgeWidth(MLON, 0.5, 2)
V(MLON)$size <- log(V(MLON)$size+1)
V(MLON)$size <- nSS(MLON, 2, 6)
ns <- V(MLON)$size
plotNet(N = MLON, tit = "mlon", nsize = ns, ewidth = ew, asize = 0.1, ecurv = 0.4, mylay = mlonlay, bpdf = T)
zcoord <- V(MLON)$fitness
plotNet3D(N = MLON, z = zcoord, ewidth = ew, asize = 0.7, mylay = mlonlay) 


# CMLON MODEL
cmlonlay <- layout_nicely(CMLON)
cmlonlayKK <- layout_with_kk(CMLON)
cmlonlayDRL <- layout_with_drl(CMLON)
cmlonlayFR <- layout_with_fr(CMLON)
#cmlonlay <- layout_nicely(CMLON)
ew <- edgeWidth(CMLON, 0.5, 1)
#V(CMLON)$size <- nodeSizeStrength(CMLON)  
#V(CMLON)$size <- log(V(CMLON)$size+1)
V(CMLON)$size <- nSS(CMLON, 2, 6)
ns <- V(CMLON)$size
plotNet(N = CMLON, tit = "cmlon", nsize = ns, ewidth = ew, asize = 0.1, ecurv = 0.2, mylay = cmlonlay, bpdf = T)
zcoord <- V(CMLON)$fitness
plotNet3D(N = CMLON, z = zcoord, ewidth = ew, asize = 0.7, mylay = cmlonlay) 


