
import igraph as ig
import zipfile
from matplotlib.pyplot import title
import plotly.graph_objs as go
import pandas as pd
import os
import numpy as np
import pandas as pd
directory = './models/'

class lonBuilder:
    def __init__(self, instance, roundDp):
        file_path = os.path.join(directory, instance)
    
        with zipfile.ZipFile(file_path+'.zip') as zip:
            with zip.open(instance+'.nodes', mode='r') as file:
                nodes = pd.read_csv(file, sep=" ", header=None, dtype={'id': int,'fitness':float, 'size':int})
                nodes.columns = ["id", "fitness", 'size']
            with zip.open(instance+'.edges', mode = 'r') as file:
                edges = pd.read_csv(file, sep=" ", header=None,  dtype={'start': int,'end':int, 'weight':int})
                edges.columns = ["start", "end", 'weight']
            with zip.open('weights.npy', mode='r') as file:
                weights = np.load(file, allow_pickle='TRUE')
                weights = weights.item()
                weights = weights.values()
        nodes.insert(3, 'params', weights)


        fits = []
        g = ig.Graph.DataFrame(edges, directed=True, vertices=nodes)
        g.simplify(multiple=True, loops=True)
        g.es['weight'] = edges['weight']
        el = g.get_edgelist()
        fits = g.vs["fitness"]
        names = g.vs["name"]



        f1 = [fits[names.index(e)] for e in [e[0] for e in el]]
        f2 = [fits[names.index(e)] for e in [e[1] for e in el]]

        k = g.copy()

        k.vs["size"]=1
        gnn = k.subgraph_edges([c for c, i in enumerate(f1) if round(f1[c],roundDp)==round(f2[c],roundDp)], delete_vertices=False)
        nn_memb = gnn.components(mode="weak").membership
        k.contract_vertices(mapping = nn_memb, combine_attrs=dict(fitness="first", params="first", size='sum'))
        k.simplify(combine_edges = dict(weight='sum'))
        self.layout=k.layout('auto')

        self.cmlon = k
        self.mlon = g
        self.name = instance

        # Caluclate metrics

        # MLON - Success and Deviation
        mBest = min(self.mlon.vs["fitness"])
        mSink_fits = [ v["fitness"] for v in self.mlon.vs if v.outdegree() == 0 ]
        mSinks = [ v.index for v in self.mlon.vs if v.outdegree() == 0 ]
        mIdGlobalSinks = [sink for sink in mSinks if mSink_fits[mSinks.index(sink)] == mBest]
        mIdLocalSinks = [sink for sink in mSinks if mSink_fits[mSinks.index(sink)] > mBest]

        Fitnessdiff = 0
        for fit in mSink_fits:
            Fitnessdiff += np.absolute(fit - mBest)
        deviation = Fitnessdiff/len(mSink_fits)
        self.deviation = deviation

        #CMLON
        self.sink_fits = [ v["fitness"] for v in self.cmlon.vs if v.outdegree() == 0 ]
        sinks = [ v.index for v in self.cmlon.vs if v.outdegree() == 0 ]
        self.numSinks = len(sinks)
        self.globalOptimimum = min(self.cmlon.vs["fitness"])
        self.neutral = (self.mlon.vcount()-self.cmlon.vcount())/self.mlon.vcount()

        idGlobalSinks = [sink for sink in sinks if self.sink_fits[sinks.index(sink)] == self.globalOptimimum]
        idLocalSinks = [sink for sink in sinks if self.sink_fits[sinks.index(sink)] > self.globalOptimimum]
        self.numGlobalSinks = len(idGlobalSinks)

    
        globalSinkStrength = sum(self.cmlon.strength(idGlobalSinks, mode='in', loops=False, weights=None))
        localSinkStrength = sum(self.cmlon.strength(idLocalSinks, mode='in', loops=False, weights=None))
        try:
            self.gstrength = globalSinkStrength/(globalSinkStrength+localSinkStrength)
            self.lstrength = localSinkStrength/(globalSinkStrength+localSinkStrength)
        except:
            self.gstrength = "NA"
            self.lstrength = "NA"
        
        self.cmlon_sinks = sinks
        self.cmlon_global_sinks = idGlobalSinks


        self.weights = {}
        for id, weight in enumerate(self.cmlon.vs["params"]):
            self.weights[id] = weight

    def get_cmlon_weights(self):
        return self.weights

    
    def get_deviation(self):
        fits = []
        global_sink_id = self.mlon_global_sinks[0]
        for id, fit in enumerate(self.mlon.vs["fitness"]):
            if id == global_sink_id:
                global_sink_fit = -418.2
            fits.append(fit)
        Fitnessdiff = 0
        for fit in fits:
            difference = np.absolute(fit - global_sink_fit)
            Fitnessdiff += difference
        deviation = Fitnessdiff/len(fits)
        return deviation

    
    def print_metrics(self):
        print("-------METRICS-------")
        print("Global optimum: "+str(self.globalOptimimum))
        print("###########################################")
        print("Number of sinks: "+str(self.numSinks))
        print("Number of global sinks: "+str(self.numGlobalSinks))
        print("Fitness of sinks: "+ str(self.sink_fits))
        print("###########################################")
        print("Neutrality: "+ str(self.neutral))
        print("###########################################")
        print("Normalised incoming strength of global sinks: "+str(self.gstrength))
        print("Normalised incoming strength of local sinks: "+str(self.lstrength))
        print("###########################################")
        print("Number of nodes in CMLON: "+str(self.cmlon.vcount()))
        print("Number of nodes in MLON: "+str(self.mlon.vcount()))
        print("###########################################")
        print("Number of edges in CMLON: "+str(self.cmlon.ecount()))
        print("Number of edges in MLON: "+str(self.mlon.ecount()))
        print("###########################################")
        print("Mean deviation: "+str(self.deviation))
     

    def load(self, minSize, maxSize):
        fits = self.cmlon.vs["fitness"]
        N = len(fits)
        Xn=[self.layout[i][0] for i in range(N)]# x-coordinates of nodes
        Yn=[self.layout[i][1] for i in range(N)]# y-coordinates of nodes
        Zn = [fits[i] for i in range(N)]
        Xe=[]
        Ye=[]
        Ze=[]

        for e in self.cmlon.es:
            Xe+=[self.layout[e.tuple[0]][0], self.layout[e.tuple[1]][0], None]# x-coordinates of edge ends
            Ye+=[self.layout[e.tuple[0]][1],self.layout[e.tuple[1]][1], None] # y-coordinates of edge ends
            Ze+=[fits[e.tuple[0]], fits[e.tuple[1]] ,None]

        xtp = []
        ytp = []
        ztp = []
        for e in self.cmlon.es:
            xtp.append(0.5*(self.layout[e.tuple[0]][0]+self.layout[e.tuple[1]][0]))
            ytp.append(0.5*(self.layout[e.tuple[0]][1]+ self.layout[e.tuple[1]][1]))
            ztp.append(0.5*(fits[e.tuple[0]]+ fits[e.tuple[1]])) 
        
        # Node text and edge text
        ftext =  [f'fitness={f}, id={id}' for id,f in enumerate(self.cmlon.vs["fitness"])]
        etext = [f'weight={w}' for w in self.cmlon.es['weight']]

        # Calculate node=size
        node_sizes = self.cmlon.degree(mode='in', loops=False)
        new_sizes = []
        for node in node_sizes:
            node = node*10
            if node > maxSize:
                new_sizes.append(maxSize)
            elif node < minSize:
                new_sizes.append(minSize)
            else:
                new_sizes.append(node)
       
        monotonicEdges=go.Scatter3d(x=Xe, y=Ye, z=Ze, mode='lines', line=dict(color='rgb(125,125,125)', width=2), hoverinfo='none')

        optima=go.Scatter3d(x=Xn,y=Yn,z=Zn,mode='markers',name='actors',
            marker=dict(symbol='circle',
                size=new_sizes,
                color=Zn,
                colorscale='Inferno',
                line=dict(color='rgb(50,50,50)', width=0.5), colorbar=dict(title="Fitness",thickness=20, len=0.1)
                ),
            text=ftext, hoverinfo='text'
            )

        weights=go.Scatter3d(x=xtp, y=ytp, z=ztp, mode='markers', marker = dict(color='rgb(125,125,125)', size=2), text = etext, hoverinfo='text')

        axis=dict(showbackground=True,
            showline=False,
            zeroline=False,
            showgrid=True,
            showticklabels=False,
            title=None,

        )
        layout = go.Layout(
            title=self.name,
            width=1800,
            height=1800,
            showlegend=True,
            scene_aspectmode='cube',
            font = dict(size=18),
            scene=dict(
                xaxis=dict(axis),
                yaxis=dict(axis),
            ),
        margin=dict(
            t=100
        ),
        hovermode='closest',
        annotations=[
            dict(
            showarrow=False,
                text=self.name,
                xref=None,
                yref=None,
                x=0,
                y=0.1,
                xanchor='left',
                yanchor='bottom',
                font=dict(
                size=14
                )
                )
            ],    )
        data=[optima, monotonicEdges, weights]
        fig=go.Figure(data=data, layout=layout)
        fig.write_html(os.path.join(directory, self.name+'.html'), auto_open=True)










