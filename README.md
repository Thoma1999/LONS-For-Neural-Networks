# Local Optima Networks for Neural Networks
Generate local optima networks models on the loss landscape of neural networks


<p align="center">
  <img width="350" height="330" src="/docs/img/iris-relu-h1.jpg">
  <img width="350" height="330" src="/docs/img/iris-sigmoid-h3.jpg">
   <img width="450" height="300" src="/docs/img/showId.jpg">
</p>

This repository a modified implementation of the paper

> Jason Adair, Gabriela Ochoa, and Katherine M. Malan. [*Local optima networks for continuous fitness landscapes*](https://doi.org/10.1145/3319619.3326852). In Proceedings of the Genetic and Evolutionary Computation Conference Companion (GECCO '19), 2019.


## How to use
```python
# 1. Define Neural network parameters

input_size = 4
output_size = 3
activation = "tanh"
hidden_layer_width = 1
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
x_train, x_test, y_train, y_test = data.load_iris_data(seed)
model = build_iris_model(hidden_layer_width, input_size, output_size, activation)

# 2. Define neural network as function
myfunc = functions.function_factory(model, loss, x_train, y_train) # Pass x_test and y_test to build test landscape


# 3. Define LON sampler parameters
name = "tanh-iris-H1" # Define Name to save model as
dimensions = input_size * hidden_layer_width + output_size*hidden_layer_width # Define dimensions
bounds = [dimensions*(-1,1)] # Define evaluation boundries
num_runs = 100 # Define number of number of samples
bsteps = 10000  # Define Basin-hopping steps
step = 0.18 # Define step size
x_tol = 1e-2 # Define X tolerance (Nodes within a distance of X in all dimensions are considered as the same node)
f_tol = 1e-3  # Define F tolerance (Termination condition for LFBGS-B)
use_jac = True # jac must be True for neural networks
logs = True # Show logs
niter_success = 1000 # Basin-hopping stops after n failed iterations

# 4. Initialise and build LON
mylon = LON.lonsSampler(name,myfunc, dimensions, num_runs, bounds, bsteps, step, x_tol, f_tol, use_jac, disp=logs, success=niter_success)
mylon.run()

# 5. View LON and metrics
mygraph = graph.lonBuilder(name, roundDp=2)   # Define graph to build *RoundDp rounds the fitness to n decimal place 
mygraph.print_metrics()
mygraph.load(8,25) # Builds graph, must specify the minmum and maximum node size

# 6. Evaluate performance of a model
all_weights = mygraph.get_cmlon_weights()
nodeID = 0 # Nodes ids can be seen be hovering on them in plotly
model = set_weights(model,all_weights[nodeID])
evaluate(model, x_train, x_test, y_train, y_test, loss)
