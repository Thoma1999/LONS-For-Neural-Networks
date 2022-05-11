import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import data
import functions
import LON
import graph

def evaluate(model, x_train, x_test, y_train, y_test, loss):
    model.compile(loss = loss, metrics=['categorical_accuracy'])
    loss_score = loss(y_train, model(x_train, training=True))
    results = model.evaluate(x_test, y_test, verbose = 0)
    print(model.summary())
    print("Training Loss score: "+str(loss_score.numpy()))
    print("Test Loss score: "+str(results[0]))
    print("Test Accuracy score: "+str(results[1]))

def set_weights(model, params_1d):
    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)
	# we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
	# prepare required information first
    count = 0
    idx = [] # stitch indices
    part = [] # partition indices
    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n
    part = tf.constant(part)
    params = tf.dynamic_partition(params_1d, part, n_tensors)
    for i, (shape, param) in enumerate(zip(shapes, params)):
        model.trainable_variables[i].assign(tf.reshape(param, shape))
    return model


def build_iris_model(units, input_size, output_size, act_fn):
	model = Sequential([
		Dense(units, activation=act_fn, input_shape=(input_size,), use_bias=False),  # input shape required
		Dense(output_size, activation='softmax', use_bias=False)
	])
	return model



if __name__ == "__main__":
    seed = 44
    tf.keras.backend.set_floatx("float64")

    # Define Name to save model as
    name = "ToyTwox1"

    # Define Neural network parameters
    input_size = 4
    output_size = 3
    activation = "tanh"
    hidden_layer_width = 1
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    x_train, x_test, y_train, y_test = data.load_iris_data(seed)
    model = build_iris_model(hidden_layer_width, input_size, output_size, activation)

    # Define neural network as function
    myfunc = functions.function_factory(model, loss, x_train, y_train)

    ###################################
    #   Basin-hopping parameters     #
    ###################################

    # Define dimensions
    dimensions = input_size * hidden_layer_width + output_size*hidden_layer_width
    # Define evaluation boundries
    bounds = [dimensions*(-1,1)]
    # Define number of number of runs
    num_runs = 100
    # Define Basin-hopping steps
    bsteps = 10000
    # Define step size
    step = 0.18
    # Define X tolerance (Nodes within a distance of X in all dimensions are considered as the same node)
    x_tol = 1e-2
    # Define F tolerance (Termination condition for LFBGS-B)
    F_tol = 1e-3
    # jac must be True for neural networks
    use_jac = True
    # Show logs
    logs = True

    ###################################
    #           Create LON            #
    ###################################

    # Initialise LON
    mylon = LON.lonsSampler(name,myfunc, dimensions, num_runs, bounds, bsteps, 0.18, 1e-2, 1e-3, False, disp=True, success=1000)
    # Build LON
    mylon.run()

    ###################################
    #           View LON              #
    ###################################
    # All CMLONS are exported as a html file in the models folder.

    # Define graph to build *RoundDp rounds the fitness to n decimal place 
    mygraph = graph.lonBuilder(name, roundDp=2)
    # Builds graph, must specify the minmum and maximum node size 
    mygraph.load(8,25)

    ###################################
    #           Metrics LON           #
    ###################################

    # Define graph to build *RoundDp rounds the fitness to n decimal place 
    mygraph = graph.lonBuilder(name, roundDp=2)
    # Success is unavailable from here since it is calculated during the construction phase
    mygraph.print_metrics()

    ###################################
    #      Evaluate CMLON nodes       #
    ###################################
    mygraph = graph.lonBuilder('Schwefel26-8D', roundDp=2)
    all_weights = mygraph.get_cmlon_weights()
    nodeID = 162
    # Nodes ids can be seen be hovering on them in plotly
    model = set_weights(model,all_weights[nodeID])
    evaluate(model, x_train, x_test, y_train, y_test, loss)
