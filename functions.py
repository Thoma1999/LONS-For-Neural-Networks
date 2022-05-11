import numpy as np
import tensorflow as tf
import math

def Test_two(chromosome):
	x=1
	y=1
	z = (chromosome[1]*x)
	phi = ((-z**2) + (2*z) -1)/8
	fun = (y-chromosome[0]*phi)**2
	return fun

def Test(chromosome):
	return ((1-(chromosome[0]*chromosome[1]))**2)

def Griewank(chromosome):
	i = np.arange(1., np.size(chromosome) + 1.)
	return sum(chromosome ** 2 / 4000) - np.prod(np.cos(chromosome / np.sqrt(i))) + 1

def Lunacek(chromosome):
	s = 1.0 - (1.0 / (2.0 * math.sqrt(len(chromosome) + 20.0) - 8.2))
	d = 1.0
	mu1 = 2.5
	mu2 = - math.sqrt(abs((mu1**2 - d) / s))
	firstSum = 0.0
	secondSum = 0.0
	thirdSum = 0.0
	for i in range(len(chromosome)):
		firstSum += (chromosome[i]-mu1)**2
		secondSum += (chromosome[i]-mu2)**2
		thirdSum += 1.0 - math.cos(2*math.pi*(chromosome[i]-mu1))
	return min(firstSum, d*len(chromosome) + s*secondSum)+10*thirdSum


def Ackley(chromosome):
	firstSum = 0.0
	secondSum = 0.0
	for c in chromosome:
		firstSum += c**2.0
		secondSum += np.cos(2.0*np.pi*c)
	n = float(len(chromosome))
	return -20.0*np.exp(-0.2*np.sqrt(firstSum/n)) - np.exp(secondSum/n) + 20 + np.e


def Rastrigin(chromosome):
	return (10*len(chromosome) + sum(chromosome**2 - 10*np.cos(2*np.pi*chromosome)))

def Schwefel26(chromosome):
	return (1/len(chromosome)) * sum(chromosome * np.sin(np.sqrt(abs(chromosome))))

def Rosenbrock(chromosome):
	b = 10
	return (chromosome[0]-1)**2 + b*(chromosome[1]-chromosome[0]**2)**2


def function_factory(model, loss, train_x, train_y):
	"""A factory to create a function required by tfp.optimizer.lbfgs_minimize.

	Args:
		model [in]: an instance of `tf.keras.Model` or its subclasses.
		loss [in]: a function with signature loss_value = loss(pred_y, true_y).
		train_x [in]: the input part of training data.
		train_y [in]: the output part of training data.

	Returns:
		A function that has a signature of:
			loss_value, gradients = f(model_parameters).
	"""

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

	def assign_new_model_parameters(params_1d):
		"""A function updating the model's parameters with a 1D tf.Tensor.

		Args:
			params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
		"""

		params = tf.dynamic_partition(params_1d, part, n_tensors)
		for i, (shape, param) in enumerate(zip(shapes, params)):
			model.trainable_variables[i].assign(tf.reshape(param, shape))

	# now create a function that will be returned by this factory
	def f(params_1d):
		"""A function that can be used by tfp.optimizer.lbfgs_minimize.

		This function is created by function_factory.

		Args:
		   params_1d [in]: a 1D tf.Tensor.

		Returns:
			A scalar loss and the gradients w.r.t. the `params_1d`.
		"""
		# update the parameters in the model
		assign_new_model_parameters(params_1d)

		# use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
		with tf.GradientTape() as tape:
			tape.watch(model.trainable_variables)
			# update the parameters in the model
			#assign_new_model_parameters(params_1d)
			# calculate the loss
			loss_value = loss(train_y, model(train_x, training=True))

		# calculate gradients and convert to 1D tf.Tensor
		grads = tape.gradient(loss_value, model.trainable_variables)
		grads = tf.dynamic_stitch(idx, grads).numpy()

		#print(grads)
		return loss_value, grads

	# store these information as members so we can use them outside the scope
	f.idx = idx
	f.part = part
	f.shapes = shapes
	f.assign_new_model_parameters = assign_new_model_parameters
	return f