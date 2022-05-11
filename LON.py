from scipy import optimize
from scipy.stats import qmc
import numpy as np
from zipfile import ZipFile
import os.path


class lonsSampler:
	def __init__(self, name, func, dimensions, size, bounds ,iter, stepsize, x_tol, f_tol, jac, disp=False, success=None):
		self.name = name
		self.dimensions = dimensions
		self.func = func
		self.iter = iter
		self.stepsize = stepsize
		self.disp = disp
		self.x_tol = x_tol
		self.jac = jac
		self.f_tol = f_tol
		self.nodes = []
		self.localOptima = {}
		self.weights = {}
		self.monotonicEdges = []
		self.success = success
		self.counter = 0
		self.bounds = bounds
		self.minimizer_kwargs = {"method" : "L-BFGS-B", "bounds":self.bounds, "tol":self.f_tol, "jac":self.jac}
		self.size = size
		self.samples = []
		self.latinHypercube()
		self.success_metric = 0


	def run(self):
		# Main program
		print("Running Basin hopping...")
		last_nodes = []
		for runs, s in enumerate(self.samples):
			self.monotonicSequence = []
			res = optimize.minimize(self.func, s, method='L-BFGS-B', bounds=self.bounds, jac=self.jac, tol=self.f_tol, options={'maxiter':self.iter})
			index = self.precision(res.x)
			if index < 0:
				self.nodes.append(res.x)
				self.localOptima[self.counter]=res.fun
				self.weights[str(self.counter)] = res.x
				self.monotonicSequence.append(self.counter)
				self.counter +=1
			else:
				self.monotonicSequence.append(index)
				if res.fun < self.localOptima.get(index):
					self.localOptima[index] = res.fun
			result = optimize.basinhopping(self.func, res.x, niter=self.iter, niter_success=self.success, stepsize=self.stepsize, minimizer_kwargs=self.minimizer_kwargs, T=0, callback=self.storeOptima, disp=self.disp)
			print("Bhop run: ", runs)
			last_nodes.append(result.fun)
		print('Runs complete')
		last_nodes = [round(num,2) for num in last_nodes]
		best = min(last_nodes)
		self.success_metric = last_nodes.count(best)/100
		print("Success rate: ",self.success_metric)
		self.export_LON()

	def latinHypercube(self):
		sampler = qmc.LatinHypercube(d=self.dimensions)
		samples = sampler.random(n=self.size)
		for dim, bound in enumerate(self.bounds):
			rmin = bound[0]
			rmax = bound[1]
			for sample in samples:
				sample[dim] = rmin + sample[dim] * (rmax- rmin)
		self.samples = samples

	def precision(self,u):
		for v in range(0,len(self.nodes)):
			if max(np.absolute(self.nodes[v] - u)) <=self.x_tol:
				return v
		return -1

	def storeOptima(self, x, f):
		if f < self.localOptima.get(self.monotonicSequence[-1]):
			index = self.precision(x)
			if index == self.monotonicSequence[-1]:
				return False
			if index < 0:
				self.monotonicEdges.append([self.monotonicSequence[-1],self.counter])
				self.nodes.append(x)
				self.localOptima[self.counter]=f
				self.weights[str(self.counter)] = x
				self.monotonicSequence.append(self.counter)
				self.counter +=1 
			else:
				if self.localOptima.get(index) < self.localOptima.get(self.monotonicSequence[-1]):
					self.monotonicEdges.append([self.monotonicSequence[-1], index])
					self.monotonicSequence.append(index)


	def export_LON(self):
		directory = './models/'
		if not os.path.isdir(directory):
			os.mkdir(directory)

		file = open(self.name+'.edges', "w")
		visited = []
		for edge in self.monotonicEdges:
			if edge not in visited:
				file.write(str(edge[0])+' '+str(edge[1])+' '+str(self.monotonicEdges.count(edge))+'\n')
				visited.append(edge)
		file.close()


		file = open(self.name+'.nodes', "w")
		for i in range(0, len(self.localOptima)):
			file.write(str(i)+' '+str(self.localOptima[i])+' '+str(0)+'\n')
		file.close()


		np.save('weights.npy', self.weights)

		# create a ZipFile object
		file_path = os.path.join(directory, self.name+'.zip')
		zipObj = ZipFile(file_path, 'w')
		# Add multiple files to the zip
		zipObj.write(self.name+'.edges')
		zipObj.write(self.name+'.nodes')
		zipObj.write('weights.npy')
		# close the Zip File
		zipObj.close()
		os.remove(self.name+'.edges')
		os.remove(self.name+'.nodes')
		os.remove('weights.npy')
		print('File saved in models as '+self.name+'.zip')