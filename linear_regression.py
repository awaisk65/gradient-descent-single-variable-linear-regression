import numpy
import math
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

#################Read CSV file and filter data######################
data = numpy.genfromtxt("housing.csv", delimiter=',')

m = data[:160,0]
engine_cap = data[:160,1]

test_m = data[160:, 0]
test_engcap = data[160:, 1]
###########################################################
# h = theta0 + theta1*x or y = mx + b 
def eq_line(m,x,b):
	y = (m*x) + b
	return y 
def error(y1,y2):
	result = math.sqrt((y2- y1)**2)
	return result 

def gradient_step(theta0,theta1,m,engine_cap, rate):
	theta0_gradient = 0
	theta1_gradient = 0
	no_of_sample = len(m)
	for i in range (0, len(engine_cap)):
		y = eq_line(theta1, engine_cap[i], theta0)
		theta0_gradient += (y - m[i] )
		theta1_gradient += (y - m[i] ) * engine_cap[i]
	
	theta0_gradient = theta0_gradient  / float(no_of_sample)
	theta1_gradient = theta1_gradient / float(no_of_sample)

	new_theta0 = theta0 - (rate * theta0_gradient)
	new_theta1 = theta1 - (rate * theta1_gradient)

	return new_theta0, new_theta1

def gradient(theta0, theta1, m, rate, itterations):
	temp_theta0 = theta0
	temp_theta1 = theta1
	for i in tqdm(range (0,itterations)):
		temp_theta0,temp_theta1 = gradient_step(temp_theta0, temp_theta1, m, engine_cap, rate)
		theta0 = temp_theta0
		theta1 = temp_theta1
	return theta0, theta1
def main():
	for i in tqdm(range(0,1400,5)):
		theta0 = i # m gussed
		theta1 = i # y-intercept guess
		print("Initial theta0 => " + str(theta0))
		print("Initial theta1 => " + str(theta1))
		rate = 1e-5#0.00000001
		itterations = 500
		before_error = 0
		after_error = 0
		w = []
		for i in range (0, len(engine_cap)):
			y = eq_line(theta1,engine_cap[i],theta0)
			w.append(y)
			before_error += error(y, m[i])
		before_error = before_error/len(m)
		print("Error Before Training => " + str(before_error))

		b, slope = gradient(theta0, theta1, m, rate, itterations)

		print("Trained Theta0 => " + str(b))
		print("Trained Theta1 => " + str(slope))
		for i in range (0,len(test_engcap)):
			y = eq_line(slope,test_engcap[i],b)
			after_error += error(y ,test_m[i])
		after_error = after_error/len(test_m)
		print("Error After Training => " + str(after_error))

	z = []
	for i in range (0,len(engine_cap)):
		y = eq_line(slope,engine_cap[i],b)
		z.append(y)
	fig, ax = plt.subplots()

	for i in range(0,len(m)):
		ax.plot(engine_cap, m, 'g+')
	for i in range(0,len(m)):
		ax.plot(engine_cap, w, 'b-')	
	for i in range(0,len(m)):
		ax.plot(engine_cap, z, 'r-')

	ax.set(xlabel='area ', ylabel='price', title='Linear Regression')
	ax.grid()

	fig.savefig("test.png")
	plt.show()

if __name__ == '__main__':
	main()