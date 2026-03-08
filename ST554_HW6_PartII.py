#import some modules needed
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from sklearn import linear_model

class SLR_slope_simulator:
    def __init__ (self, beta_0, beta_1, x, sigma, seed):
        #create initial attributes
        self.beta_0 = beta_0           #The true intercept of the linear model
        self.beta_1 = beta_1           #The true slope of the linear model
        self.x = x                     #The array of independent variable values
        self.sigma = sigma             #The standard deviation of the error
        self.seed = seed               #The seed for the random number generator
        self.n = len(x)                #The number of data points in x
        self.rng = default_rng(seed)   #Initialize rng with the provided seed
        self.beta_array = np.array([]) #Stores estimated intercepts and slopes from simulations
        self.slopes = []               #An empty list to store slopes

    #define required methods
    def generate_data(self):
        n = len(self.x)
        #create the 'responses' modeled from the line plus a random deviation
        self.y = self.beta_0 + self.beta_1 * self.x + self.sigma*self.rng.standard_normal(size=n)
        return self.y

    def fit_slope (self):
        #prepare for the LR fit
        reg = linear_model.LinearRegression()
        #Create a reg object
        fit = reg.fit(self.x.reshape(-1, 1), self.y)
        return [fit.intercept_, fit.coef_[0]]

    def run_simulations(self, num_simulation):
        #initialize array to save estimates
        self.beta_array = np.zeros(shape = (num_simulation, 2))
        for i in range(num_simulation):
            self.generate_data()
            self.beta_array[i, :] = self.fit_slope()
        return self.beta_array

    def plot_sampling_distribution(self):
        if self.beta_array.size > 0:           #Check if beta_array has data
            plt.hist(self.beta_array[:, 1])    #Plotting the slope distribution
            plt.show()
        else:
            print(f'run_simulations() must be called first')

    def find_prob(self, value, sided = "two-sided"):
        if self.beta_array.size > 0:
            n = len(self.beta_array)
            slopes = self.beta_array[:, 1]     # Extracting slopes
            if sided == "above":
                count = sum(1 for x_val in slopes if x_val >= value)
            elif sided == "below":
                count = sum(1 for x_val in slopes if x_val <= value)
            else:  # two-sided
                abs_value = abs(value)
                count = sum(1 for x_val in slopes if abs(x_val) >= abs_value)
            return count / n
        else:
            print(f'run_simulations() must be called first')
            return None
        
# Creates an instance of the SLR_slope_simulator object
simulator_instance = SLR_slope_simulator(
    beta_0 = 12,
    beta_1 = 2,
    x = np.array(list(np.linspace(start= 0, stop = 10, num = 11))*3),
    sigma = 1,
    seed = 10
)

#Call the run_simulation() method 
#Run 10000 simulations
#num_simulation = 1000
#run_simulations(num_simulation)    #return the error message

#create an instance with run_simulations method
#Run 10000 simulations
num_simulation = 10000
simulator_instance.run_simulations(num_simulation)

#Plot the sampling distribution
simulator_instance.plot_sampling_distribution()

#Approximate the two-sided probability of being larger than 2.1
probability = simulator_instance.find_prob(value=2.1, sided="two-sided")
print(f"Two-sided probability of the slope being larger than 2.1: {probability}")

#Print out the value of the simulated slopes using the attribute
print("Simulated slopes (first 10 values):\n", simulator_instance.beta_array[:10, 1])