import torch
import torchga
import pygad
import numpy


class PopulationOptim:
  def __init(self, loss_function, data_inputs, data_outputs, torch_ga, model):
    self.loss_function = loss_function
    self.data_inputs = data_inputs
    self.data_outputs = data_outputs
    # self.torch_ga = torch_ga
    self.model = model
    self.torch_ga = torchga.TorchGA(model=model,
                           num_solutions=10)
    num_generations = 200 # Number of generations.
    num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
    initial_population = torch_ga.population_weights # Initial population of network weights.
    parent_selection_type = "sss" # Type of parent selection.
    crossover_type = "single_point" # Type of the crossover operator.
    mutation_type = "random" # Type of the mutation operator.
    mutation_percent_genes = 10 # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
    keep_parents = -1 # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.

    # Create an instance of the pygad.GA class
    self.ga_instance = pygad.GA(num_generations=num_generations, 
                           num_parents_mating=num_parents_mating, 
                           initial_population=initial_population,
                           fitness_func=self.fitness_func,
                           parent_selection_type=parent_selection_type,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           keep_parents=keep_parents,
                           on_generation=self.callback_generation)

  def fitness_func(self, solution, sol_idx):

      model_weights_dict = torchga.model_weights_as_dict(model=self.model,
                                                         weights_vector=solution)

      model.load_state_dict(model_weights_dict)

      predictions = self.model(self.data_inputs)

      solution_fitness = 1.0 / (self.loss_function(predictions, self.data_outputs).detach().numpy() + 0.00000001)

      return solution_fitness

  def callback_generation(self, ga_instance):
      print("Generation = {generation}".format(generation=ga_instance.generations_completed))
      print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

# Build the PyTorch model.
# input_layer = torch.nn.Conv2d(in_channels=3, out_channels=5, kernel_size=7)
# relu_layer1 = torch.nn.ReLU()
# max_pool1 = torch.nn.MaxPool2d(kernel_size=5, stride=5)

# conv_layer2 = torch.nn.Conv2d(in_channels=5, out_channels=3, kernel_size=3)
# relu_layer2 = torch.nn.ReLU()

# flatten_layer1 = torch.nn.Flatten()
# # The value 768 is pre-computed by tracing the sizes of the layers' outputs.
# dense_layer1 = torch.nn.Linear(in_features=768, out_features=15)
# relu_layer3 = torch.nn.ReLU()

# dense_layer2 = torch.nn.Linear(in_features=15, out_features=4)
# output_layer = torch.nn.Softmax(1)

# model = torch.nn.Sequential(input_layer,
#                             relu_layer1,
#                             max_pool1,
#                             conv_layer2,
#                             relu_layer2,
#                             flatten_layer1,
#                             dense_layer1,
#                             relu_layer3,
#                             dense_layer2,
#                             output_layer)

# Create an instance of the pygad.torchga.TorchGA class to build the initial population.


# loss_function = torch.nn.CrossEntropyLoss()

# Data inputs
# data_inputs = torch.from_numpy(numpy.load("dataset_inputs.npy")).float()
# data_inputs = data_inputs.reshape((data_inputs.shape[0], data_inputs.shape[3], data_inputs.shape[1], data_inputs.shape[2]))

# # Data outputs
# data_outputs = torch.from_numpy(numpy.load("dataset_outputs.npy")).long()

# Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class


# Start the genetic algorithm evolution.
# ga_instance.run()

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
# ga_instance.plot_result(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

# # Returning the details of the best solution.
# solution, solution_fitness, solution_idx = ga_instance.best_solution()
# print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
# print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

# # Fetch the parameters of the best solution.
# best_solution_weights = torchga.model_weights_as_dict(model=model,
#                                                       weights_vector=solution)
# model.load_state_dict(best_solution_weights)
# predictions = model(data_inputs)
# # print("Predictions : \n", predictions)

# # Calculate the crossentropy for the trained model.
# print("Crossentropy : ", loss_function(predictions, data_outputs).detach().numpy())

# # Calculate the classification accuracy for the trained model.
# accuracy = torch.sum(torch.max(predictions, axis=1).indices == data_outputs) / len(data_outputs)
# print("Accuracy : ", accuracy.detach().numpy())