import random

class Chromosome:
    def __init__(self, gens):
        self.gens = gens
        self.fitness = 0.0
        self.fitness_ratio = 0.0
        self.probability_range = [0.0, 0.0]
        self.summation = 0.0

    def intersects(self):
        conflicts = 0
        for i in range(len(self.gens) - 1):
            for j in range(i + 1, len(self.gens)):
                if self.gens[i] == self.gens[j]:
                    conflicts += 1
        return conflicts

    def fitness_func(self):
        if self.fitness != 0.0:
            return self.fitness
        self.fitness = 1.0 / (self.intersects() + self.epsilon())
        return self.fitness

    def fitness_ratio_func(self):
        return self.fitness_func() / self.summation

    def set_probability_range(self, range_value):
        self.probability_range = range_value

    def is_chosen(self, number):
        if self.probability_range == [0.0, 0.0]:
            raise ValueError("Probability range must be valid in Chromosome structure!")
        return self.probability_range[0] <= number < self.probability_range[1]

    def epsilon(self):
        return 0.00000001

    def print_chromosome(self):
        print(f"gens: {self.gens}, intersects: {self.intersects()}, fitness: {self.fitness}")

class Genetic:
    def __init__(self, chromosome_length, population_size, per_mutation, max_iter, crossover_type):
        if per_mutation > 1.0:
            raise ValueError("per_mutation argument cannot be greater than 1!")
        self.chromosome_length = chromosome_length
        self.population_size = population_size
        self.per_mutation = per_mutation
        self.population = []
        self.max_iter = max_iter
        self.crossover_type = crossover_type
        self.init_population()

    def init_population(self):
        # Initialize population with some predefined data for testing
        dataset = [
            [5, 5, 15, 15, 2, 2, 7, 7, 17, 17],
            [1, 15, 15, 25, 25, 25, 12, 17, 22, 3],
            [10, 10, 10, 20, 20, 7, 2, 12, 12, 3],
            [1, 10, 15, 15, 2, 2, 7, 7, 12, 22],
            [5, 5, 20, 25, 20, 25, 12, 12, 17, 3],
            [10, 10, 20, 15, 25, 25, 2, 7, 22, 22],
            [5, 5, 15, 20, 20, 7, 7, 7, 12, 22],
            [10, 15, 15, 25, 20, 2, 2, 12, 17, 17],
            [10, 10, 20, 25, 25, 2, 7, 12, 12, 17],
            [5, 15, 20, 20, 2, 2, 2, 22, 22, 17],
        ]

        for item in dataset:
            self.population.append(Chromosome(item))

    def random_chromosome(self):
        rand_vec = [random.randint(0, 25) for _ in range(self.chromosome_length)]
        return Chromosome(rand_vec)

    def fitness_summation(self):
        total_sum = sum(chromosome.fitness_func() for chromosome in self.population)
        for chromosome in self.population:
            chromosome.summation = total_sum
        return total_sum

    def init_probability_range(self):
        temp = 0.0
        for chromo in self.population:
            temp_ = temp + chromo.fitness_ratio_func()
            chromo.set_probability_range([temp, temp_])
            temp = temp_

    def parent_selection(self):
        self.fitness_summation()
        self.init_probability_range()

        new_parents = []
        while len(new_parents) < self.population_size:
            for chromo in self.population:
                rand_no = random.uniform(0, 1)
                if chromo.is_chosen(rand_no):
                    new_parents.append(chromo)
                    break
        return new_parents

    def one_point_crossover(self, parent1, parent2):
        crossover_point = random.randint(0, self.chromosome_length - 1)

        new_child1 = parent1.gens[:crossover_point] + parent2.gens[crossover_point:]
        new_child2 = parent2.gens[:crossover_point] + parent1.gens[crossover_point:]

        return Chromosome(new_child1), Chromosome(new_child2)

    def two_point_crossover(self, parent1, parent2):
        ind1 = random.randint(0, self.chromosome_length - 1)
        ind2 = random.randint(ind1, self.chromosome_length - 1)

        new_child1 = parent1.gens[:ind1] + parent2.gens[ind1:ind2] + parent1.gens[ind2:]
        new_child2 = parent2.gens[:ind1] + parent1.gens[ind1:ind2] + parent2.gens[ind2:]

        return Chromosome(new_child1), Chromosome(new_child2)

    def uniform_crossover(self, parent1, parent2):
        child1 = []
        child2 = []

        for i in range(len(parent1.gens)):
            if random.randint(0, 1) == 1:
                child1.append(parent1.gens[i])
                child2.append(parent2.gens[i])
            else:
                child1.append(parent2.gens[i])
                child2.append(parent1.gens[i])

        return Chromosome(child1), Chromosome(child2)

    def crossover(self, parent1, parent2):
        if self.crossover_type == CrossoverType.OnePoint:
            return self.one_point_crossover(parent1, parent2)
        elif self.crossover_type == CrossoverType.TwoPoint:
            return self.two_point_crossover(parent1, parent2)
        elif self.crossover_type == CrossoverType.Uniform:
            return self.uniform_crossover(parent1, parent2)
        else:
            return self.two_point_crossover(parent1, parent2)

    def recombination(self, parents):
        offsprings = []
        for i in range(0, len(parents)-1, 2):
            child1, child2 = self.crossover(parents[i], parents[i+1])
            offsprings.extend([child1, child2])
        return offsprings

    def swap_mutation(self, chromosome):
        new_gens = chromosome.gens.copy()

        if random.uniform(0, 1) <= self.per_mutation:
            i = random.randint(0, len(new_gens) - 1)
            j = random.randint(0, len(new_gens) - 1)
            new_gens[i], new_gens[j] = new_gens[j], new_gens[i]

        return Chromosome(new_gens)

    def mutation(self, offsprings):
        return [self.swap_mutation(chromosome) for chromosome in offsprings]

    def maximum_fitness(self, population):
        max_i = 0
        max_fit = population[0].fitness_func()
        for i in range(len(population)):
            if population[i].fitness_func() > max_fit:
                max_fit = population[i].fitness_func()
                max_i = i
        return max_i, max_fit

    def start_loop(self):
        if self is None:
            raise ValueError("Genetic instance is None.")

        best_fitnesses = []
        best = self.random_chromosome()

        for i in range(1, self.max_iter + 1):
            parents = self.parent_selection()
            offsprings = self.recombination(parents)
            offsprings = self.mutation(offsprings)
            self.population = offsprings

            best_index, best_fitness = self.maximum_fitness(self.population)
            if best_fitness > best.fitness_func():
                best = self.population[best_index]

            best_fitnesses.append(best_fitness)

        return best, best_fitnesses

class CrossoverType:
    OnePoint = 0
    TwoPoint = 1
    Uniform = 2

if __name__ == "__main__":
    chromosome_length = 10
    population_size = 10
    per_mutation = 0.1
    max_iter = 300
    crossover_type = CrossoverType.Uniform

    genetic = Genetic(chromosome_length, population_size, per_mutation, max_iter, crossover_type)

    best_chromosome, best_fitnesses = genetic.start_loop()

    print(f"Best Chromosome: {best_chromosome.gens}, intersects: {best_chromosome.intersects()}")
    best_chromosome.print_chromosome()

    print(f"Best Fitnesses: {best_fitnesses}")
