from random import randint, random, randrange

import numpy

from Utils.constants import (CROSSOVER_OPERATOR, ELITIST_SELECTION,
                             MUTATION_OPERATOR, PROPORTIONAL_SELECTION,
                             RANDOM_SELECTION, RANK_SELECTION,
                             TOURNAMENT_SELECTION)
from Utils.individual import Individual
from Utils.selections import (elitist, proportional, randomSelection, rank,
                              tournament)


class HAEA():
	def __init__( self, function, chromosomeLength, populationLength ):
		self.function = function
		self.chromosomeLength = chromosomeLength
		self.populationLength = populationLength
		self.population = []

	def __init__( self, function, dimensionsLength, populationLength, limits ):
		self.function = function
		self.dimensionsLength = dimensionsLength
		self.populationLength = populationLength
		self.limits = limits

	def crossover( self, parent1, parent2, crossPoint ):
		mask1 = sum( [2 ** i for i in range( crossPoint )] )
		mask2 = sum( [2 ** i for i in range( crossPoint, self.chromosomeLength )] )
		crossParent11 = parent1 & mask1
		crossParent12 = parent1 & mask2
		crossParent21 = parent2 & mask1
		crossParent22 = parent2 & mask2
		child1 = Individual(
			chromosome = crossParent11 | crossParent22,
			function = self.function
		)
		child2 = Individual(
			chromosome = crossParent21 | crossParent12,
			function = self.function
		)

		return ( child1, child2 )

	def realCrossover( self, parent1, parent2, crossPoint ):
		child1 = Individual(
			chromosome = numpy.append( parent1[:crossPoint], parent2[crossPoint:] ),
			function = self.function
		)
		child2 = Individual(
			chromosome = numpy.append( parent2[:crossPoint], parent1[crossPoint:] ),
			function = self.function
		)

		return ( child1, child2 )

	def lcCrossover( self, parent1, parent2 ):
		alpha = numpy.random.uniform()
		c1 = alpha * parent1 + ( 1 - alpha ) * parent2
		c2 = ( 1 - alpha ) * parent1 + alpha * parent2
		child1 = Individual(
			chromosome = c1,
			function = self.function
		)
		child2 = Individual(
			chromosome = c2,
			function = self.function
		)

		return ( child1, child2 )

	def mutation( self, chromosome ):
		position = randint( 0, self.chromosomeLength )
		child = Individual(
			chromosome = chromosome ^ 2 ** position,
			function = self.function
		)

		return child

	def realMutation( self, chromosome ):
		pos = randrange( 0, self.dimensionsLength )
		lim_min, lim_max = self.limits[0], self.limits[1]
		sigma = (lim_max - lim_min) / 100
		new_chrom = numpy.array( chromosome[:] )
		new_chrom[pos] += numpy.random.randn() * sigma
		if new_chrom[pos] > lim_max or new_chrom[pos] < lim_min:
			return Individual(chromosome = chromosome, function = self.function)
		
		child = Individual(
			chromosome = new_chrom,
			function = self.function
		)

		return child

	def createPopulation( self ):
		self.population = [Individual(
			chromosome = randint( 0, sum( [2 ** i for i in range( self.chromosomeLength )] ) ),
			function = self.function
		) for _ in range( self.populationLength )]

	def dcCreatePopulation( self ):
		minLim, maxLim = self.limits[0], self.limits[1]
		self.population = numpy.array( [Individual(
			chromosome = numpy.random.uniform( minLim, maxLim, self.dimensionsLength ),
			function = self.function
		) for _ in range( self.populationLength )] )

	def selectOperator( self, individual ):
		probability = random()

		if probability < individual.mutationRate:
			return CROSSOVER_OPERATOR

		return MUTATION_OPERATOR

	def selectParents( self, selection, type = None ):
		if type != "M":
			if selection == PROPORTIONAL_SELECTION:
				parents = proportional( self.population, 1 )
			elif selection == RANK_SELECTION:
				parents = rank( self.population, 1 )
			elif selection == TOURNAMENT_SELECTION:
				parents = tournament( self.population, 1 )
			elif selection == RANDOM_SELECTION:
				parents = randomSelection( self.population, 1 )
			else: # selection == ELITIST_SELECTION
				parents = elitist( self.population, 1 )
		else:
			parents = randomSelection( self.population, 1 )

		return parents

	def applyOperator( self, operator, parents ):
		offspring = parents

		if operator == CROSSOVER_OPERATOR:
			crossPoint = randint( 1, self.chromosomeLength )
			children = self.crossover( parents[0].chromosome, parents[1].chromosome, crossPoint )
		else: # operator == MUTATION_OPERATOR
			children = [self.mutation( parents[0].chromosome )]
		offspring += children

		return offspring

	def dcApplyOperator( self, operator, parents ):
		offspring = []

		if operator == CROSSOVER_OPERATOR:
			children = self.lcCrossover( parents[0].chromosome, parents[1].chromosome )
		else: # operator == MUTATION_OPERATOR
			children = [self.realMutation( parents[0].chromosome )]
		offspring += children

		return offspring

	def best( self, offspring ):
		return max( offspring, key = lambda individual: individual.fitness )

	def euclideanDistance( self, individual1, individual2 ):
		return numpy.linalg.norm( individual1 - individual2 )

	def dcBest( self, offspring, individual, type = None ):
		total = len( offspring )
		best = offspring[0]
		minDistance = self.euclideanDistance( offspring[0].chromosome, individual.chromosome )
		for i in range( 1, total ):
			actual = offspring[i]
			distance = self.euclideanDistance( individual.chromosome, actual.chromosome )
			if distance > 0 and distance < minDistance:
				best = offspring[i]
				minDistance = distance

		distance = self.euclideanDistance( individual.chromosome, best.chromosome )
		if individual.fitness > best.fitness:
			best = individual
		if type == "R" and distance > 0.1:
			best = individual

		return best

	def recalculateRates( self, operator, child, individual ):
		sigma = random()
		crossoverRate = individual.crossoverRate
		mutationRate = individual.mutationRate

		if child.fitness >= individual.fitness:
			if operator == CROSSOVER_OPERATOR:
				crossoverRate *= ( 1 + sigma )
			else: # operator == MUTATION_OPERATOR
				mutationRate *= ( 1 + sigma )
		else:
			if operator == CROSSOVER_OPERATOR:
				crossoverRate *= ( 1 - sigma )
			else: # operator == MUTATION_OPERATOR
				mutationRate *= ( 1 - sigma )

		total = crossoverRate + mutationRate
		crossoverRate /= total
		mutationRate /= total

		child.crossoverRate = crossoverRate
		child.mutationRate = mutationRate

	def init( self, generations, selection ):
		# Create initial population
		self.createPopulation()

		# Initialize data to return
		data = []

		# Generations
		for _ in range( generations ):
			newPopulation = []
			for individual in self.population:
				# Select operator to apply
				operator = self.selectOperator( individual )

				# Apply operator
				if operator == CROSSOVER_OPERATOR:
					parents = self.selectParents( selection )
					parents = [parents[randint( 0, len( parents ) - 1 )]] + [individual]
				else: # operator == MUTATION_OPERATOR
					parents = [individual]
				offspring = self.applyOperator( operator, parents )

				# Choose the best individual
				child = self.best( offspring )

				# Recalculate operator rates
				self.recalculateRates( operator, child, individual )

				# Add child to newPopulation
				newPopulation.append( child )
			self.population = newPopulation

			data.append( self.best( self.population ) )

		return data

	def realDCInit( self, generations, selection, type = None ):
		# Create initial population
		self.dcCreatePopulation()

		self.initialPopulation = numpy.array( self.population[:] )

		# Generations
		for i in range( generations ):
			newPopulation = []
			for individual in self.population:
				# Select operator to apply
				operator = self.selectOperator( individual )

				# Apply operator
				if operator == CROSSOVER_OPERATOR:
					parents = self.selectParents( selection, type )
					parents = [parents[randint( 0, len( parents ) - 1 )]] + [individual]
				else: # operator == MUTATION_OPERATOR
					parents = [individual]
				offspring = self.dcApplyOperator( operator, parents )

				# Choose the best individual
				child = self.dcBest( offspring, individual, type )

				# Recalculate operator rates
				self.recalculateRates( operator, child, individual )

				# Add child to newPopulation
				newPopulation.append( child )
			self.population = newPopulation

			if i == 39:
				self.population40 = numpy.array( self.population[:] )
			elif i == 79:
				self.population80 = numpy.array( self.population[:] )
			elif i == 119:
				self.population120 = numpy.array( self.population[:] )
			elif i == 159:
				self.population160 = numpy.array( self.population[:] )

		return ( self.initialPopulation, self.population40, self.population80, self.population120, self.population160, self.population )