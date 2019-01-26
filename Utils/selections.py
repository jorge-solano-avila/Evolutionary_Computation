import numpy
from random import random, randint

def proportional( population, parentsLength ):
	parents = []
	totalFitness = sum( individual.fitness for individual in population )

	for i in range( parentsLength ):
		randomNumber = random()
		probabilityAcumulated = 0
		for individual in population:
			probability = individual.fitness / totalFitness
			probabilityAcumulated += probability

			if randomNumber <= probabilityAcumulated:
				parents.append( individual )

				break

	return parents

def rank( population, parentsLength ):
	parents = []
	populationLength = len( population )
	sorted( population, key = lambda individual: individual.fitness, reverse = True )

	for i in range( parentsLength ):
		randomNumber = random()
		for j in range( populationLength ):
			individual = population[j]
			probability = ( j + 1 ) / populationLength

			if randomNumber <= probability:
				parents.append( individual )

				break

	return parents

def tournament( population, parentsLength ):
	parents = []
	populationLength = len( population )

	for i in range( parentsLength ):
		best = None
		for j in range( int( populationLength / 2 ) ):
			individual = population[randint( 0, populationLength - 1 )]
			if not best or individual.fitness > best.fitness:
				best = individual
		parents.append( best )
	
	return parents

def elitist( population, parentsLength ):
	return sorted( population, key = lambda individual: individual.fitness, reverse = True )[:parentsLength]

def randomSelection( population, parentsLength ):
	return numpy.random.choice( population, parentsLength )
