from random import random

class Individual():
	def __init__( self, chromosome, function ):
		self.chromosome = chromosome
		self.function = function
		self.fitness = self.function( self.chromosome )
		self.crossoverRate = random()
		self.mutationRate = 1 - self.crossoverRate

	def print( self ):
		print( ( bin( self.chromosome ), self.fitness ) )