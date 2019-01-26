from individual import Individual
from random import randint

class Classic():
	def __init__( self, chromosomeLength ):
		self.chromosomeLength = chromosomeLength

	def createIndividual( self ):
		return Individual( randint( 0, sum( [2**i for i in range( self.chromosomeLength )] ) ) )

	def mutation( self, chromosome ):
		position = randint( 0, self.chromosomeLength )

		return chromosome ^ 2 ** position

	def crossover( self, parent1, parent2, crossPoint ):
		mask1 = sum( [2**i for i in range( crossPoint )] )
		mask2 = sum( [2**i for i in range( crossPoint, self.chromosomeLength )] )
		crossParent11 = parent1 & mask1
		crossParent12 = parent1 & mask2
		crossParent21 = parent2 & mask1
		crossParent22 = parent2 & mask2
		child1 = crossParent11 | crossParent22
		child2 = crossParent21 | crossParent12

		return ( child1, child2 )