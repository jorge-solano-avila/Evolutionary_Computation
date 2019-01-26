def generational( parents, children ):
	return children

def stableState( parents, children ):
	parentsLength = len( parents )

	return sorted( parents + children, key = lambda individual: individual.fitness, reverse = True )[:parentsLength]