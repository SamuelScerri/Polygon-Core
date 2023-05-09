from numba.experimental import jitclass

@jitclass
class UV:
	x: float
	y: float

	def __init__(self, x, y):
		self.x = x
		self.y = y