from numba.experimental import jitclass

@jitclass
class UV:
	x: float
	y: float

	def __init__(self, x, y):
		self.x = x
		self.y = y

	def copy(self):
		return UV(self.x, self.y)

	def interpolate(self, uv, factor):
		return UV(
			self.x * (1 - factor) +  uv.x * factor,
			self.y * (1 - factor) +  uv.y * factor,
		)