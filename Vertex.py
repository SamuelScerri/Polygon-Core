from numba.experimental import jitclass

@jitclass
class Vertex:
	x: float
	y: float
	z: float
	w: float

	def __init__(self, x, y, z, w=1):
		self.x = x
		self.y = y
		self.z = z
		self.w = w

	def get_matrix(self):
		return(self.x, self.y, self.z, self.w)

	def matrix_multiply(self, matrix):
		self.x = (self.x * matrix[0][0]) + (self.y * matrix[1][0]) + (self.z * matrix[2][0]) + (self.w * matrix[3][0])
		self.y = (self.x * matrix[0][1]) + (self.y * matrix[1][1]) + (self.z * matrix[2][1]) + (self.w * matrix[3][1])
		self.z = (self.x * matrix[0][2]) + (self.y * matrix[1][2]) + (self.z * matrix[2][2]) + (self.w * matrix[3][2])
		self.w = (self.x * matrix[0][3]) + (self.y * matrix[1][3]) + (self.z * matrix[2][3]) + (self.w * matrix[3][3])

	def convert_to_screen_space(self, size):
		self.x = ((self.x + 1) * size[0]) / 2
		self.y = ((-self.y + 1) * size[1]) / 2

	def normalize(self):
		self.x /= self.w
		self.y /= self.w
		self.z /= self.w

	def interpolate(self, vertex, factor):
		return Vertex(
			self.x * (1 - factor) + vertex.x * factor,
			self.y * (1 - factor) + vertex.y * factor,
			self.z * (1 - factor) + vertex.z * factor,
			self.w * (1 - factor) + vertex.w * factor
		)

	def copy(self):
		return Vertex(self.x, self.y, self.z, self.w)