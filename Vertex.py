from numba.experimental import jitclass
import math

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

	def matrix_multiply(self, matrix, matrix_first):
		temp_x = self.x
		temp_y = self.y
		temp_z = self.z
		temp_w = self.w

		if matrix_first == False:
			self.x = (temp_x * matrix[0][0]) + (temp_y * matrix[1][0]) + (temp_z * matrix[2][0]) + (temp_w * matrix[3][0])
			self.y = (temp_x * matrix[0][1]) + (temp_y * matrix[1][1]) + (temp_z * matrix[2][1]) + (temp_w * matrix[3][1])
			self.z = (temp_x * matrix[0][2]) + (temp_y * matrix[1][2]) + (temp_z * matrix[2][2]) + (temp_w * matrix[3][2])
			self.w = (temp_x * matrix[0][3]) + (temp_y * matrix[1][3]) + (temp_z * matrix[2][3]) + (temp_w * matrix[3][3])

		else:
			self.x = (temp_x * matrix[0][0]) + (temp_y * matrix[0][1]) + (temp_z * matrix[0][2]) + (temp_w * matrix[0][3]) 
			self.y = (temp_x * matrix[1][0]) + (temp_y * matrix[1][1]) + (temp_z * matrix[1][2]) + (temp_w * matrix[1][3]) 
			self.z = (temp_x * matrix[2][0]) + (temp_y * matrix[2][1]) + (temp_z * matrix[2][2]) + (temp_w * matrix[2][3]) 
			self.w = (temp_x * matrix[3][0]) + (temp_y * matrix[3][1]) + (temp_z * matrix[3][2]) + (temp_w * matrix[3][3])

	def convert_to_screen_space(self, size):
		self.x = ((self.x + 1) * size[0]) / 2
		self.y = ((-self.y + 1) * size[1]) / 2

	def convert_to_normalized_device_coordinates(self):
		self.x /= self.w
		self.y /= self.w
		self.z /= self.w

	def normalized(self):
		if math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z) == 0:
			return self.copy()

		else:
			magnitude = 1 / math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
			return Vertex(self.x * magnitude, self.y * magnitude, self.z * magnitude, self.w)

	def dot(self, vertex):
		return self.x * vertex.x + self.y * vertex.y + self.z * vertex.z

	def cross(self, vertex):
		return Vertex(
			self.y * vertex.z - self.z * vertex.y,
			self.z * vertex.x - self.x * vertex.z,
			self.x * vertex.y - self.y * vertex.x,
			1
		)

	def interpolate(self, vertex, factor):
		return Vertex(
			self.x * (1 - factor) + vertex.x * factor,
			self.y * (1 - factor) + vertex.y * factor,
			self.z * (1 - factor) + vertex.z * factor,
			self.w * (1 - factor) + vertex.w * factor
		)

	def copy(self):
		return Vertex(self.x, self.y, self.z, self.w)