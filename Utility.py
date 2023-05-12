from Vertex import Vertex
from Triangle import Triangle

from UV import UV
import numba.typed

class Utility:
	def __init__(self, filename):
		self.vertex_data = []
		self.uv_data = []
		self.face_data = []
		self.filename = filename

		self.triangle_data = numba.typed.List()

		for line in open(filename, "r"):
			if line.startswith('#'):
				continue
			values = line.split()

			if not values:
				continue

			if values[0] == 'v':
				self.vertex_data.append(Vertex(
					float(values[1]), float(values[2]), float(values[3]), 1
				))

			elif values[0] == 'vt':
				self.uv_data.append(UV(
					float(values[1]), float(values[2])
				))

			elif values[0] == 'f':
				faces = []
				uv_data = []

				for v in values[1:]:
					w = v.split('/')
					faces.append(int(w[0]))

					if len(w) >= 2 and len(w[1]) > 0:
						uv_data.append(int(w[1]))
					else:
						uv_data.append(0)

				self.face_data.append((faces, uv_data))

	def build_triangle_data(self):
		for face in self.face_data:
			self.triangle_data.append(Triangle(
				self.vertex_data[face[0][0] - 1],
				self.vertex_data[face[0][1] - 1],
				self.vertex_data[face[0][2] - 1],

				self.uv_data[face[1][0] - 1],
				self.uv_data[face[1][1] - 1],
				self.uv_data[face[1][2] - 1]
			))

			if len(face[0]) > 3:
				self.triangle_data.append(Triangle(
						self.vertex_data[face[0][0] - 1],
						self.vertex_data[face[0][2] - 1],
						self.vertex_data[face[0][3] - 1],

						self.uv_data[face[1][0] - 1],
						self.uv_data[face[1][1] - 1],
						self.uv_data[face[1][2] - 1]
			))

		print("Successfully Built Triangle Data For:", self.filename)