from Vertex import Vertex
from Triangle import Triangle

from UV import UV
from numba.experimental import jitclass
import numba
from typing import List

from functools import cache

class Utility:
	def __init__(self, filename):
		self.vertex_data = []
		self.uv_data = []
		self.face_data = []
		self.filename = filename

		self.triangle_data = []

		for line in open(filename, "r"):
			if line.startswith('#'):
				continue
			values = line.split()

			if not values:
				continue

			if values[0] == 'v':
				self.vertex_data.append(Vertex(
					float(values[1]), float(values[2]), float(values[3])
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
			vertex_a = self.vertex_data[face[0][0] - 1].copy()
			vertex_b = self.vertex_data[face[0][1] - 1].copy()
			vertex_c = self.vertex_data[face[0][2] - 1].copy()

			vertex_a.z += 128
			vertex_b.z += 128
			vertex_c.z += 128

			vertex_a.x -= 16
			vertex_b.x -= 16
			vertex_c.x -= 16

			self.triangle_data.append(Triangle(
				vertex_a,
				vertex_b,
				vertex_c,

				self.uv_data[face[1][0] - 1],
				self.uv_data[face[1][1] - 1],
				self.uv_data[face[1][2] - 1]
			))

		self.triangle_data = tuple(self.triangle_data)

		print("Successfully Built Triangle Data For:", self.filename)