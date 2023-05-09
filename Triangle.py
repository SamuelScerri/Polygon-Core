from Vertex import Vertex
from UV import UV

from numba.experimental import jitclass

@jitclass
class Triangle:
	vertex_a: Vertex
	vertex_b: Vertex
	vertex_c: Vertex

	uv_a: UV
	uv_b: UV
	uv_c: UV

	def __init__(self, vertex_a, vertex_b, vertex_c, uv_a, uv_b, uv_c):
		self.vertex_a = vertex_a
		self.vertex_b = vertex_b
		self.vertex_c = vertex_c

		self.uv_a = uv_a
		self.uv_b = uv_b
		self.uv_c = uv_c



	def matrix_multiply(self, matrix):
		return Triangle(
			self.vertex_a.matrix_multiply(matrix),
			self.vertex_b.matrix_multiply(matrix),
			self.vertex_c.matrix_multiply(matrix),

			self.uv_a,
			self.uv_b,
			self.uv_c
		)

	def get_vertex_span(self):
		vertex_span_1 = (self.vertex_b.x - self.vertex_a.x, self.vertex_b.y - self.vertex_a.y)
		vertex_span_2 = (self.vertex_c.x - self.vertex_a.x, self.vertex_c.y - self.vertex_a.y)
		span = vertex_span_1[0] * vertex_span_2[1] - vertex_span_1[1] * vertex_span_2[0]

		return vertex_span_1, vertex_span_2, span

	def convert_to_screen_space(self, size):
		self.vertex_a.convert_to_screen_space(size)
		self.vertex_b.convert_to_screen_space(size)
		self.vertex_c.convert_to_screen_space(size)

	def normalize(self):
		self.vertex_a.normalize()
		self.vertex_b.normalize()
		self.vertex_c.normalize()

	def get_barycentric_coordinates(self, vertex_span_1, vertex_span_2, span, x, y):
		q = (x - self.vertex_a.x, y - self.vertex_a.y)
		s = (q[0] * vertex_span_2[1] - q[1] * vertex_span_2[0]) / span
		t = (vertex_span_1[0] * q[1] - vertex_span_1[1] * q[0]) / span

		return s, t, 1 - s - t