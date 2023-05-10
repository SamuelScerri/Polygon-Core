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

	def clip_axis(self, vertex_data, uv_data, axis, opposite):
		vertex_list = []
		uv_list = []

		previous_vertex = vertex_data[len(vertex_data) - 1]

		if axis == 0:
			previous_component = previous_vertex.x
		elif axis == 1:
			previous_component = previous_vertex.y
		elif axis == 2:
				previous_component = previous_vertex.z		

		if opposite:
			previous_inside = previous_component >= -previous_vertex.w
		else:
			previous_inside = previous_component <= previous_vertex.w

		previous_uv = uv_data[len(uv_data) - 1]

		for vertex in range(len(vertex_data)):
			current_vertex = vertex_data[vertex]

			if axis == 0:
				current_component = current_vertex.x
				previous_component = previous_vertex.x
			elif axis == 1:
				current_component = current_vertex.y
				previous_component = previous_vertex.y
			elif axis == 2:
				current_component = current_vertex.z
				previous_component = previous_vertex.z				

			if opposite:
				current_inside = current_component >= -current_vertex.w
			else:
				current_inside = current_component <= current_vertex.w
			current_uv = uv_data[vertex]

			if current_inside ^ previous_inside:
				if opposite:
					factor = (previous_vertex.w + previous_component) / (
						(previous_vertex.w + previous_component) -
						(current_vertex.w + current_component)
					)
				else:
					factor = (previous_vertex.w - previous_component) / (
						(previous_vertex.w - previous_component) -
						(current_vertex.w - current_component)
					)					

				vertex_list.append(previous_vertex.interpolate(current_vertex, factor))
				uv_list.append(previous_uv.interpolate(current_uv, factor))

			if current_inside:
				vertex_list.append(current_vertex)
				uv_list.append(current_uv)

			previous_vertex = current_vertex
			previous_inside = current_inside
			previous_uv = current_uv

		return vertex_list, uv_list

	def get_boundaries(self):
		max_x = max(self.vertex_a.x, max(self.vertex_b.x, self.vertex_c.x))
		min_x = min(self.vertex_a.x, min(self.vertex_b.x, self.vertex_c.x))

		max_y = max(self.vertex_a.y, max(self.vertex_b.y, self.vertex_c.y))
		min_y = min(self.vertex_a.y, min(self.vertex_b.y, self.vertex_c.y))

		return max_x, min_x, max_y, min_y

	#Note, This Is A Very Expensive Operation, This Will Also Normalize Every Vertex And Convert It To Screen-Space Automatically
	def clip(self, size):
		#We Convert To A Tuple For Cleaner Code
		vertex_data = (self.vertex_a, self.vertex_b, self.vertex_c)
		uv_data = (self.uv_a, self.uv_b, self.uv_c)

		vertex_data, uv_data = self.clip_axis(vertex_data, uv_data, 0, False)

		if len(vertex_data) > 0:
			vertex_data, uv_data = self.clip_axis(vertex_data, uv_data, 0, True)

			if len(vertex_data) > 0:
				vertex_data, uv_data = self.clip_axis(vertex_data, uv_data, 1, False)

				if len(vertex_data) > 0:
					vertex_data, uv_data = self.clip_axis(vertex_data, uv_data, 1, True)


		triangles = []

		if len(vertex_data) > 0:
			for v in vertex_data:
				v.normalize()
				v.convert_to_screen_space(size)

			for index in range(len(vertex_data) - 2):
				triangles.append(
					Triangle(
						vertex_data[0], vertex_data[index + 1], vertex_data[index + 2],
						uv_data[0], uv_data[index + 1], uv_data[index + 2])
				)

		return triangles