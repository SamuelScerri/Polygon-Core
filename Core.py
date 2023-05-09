import functools
import time
import numpy
import math
import pygame
from numba.experimental import jitclass
import numba

SIZE = (320, 240)

def lerp(a, b, factor):
	return a * (1 - t) + b * t

def create_projection_matrix(fov, near, far, size):
	aspect_ratio = size[0] / size[1]
	top = math.tan(math.radians(fov) / 2) * near
	bottom = -top * aspect_ratio
	right = top * aspect_ratio
	left = bottom

	return (
		((2 * near) / (right - left), 0, (right + left) / (right - left), 0),
		(0, (2 * near) / (top - bottom), (top + bottom) / (top - bottom), 0),
		(0, 0,-(far + near) / (far - near),-(2 * far * near) / (far - near)),
		(0, 0, -1, 0)
	)

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
		return Vertex(
			(self.x * matrix[0][0]) + (self.y * matrix[1][0]) + (self.z * matrix[2][0]) + (self.w * matrix[3][0]),
			(self.x * matrix[0][1]) + (self.y * matrix[1][1]) + (self.z * matrix[2][1]) + (self.w * matrix[3][1]),
			(self.x * matrix[0][2]) + (self.y * matrix[1][2]) + (self.z * matrix[2][2]) + (self.w * matrix[3][2]),
			(self.x * matrix[0][3]) + (self.y * matrix[1][3]) + (self.z * matrix[2][3]) + (self.w * matrix[3][3])
		)

	def convert_to_screen_space(self, size):
		return Vertex(
			((self.x + 1) * size[0]) / 2, ((-self.y + 1) * size[1]) / 2, self.z, self.w
		)

	def normalize(self):
		return Vertex(
			self.x / self.w, self.y / self.w, self.z / self.w, self.w
		)

	def lerp(self, vertex, factor):
		return (
			lerp(self.x, vertex.x, factor),
			lerp(self.y, vertex.y, factor),
			lerp(self.z, vertex.z, factor),
			lerp(self.w, vertex.w, factor)
		)


@jitclass
class UV:
	x: float
	y: float

	def __init__(self, x, y):
		self.x = x
		self.y = y

	def lerp(self, uv, factor):
		return (
			lerp(self.x, uv.x, factor),
			lerp(self.y, uv.y, factor)
		)


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

	def convert_to_screen_space(self, size):
		return Triangle(
			self.vertex_a.convert_to_screen_space(size),
			self.vertex_b.convert_to_screen_space(size),
			self.vertex_c.convert_to_screen_space(size),

			self.uv_a,
			self.uv_b,
			self.uv_c
		)

	def normalize(self):
		return Triangle(
			self.vertex_a.normalize(),
			self.vertex_b.normalize(),
			self.vertex_c.normalize(),

			self.uv_a,
			self.uv_b,
			self.uv_c
		)

	def get_barycentric_coordinates(self, x, y):
		vertex_span_1 = (self.vertex_b.x - self.vertex_a.x, self.vertex_b.y - self.vertex_a.y)
		vertex_span_2 = (self.vertex_c.x - self.vertex_a.x, self.vertex_c.y - self.vertex_a.y)

		span = vertex_span_1[0] * vertex_span_2[1] - vertex_span_1[1] * vertex_span_2[0]

		q = (x - self.vertex_a.x, y - self.vertex_a.y)
		s = (q[0] * vertex_span_2[1] - q[1] * vertex_span_2[0]) / span
		t = (vertex_span_1[0] * q[1] - vertex_span_1[1] * q[0]) / span

		return s, t, 1 - s - t

#This Function Is Responsible Only For Rendering The Triangle
@numba.njit
def render_triangle(triangle, screen_buffer):
	for x in range(screen_buffer.shape[0]):
		for y in range(screen_buffer.shape[1]):
			s, t, w = triangle.get_barycentric_coordinates(x, y)

			if s > 0 and t > 0 and s + t <= 1:
				screen_buffer[x][y] = 255

pygame.init()

projection_matrix = create_projection_matrix(90, .1, 1000, SIZE)
screen = pygame.display.set_mode(SIZE, pygame.SCALED, vsync=False)

screen_buffer = numpy.zeros(SIZE, dtype=numpy.int32)

running = True
clock = pygame.time.Clock()
font = pygame.font.SysFont("Monospace" , 24 , bold = False)

triangle = Triangle(
	Vertex(+0.0, +0.5, -4.0),
	Vertex(-0.5, -0.5, -4.0),
	Vertex(+0.5, -0.5, -4.0),

	UV(-0.5, -0.5),
	UV(+0.5, -0.5),
	UV(+0.5, +0.5)
)

while running:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

	keys = pygame.key.get_pressed()

	render_triangle(triangle.matrix_multiply(projection_matrix).normalize().convert_to_screen_space(screen_buffer.shape), screen_buffer)

	pygame.surfarray.blit_array(pygame.display.get_surface(), screen_buffer)
	screen.blit(font.render("FPS: " + str(clock.get_fps()), False, (255, 255, 255)), (0, 0))

	pygame.display.flip()
	screen_buffer.fill(0)

	clock.tick()

pygame.quit()