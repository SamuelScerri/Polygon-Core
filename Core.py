import functools
import time
from numpy import zeros, int32
import math
import pygame

SIZE = (320, 240)

@functools.cache
def lerp(a, b, factor):
	return a * (1 - t) + b * t

@functools.cache
def matrix_multiply(x, y):
	return [[sum(a*b for a,b in zip(x_row,y_col)) for y_col in zip(*y)] for x_row in x]

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

class Vertex:
	def __init__(self, x, y, z, w=1):
		self.x = x
		self.y = y
		self.z = z
		self.w = w

	@functools.cache
	def get_matrix(self):
		return (
			(self.x, 0, 0, 0),
			(0, self.y, 0, 0),
			(0, 0, self.z, 0),
			(0, 0, 0, self.w)
		)

	@functools.cache
	def lerp(self, vertex, factor):
		return (
			lerp(self.x, vertex.x, factor),
			lerp(self.y, vertex.y, factor),
			lerp(self.z, vertex.z, factor),
			lerp(self.w, vertex.w, factor)
		)


class UV:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	@functools.cache
	def get_matrix(self):
		return (
			(self.x, 0),
			(0, self.y)
		)

	@functools.cache
	def lerp(self, uv, factor):
		return (
			lerp(self.x, uv.x, factor),
			lerp(self.y, uv.y, factor)
		)

class Triangle:
	def __init__(self, vertex, uv):
		self.vertex = vertex
		self.uv = uv

	@functools.cache
	def matrix_multiply(self, matrix):
		return (
			matrix_multiply(self.vertex[0].get_matrix(), matrix),
			matrix_multiply(self.vertex[1].get_matrix(), matrix),
			matrix_multiply(self.vertex[2].get_matrix(), matrix)
		)

pygame.init()
screen_buffer = zeros(SIZE, dtype=int32)
projection_matrix = create_projection_matrix(90, .1, 1000, SIZE)
screen = pygame.display.set_mode(SIZE, pygame.SCALED, vsync=True)

running = True

triangle = Triangle(
	(
		Vertex(+0.0, +0.5, -1.0),
		Vertex(-0.5, -0.5, -1.0),
		Vertex(+0.5, -0.5, -1.0)
	),

	(
		UV(-0.5, -0.5),
		UV(+0.5, -0.5),
		UV(+0.5, +0.5)
	)
)

iteration = 0

while running:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

	keys = pygame.key.get_pressed()
	triangle.matrix_multiply(projection_matrix)

	pygame.surfarray.blit_array(pygame.display.get_surface(), screen_buffer)
	pygame.display.flip()

	screen_buffer.fill(0)

pygame.quit()