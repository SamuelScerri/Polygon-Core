import functools
import time
import numpy
import math
import pygame

SIZE = (640, 360)
screen_information = []

for y in range(SIZE[0]):
	for x in range(SIZE[1]):
		screen_information.append((x, y))

@functools.cache
def lerp(a, b, factor):
	return a * (1 - t) + b * t

@functools.cache
def matrix_multiply(x, y):
	return [[sum(a*b for a,b in zip(x_row,y_col)) for y_col in zip(*y)] for x_row in x]

@functools.cache
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
			(self.x, self.y, self.z, self.w),
		)

	@functools.cache
	def matrix_multiply(self, matrix):
		vertex = matrix_multiply(self.get_matrix(), matrix)
		return Vertex(vertex[0][0], vertex[0][1], vertex[0][2], vertex[0][3])

	@functools.cache
	def convert_to_screen_space(self, size):
		return Vertex(
			((self.x + 1) * size[0]) / 2, ((-self.y + 1) * size[1]) / 2, self.z, self.w
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
		return Triangle(
			(
				self.vertex[0].matrix_multiply(matrix),
				self.vertex[1].matrix_multiply(matrix),
				self.vertex[2].matrix_multiply(matrix),
			),

			self.uv
		)

	@functools.cache
	def convert_to_screen_space(self, size):
		return Triangle(
			(
				self.vertex[0].convert_to_screen_space(size),
				self.vertex[1].convert_to_screen_space(size),
				self.vertex[2].convert_to_screen_space(size)
			),

			self.uv
		)

def test_coordinate(information):
	return 255

@functools.cache
def render_triangle():
	screen_iter = map(test_coordinate, screen_information)
	return numpy.fromiter(screen_iter, count=SIZE[0] * SIZE[1], dtype=numpy.int32).reshape((SIZE[0], SIZE[1]))

pygame.init()

projection_matrix = create_projection_matrix(90, .1, 1000, SIZE)
screen = pygame.display.set_mode(SIZE, pygame.SCALED, vsync=False)

running = True
clock = pygame.time.Clock()
font = pygame.font.SysFont("Monospace" , 24 , bold = False)

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


while running:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

	keys = pygame.key.get_pressed()

	surf = render_triangle()

	pygame.surfarray.blit_array(pygame.display.get_surface(), surf)
	screen.blit(font.render("FPS: " + str(clock.get_fps()), False, (255, 255, 255)), (0, 0))

	pygame.display.flip()

	clock.tick()

pygame.quit()