import time
import numpy
import math
import pygame
import numba

from Triangle import Triangle
from Vertex import Vertex
from UV import UV

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