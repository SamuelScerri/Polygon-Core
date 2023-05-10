import time
import numpy
import math
import pygame
import numba

from Triangle import Triangle
from Vertex import Vertex
from UV import UV
from Utility import Utility

SIZE = (320, 180)

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
def render_triangle(triangle, texture, screen_buffer, depth_buffer):
	vertex_span_1, vertex_span_2, span = triangle.get_vertex_span()
	max_x, min_x, max_y, min_y = triangle.get_boundaries()

	normalized_uv_a = (triangle.uv_a.x / triangle.vertex_a.w, triangle.uv_a.y / triangle.vertex_a.w)
	normalized_uv_b = (triangle.uv_b.x / triangle.vertex_b.w, triangle.uv_b.y / triangle.vertex_b.w)
	normalized_uv_c = (triangle.uv_c.x / triangle.vertex_c.w, triangle.uv_c.y / triangle.vertex_c.w)

	inverse_vertex_a = 1 / triangle.vertex_a.w
	inverse_vertex_b = 1 / triangle.vertex_b.w
	inverse_vertex_c = 1 / triangle.vertex_c.w

	if span != 0:
		for x in range(min_x, max_x):
			for y in range(min_y, max_y):
				s, t, w = triangle.get_barycentric_coordinates(vertex_span_1, vertex_span_2, span, x, y)

				#If The Current Point Is In The Triangle, Then We Render It
				if s > 0 and t > 0 and s + t <= 1:
					depth = w * inverse_vertex_a + s * inverse_vertex_b + t * inverse_vertex_c

					if depth > depth_buffer[x][y]:
						#Texture Mapping With Perspective Correction
						uv_x = w * normalized_uv_a[0] + s * normalized_uv_b[0] + t * normalized_uv_c[0]
						uv_y = w * normalized_uv_a[1] + s * normalized_uv_b[1] + t * normalized_uv_c[1]
						z = 1 / (w * inverse_vertex_a + s * inverse_vertex_b + t * inverse_vertex_c)

						screen_buffer[x][y] = texture[int(uv_x * texture.shape[0] * z)][int(1 - uv_y * texture.shape[1] * z)]
						depth_buffer[x][y] = depth

@numba.njit
def render_triangles(triangles, texture, screen_buffer, depth_buffer):
	for triangle in triangles:
		new_triangle = triangle.matrix_multiply(projection_matrix)
		clipped_triangles = new_triangle.clip(SIZE)

		for t in range(len(clipped_triangles)):
			render_triangle(clipped_triangles[t], texture, screen_buffer, depth_buffer)


pygame.init()

projection_matrix = create_projection_matrix(90, .1, 1000, SIZE)
screen = pygame.display.set_mode(SIZE, pygame.SCALED, vsync=False)

screen_buffer = numpy.zeros(SIZE, dtype=numpy.int32)
depth_buffer = numpy.zeros(SIZE, dtype=numpy.float32)

running = True
clock = pygame.time.Clock()
font = pygame.font.SysFont("Monospace" , 24 , bold=False)

triangle = Triangle(
	#Vertex(+0.0, +0.5, -4.0),
	#Vertex(-0.5, -0.5, -4.0),
	#Vertex(+0.5, -0.5, -4.0),

	Vertex(+0.1, +0.1, -1.0),
	Vertex(+0.1, -0.1, -1.5),
	Vertex(+0.1, -0.1, -0.5),

	UV(-0.5, -0.5),
	UV(+0.5, -0.5),
	UV(+0.5, +0.5)
)

model = Utility("player.obj")
model.build_triangle_data()

texture = pygame.surfarray.pixels2d(pygame.image.load("player_0.png").convert())

while running:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

	keys = pygame.key.get_pressed()

	render_triangles(model.triangle_data, texture, screen_buffer, depth_buffer)

	pygame.surfarray.blit_array(pygame.display.get_surface(), screen_buffer)
	screen.blit(font.render("FPS: " + str(clock.get_fps()), False, (255, 255, 255)), (0, 0))

	pygame.display.flip()
	screen_buffer.fill(0)
	depth_buffer.fill(0)

	clock.tick()

pygame.quit()