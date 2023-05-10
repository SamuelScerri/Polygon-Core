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

def quick_matrices_multiply(matrix_1, matrix_2):
	return [[sum(a*b for a,b in zip(x_row,y_col)) for y_col in zip(*matrix_2)] for x_row in matrix_1]

def lerp(a, b, factor):
	return a * (1 - t) + b * t

def create_translation_matrix(translation):
	return(
		(1.0, 0.0, 0.0, translation.x),
		(0.0, 1.0, 0.0, translation.y),
		(0.0, 0.0, 1.0, translation.z),
		(0.0, 0.0, 0.0, 1.0)
	)

def create_rotation_matrix(amount, x, y, z):
	c = math.cos(math.radians(amount))
	s = math.sin(math.radians(amount))

	#return(
	#	(1.0, 0.0, 0.0, 0.0),
	#	(0.0, c, -s, 0.0),
	#	(0.0, s, c, 8.0),
	#	(0.0, 0.0, 0.0, 1.0)
	#)

	#return(
	#	((1 - c) * x * x + c, (1 - c) * x * y - s * z, (1 - c) * x * z + s * y, 0),
	#	((1 - c) * x * y + s * z, (1 - c) * y * y + c, (1 - c) * y * z - s * x, 0),
	#	((1 - c) * x * z - s * y, (1 - c) * y * z + s * x, (1 - c) * z * z + c, 0),
	#	(0.0, 0.0, 0.0, 1.0)
	#)

	#return(
	#	(0.0, -z, )
	#)

	#return(
	#	(1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * s * z, 2 * x * z + 2 * s * y, 0.0),
	#	(2 * x * y + 2 * s * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * s * x, 0.0),
	#	(2 * x * z - 2 * s * y, 2 * y * z + 2 * s * x, 1 - 2 * x * x - 2 * y * y, 0.0),
	#	(0.0, 0.0, 0.0, 1.0)
	#)

def create_projection_matrix(fov, near, far, size):
	aspect_ratio = size[0] / size[1]
	top = math.tan(math.radians(fov) / 2)
	bottom = -top
	right = top * aspect_ratio
	left = bottom

	return (
		(1 / right, 0, 0, 0),
		(0, 1 / top, 0, 0),
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
		for x in range(min_x - 1, max_x + 1):
			for y in range(min_y - 1, max_y + 1):
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
def render_triangles(triangles, texture, screen_buffer, depth_buffer, matrix, world_matrix):
	for triangle in triangles:
		new_triangle = triangle.copy()	
		
		#for queue in transformation_queue:
		new_triangle.matrix_multiply(world_matrix, True)

		new_triangle.matrix_multiply(matrix, True)
		clipped_triangles = new_triangle.clip(screen_buffer.shape)

		for t in range(len(clipped_triangles)):
			render_triangle(clipped_triangles[t], texture, screen_buffer, depth_buffer)


pygame.init()

projection_matrix = create_projection_matrix(90, .1, 100, SIZE)
screen = pygame.display.set_mode(SIZE, pygame.SCALED, vsync=True)

screen_buffer = numpy.zeros(SIZE, dtype=numpy.int32)
depth_buffer = numpy.zeros(SIZE, dtype=numpy.float32)

running = True
clock = pygame.time.Clock()
font = pygame.font.SysFont("Monospace" , 24 , bold=False)

model = Utility("player.obj")
model.build_triangle_data()

z = 3
x = 0
y = 0
r = 0
q = 0

texture = pygame.surfarray.pixels2d(pygame.image.load("player_0.png").convert())

while running:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

	keys = pygame.key.get_pressed()

	x += (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * .1
	y += (keys[pygame.K_UP] - keys[pygame.K_DOWN]) * .1
	z += (keys[pygame.K_w] - keys[pygame.K_s]) * .1

	r += (keys[pygame.K_d] - keys[pygame.K_a]) * .4

	transformation_queue = []

	rv = Vertex(1, 1, 1)
	rv = rv.normalized()

	p = Vertex(x, y, z)

	cross = rv.cross(p)

	world_matrix = create_translation_matrix(Vertex(x, y, z))
	#world_matrix = quick_matrices_multiply(create_rotation_matrix(r, 1, 0, 0), world_matrix)
	world_matrix = quick_matrices_multiply(create_translation_matrix(Vertex(x, y, z)), world_matrix)
	
	render_triangles(model.triangle_data, texture, screen_buffer, depth_buffer, projection_matrix, tuple(world_matrix))

	pygame.surfarray.blit_array(pygame.display.get_surface(), screen_buffer)
	screen.blit(font.render("FPS: " + str(clock.get_fps()), False, (255, 255, 255)), (0, 0))

	pygame.display.flip()
	screen_buffer.fill(0)
	depth_buffer.fill(0)

	clock.tick()

pygame.quit()