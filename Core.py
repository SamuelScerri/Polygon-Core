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

def lerp(a, b, t):
	return a * (1 - t) + b * t

def create_identity_matrix():
	return (
		(1.0, 0.0, 0.0, 0.0),
		(0.0, 1.0, 0.0, 0.0),
		(0.0, 0.0, 1.0, 0.0),
		(0.0, 0.0, 0.0, 1.0)
	)

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

	return(
		((1.0 - c) * x * x + c, (1.0 - c) * x * y - s * z, (1.0 - c) * x * z + s * y, 0.0),
		((1.0 - c) * x * y + s * z, (1.0 - c) * y * y + c, (1.0 - c) * y * z - s * x, 0.0),
		((1.0 - c) * x * z - s * y, (1.0 - c) * y * z + s * x, (1.0 - c) * z * z + c, 0.0),
		(0.0, 0.0, 0.0, 1.0)
	)

def create_projection_matrix(fov, near, far, size):
	aspect_ratio = size[0] / size[1]
	top = math.tan(math.radians(fov) / 2)
	bottom = -top
	right = top * aspect_ratio
	left = bottom

	return (
		((2 * near) / (right - left), 0, (right + left) / (right - left), 0),
		(0, (2 * near) / (top - bottom), (top + bottom) / (top - bottom), 0),
		(0, 0, (- 1 * (far + near)) / (far - near), (-2 * far * near) / (far - near)),
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
		for x in range(min_x, max_x + 1):
			for y in range(min_y, max_y + 1):
				s, t, w = triangle.get_barycentric_coordinates(vertex_span_1, vertex_span_2, span, x, y)

				#If The Current Point Is In The Triangle, Then We Render It
				if s >= 0 and t >= 0 and s + t <= 1:
					depth = w * inverse_vertex_a + s * inverse_vertex_b + t * inverse_vertex_c
					#print(depth)

					if depth > depth_buffer[x][y]:
						#Texture Mapping With Perspective Correction
						uv_x = w * normalized_uv_a[0] + s * normalized_uv_b[0] + t * normalized_uv_c[0]
						uv_y = w * normalized_uv_a[1] + s * normalized_uv_b[1] + t * normalized_uv_c[1]
						z = 1 / depth

						screen_buffer[x][y] = texture[int(uv_x * texture.shape[0] * z)][int(1 - uv_y * texture.shape[1] * z)]
						depth_buffer[x][y] = depth

@numba.njit
def render_triangles(triangles, texture, screen_buffer, depth_buffer, matrix, model_matrix):
	for triangle in triangles:
		new_triangle = triangle.copy()

		new_triangle.matrix_multiply(model_matrix, True)
		new_triangle.matrix_multiply(matrix, False)

		clipped_triangles = new_triangle.clip(screen_buffer.shape)

		for t in range(len(clipped_triangles)):
			render_triangle(clipped_triangles[t], texture, screen_buffer, depth_buffer)


pygame.init()

projection_matrix = create_projection_matrix(60, .1, 100, SIZE)
pygame.display.set_caption("Polygon Core - Unfinished Build")
screen = pygame.display.set_mode(SIZE, pygame.SCALED, vsync=True)

screen_buffer = numpy.zeros(SIZE, dtype=numpy.int32)
depth_buffer = numpy.zeros(SIZE, dtype=numpy.float32)

running = True
clock = pygame.time.Clock()
font = pygame.font.SysFont("Monospace" , 24 , bold=False)

model = Utility("Megaman.obj")
model.build_triangle_data()

position = Vertex(0, 0, -64)
velocity = Vertex(0, 0, 0)

rotation_velocity_x = 0
rotation_velocity_y = 0
rotation_velocity_z = 0

rx = 0
ry = 180
rz = 0

texture = pygame.surfarray.pixels2d(pygame.image.load("Megaman.png").convert())

while running:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

	keys = pygame.key.get_pressed()

	velocity = velocity.interpolate(Vertex(
		(keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]),
		(keys[pygame.K_UP] - keys[pygame.K_DOWN]),
		(keys[pygame.K_z] - keys[pygame.K_x])),
		.1
	)

	rotation_velocity_x = lerp(rotation_velocity_x, (keys[pygame.K_s] - keys[pygame.K_w]) * 2, .1)
	rotation_velocity_y = lerp(rotation_velocity_y, (keys[pygame.K_d] - keys[pygame.K_a]) * 2, .1)

	position.x += velocity.x
	position.y += velocity.y
	position.z += velocity.z

	rx += rotation_velocity_x
	ry += rotation_velocity_y

	world_matrix = quick_matrices_multiply(create_rotation_matrix(rx, 1, 0, 0), create_rotation_matrix(ry, 0, 1, 0))
	world_matrix = quick_matrices_multiply(create_translation_matrix(position), world_matrix)

	render_triangles(model.triangle_data, texture, screen_buffer, depth_buffer, projection_matrix, tuple(world_matrix))

	pygame.surfarray.blit_array(pygame.display.get_surface(), screen_buffer)
	screen.blit(font.render("FPS: " + str(clock.get_fps()), False, (255, 255, 255)), (0, 0))

	pygame.display.flip()
	screen_buffer.fill(0)
	depth_buffer.fill(0)

	clock.tick()

pygame.quit()