import time
import numpy
import math
import pygame
import numba

from Triangle import Triangle
from Vertex import Vertex
from UV import UV
from Utility import Utility

SIZE = (640, 360)

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

	if span != 0:
		for x in range(screen_buffer.shape[0]):
			for y in range(screen_buffer.shape[1]):
				s, t, w = triangle.get_barycentric_coordinates(vertex_span_1, vertex_span_2, span, x, y)

				#If The Current Point Is In The Triangle, Then We Render It
				if s > 0 and t > 0 and s + t <= 1:
					#print(triangle.vertex_a.w)

					#depth = w * triangle.vertex_a.w + s * triangle.vertex_b.w + t * triangle.vertex_c.w

					depth = w * (1 / triangle.vertex_a.z) + s * (1 / triangle.vertex_b.z) + t * (1 / triangle.vertex_c.z)

					#print(depth_buffer[x][y], depth)

					if depth > depth_buffer[x][y]:
						#Texture Mapping With Perspective Correction
						uv_x = w * (triangle.uv_a.x / triangle.vertex_a.w) + s * (triangle.uv_b.x / triangle.vertex_b.w) + t * (triangle.uv_c.x / triangle.vertex_c.w)
						uv_y = w * (triangle.uv_a.y / triangle.vertex_a.w) + s * (triangle.uv_b.y / triangle.vertex_b.w) + t * (triangle.uv_c.y / triangle.vertex_c.w)
						z = 1 / (w * 1 / triangle.vertex_a.w + s * 1 / triangle.vertex_b.w + t * 1 / triangle.vertex_c.w)

						screen_buffer[x][y] = texture[int(uv_x * texture.shape[0] * z)][int(uv_y * texture.shape[1] * z)]
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
screen = pygame.display.set_mode(SIZE, pygame.SCALED, vsync=True)

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

model = Utility("Cube.obj")
model.build_triangle_data()

texture = pygame.surfarray.pixels2d(pygame.image.load("Brick.bmp").convert())

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