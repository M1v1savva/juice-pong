from protocol_clean import default_shot
import numpy as np

def throw_to_coordinates(coord=(600, 400)):
	print('executed')
	x, y = coord

	print(coord)

	xs = [444, 527, 597, 670]
	ys = [201, 273, 358, 434, 505]

	ds = []
	for i in xs:
		cur = []
		for j in ys:
			cur.append((i, j))
		ds.append(cur)

	for i in ds:
		print(i)

	conf = []

	conf.append([(7, 50), (4, 50), (-2, 50), (-8, 50), (-13, 50)])
	conf.append([(9, 135), (4, 135), (-2, 135), (-9, 135), (-13, 135)])
	conf.append([(9, 165), (5, 165), (-2, 165), (-10, 165), (-13, 165)])
	conf.append([(13, 185), (7, 200), (-2, 200), (-11, 200), (-15, 185)])	

	for i in conf:
		print(i)

	id0 = 0
	for i in range(len(xs)):
		if abs(x - xs[i]) < abs(x - xs[id0]):
			id0 = i

	id1 = 0
	for i in range(len(ys)):
		if ys[i] < y:
			id1 = i
	print('id1 ' + str(id1))

	y0 = ys[id1]
	y1 = ys[id1 + 1]
	val0 = conf[id0][id1]
	val1 = conf[id0][id1 + 1]
	b, a = val0
	d, c = val1	

	print(y0)
	print(y1)

	ans0 = a
	ans1 = b + (y - y0)/(y1 - y0)*(d - b)

	ans0 = round(ans0)
	ans1 = round(ans1)

	print(ans0)
	print(ans1)

	default_shot(bot_angle=ans1, delay=ans0, wait=5000)

	# id0 = 0
	# id1 = 0

	# for i in range(len(xs)):
	# 	if xs[i] < x:
	# 		id0 = i
	# for i in range(len(ys)):
	# 	if ys[i] < y:
	# 		id1 = i

	# c0 = ds[id0][id1]
	# c1 = ds[id0 + 1][id1 + 1]
	# x0, y0 = c0
	# x1, y1 = c1

	# val0 = conf[id0][id1]
	# val1 = conf[id0 + 1][id1 + 1]
	# b, a = val0
	# d, c = val1

	# print(c0)
	# print(c1)
	# print(val0)
	# print(val1)

	# ans0 = a + (x - x0)/(x1 - x0)*(c - a)
	# ans1 = b + (y - y0)/(y1 - y0)*(d - b)

	# print(ans0)
	# print(ans1)

	# cur = 10000000
	# final_id0 = 0
	# final_id1 = 0
	# for i in range(len(ds)):
	# 	for j in range(len(ds[i])):
	# 		x0, y0 = ds[i][j]
	# 		if abs(x0 - x) + abs(y0 - y) < cur:
	# 			final_id0 = i
	# 			final_id1 = j
	# 			cur = abs(x0 - x) + abs(y0 - y)
	# print(conf[final_id0][final_id1])
	# ans0, ans1 = conf[final_id0][final_id1]
	# print(final_id0)
	# print(final_id1)

	# take closest x - it is the row
	# take 2 closest y
	# now interpolate y and get results
	# then shoot
	# then try to adjust the angle by 1 in each direction

	#default_shot(bot_angle=int(ans0), delay=int(ans1), wait=5000)		
	#default_shot(bot_angle=int(ans1), delay=int(ans0), wait=5000)