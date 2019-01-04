import cv2
for i in range(1, 154):
	image = cv2.imread('/home/labuser/sid/research/spalling/data/non_spalling/' + str(i) +'.jpg')
	print(image)
	resized_image = cv2.resize(image, (256,256))
	cv2.imwrite('/home/labuser/sid/research/spalling/data_256/non_spalling/' + str(i) + '.jpg', resized_image)
