#Generates Regular Polygons and saves the data

from skimage.draw import polygon_perimeter
import matplotlib.pyplot as plt
import numpy as np

#Data Set Parameters
perType = int(1e3) 		#Number of data points per type
grid_size = 128			#Size of coordinate system
num_sides = 10			#Maximum amount of sides for generated polygons
showPoly = 0			#Amount to display

#Save Destinations
imgs = "Test_RegularPolyImgs" + str(grid_size)
crvs = "Test_RegularPolyCurves" + str(grid_size)
labels = "Test_RegularPolyLabels" + str(grid_size)

#Set up
shapeSides = np.arange(3,num_sides+1)						#Number of sides per shape
dataIMGs = np.zeros((np.size(shapeSides)*perType, grid_size, grid_size),\
	dtype = np.uint8)	#Store data images in here
dataCurves = []			#Store data curves in here
dataLabels = np.zeros((perType*(num_sides-2),num_sides-2))	#Store data labels here
#dataLabels is one hot with first index indicating if triangle, second if square, etc

def segmentation(curves):
	#Break each sides into a random amount of segments
	shp = np.shape(curves)
	segments = np.random.randint(0,10,shp[0])	

	newCurves = np.zeros((np.sum(segments) + shp[0],2,2))
	ind = 0
	for i in range(shp[0]):
		t = np.append([0], np.random.uniform(0,1,segments[i]))
		t = np.sort(np.append(t,[1]))
		diff = curves[i,:,1] - curves[i,:,0]

		points = np.zeros((2,np.size(t)))
		points[0,:] = curves[i,0,0] + diff[0]*t
		points[1,:] = curves[i,1,0] + diff[1]*t

		for j in range(np.size(t)-1):
			newCurves[ind,:,:] = points[:,j:(j+2)]
			ind += 1 

	return newCurves
		

#Data Generation
for i in range(np.size(shapeSides)):
	print('Shape ', i+1, ' of ', num_sides-2)
	dataLabels[(i*perType):((i+1)*perType),i] = 1			#One hot encoding

	#5 in rand uniform to get distinguishable shapes (min radius)
	radius = np.random.uniform(5,grid_size/2,perType)		#Random radius for each sample
	thetas = np.linspace(0,2*np.pi,shapeSides[i]+1)			#Linear spacing of theta (Regular Poly)

	for j in range(perType):
		thetas += np.random.uniform(0,2*np.pi)				#Random phase (rotation)

		#Calculate vertices
		points = np.empty((2,shapeSides[i]+1))
		points[0,:] = (radius[j]*np.cos(thetas) + grid_size/2)
		points[0,:] = [int(pt) for pt in points[0,:]]				#Convert to int
		points[1,:] = (radius[j]*np.sin(thetas) + grid_size/2)
		points[1,:] = [int(pt) for pt in points[1,:]]				#Convert to int

		rows, columns = polygon_perimeter(points[0,:], points[1,:],\
			shape = [grid_size, grid_size], clip = True)

		dataIMGs[i*perType + j,rows,columns] = 1		#Image

		curves = np.zeros((shapeSides[i],2,2))
		for k in range(shapeSides[i]):
			curves[k] = points[:,k:(k+2)]

		dataCurves.append(curves)

#Segmentation of the curves
for i in range(len(dataCurves)):
	dataCurves[i] = segmentation(dataCurves[i])
			
np.save(imgs, dataIMGs)			#Save Generated Data Images
np.save(crvs, dataCurves)		#Save Generated Bezier Curves
np.save(labels, dataLabels)		#Save Data Labels

#Enjoy the beauty!
for i in range(showPoly):
	ind = np.random.randint(np.shape(dataIMGs)[0])
	img = dataIMGs[ind,:,:]
	curves = dataCurves[ind]
	label = dataLabels[ind]

	print('Shape ', i, '\n', 'Label: ', label,'\n' , curves, '\n')
	plt.figure(i)
	plt.imshow(img)

plt.show()

