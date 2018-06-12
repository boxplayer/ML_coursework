import numpy as np


def euclidian_distance(vec1, vec2):
    joined_vectors = zip(vec1, vec2)
    distance = 0
    for element in joined_vectors:
        distance += (element[1] - element[0]) ** 2
    
    return math.sqrt(distance)

def manhattan_distance(vec1, vec2):
    joined_vectors = zip(vec1, vec2)
    distance = 0
    for element in joined_vectors:
        distance += abs(element[1] - element[0])
    return distance

    # p determines manhattan or euclidian distance
def minkowski_distance(vec1, vec2, p):
    joined_vectors = zip(vec1, vec2)
    distance = 0
    for element in joined_vectors:
        distance += abs(element[1] - element[0]) ** p
    
    distance = distance ** (1/p)
    return distance

def main():
    x = [0,0]
    x = np.asarray(x)
    
    distance = lambda x,y: (x-y)**2
    
    y = [1,10]
    y = np.asarray(y)
    
    distances = distance(x, y)
    dist = np.linalg.norm(x-y)
    
    print("distance check")
    print("manh",manhattan_distance(x, y))
    print("manh_mink",minkowski_distance(x, y, 1))
    
    print("eucl",euclidian_distance(x, y))
    print("eucl_mink",minkowski_distance(x, y, 2))
    print("eucl_luke",distances)
    print("eucl_np",dist)
    
if __name__ == '__main__':

  main()