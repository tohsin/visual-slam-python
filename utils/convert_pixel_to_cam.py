def pixel_to_cam(point2d , k):
    return ((point2d[0] - k[0][2]) / k[0][0] , (point2d[1]- k[1][2])/k[1][1])