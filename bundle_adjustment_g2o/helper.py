data_file_path = "/Users/emma/dev/visual-slam-python/bundle_adjustment_g2o/problem-16-22106-pre.txt"
class BAproblem:
    def __init__(self, data_file_path) -> None:
        lines = []
        with open(data_file_path) as file:
            lines = file.readlines()

        #extracct number of cameras and 
        head_data =  lines[0]
        head_data = head_data.split('\n')[0]
        num_cameras, num_points, num_observations = head_data.split(" ")
        self.num_cameras = int(num_cameras)
        self.num_points = int(num_points)
        self.num_observations = int(num_observations)
        print("number of cameras ", num_cameras, " Number of points " , num_points, " Number of observations ", num_observations)

        observations = lines[1:self.num_observations+1]
        
        camera_arr = [0] * self.num_observations
        point_arr = [0] * self.num_observations
        v =  self.num_observations * 2
        observations_arr = [0] * v
        b = 9 * self.num_cameras + 3 * self.num_points
        parameters_arr = [0] * b
        print("NUMBER of observation ",observations_arr )
        print(len(parameters_arr))
        for obs in observations:
            pass
        

baproblem = BAproblem(data_file_path = data_file_path)