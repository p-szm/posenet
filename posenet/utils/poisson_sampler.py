import math
from random import uniform, randint


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class PoissonSampler:
    def __init__(self, xlim, ylim, r):
        self.xlim = xlim
        self.ylim = ylim
        self.r = r
        self.cell_size = r / math.sqrt(2)

        self.n_cells_x = int(math.ceil((xlim[1] - xlim[0]) / self.cell_size))
        self.n_cells_y = int(math.ceil((ylim[1] - ylim[0]) / self.cell_size))

        self.grid = {}
        self.samples = []
        self.process_list = []

    def reset(self):
        self.grid = {}
        self.samples = []
        self.process_list = []

    def point_to_grid(self, point):
        grid_x = int(math.floor((point[0] - self.xlim[0]) / self.cell_size))
        grid_y = int(math.floor((point[1] - self.ylim[0]) / self.cell_size))
        return (grid_x, grid_y)

    def add_sample(self, sample):
        self.samples.append(sample)
        idx = len(self.samples) - 1
        self.process_list.append(idx)
        grid_x, grid_y = self.point_to_grid(sample)
        self.grid[(grid_x, grid_y)] = idx

    def get_random_candidate(self):
        idx = randint(0, len(self.process_list)-1)
        candidate_idx = self.process_list[idx]
        return candidate_idx, self.samples[candidate_idx]

    def remove_candidate(self, candidate_idx):
        idx = self.process_list.index(candidate_idx)
        if idx == -1:
            raise ValueError('Candidate {} doesn\'t exist'.format(candidate_idx))
        self.process_list[-1], self.process_list[idx] = self.process_list[idx], self.process_list[-1]
        self.process_list.pop()

    def generate_neighbour(self, point):
        R = uniform(self.r, 2*self.r)
        theta = uniform(0, 2*math.pi)
        return [point[0] + R*math.cos(theta), point[1] + self.r*math.sin(theta)]

    def in_domain(self, point):
        return point[0] > self.xlim[0] and point[0] < self.xlim[1] and point[1] > self.ylim[0] and point[1] < self.ylim[1]

    def neighbours(self, point):
        grid_x, grid_y = self.point_to_grid(point)
        points = []
        for x in range(grid_x-2, grid_x+2):
            for y in range(grid_y-2, grid_y+2):
                idx = self.grid.get((x, y), -1)
                if idx != -1:
                    points.append(self.samples[idx])
        return points

    def sample(self):
        self.reset()

        # Choose first point
        first_point = (uniform(*self.xlim), uniform(*self.ylim))
        self.add_sample(first_point)

        while self.process_list:
            idx, point = self.get_random_candidate()
            for i in range(30):
                new_point = self.generate_neighbour(point)
                neighs = self.neighbours(new_point)
                if self.in_domain(new_point) and all(map(lambda p: distance(p, new_point) > self.r, neighs)):
                    self.add_sample(new_point)
                    break
            else:
                self.remove_candidate(idx)
        return self.samples
