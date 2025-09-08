import random
import math

class ACO_TSP:
    def __init__(self, distances, n_ants=10, n_iterations=100, alpha=1, beta=5, rho=0.5, Q=100):
        self.distances = distances
        self.n = len(distances)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.pheromone = [[1 for _ in range(self.n)] for _ in range(self.n)]

        self.best_length = float("inf")
        self.best_tour = None

    def run(self):
        for it in range(self.n_iterations):
            all_tours = []
            all_lengths = []

            for ant in range(self.n_ants):
                tour = self.construct_solution()
                length = self.compute_length(tour)

                all_tours.append(tour)
                all_lengths.append(length)

                if length < self.best_length:
                    self.best_length = length
                    self.best_tour = tour

            self.update_pheromones(all_tours, all_lengths)

        return self.best_tour, self.best_length

    def construct_solution(self):
        start = random.randint(0, self.n - 1)
        tour = [start]
        unvisited = set(range(self.n))
        unvisited.remove(start)

        current = start
        while unvisited:
            next_city = self.choose_next_city(current, unvisited)
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city

        return tour

    def choose_next_city(self, current, unvisited):
        probs = []
        total = 0
        for city in unvisited:
            tau = self.pheromone[current][city] ** self.alpha
            eta = (1.0 / self.distances[current][city]) ** self.beta
            value = tau * eta
            probs.append((city, value))
            total += value

        r = random.random() * total
        cumulative = 0
        for city, value in probs:
            cumulative += value
            if cumulative >= r:
                return city
        return probs[-1][0]

    def compute_length(self, tour):
        length = 0
        for i in range(len(tour) - 1):
            length += self.distances[tour[i]][tour[i+1]]
        length += self.distances[tour[-1]][tour[0]]
        return length

    def update_pheromones(self, all_tours, all_lengths):
        for i in range(self.n):
            for j in range(self.n):
                self.pheromone[i][j] *= (1 - self.rho)

        for tour, length in zip(all_tours, all_lengths):
            for i in range(len(tour) - 1):
                a, b = tour[i], tour[i+1]
                self.pheromone[a][b] += self.Q / length
                self.pheromone[b][a] += self.Q / length
            a, b = tour[-1], tour[0]
            self.pheromone[a][b] += self.Q / length
            self.pheromone[b][a] += self.Q / length


# Example usage
if __name__ == "__main__":
    distances = [
        [0, 2, 9, 10, 7],
        [1, 0, 6, 4, 3],
        [15, 7, 0, 8, 3],
        [6, 3, 12, 0, 11],
        [9, 7, 5, 6, 0]
    ]

    aco = ACO_TSP(distances, n_ants=10, n_iterations=50, alpha=1, beta=5, rho=0.5, Q=100)
    best_tour, best_length = aco.run()

    # Format tour as edges
    path_str = " -> ".join(map(str, best_tour)) + f" -> {best_tour[0]}"
    print("\n=== Final Best Solution ===")
    print("Best path:", path_str)
    print("Best path length:", best_length)
