import math
import random
import matplotlib.pyplot as plt


def parse_tsp_file(file_path):
    try:
        cities = {}
        with open(file_path, "r") as file:
            lines = file.readlines()
            reading_coordinates = False  # вказує чи почалось читання координат
            for line in lines:
                if line.strip() == "NODE_COORD_SECTION":
                    reading_coordinates = True
                elif line.strip() == "EOF":
                    break
                elif reading_coordinates:
                    parts = line.split()
                    if len(parts) == 3:
                        city_id = int(parts[0])
                        x, y = float(parts[1]), float(parts[2])
                        cities[city_id] = (x, y)
        return cities
    except FileNotFoundError:
        print(f"File {file_path} not found. Please check the file")
        return None


def distance(cities, name_1, name_2):
    x1, y1 = cities[name_1]
    x2, y2 = cities[name_2]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def fitness(cities, solution):
    total_distance = 0
    for i in range(len(cities) - 1):  # використовується щоб не вийти за межі масиву
        name_1 = solution[i]
        name_2 = solution[i + 1]
        total_distance += distance(cities, name_1, name_2)
    total_distance += distance(cities, solution[-1], solution[0])
    return total_distance


def random_solution(cities):
    city_ids = list(cities.keys())  # шукає ключі індетифікатори
    random.shuffle(city_ids)
    return city_ids


def greedy_algorithm(cities, start_city=None, return_to_start=True, show_fitness=True):
    if start_city is None:
        start_city = list(cities.keys())[0]  # За замовчуванням перше місто
    route = [start_city]
    unvisited = set(cities.keys()) - {start_city}  # уникання дублікатів
    while unvisited:
        current_city = route[-1]
        nearest_city = min(unvisited, key=lambda city: distance(cities, city,
                                                                current_city))
        # Для кожного міста в множині
        # unvisited викликається функція
        # distance(cities, city, current_city)
        route.append(nearest_city)
        unvisited.remove(nearest_city)
    if return_to_start:
        route.append(start_city)
    if show_fitness:
        total_fitness = fitness(cities, route)
        print(f"Greedy algorithm route: {route}")
        print(f"Fitness score: {total_fitness} units")
    return route


def run_greedy_for_all_cities(cities):
    best_score = float('inf')  # фітнес зажди стає найкращим
    best_route = None
    city_ids = list(cities.keys())
    scores = []

    for city in city_ids:
        route = greedy_algorithm(cities, start_city=city, return_to_start=True, show_fitness=False)
        score = fitness(cities, route)
        scores.append((city, score, route))

        # Виведення для кожного міста
        print(f"Starting city: {city}, Route: {', '.join(map(str, route))}, Fitness: {score} units.")

        # Перевірка на найкращий результат
        if score < best_score:
            best_score = score
            best_route = route

    print(f"\nBest starting city: {best_route[0]} with score {best_score} units.")
    print(f"Best route: {info(cities, best_route)}")


def info(cities, solution):
    fitness_value = fitness(cities, solution)
    solution_res = '->'.join(map(str, solution))  # перебирає кожний елемент в ітерації
    return f"The route: {solution_res}\nFitness (Total Distance): {fitness_value} units."


def hundred_random_solutions(cities, num_solution=100):
    random_solutions = []
    for _ in range(num_solution):
        sol = random_solution(cities)
        fit = fitness(cities, sol)
        random_solutions.append((sol, fit))

    for i, (sol, fit) in enumerate(random_solutions):
        way = ', '.join(map(str, sol))
        print(f"Solution {i + 1}: {fit} units | Route: {way}")

    best_solution = None
    best_fitness = float('inf')

    for sol, fit in random_solutions:
        if fit < best_fitness:
            best_fitness = fit
            best_solution = sol

    best_way = ', '.join(map(str, best_solution))
    print(f"\nBest Solution: {best_fitness:.2f} units | Route: {best_way}")


def initial_population(cities, num_individuals=25, num_greedy=0):
    population = []
    city_ids = list(cities.keys())

    for i in range(num_greedy):
        start_city = city_ids[i % len(city_ids)]
        # обраїовує залишок для визначення
        # індексу і якщо індекс i перевищує кількість міст,
        # то ми починаємо вибір спочатку (тобто по колу).
        greedy_solution = greedy_algorithm(cities, start_city, return_to_start=False, show_fitness=False)
        population.append(greedy_solution)

    for i in range(num_individuals - num_greedy):
        rand_solution = random_solution(cities)
        population.append(rand_solution)

    return population


def population_info(cities, population):
    fitness_scores = [fitness(cities, sol) for sol in population]

    best_score = min(fitness_scores)
    best_index = fitness_scores.index(best_score)
    best_route = population[best_index]
    worst_score = max(fitness_scores)
    median_score = sorted(fitness_scores)[len(fitness_scores) // 2]
    average_score = sum(fitness_scores) / len(fitness_scores)

    for i, (sol, fit) in enumerate(zip(population, fitness_scores)):  # єднає два списки
        solution_str = '->'.join(map(str, sol))
        print(f"Solution {i + 1}: {solution_str} | Fitness: {fit} units")

    print(f"\nPopulation size: {len(population)}")
    print(f"Best fitness score: {best_score} units")
    print(f"Worst fitness score: {worst_score} units")
    print(f"Median fitness score: {median_score} units")
    print(f"Average fitness score: {average_score} units")

    best_route_str = ' -> '.join(map(str, best_route))
    print(f"\nBest route: {best_route_str}")
    print(f"Best route fitness: {best_score} units")


def tournament_selection(cities, population, k=20):
    k = min(k, len(population))
    chosen = random.sample(population, k)
    best = min(chosen, key=lambda sol: fitness(cities, sol))
    return best


def ordered_crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [None] * len(parent1)  # поступово заповнюємо дитину
    child[start:end + 1] = parent1[start:end + 1]  # беремо з поч до кін елементи батька
    # визначаємо першу позицію після першого баті і слідкуємо щоб не перевищити довжину масиву
    current_position = (end + 1) % len(parent1)

    for i in parent2:
        if i not in child:
            while child[current_position] is not None:
                current_position = (current_position + 1) % len(parent1)  # оновлюємо
            child[current_position] = i

    return child


def swap_mutation(solution):
    idx1, idx2 = random.sample(range(len(solution)), 2)
    mutated_solution = solution[:]  # копія
    mutated_solution[idx1], mutated_solution[idx2] = mutated_solution[idx2], mutated_solution[idx1]
    return mutated_solution


def create_new_epoch(cities, population, mutation_prob=0.1, crossover_prob=0.9, pop_size=25, tournament_k=5):
    new_population = []

    while len(new_population) < pop_size:
        parent1 = tournament_selection(cities, population, k=tournament_k)
        parent2 = tournament_selection(cities, population, k=tournament_k)

        if parent1 == parent2:
            parent2 = tournament_selection(cities, population, k=tournament_k)

        if random.random() < crossover_prob:
            child = ordered_crossover(parent1, parent2)
        else:
            child = parent1[:]  # якщо немає кросовера, дитина це копія parent1

        if random.random() < mutation_prob:
            child = swap_mutation(child)

        new_population.append(child)
    return new_population


def evolutionary_algorithm_with_plot(cities, epochs, population_size, mutation_prob, crossover_prob, tournament_k):
    population = initial_population(cities, population_size)

    best_solution = None
    best_fitness = float('inf')
    fitness_per_epoch = []

    for epoch in range(epochs):
        population = create_new_epoch(cities, population, mutation_prob, crossover_prob, population_size, tournament_k)

        for sol in population:
            fit = fitness(cities, sol)
            if fit < best_fitness:
                best_fitness = fit
                best_solution = sol
        fitness_per_epoch.append(best_fitness)
        route_str = ', '.join(map(str, best_solution))
        print(f"Epoch {epoch + 1}: {route_str} | Best fitness(total distance) = {best_fitness}")

    return best_solution, best_fitness, fitness_per_epoch


def plot_fitness_and_route(fitness_per_epoch_1, fitness_per_epoch_2, fitness_per_epoch_3,
                           cities, best_solution_1):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    best_fit_1 = min(fitness_per_epoch_1)
    best_fit_2 = min(fitness_per_epoch_2)
    best_fit_3 = min(fitness_per_epoch_3)

    ax1.plot(fitness_per_epoch_1, marker='o', linestyle='-', color='orange', linewidth=2,
             label=f'Mutation=0.01, Crossover=0.9 (Best: {best_fit_1:.2f})')
    ax1.plot(fitness_per_epoch_2, marker='o', linestyle='-', color='purple', linewidth=2,
             label=f'Mutation=0.01, Crossover=0.8 (Best: {best_fit_2:.2f})')
    ax1.plot(fitness_per_epoch_3, marker='o', linestyle='-', color='red', linewidth=2,
             label=f'Mutation=0.01, Crossover=0.7 (Best: {best_fit_3:.2f})')

    min_epoch_1 = fitness_per_epoch_1.index(best_fit_1)
    min_epoch_2 = fitness_per_epoch_2.index(best_fit_2)
    min_epoch_3 = fitness_per_epoch_3.index(best_fit_3)

    ax1.scatter(min_epoch_1, best_fit_1, color='yellow', s=150, zorder=5,
                edgecolor='black', linewidth=1.5, label='Best Points of fitness over epochs')
    ax1.scatter(min_epoch_2, best_fit_2, color='yellow', s=150, zorder=5,
                edgecolor='black', linewidth=1.5)
    ax1.scatter(min_epoch_3, best_fit_3, color='yellow', s=150, zorder=5,
                edgecolor='black', linewidth=1.5)

    ax1.set_title("Best Fitness over Epochs with Different Probabilities")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Best Fitness (Total Distance)")
    ax1.legend()
    ax1.grid(True)

    x_coords = [cities[id][0] for id in best_solution_1]
    y_coords = [cities[id][1] for id in best_solution_1]

    x_coords.append(x_coords[0])
    y_coords.append(y_coords[0])

    ax2.plot(x_coords, y_coords, 'purple', alpha=0.5)
    ax2.scatter(x_coords[:-1], y_coords[:-1], c='blue', s=100)

    for i, id in enumerate(best_solution_1):
        # додає анотацію на точках графіку
        # стр то текст і координати
        # зсув анотаціі 'offset points' означає, що зсув (5, 5) буде вимірюватися в одиницях "точок"
        ax2.annotate(str(id), (x_coords[i], y_coords[i]),
                     xytext=(5, 5), textcoords='offset points')

    ax2.set_title(f'The best route. \nBest Fitness: {best_fit_1:.2f}')
    ax2.set_xlabel("X-coordinates")
    ax2.set_ylabel("Y-coordinates")
    ax2.grid(True)

    plt.tight_layout()  # вітоматично підлаштовує графіки
    plt.show()


file_path1 = "berlin11_modified.tsp"
file_path2 = "berlin52.tsp"

cities_from_file1 = parse_tsp_file(file_path1)
cities_from_file2 = parse_tsp_file(file_path2)

population1 = initial_population(cities_from_file1)
population2 = initial_population(cities_from_file2)

print("------------ Calculating distance between two cities in berlin11_modified -----------")
city1 = 2
city2 = 4
distance_between = distance(cities_from_file1, city1, city2)
print(f"The distance between city {city1} and {city2} is: {distance_between:.2f} units")

print("------------ Calculating distance between two cities in berlin52 -----------")
city3 = 3
city4 = 1
distance_between2 = distance(cities_from_file2, city3, city4)
print(f"The distance between city {city3} and {city4} is: {distance_between2:.2f} units")

print("\n------------ Random solution ------------")
random_solution_list = random_solution(cities_from_file1)
print("Random for berlin11_modified.tsp:", random_solution_list)
print(info(cities_from_file1, random_solution_list))

print("\n------------ Reading data from berlin11_modified.tsp ------------")
print(f"Cities loaded from {file_path1}: {cities_from_file1}")

print("\n------------ Greedy algorithm for berlin11_modified.tsp ------------")
greedy_algorithm(cities_from_file1)

print("\n------------ Run greedy algorithm for all starting cities for berlin11_modified.tsp ------------")
run_greedy_for_all_cities(cities_from_file1)

print("\n------------ Generate 100 random solutions for berlin11_modified.tsp------------")
hundred_random_solutions(cities_from_file1)

print("\n------------ Reading data from berlin52 ------------")
print(f"Cities loaded from {file_path2}: {cities_from_file2}")

print("\n------------ Greedy algorithm for berlin52.tsp ------------")
greedy_algorithm(cities_from_file2)

print("\n------------ Run greedy algorithm for all starting cities for berlin52.tsp ------------")
run_greedy_for_all_cities(cities_from_file2)

print("\n------------ Generate 100 random solutions for berlin52.tsp------------")
hundred_random_solutions(cities_from_file2)
#
# print("\n------------ Population ------------")
# print("\n== Population for berlin11_modified.tsp: ==")
# population_info(cities_from_file1, population1)
#
# print("\n------------ Tournament selection ------------")
# selected = tournament_selection(cities_from_file1, population1)
# print(f"Selected route for berlin11_modified.tsp: {selected}")
#
# print("\n------------ Ordered crossover ------------")
# parent1 = tournament_selection(cities_from_file1, population1)
# parent2 = tournament_selection(cities_from_file1, population1)
# child1 = ordered_crossover(parent1, parent2)
# print(f"berlin11_modified.tsp - Parent 1: {parent1}")
# print(f"berlin11_modified.tsp - Parent 2: {parent2}")
# print(f"berlin11_modified.tsp - Child: {child1}")
#
# print("\n------------ Swap Mutation ------------")
# child1_swap = swap_mutation(child1)
# print(f"Swap mutation for berlin11_modified.tsp: {child1_swap}")
#
# print("\n------------ Solving for berlin11_modified ------------")
# # Configuration 1
# best_solution_1, best_fitness_1, fitness_per_epoch_1 = evolutionary_algorithm_with_plot(
#     cities_from_file1, epochs=50, population_size=25, mutation_prob=0.99, crossover_prob=0.9, tournament_k=20)
#
# print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
# print("Best Solution for Configuration 1:", best_solution_1)
# print("Best Fitness (Total Distance) for Configuration 1:", best_fitness_1)
# print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
#
# # Configuration 2
# best_solution_2, best_fitness_2, fitness_per_epoch_2 = evolutionary_algorithm_with_plot(
#     cities_from_file1, epochs=50, population_size=25, mutation_prob=0.99, crossover_prob=0.8, tournament_k=20)
#
# print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
# print("Best Solution for Configuration 2:", best_solution_2)
# print("Best Fitness (Total Distance) for Configuration 2:", best_fitness_2)
# print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
#
# # Configuration 3
# best_solution_3, best_fitness_3, fitness_per_epoch_3 = evolutionary_algorithm_with_plot(
#     cities_from_file1, epochs=50, population_size=25, mutation_prob=0.99, crossover_prob=0.7, tournament_k=20)
#
# print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
# print("Best Solution for Configuration 3:", best_solution_3)
# print("Best Fitness (Total Distance) for Configuration 3:", best_fitness_3)
# print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
#
# plot_fitness_and_route(fitness_per_epoch_1, fitness_per_epoch_2, fitness_per_epoch_3, cities_from_file1,
#                        best_solution_1)

print("\n------------ Population ------------")
print("\n== Population for berlin52.tsp: ==")
population_info(cities_from_file2, population2)

print("\n------------ Tournament selection ------------")
selected2 = tournament_selection(cities_from_file2, population2)
print(f"Selected route for berlin52.tsp: {selected2}")

print("\n------------ Ordered crossover ------------")
parent3 = tournament_selection(cities_from_file2, population2)
parent4 = tournament_selection(cities_from_file2, population2)
child2 = ordered_crossover(parent3, parent4)
print(f"berlin52.tsp - Parent 1: {parent3}")
print(f"berlin52.tsp - Parent 2: {parent4}")
print(f"berlin52.tsp - Child: {child2}")

print("\n------------ Swap Mutation ------------")
child2_swap = swap_mutation(child2)
print(f"Swap mutation for berlin52.tsp: {child2_swap}")

print("\n------------ Solving for berlin52 ------------")
best_solution_2_1, best_fitness_2_1, fitness_per_epoch_2_1 = evolutionary_algorithm_with_plot(
    cities_from_file2, epochs=50, population_size=100, mutation_prob=0.01, crossover_prob=0.9, tournament_k=20)

print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
print("Best Solution for Configuration 1:", best_solution_2_1)
print("Best Fitness (Total Distance) for Configuration 1:", best_fitness_2_1)
print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")

# Configuration 2
best_solution_2_2, best_fitness_2_2, fitness_per_epoch_2_2 = evolutionary_algorithm_with_plot(
    cities_from_file2, epochs=50, population_size=100, mutation_prob=0.01, crossover_prob=0.8, tournament_k=20)

print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
print("Best Solution for Configuration 2:", best_solution_2_2)
print("Best Fitness (Total Distance) for Configuration 2:", best_fitness_2_2)
print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")

# Configuration 3
best_solution_2_3, best_fitness_2_3, fitness_per_epoch_2_3 = evolutionary_algorithm_with_plot(
    cities_from_file2, epochs=50, population_size=100, mutation_prob=0.01, crossover_prob=0.7, tournament_k=20)

print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
print("Best Solution for Configuration 3:", best_solution_2_3)
print("Best Fitness (Total Distance) for Configuration 3:", best_fitness_2_3)
print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")

plot_fitness_and_route(fitness_per_epoch_2_1, fitness_per_epoch_2_2, fitness_per_epoch_2_3,
                       cities_from_file2, best_solution_2_1)
