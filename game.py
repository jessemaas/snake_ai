import random
import enum

world_width = 9
world_height = 9
initial_snake_length = 4
directions = [
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
]

class World:
    def __init__(self, width = world_width, height = world_height, initial_snake_length = initial_snake_length):
        self.width = width
        self.height = height
        self.snake_alive = True

        snake_head = (
            random.randint(initial_snake_length - 1, width - initial_snake_length), # random x
            random.randint(initial_snake_length - 1, height - initial_snake_length) # random y
        )

        self.snake_direction = random_direction()

        # create snake
        self.snake = [snake_head]

        for i in range(1, initial_snake_length):
            self.snake.append(
                (snake_head[0] + self.snake_direction[0] * -i, snake_head[1] + self.snake_direction[1] * -i)
            )
        
        self.place_food()
        
        
    def forward(self):
        snake_head = self.snake[0]
        new_head = (snake_head[0] + self.snake_direction[0], snake_head[1] + self.snake_direction[1])

        if new_head == self.food:
            self.snake.insert(0, new_head)
            self.place_food()
            return MoveResult.eat           
        elif new_head[0] < 0 or new_head[0] >= self.width or new_head[1] < 0 or new_head[1] >= self.height or new_head in self.snake:
            self.snake_alive = False
            return MoveResult.death
        else:
            del self.snake[-1]
            self.snake.insert(0, new_head)
            return MoveResult.normal

    def place_food(self):
        self.food = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
        
        while self.food in self.snake:
            self.food = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))

    def set_direction(self, direction):
        # if -self.snake_direction[0] != direction[0] or -self.snake_direction[1] != direction[1]:
            self.snake_direction = direction

# returns a random one out of four possible directions (up, down, left, right)
def random_direction():
    return directions[random.randint(0, 3)]

class MoveResult(enum.Enum):
    normal = 0
    eat = 1
    death = 2