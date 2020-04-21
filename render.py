import pygame
import time
import game
from lstm_ai import LSTMAi

tile_size = 50
tile_margin = 5

screen_size = (game.world_width * tile_size, game.world_height * tile_size)


class Renderer:
    def __init__(self, ai = 'player'):
        self.world = game.World()
        self.ai = ai
        self.screen = pygame.display.set_mode(screen_size)
        
    def render_loop(self):
        last_update = time.time()
        status = game.MoveResult.normal

        stop = False
        score = 0

        while not stop:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    stop = True
            pressed = pygame.key.get_pressed()

            if self.ai == 'player':
                if pressed[pygame.K_w]:
                    self.world.set_direction((0, -1))
                elif pressed[pygame.K_s]:
                    self.world.set_direction((0, 1))
                if pressed[pygame.K_a]:
                    self.world.set_direction((-1, 0))
                elif pressed[pygame.K_d]:
                    self.world.set_direction((1, 0))
            if pressed[pygame.K_ESCAPE]:
                stop = True


            if time.time() - last_update > 0.2:
                if self.ai != 'player':
                    predictions = self.ai.predict_best_moves([self.world])
                    print(game.directions[predictions[0]])

                    self.world.set_direction(
                        game.directions[predictions[0]]
                    )
                
                last_update = time.time()
                
                if(status != game.MoveResult.death):
                    status = self.world.forward()
                    print(status)
                    print(score)

                    if status == game.MoveResult.eat:
                        score += 1

                if(status == game.MoveResult.death):
                    self.screen.fill((200, 50, 50))
                else:
                    self.screen.fill((255, 255, 255))

                for x, y in self.world.snake:
                    color = (100, 100, 100) if (x, y) == self.world.snake[0] else (0, 0, 0)
                    self.screen.fill(color, pygame.Rect(x * tile_size + tile_margin, y * tile_size + tile_margin, tile_size - 2 * tile_margin, tile_size - 2 * tile_margin))
                    
                x, y = self.world.food
                self.screen.fill((50, 200, 50), pygame.Rect(x * tile_size + tile_margin, y * tile_size + tile_margin, tile_size - 2 * tile_margin, tile_size - 2 * tile_margin))

                pygame.display.flip()
                
        pygame.display.quit()