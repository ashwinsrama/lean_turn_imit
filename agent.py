import torch
import random
import pygame
import numpy as np
from collections import deque
from game import CarGameAI
from model import Linear_QNet, QTrainer


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.0001


class Agent:

    def __init__(self):
        self.n_games = 0
        self.gamma = 0.80 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.outputs = 4
        self.model = Linear_QNet(431, 512, 256, self.outputs)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):

        #state of the environment, used for features
        state = [
        
            # front location
            game.car.front[0], game.car.front[1],
	
	        game.frame_iteration,
	    
            # vertices
            game.car.x_fl, game.car.y_fl,
            game.car.x_fr, game.car.y_fr,
            game.car.x_br, game.car.y_br,
            game.car.x_bl, game.car.y_bl,
            
            game.food[0] - game.car.front[0],  # x_dist to food
            game.food[1] - game.car.front[1]   # y_dist to food

            ] + [game.view_ahead_pt(dist, ang) for dist in range(20, 121, 10) for ang in range(-120, 121, 10)] + [game.view_ahead_pt(dist, ang) for dist in range(20, 121, 10) for ang in range(120, 241, 10)] #perception data

        return np.array(state, dtype=int)


    def remember(self, state, action, reward, next_state, done): #add "sarsa" to memory
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones) #train model on batch of "sarsa"s


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done) #train model on single "sarsa"


    def get_action(self, state):

        #explore random action if the number of games played < 250
        if self.n_games < 10:

            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                    key = pygame.key.get_pressed()
                    if key[pygame.K_LEFT]:
                        return [0, 0, 0, 1]
                    if key[pygame.K_RIGHT]:
                        return [0, 0, 1, 0]
                    if key[pygame.K_UP]:
                        return [1, 0, 0, 0]
                    if key[pygame.K_DOWN]:
                        return [0, 1, 0, 0]
    

        #use model to predict best action if number of games >= 250
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move = [0] * self.outputs
            final_move[move] = 1

        return final_move
