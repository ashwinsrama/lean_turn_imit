import pygame
import numpy as np
from math import sin, cos, radians
from random import randint, choice
from car import Car
from calc import poly_points, distance, in_rect

pygame.init()

colors = {   
            "black"  : (0, 0, 0),
            "grey"   : (70, 70, 70), 
            "blue"   : (0, 0, 255),
            "green"  : (0, 255, 0), 
            "yellow" : (253, 218, 22), 
            "red"    : (255, 0, 0), 
            "beige"  : (247, 233, 210),
            "white"  : (255, 255, 255),
        }

FRAME_RATE = 10000

class CarGameAI:
    def __init__(self, w=600, h=400):
    	#initialize the Pygame GUI frame
        self.w = w
        self.h = h
        self.gameDisplay = pygame.display.set_mode((self.w, self.h))
        self.clock=pygame.time.Clock()
        self.curr_cars = []
        self.game_iteration = 0
        self.reset()


    def reset(self):
    	#initialize the car object on the frame
        self.car = Car("red", 160, 270, 180, 270, 180, 310, 160, 310, deg=90)
        self.food = (self.w-10, 188) #location of center of food pellet
        self.min_dist = float("inf") #minimum distance so far to food
        self.frame_iteration = 0


    def is_collision(self): #check if vertexes of car has collision with boundary of frame or goes off-road
        #on-road collision
        all_poly_points = []
        for car_obs in self.curr_cars:
            all_poly_points += poly_points(car_obs.vertices)

        car_poly_points = poly_points(self.car.vertices)
        for pts in car_poly_points:
            pt = (int(pts[0]), int(pts[1]))
            if pt in all_poly_points:
                return True
        
        #off-road collision
        for vtx in poly_points(self.car.vertices):
            if not in_rect([vtx], 0, self.w, 150, 225) and not in_rect([vtx], 150, 195, 150, self.h):
                return True
        
        return False


    def view_ahead_pt(self, dist, angle): #check if a point ahead of car front-center points has off-road collision or frame boundary collision
        x, y = self.car.front
        x_new = x + cos(radians(self.car.deg + angle)) * dist
        y_new = y - sin(radians(self.car.deg + angle)) * dist

        vtx = (x_new, y_new)
        
        if not in_rect([vtx], 0, self.w, 150, 225) and not in_rect([vtx], 150, 195, 150, self.h): #boundaries of road
            return 0
        
        elif distance(vtx, self.food) <= 5: #x_new, y_new near food
            return 3
        
        for car_obs in self.curr_cars:
            if in_rect([vtx], car_obs.back[0], car_obs.front[0], car_obs.y_fl, car_obs.y_fr, pad=5):
                return 1

        else:
            return 2


    def ate_food(self): #checks if FRONT (left, right, or center) of the car has intersected with the food
        car_front_right = int(self.car.x_fr), int(self.car.y_fr)
        car_front_left = int(self.car.x_fl), int(self.car.y_fl)
        car_front_center = int(self.car.front[0]), int(self.car.front[1])
        return distance(car_front_center, self.food) < 5 or distance(car_front_right, self.food) < 5 or distance(car_front_left, self.food) < 5


    def draw_point(self, center, size=1, color="green"): #draw a point at position (x, y)
        x, y = center
        pygame.draw.polygon(self.gameDisplay, colors[color], [(x-size, y-size), (x+size, y-size), (x+size, y+size), (x-size, y+size)])


    def draw_road(self): #draw the road using Pygame polygons customized to exact pixel locations
        y = 150
        x = 150
        pygame.draw.polygon(self.gameDisplay, colors["grey"], [(0, y), (self.w, y), (self.w, y+75), (0, y+75)])
        pygame.draw.polygon(self.gameDisplay, colors["grey"], [(x, y), (x+45, y), (x+45, self.h), (x, self.h)])
        for i in range(0, self.w, 40):
            pygame.draw.polygon(self.gameDisplay, colors["white"], [(i, y+37), (i+10, y+37), (i+10, y+38), (i, y+38)])

    def update_ui(self, perception=True): #generate the GUI on Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.gameDisplay.fill(colors["beige"]) #draw background
        self.clock.tick(FRAME_RATE)

        self.draw_road() #draw road/infrastructure
        pygame.draw.polygon(self.gameDisplay, colors[self.car.color], self.car.vertices) #draw car
        self.draw_point(self.car.front, size=2, color="black") #draw front indicator
        pygame.draw.rect(self.gameDisplay, colors["yellow"], pygame.Rect(self.food[0]-5, self.food[1]-5, 10, 10)) #draw food pellet
        

        #######################################################################
        if self.game_iteration % 50 == 0:
            if randint(0, 51) < 25:
                new_obs = Car(choice(["blue", "black", "white", "yellow"]), 0, 160, 0, 180, -40, 180, -40, 160, deg=0)
            else:
                new_obs = Car(choice(["blue", "black", "white", "yellow"]), 0, 195, 0, 215, -40, 215, -40, 195, deg=0)
            self.curr_cars.append(new_obs)

        for elem in self.curr_cars:
            if elem.x_bl < self.w+20:
                self.clock.tick(FRAME_RATE)
                pygame.draw.polygon(self.gameDisplay, colors[elem.color], elem.vertices)
                elem.move_lin(True, randint(1, 6))

        self.curr_cars = [i for i in self.curr_cars if i.x_bl < self.w+20]
        #######################################################################

        
        #if user enables perception, draw perception samples
        if perception:
            x, y = self.car.front
            for angle in range(-120, 121, 10):
                for freq in range(20, 121, 10):
                    x_new = x + cos(radians(self.car.deg+angle))*freq
                    y_new = y - sin(radians(self.car.deg+angle))*freq
                    
                    self.draw_point((x_new, y_new), 1)

                    for car_obs in self.curr_cars:
                        if in_rect([(x_new, y_new)], car_obs.back[0], car_obs.front[0], car_obs.y_fl, car_obs.y_fr, pad=5):
                            self.draw_point((x_new, y_new), 1, "red")

        if perception:
            x, y = self.car.back
            for angle in range(120, 241, 10):
                for freq in range(20, 121, 10):
                    x_new = x + cos(radians(self.car.deg+angle))*freq
                    y_new = y - sin(radians(self.car.deg+angle))*freq
                    
                    self.draw_point((x_new, y_new), 1)

                    for car_obs in self.curr_cars:
                        if in_rect([(x_new, y_new)], car_obs.back[0], car_obs.front[0], car_obs.y_fl, car_obs.y_fr, pad=5):
                            self.draw_point((x_new, y_new), 1, "red")

        pygame.display.flip()


    def play_step(self, action):
    	#given an action passed by the agent, perform that action
        self.frame_iteration += 1
        self.game_iteration += 1
        self.move(action)
        
        reward = 0
        game_over = False

        #end the game upon collision
        if self.frame_iteration > 500:
            game_over = True
            reward = -2500
            return reward, game_over
            
        if self.is_collision():
            game_over = True
            reward = -1250
            return reward, game_over

        #update the reward according to food and distance heuristics
        if self.ate_food():
            game_over = True
            reward = 1500
            return reward, game_over
            
        else: 
            dst = distance(self.car.front, self.food)
            if dst < self.min_dist: #reward when car gets closer to food
                self.min_dist = dst 
                reward = 50
            else:
                reward = -50 #penalize when car gets further from food
            return reward, game_over

    def move(self, action):
        if   np.array_equal(action, [1, 0, 0, 0]):
            self.car.move_lin(True, 2) #linear forward at vel=2
        elif np.array_equal(action, [0, 1, 0, 0]):
            self.car.move_lin(True, 4) #linear forward at vel=4
        elif np.array_equal(action, [0, 0, 1, 0]):
            self.car.rotate(True, True, 6) #rotate clockwise and forward
        elif np.array_equal(action, [0, 0, 0, 1]):
            self.car.rotate(False, True, 6) #rotate counter-clockwise and forward
        else:
            pass
        self.clock.tick(FRAME_RATE)
        self.update_ui()
