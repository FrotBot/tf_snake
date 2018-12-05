# SNAKES GAME
# Use ARROW KEYS to play, SPACE BAR for pausing/resuming and Esc Key for exiting
import sys
from operator import itemgetter
import tensorflow as tf
from tensorflow import keras
import numpy as np
import curses
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
from random import randint
import time

individual_count = 20
#generations = 10000
mutation_rate = 1.2
true_random_food = True
long_time = 5
use_distance = False
rounds_per_individual = 7
max_foodless_distance = 800
load_weights = False

print_gen = True
show_gen = 1
show_all_individual = False

foodlist = [[randint(1, 18), randint(1, 58)]]
for i in range(200):
    foodlist.append([randint(1, 18), randint(1, 58)])


counter = [0 for i in range(individual_count)]
score = [0 for i in range(individual_count)]

net_input_useless = np.zeros((1, 5))
model_list = [keras.Sequential() for i in range(individual_count)]
model_save_list = [keras.Sequential() for i in range(individual_count)]
for i in range(individual_count):
    model_list[i].add(keras.layers.Dense(
        6, activation=keras.layers.LeakyReLU(alpha=0.3)))
    model_list[i].add(keras.layers.Dense(
        4, activation=keras.layers.LeakyReLU(alpha=0.3)))
    model_list[i].add(keras.layers.Dense(3, activation=tf.nn.softmax))
    model_list[i].compile(optimizer=tf.train.AdamOptimizer(),
                          loss='mean_squared_error',
                          metrics=['accuracy'])
    # no idea why this is needed to initialize...
    model_list[i].predict(net_input_useless)


for i in range(individual_count):
    model_save_list[i].add(keras.layers.Dense(
        6, activation=keras.layers.LeakyReLU(alpha=0.3)))
    model_save_list[i].add(keras.layers.Dense(
        4, activation=keras.layers.LeakyReLU(alpha=0.3)))
    model_save_list[i].add(keras.layers.Dense(3, activation=tf.nn.softmax))
    model_save_list[i].compile(optimizer=tf.train.AdamOptimizer(),
                               loss='mean_squared_error',
                               metrics=['accuracy'])
    # no idea why this is needed to initialize...
    model_save_list[i].predict(net_input_useless)


highscore = 0
highcount = 0
total_highscore = 0

if(load_weights):
    model_list[0].load_weights('long_train_indiv_0.h5')
    model_list[1].load_weights('long_train_indiv_1.h5')
    model_list[2].load_weights('long_train_indiv_2.h5')
    model_list[3].load_weights('long_train_indiv_3.h5')
    model_list[4].load_weights('long_train_indiv_4.h5')

curses.initscr()
current_generation = 0

# for current_generation in range(1, generations+1):
while True:
    current_generation = current_generation+1
    for individual in range(individual_count):

        timer = 0
        score[individual] = 0
        counter[individual] = 0

        dist = 0
        for round_for_this_individual in range(rounds_per_individual):

            print_gen = (current_generation -
                         1) % show_gen == 0 and individual == 0
            if(show_all_individual):
                print_gen = True
            #print_gen = True

            #if(print_gen):
            win = curses.newwin(20, 60, 0, 0)
            win.keypad(1)
            curses.noecho()
            curses.curs_set(0)
            win.border(0)
            win.nodelay(1)
            

            # Initializing values
            key = KEY_RIGHT

            # Initial snake co-ordinates
            snake = [[4, 10], [4, 9], [4, 8], [4, 7], [4, 6], [4, 5], [4, 4], [4, 3], [
                4, 2], [4, 9], [4, 8], [4, 7], [4, 6], [4, 5], [4, 4], [4, 3], [4, 2]]
            # First food co-ordinates
            if(not true_random_food):
                food = foodlist[0]
            else:
                food = [randint(1, 18), randint(1, 58)]

            # Prints the food
            if(print_gen):
                win.addch(food[0], food[1], 'X')

            hand_control = False
            genome_str = 'Generation ' + \
                str(current_generation) + ' Individual ' + str(individual)

            x_food_dist = 0
            y_food_dist = 0

            loop_counter = 0
            pred_key = 0
            net_input = np.zeros((1, 5))

            timer = long_time if individual == 0 else 0

            if(not print_gen):
                win.border(0)
                win.addstr(5, 18, 'EVALUATING GENERATION ' + str(current_generation))
                win.addstr(9, 20, 'INDIVIDUAL  ' + str(individual+1) + "/20")
                #win.addstr(5, 18, '____________________')
                win.addch(11, 18, '[')
                for i in range(individual_count):
                    if(i < individual):
                        win.addch(11,19+i, '█')
                    else:
                        win.addch(11,19+i, ' ')
                win.addch(11,38, ']')                    
                win.addstr(19, 5, 'LastGen HiScore: ' + str(highscore))
                win.addstr(19, 30, 'Total HiScore: ' + str(total_highscore))
                win.timeout(timer)
                win.getch()

            while True:                                                   # While Esc key is not pressed

                if(print_gen):
                    win.border(0)
                    # Printing 'Score' and
                    win.addstr(
                        0, 2, 'Score: ' + str(int(score[individual])))
                    win.addstr(19, 5, 'LastGen HiScore: ' + str(highscore))
                    win.addstr(19, 30, 'Total HiScore: ' + str(total_highscore))
                    # 'SNAKE' strings
                    win.addstr(0, 27, genome_str)
                    # Increases the speed of Snake as its length increases
                    win.timeout(timer)
                

                counter[individual] += 1
                loop_counter += 1

                # calc distance to next hurdle
                dist = 0
                left_dist = 0
                will_bite_thyself = False
                if(key == KEY_DOWN):
                    for ray in range(snake[0][0]+1, 19):
                        if [ray, snake[0][1]] in snake[1:]:
                            # check for nearest segment
                            if(will_bite_thyself):
                                if(dist > abs(ray-snake[0][0])):
                                    dist = abs(ray-snake[0][0])
                            else:
                                dist = abs(ray-snake[0][0])
                                will_bite_thyself = True
                    if(not will_bite_thyself):
                        dist = abs(19-snake[0][0])

                    will_bite_thyself=True
                    for ray in range(snake[0][1]+1, 59):
                        if [snake[0][0], ray] in snake[1:]:
                            # check for nearest segment
                            if(will_bite_thyself):
                                if(left_dist > abs(ray-snake[0][1])):
                                    left_dist = abs(ray-snake[0][1])
                            else:
                                left_dist = abs(ray-snake[0][1])
                                will_bite_thyself = True
                    if(not will_bite_thyself):
                        left_dist = abs(59-snake[0][1])

                    will_bite_thyself=True
                    for ray in range(0, snake[0][1]):
                        if [snake[0][0], ray] in snake[1:]:
                            # check for nearest segment
                            if(will_bite_thyself):
                                if(right_dist > abs(ray-snake[0][1])):
                                    right_dist = abs(ray-snake[0][1])
                            else:
                                right_dist = abs(ray-snake[0][1])
                                will_bite_thyself = True
                    if(not will_bite_thyself):
                        right_dist = abs(snake[0][1])

                if(key == KEY_UP):
                    for ray in range(0, snake[0][0]):
                        if [ray, snake[0][1]] in snake[1:]:
                            # check for nearest segment
                            if(will_bite_thyself):
                                if(dist > abs(ray-snake[0][0])):
                                    dist = abs(ray-snake[0][0])
                            else:
                                dist = abs(ray-snake[0][0])
                                will_bite_thyself = True
                    if(not will_bite_thyself):
                        dist = abs(snake[0][0])

                    for ray in range(snake[0][1]+1, 59):
                        if [snake[0][0], ray] in snake[1:]:
                            # check for nearest segment
                            if(will_bite_thyself):
                                if(right_dist > abs(ray-snake[0][1])):
                                    right_dist = abs(ray-snake[0][1])
                            else:
                                right_dist = abs(ray-snake[0][1])
                                will_bite_thyself = True
                    if(not will_bite_thyself):
                        right_dist = abs(59-snake[0][1])

                    for ray in range(0, snake[0][1]):
                        if [snake[0][0], ray] in snake[1:]:
                            # check for nearest segment
                            if(will_bite_thyself):
                                if(left_dist > abs(ray-snake[0][1])):
                                    left_dist = abs(ray-snake[0][1])
                            else:
                                left_dist = abs(ray-snake[0][1])
                                will_bite_thyself = True
                    if(not will_bite_thyself):
                        left_dist = abs(snake[0][1])

                if(key == KEY_RIGHT):
                    for ray in range(snake[0][1]+1, 59):
                        if [snake[0][0], ray] in snake[1:]:
                            # check for nearest segment
                            if(will_bite_thyself):
                                if(dist > abs(ray-snake[0][1])):
                                    dist = abs(ray-snake[0][1])
                            else:
                                dist = abs(ray-snake[0][1])
                                will_bite_thyself = True
                    if(not will_bite_thyself):
                        dist = abs(59-snake[0][1])

                    for ray in range(0, snake[0][0]):
                        if [ray, snake[0][1]] in snake[1:]:
                            # check for nearest segment
                            if(will_bite_thyself):
                                if(left_dist > abs(ray-snake[0][0])):
                                    left_dist = abs(ray-snake[0][0])
                            else:
                                left_dist = abs(ray-snake[0][0])
                                will_bite_thyself = True
                    if(not will_bite_thyself):
                        left_dist = abs(snake[0][0])

                    for ray in range(snake[0][0]+1, 19):
                        if [ray, snake[0][1]] in snake[1:]:
                            # check for nearest segment
                            if(will_bite_thyself):
                                if(right_dist > abs(ray-snake[0][0])):
                                    right_dist = abs(ray-snake[0][0])
                            else:
                                right_dist = abs(ray-snake[0][0])
                                will_bite_thyself = True
                    if(not will_bite_thyself):
                        right_dist = abs(19-snake[0][0])

                if(key == KEY_LEFT):
                    for ray in range(0, snake[0][1]):
                        if [snake[0][0], ray] in snake[1:]:
                            # check for nearest segment
                            if(will_bite_thyself):
                                if(dist > abs(ray-snake[0][1])):
                                    dist = abs(ray-snake[0][1])
                            else:
                                dist = abs(ray-snake[0][1])
                                will_bite_thyself = True
                    if(not will_bite_thyself):
                        dist = abs(snake[0][1])

                    for ray in range(snake[0][0]+1, 19):
                        if [ray, snake[0][1]] in snake[1:]:
                            # check for nearest segment
                            if(will_bite_thyself):
                                if(left_dist > abs(ray-snake[0][0])):
                                    left_dist = abs(ray-snake[0][0])
                            else:
                                left_dist = abs(ray-snake[0][0])
                                will_bite_thyself = True
                    if(not will_bite_thyself):
                        left_dist = abs(19-snake[0][0])

                    for ray in range(0, snake[0][0]):
                        if [ray, snake[0][1]] in snake[1:]:
                            # check for nearest segment
                            if(will_bite_thyself):
                                if(right_dist > abs(ray-snake[0][0])):
                                    right_dist = abs(ray-snake[0][0])
                            else:
                                right_dist = abs(ray-snake[0][0])
                                will_bite_thyself = True
                    if(not will_bite_thyself):
                        right_dist = abs(snake[0][0])

                # calc distance to food
                x_food_dist = snake[0][1]-food[1]
                y_food_dist = snake[0][0]-food[0]

                side_food_dist = 0
                front_food_dist = 0

                if(key == KEY_RIGHT):
                    side_food_dist = y_food_dist
                    front_food_dist = x_food_dist
                if(key == KEY_LEFT):
                    side_food_dist = -y_food_dist
                    front_food_dist = -x_food_dist
                if(key == KEY_DOWN):
                    side_food_dist = -x_food_dist
                    front_food_dist = y_food_dist
                if(key == KEY_RIGHT):
                    side_food_dist = x_food_dist
                    front_food_dist = -y_food_dist

                # neural net input
                net_input[0, 0] = float(dist)
                net_input[0, 1] = float(side_food_dist)
                net_input[0, 2] = float(front_food_dist)
                net_input[0, 3] = float(left_dist)
                net_input[0, 4] = float(right_dist)
                # net_input[0,1]=float(x_food_dist)
                # net_input[0,2]=float(y_food_dist)
                #net_input[0,3]= 30.0 if (key==KEY_UP) else 0.0
                #net_input[0,4]= 30.0 if (key==KEY_DOWN) else 0.0
                #net_input[0,5]= 30.0 if (key==KEY_LEFT) else 0.0
                #net_input[0,6]= 30.0 if (key==KEY_RIGHT) else 0.0

                # calc net

                predictions = model_list[individual].predict(net_input)

                pred_key = np.argmax(predictions[0])

                # Previous key pressed
                prevKey = key

                if(print_gen):
                    event = win.getch()
                    if (event == 27):
                        curses.endwin()
                        sys.exit()
                if(hand_control):
                    key = key if event == -1 else event
                else:
                    #key = KEY_UP if (pred_key == 1) else key
                    #key = KEY_DOWN if (pred_key == 2) else key
                    #key = KEY_RIGHT if (pred_key == 3) else key
                    #key = KEY_LEFT if (pred_key == 0) else key
                    if(pred_key == 0):  # links
                        if(key == KEY_RIGHT):
                            key = KEY_UP
                        elif(key == KEY_UP):
                            key = KEY_LEFT
                        elif(key == KEY_LEFT):
                            key = KEY_DOWN
                        elif(key == KEY_DOWN):
                            key = KEY_RIGHT
                    if(pred_key == 1):  # rechts
                        if(key == KEY_RIGHT):
                            key = KEY_DOWN
                        elif(key == KEY_UP):
                            key = KEY_RIGHT
                        elif(key == KEY_LEFT):
                            key = KEY_UP
                        elif(key == KEY_DOWN):
                            key = KEY_LEFT

                # If SPACE BAR is pressed, wait for another
                if key == ord(' '):
                    # one (Pause/Resume)
                    key = -1
                    while key != ord(' '):
                        key = win.getch()
                    key = prevKey
                    continue

                # If an invalid key is pressed
                if key not in [KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN, 27]:
                    key = prevKey

                # Calculates the new coordinates of the head of the snake. NOTE: len(snake) increases.
                # This is taken care of later at [1].
                snake.insert(0, [snake[0][0] + (key == KEY_DOWN and 1) + (key == KEY_UP and -1),
                                 snake[0][1] + (key == KEY_LEFT and -1) + (key == KEY_RIGHT and 1)])

                # If snake crosses the boundaries, kill it
                if snake[0][0] == 0:  # y
                    break
                if snake[0][1] == 0:  # x
                    break
                if snake[0][0] == 19:  # y
                    break
                if snake[0][1] == 59:  # x
                    break

                # Exit if snake crosses the boundaries (Uncomment to enable)
                # if snake[0][0] == 0 or snake[0][0] == 19 or snake[0][1] == 0 or snake[0][1] == 59: break

                # if snake caught in infinite loop
                if loop_counter > max_foodless_distance:
                    counter[individual] = 10
                    break

                # If snake runs over itself
                if snake[0] in snake[1:]:
                    break

                if snake[0] == food:
                    loop_counter = 0
                    # When snake eats the food
                    counter[individual] = 0
                    food = []
                    score[individual] += 1
                    while food == []:
                        # Calculating next food's coordinates
                        #food = [randint(1, 18), randint(1, 58)]
                        if(not true_random_food):
                            food = foodlist[score[individual] % 200]
                        else:
                            food = [randint(1, 18), randint(1, 58)]
                        if food in snake:
                            food = []
                    if(print_gen):
                        win.addch(food[0], food[1], 'X')
                else:
                    # [1] If it does not eat the food, length decreases
                    last = snake.pop()
                    if(print_gen):
                        win.addch(last[0], last[1], ' ')
                if(print_gen):
                    win.addch(snake[0][0], snake[0][1], '█')

                

    if(not use_distance):
        for individual in range(individual_count):
            counter[individual] = 0

    #yz = zip(score, model_list)
    #sorted(yz, key=lambda pair: pair[0])
    #combined_list = [model_list, score, counter]
    combined_list = [(model_list[0], score[0], counter[0])]
    for individual in range(1, individual_count):
        combined_list.append(
            (model_list[individual], score[individual], counter[individual]))

    combined_list.sort(key=itemgetter(1, 2))
    # print(combined_list)

    highscore = combined_list[individual_count-1][1]
    highcount = combined_list[individual_count-1][2]
    total_highscore = highscore if highscore > total_highscore else total_highscore

    # for i in range(0,20):
    #    print(combined_list[i][2])
    #    print(len(model_save_list[0].get_weights()))
    #    print(len(model_list[0].get_weights()))

    # time.sleep(10)

    # do selection
    model_save_list[0].set_weights(
        combined_list[individual_count-1][0].get_weights())
    model_save_list[1].set_weights(
        combined_list[individual_count-1][0].get_weights())
    model_save_list[2].set_weights(
        combined_list[individual_count-1][0].get_weights())
    model_save_list[3].set_weights(
        combined_list[individual_count-1][0].get_weights())
    model_save_list[4].set_weights(
        combined_list[individual_count-2][0].get_weights())
    model_save_list[5].set_weights(
        combined_list[individual_count-2][0].get_weights())
    model_save_list[6].set_weights(
        combined_list[individual_count-2][0].get_weights())
    model_save_list[7].set_weights(
        combined_list[individual_count-3][0].get_weights())
    model_save_list[8].set_weights(
        combined_list[individual_count-3][0].get_weights())
    model_save_list[9].set_weights(
        combined_list[individual_count-3][0].get_weights())
    model_save_list[10].set_weights(
        combined_list[individual_count-4][0].get_weights())
    model_save_list[11].set_weights(
        combined_list[individual_count-4][0].get_weights())
    model_save_list[12].set_weights(
        combined_list[individual_count-5][0].get_weights())
    model_save_list[13].set_weights(
        combined_list[individual_count-5][0].get_weights())
    model_save_list[14].set_weights(
        combined_list[individual_count-6][0].get_weights())
    model_save_list[15].set_weights(
        combined_list[individual_count-7][0].get_weights())
    model_save_list[16].set_weights(
        combined_list[individual_count-8][0].get_weights())
    model_save_list[17].set_weights(
        combined_list[individual_count-9][0].get_weights())
    model_save_list[18].set_weights(
        combined_list[individual_count-10][0].get_weights())
    model_save_list[19].set_weights(
        combined_list[individual_count-11][0].get_weights())

    #print(np.array(model_save_list[1].get_weights())- np.array(combined_list[individual_count-1][0].get_weights()))

    for indiv in range(20):
        model_list[indiv].set_weights(model_save_list[indiv].get_weights())

    #model_list[0] = combined_list[individual_count-1][0]
    # model_list[1] = combined_list[individual_count-1][0]#mod
    # model_list[2] = combined_list[individual_count-1][0]#mod
    # model_list[3] = combined_list[individual_count-1][0]#mod
    #model_list[4] = combined_list[individual_count-2][0]
    # model_list[5] = combined_list[individual_count-2][0]#mod
    # model_list[6] = combined_list[individual_count-2][0]#mod
    #model_list[7] = combined_list[individual_count-3][0]
    # model_list[8] = combined_list[individual_count-3][0]#mod
    # model_list[9] = combined_list[individual_count-3][0]#mod
    #model_list[10] = combined_list[individual_count-4][0]
    # model_list[11] = combined_list[individual_count-4][0]#mod
    # model_list[12] = combined_list[individual_count-5][0]#mod
    # model_list[13] = combined_list[individual_count-5][0]#mod
    # model_list[14] = combined_list[individual_count-6][0]#mod
    # model_list[15] = combined_list[individual_count-7][0]#mod
    # model_list[16] = combined_list[individual_count-8][0]#mod
    # model_list[17] = combined_list[individual_count-9][0]#mod
    # model_list[18] = combined_list[individual_count-10][0]#mod
    # model_list[19] = combined_list[individual_count-11][0]#mod

    #save 5 best:
    combined_list[individual_count-1][0].save_weights('indiv_0.h5')
    combined_list[individual_count-2][0].save_weights('indiv_1.h5')
    combined_list[individual_count-3][0].save_weights('indiv_2.h5')
    combined_list[individual_count-4][0].save_weights('indiv_3.h5')
    combined_list[individual_count-5][0].save_weights('indiv_4.h5')
    #c = np.fromfile('test2.dat', dtype=int)
    #c == a
    #array([ True,  True,  True,  True], dtype=bool)

    for indiv in range(1, 20):
        if(indiv == 0):
            sys.exit()
        if(indiv != 4 and indiv != 7 and indiv != 10):
            # save weights in a np.array of np.arrays
            weights = np.array(model_list[indiv].get_weights())
            for i in range(6):
                weights[i] = weights[i] + \
                    np.random.normal(0, mutation_rate, weights[i].shape)
            model_list[indiv].set_weights(weights)


curses.endwin()

# weights = np.array(model_list[1].get_weights())         # save weights in a np.array of np.arrays
# for i in range(6) :
#    weights[i] = weights[i] + np.random.normal(0,mutation_rate,weights[i].shape)
# model_list[1].set_weights(weights)


#sorted_combined_list = sorted(combined_list, key=lambda pair: (pair[1], pair[2]))
#sorted_counter = [x for _,x in sorted(zip(score, counter), key=lambda pair: pair[0])]
#print("\nScore: " + str(score) + ', Counter: ' + str(counter))
#print("\nScore: " + str(sorted_score) + ', Counter: ' + str(counter))
