from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime

import pygame, random
from numpy.polynomial.polynomial import polyval

import numpy as np
import tensorflow as tf
from math import floor
mnist = tf.keras.datasets.mnist

class ghost(pygame.sprite.Sprite,object):
    def __init__(self,surface,observer,x=0,y=0,w=15,h=15):
        pygame.sprite.Sprite.__init__(self)
        self.surf = surface
        self.obs = observer
        self.rect = pygame.Rect(x,y,w,h)
        self.surf_rect = surface.get_rect()
        self.R = random.randrange(0,255)
        self.G = random.randrange(0,255)
        self.B = random.randrange(0,255)
        self.symbol = 'Y'
        
        self.grid = {
            'height': self.surf_rect.h/self.rect.h,
            'width': self.surf_rect.w/self.rect.w
            }
    def update(self):
        self.on_move()
    def on_move(self):
        pass
    def on_do(self):
        pass
    def switch_color(self):
        self.R = random.randrange(0,255)
        self.G = random.randrange(0,255)
        self.B = random.randrange(0,255)

class test(ghost):
    def __init__(self,surface,observer,x=0,y=0,w=15,h=15):
        super(test,self).__init__(surface,observer,x,y,w,h)
    
    def on_move(self):
        self.rect.x = 20
        self.rect.y = 1
        self.obs.set_tile(self.rect.x,self.rect.y,'color',(self.R,self.G,self.B))

class bakemono(ghost):
    def __init__(self,surface,observer,x=0,y=0,w=15,h=15):
        super(bakemono,self).__init__(surface,observer,x,y,w,h)
        self.symbol = 'B'
    def on_move(self):
        if self.rect.y + 1 < self.grid['height']:
            self.rect.y += 1
        else:
            self.rect.y = 0
        
        dxx = polyval(self.rect.y, [-self.rect.x,0.5,0.06,-0.0008])
        if dxx < self.grid['width']:
            self.rect.x = dxx
        else:
            self.rect.x = 0
            
        self.obs.set_tile(self.rect.x,self.rect.y,'color',(self.R,self.G,self.B))

class yurei(ghost,object):
    def __init__(self,surface,observer,x=0,y=0,w=15,h=15):
        super(yurei,self).__init__(surface,observer,x,y,w,h)
        self.random_choice()
        self.dxx = x
        self.dyy = y
    def on_move(self):
        self.random_move()
        self.on_do()
    def on_do(self):
        self.change_tile()
    def random_choice(self):
        self.dx = random.choice([-2,-1,0,1,2]) 
        self.dy = random.choice([-2,-1,0,1,2])
    def random_move(self):
        if (self.dx == 0 and self.dy == 0) or (abs(self.dx) == 2 and abs(self.dy) == 2):
            self.random_choice()
        if 0 < self.rect.x+self.dx < self.surf_rect.w/self.rect.w:
            self.dxx = self.rect.x
            self.rect.x += self.dx
            if 0 < self.rect.y+self.dy < self.surf_rect.h/self.rect.h:
                self.dyy = self.rect.y
                self.rect.y += self.dy
            else:
                self.dyy = self.rect.y
                #self.dy *= -1
                self.random_choice()
                self.switch_color()
        else:
            self.dxx = self.rect.x
            #self.dx *= -1
            self.random_choice()
            self.switch_color()
    def change_tile(self,surface,observer,x=0,y=0,w=15,h=15):
        self.obs.set_tile(self.rect.x,self.rect.y,'color',(self.R,self.G,self.B))
        self.obs.set_tile(self.rect.x,self.rect.y,'symbol',self.symbol)
        self.obs.set_tile(self.dxx,self.dyy,'symbol','O')

class virus(ghost,object):
    def __init__(self,surface,observer,x=0,y=0,w=15,h=15):
        super(virus,self).__init__(surface,observer,x,y,w,h)
        self.dy = 1
        self.rect.x = random.randrange(0,self.surf_rect.w/self.rect.w)
        #self.rect.y = random.randrange(0,(self.surf_rect.w/self.rect.w)-(self.surf_rect.w/self.rect.w)/7)
        self.rect.y = random.randrange(0,self.surf_rect.h/self.rect.h)
    def on_move(self):
        self.on_fall()
        self.on_do()
    def on_do(self):
        self.change_tile()
    def on_fall(self):
            if 0 < self.rect.y+self.dy < self.surf_rect.h/self.rect.h:
                self.rect.y += self.dy
            else:
                self.rect.x = random.randrange(0,self.surf_rect.w/self.rect.w)
                #self.rect.y = random.randrange(0,(self.surf_rect.h/self.rect.h)-(self.surf_rect.h/self.rect.h)/7)
                self.rect.y = random.randrange(0,self.surf_rect.h/self.rect.h)
                self.R = random.randrange(0,255)
                self.G = random.randrange(0,255)
                self.B = random.randrange(0,255)
    def change_tile(self):
        self.obs.set_tile(self.rect.x,self.rect.y,'color',(self.R,self.G,self.B))
        if self.obs.get_tile(self.rect.x,self.rect.y,'symbol') == 'O':
            self.obs.set_tile(self.rect.x,self.rect.y,'symbol','X')
        elif self.obs.get_tile(self.rect.x,self.rect.y,'symbol') == 'X':
            self.obs.set_tile(self.rect.x,self.rect.y,'symbol','G')
        else:
            self.obs.set_tile(self.rect.x,self.rect.y,'symbol','O')

class rojinbi(ghost,object):
    def __init__(self,surface,observer,x=0,y=0,w=15,h=15):
        super(rojinbi,self).__init__(surface,observer,x,y,w,h)
        self.symbol = 'R'
    def on_move(self):
        self.spark()
        self.on_do()
    def on_do(self):
        self.change_tile()
    def spark(self):
        self.rect.x = random.randrange(0,self.surf_rect.w/self.rect.w)
        self.rect.y = random.randrange(0,self.surf_rect.h/self.rect.h)
        self.R = random.randrange(0,255)
        self.G = random.randrange(0,255)
        self.B = random.randrange(0,255)
    def change_tile(self):
        self.obs.set_tile(self.rect.x,self.rect.y,'color',(self.R,self.G,self.B))
        if self.obs.get_tile(self.rect.x,self.rect.y,'symbol') != self.symbol:
            self.obs.set_tile(self.rect.x,self.rect.y,'symbol',self.symbol)
        else:
            self.obs.set_tile(self.rect.x,self.rect.y,'symbol','O')

class Mononoke(ghost):
    '''
    @param model: TensorFlow model
    
    ---
    data_sets:
    @param train_input: training input data set
    @param train_output: trainung output data set
    
    @param validate_input: validation input data set
    @param validate_output: validation output data set
    ---
        
    '''
    
    def __init__(self,surface,observer,x=0,y=0,w=15,h=15):
        super(Mononoke,self).__init__(surface,observer,x,y,w,h)
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(2),
            tf.keras.layers.LayerNormalization(),
            #tf.keras.layers.LayerNormalization(axis=1 , center=True , scale=True),
            #tf.keras.layers.Dense(8),
            #tf.keras.layers.Dense(64, activation='relu'),
            #tf.keras.layers.Dense(64),
            tf.keras.layers.Dense(120),
            #tf.keras.layers.Dropout(0.2),
            #tf.keras.layers.Dense(64),
            #tf.keras.layers.Dense(8),
            #tf.keras.layers.Dense(3),
            #tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(2)
            #tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        '''
        [
            [x,y],
            ...
            [xx,yy]
        ]
        '''
        
        self.data_sets = {
            'train_input': {
                'bounce': [[0,0],[0,1],[1,0],[1,1]],
                'random': np.random.random((1000, 2)),
                },
            'train_output': {
                'bounce': [[1,1],[1,-1],[-1,1],[-1,-1]],
                'random': np.random.random((1000, 2)),
                },
            'validate_input': {
                'bounce': [[0.1,0],[0.1,1],[1,0.1],[1,1]],
                'random': np.random.random((1000, 2)),
                },
            'validate_output': {
                'bounce': [[0.1,1],[0.1,-1],[-1,0.1],[-1,-1]],
                'random': np.random.random((1000, 2)),
                }            
            }
        
        self.trainDSGen()
        
        #self.train_input_data = [[0,0],[0,1],[1,0],[1,1]]
        #self.train_output_data = [[1,1],[1,-1],[-1,1],[-1,-1]]
    
        #self.validate_input_data = [[0.1,0],[0.1,1],[1,0.1],[1,1]]
        #self.validate_output_data = [[0.1,1],[0.1,-1],[-1,0.1],[-1,-1]]
        
        #self.train_input_data = np.random.random((1000, 2))
        #self.train_output_data = np.random.random((1000, 2))
        
        #self.validate_input_data = np.random.random((100, 2))
        #self.validate_output_data = np.random.random((100, 2))
        
        #self.model.compile(optimizer='adam',
        #    loss='sparse_categorical_crossentropy',
        #    metrics=['accuracy'])
        
        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        self.train()
        
        #data = np.random.random((1000, 2))
        #labels = np.random.random((1000, 2))
        
        #val_data = np.random.random((100, 2))
        #val_labels = np.random.random((100, 2))

        
        #self.model.fit(data, labels, epochs=10, validation_data=(val_data, val_labels))
        #self.model.fit(data, labels, epochs=10)
        #self.model.fit(data, labels, epochs=10)

    def on_move(self):
        x = self.rect.x / self.grid['width']
        y = self.rect.y / self.grid['height']
        dx, dy = np.floor(np.multiply(self.predict([x, y]),10))
        #dx, dy = np.floor(self.predict([x, y]))
        #xx, yy = self.predict([x, y])
        
        #dx = 1 if xx > 0 else -1 if xx < 0 else 0
        #dy = 1 if yy > 0 else -1 if yy < 0 else 0
        
        if self.rect.y + dy < self.grid['height']:
            self.rect.y += dy
        else:
            #pass
            #self.rect.y = 0
            #self.rect.y = floor(self.grid['height'] / 2)
            self.rect.y = random.randrange(0,self.surf_rect.h/self.rect.h)
                
        #dxx = polyval(self.rect.y, [-self.rect.x,0.5,0.06,-0.0008])
        if self.rect.x + dx < self.grid['width']:
            self.rect.x += dx
        else:
            #pass
            #self.rect.x = floor(self.grid['width'] / 2)
            self.rect.x = random.randrange(0,self.surf_rect.w/self.rect.w)
            
        #self.obs.set_tile(self.rect.x,self.rect.y,'color',(self.R,self.G,self.B))
        self.change_tile()
        
    def change_tile(self):
        self.obs.set_tile(self.rect.x,self.rect.y,'color',(self.R,self.G,self.B))
        if self.obs.get_tile(self.rect.x,self.rect.y,'symbol') == 'O':
            self.obs.set_tile(self.rect.x,self.rect.y,'symbol','X')
        elif self.obs.get_tile(self.rect.x,self.rect.y,'symbol') == 'X':
            self.obs.set_tile(self.rect.x,self.rect.y,'symbol','G')
        else:
            self.obs.set_tile(self.rect.x,self.rect.y,'symbol','O')
    
    def trainDSGen(self):
        i = 0
        
        data_set = []
        
        while i < 1000:
            val = round(np.random.uniform(),3)
            delta = 1 if val < 0.2 else -1 if val > 0.8 else round(np.random.uniform(),3)
            data_set.insert(i, {
                    'input': val,
                    'output': delta
                })

            i += 1
        
        self.data_sets['train_input']['bounce_gen'] = []
        self.data_sets['train_output']['bounce_gen'] = []
        self.data_sets['validate_input']['bounce_gen'] = []
        self.data_sets['validate_output']['bounce_gen'] = []
        
        for i, x in enumerate(data_set):
            self.data_sets['train_input']['bounce_gen'].append([x['input']])
            self.data_sets['train_output']['bounce_gen'].append([x['output']])
        
        np.random.shuffle(data_set)
        
        for i, y in enumerate(data_set):
            self.data_sets['train_input']['bounce_gen'][i].append(y['input'])
            self.data_sets['train_output']['bounce_gen'][i].append(y['output'])
            
        np.random.shuffle(data_set)
        
        for i, x in enumerate(data_set):
            self.data_sets['validate_input']['bounce_gen'].append([x['input']])
            self.data_sets['validate_output']['bounce_gen'].append([x['output']])
        
        np.random.shuffle(data_set)
        
        for i, y in enumerate(data_set):
            self.data_sets['validate_input']['bounce_gen'][i].append(y['input'])
            self.data_sets['validate_output']['bounce_gen'][i].append(y['output'])

        
    def train(self):
        # Define the Keras TensorBoard callback.
        logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        
        #ds_type = 'bounce_gen'
        ds_type = 'bounce'
        #ds_type = 'random'
        
        self.model.fit(
            self.data_sets['train_input'][ds_type],
            self.data_sets['train_output'][ds_type], 
            epochs=10, 
            validation_data=(
                self.data_sets['validate_input'][ds_type],
                self.data_sets['validate_output'][ds_type], 
                ),
            callbacks=[tensorboard_callback])
        
        self.df_max = self.model.predict([[0,0]])
        self.df_min = self.model.predict([[1,1]])
        self.df_mean = [
            np.mean([self.df_min.item(0),self.df_max.item(0)]),
            np.mean([self.df_min.item(1),self.df_max.item(1)])
            ]
        
    def getModel(self):
        return self.model
    
    def predict(self, data):
        '''
        normalisation formula:
        df_norm = (df - df.mean()) / (df.max() - df.min())
        '''
        prediction = self.model.predict([data])
        x = prediction.item(0)
        y = prediction.item(1)
        
        # Normalisation
        df_x = (x - self.df_mean[0]) / (self.df_max.item(0) - self.df_min.item(0))
        df_y = (y - self.df_mean[1]) / (self.df_max.item(1) - self.df_min.item(1))
        return [round(df_x, 3), round(df_y,3)]