import pygame
import math, random, scipy, numpy, Queue, threading, logging, os
from numpy import linalg
from objects import map
from decimal import *
from Queue import Queue
from threading import Thread

module_logger = logging.getLogger('App.Grid')

#Return grid with tiles
class Grid(object):
    def __init__(self,surface,font,notifier,tile_size=50,tile_value=False,neighbor_rad=False,nt=1,lb=1,rb=10):
        self.logger = logging.getLogger('App.Grid.g')
        self.font = font
        self.notifier = notifier
        self.surface = surface
        self.s_rect = surface.get_rect()
        self.tile_s = tile_size
        self.nt = nt
        self.lb = lb
        self.rb = rb
        self.q = Queue()

        self.w = self.s_rect.w/tile_size #grid width
        self.h = self.s_rect.h/tile_size #grid height
        self.v = tile_value #tile value: [1] - color, [2] - type
        self.g = [] #grid
        self.tiles = pygame.sprite.Group()

        self.on_generate()

        if neighbor_rad != False:
            self.sfn(neighbor_rad)
            self.tiles.update()


    def on_generate(self):
        for x in range(self.w):
            self.g.append([])
#            print '---------',x
            for y in range(self.h):
#                print self.a
                if self.v==False:
                    newtile = Dummy_Tile(self.surface,self.font,self.tile_s,self.notifier,x,y)
                    self.tiles.add(newtile)
                    self.g[x].append(newtile)
                elif self.v == 'grass':
                    newtile = Grass(self.surface,self.font,self.tile_s,self.notifier,x,y,lb=self.lb,rb=self.rb)
                    self.tiles.add(newtile)
                    self.g[x].append(newtile)
#                    self.t = Thread(target=newtile.update)
#                    self.t.daemon = True
#                    self.t.start()
#                    self.t.join()

        
#        for t in self.g:
#            t.color = 120,120,0
#        pygame.display.update()
#        self.tiles.update()

    #search for neighbors algoritm
    #Radius for search: neighbor_rad = (x,y)
    def sfn(self,neighbor_rad):
        progr = 0
        dx = neighbor_rad[0]
        dy = neighbor_rad[1]
        for dxx in range(dx):
            for dyy in range(dy):
                for x in range(self.w):
                    for y in range(self.h):
                        self.g[x][y].neighbor_rad = neighbor_rad
                        d1 = (x+(dxx+1),y)
                        d1d = dxx+1,0
                        d2 = (x+(dxx+1),y+(dyy+1))
                        d2d = dxx+1,dyy+1
                        d3 = (x,y+(dyy+1))
                        d3d = 0,dyy+1
                        d4 = (x-(dxx+1),y+(dyy+1))
                        d4d = -1*(dxx+1),dyy+1
                        d5 = (x-(dxx+1),y)
                        d5d = -1*(dxx+1),0
                        d6 = (x-(dxx+1),y-(dyy+1))
                        d6d = -1*(dxx+1),-1*(dyy+1)
                        d7 = (x,y-(dyy+1))
                        d7d = 0,-1*(dyy+1)
                        d8 = (x+(dxx+1),y-(dyy+1))
                        d8d = dxx+1,-1*(dyy+1)
                        if self.nt == 1:
                            d = (d1,d1d),(d2,d2d),(d3,d3d),(d4,d4d),(d5,d5d),(d6,d6d),(d7,d7d),(d8,d8d)
                        elif self.nt == 2:
                            d = (d2,d2d),(d4,d4d),(d6,d6d),(d8,d8d)
                        elif self.nt == 3:
                            d = (d1,d1d),(d3,d3d),(d5,d5d),(d7,d7d)
                        elif self.nt == 4:
                            d = (d1,d1d),(d5,d5d)
                        elif self.nt == 5:
                            d = (d3,d3d),(d7,d7d)
                        elif self.nt == 6:
                            d = (d2,d2d),(d3,d3d),(d4,d4d)
                        for dd in d:
                            if 0 <= dd[0][0] <= self.w-1 and 0 <= dd[0][1] <= self.h-1:
                                ul = self.unit_vector((dx,dy),(dd[1][0],dd[1][1]))
                                app = (dd[0][0],dd[0][1]),((dxx+1),(dyy+1)),ul
                                self.g[x][y].neighbors.append(app)
            progr += 1
            self.logger.info(msg=str(progr))
#            print progr


    def unit_vector(self,rad,tile):
        b = numpy.array(tile)
        a = numpy.array(rad)
        al = numpy.linalg.norm(a)
        u = b/al
        ul = numpy.linalg.norm(u)
        return ul

        

class Dummy_Tile(pygame.sprite.Sprite,object):
    def __init__(self,surface,font,tile_size,notifier,x,y):
        pygame.sprite.Sprite.__init__(self)
        self.surface = surface
        self.font = font
        self.notifier = notifier
        self.rect = pygame.Rect ((x*tile_size,y*tile_size),(tile_size,tile_size))
        self.me = x,y
        self.var_tuple = 'grid.g[',str(x),'][',str(y),']'
        self.var_string = ''.join(self.var_tuple)
        self.notifier.bind_on(x,y,'color',callback=self.callback)
        self.notifier.bind_on(x,y,'symbol',callback=self.callback)
        self.q = Queue()
#        print 'Create callback',self.var_string
        
        self.color = 0,15,0
        self.symbol = 'O'

    def log_tile(self,**kwarg):
        name = ('App.Grid.g',str(self.me))
        logger = logging.getLogger(''.join(name))
        keys = sorted(kwarg.keys())
        for kw in keys: #kwarg:
            msg = (str(kw),':',str(kwarg[kw]))
            msg_s = ''.join(msg)
            logger.info(msg_s)

    def callback(self,tile,var_name,new_value):
#        print 'Ouch! My name is ',self.var_string
        self.update()
    #Update model
    def set_symbol(self):
        self.symbol = 'O'
#        if self.symbol == 'O':
#            self.symbol = 'H'
#        elif self.symbol == 'X':
#            self.symbol = 'V'
#        elif self.symbol == 'G':
#            self.symbol = 'K'
#        else:
#            self.symbol = 'O'

    def set_color(self,color):
        self.color = color
#        self.set_symbol()
        self.render(self.surface)
#        self.update()

    def update(self):
#        self.q.put(True)
        self.render(self.surface)
#        pass
    #Update view
    def render(self,surface):
        surface.fill(color=(0,0,0),rect=self.rect)
        self.text = self.font.render(self.symbol,True,self.color)
        self.text_x = self.rect.centerx-self.text.get_size()[0]*0.5
        self.text_y = self.rect.centery-self.text.get_size()[1]*0.5
        surface.blit(self.text,(self.text_x,self.text_y))
        pygame.display.update(self.rect)

class Grass(Dummy_Tile,object):
    def __init__(self,surface,font,tile_size,notifier,x,y,lb=1,rb=10,debug=False):
        super(Grass,self).__init__(surface,font,tile_size,notifier,x,y)
        self.neighbors = []
        self.lb = lb
        self.rb = rb
        self.dbg = debug
#        print 'Init complete:',self.me
        self.log_tile(Init='complete')

    def neighborhood(self):
        pass

    def update(self):
        super(Grass,self).update()
        self.voayor()

    def voayor(self):
        for neighbor in self.neighbors:
            self.notifier.bind_on(neighbor[0][0],neighbor[0][1],'color',callback=self.spoocked)



    def spoocked(self,tile,var_name,new_value):
#        os.system('clear')
#        self.log_tile(Tile=self.me)
#        self.q.put(True)
#        msg = '----------','\n',str(self.me),'----------'
#        msg_s = ''.join(msg)
#        logging.info(msg)
#        logging.info(msg_s)
#        if self.dbg != False:
#            print '----------',self.me,'----------'
        grad = 0.8
        if var_name == 'color':
            for ne in self.neighbors:
#                print ne
                if ne[0][0] == tile[0] and ne[0][1] == tile[1]:
                    dist = ne[2]
                    #self.logger.info(ne)
                    #self.logger.info(round(ne[2],2))
#                    if self.dbg != False:
#                        print ne
#                        print round(ne[2],2)
            nR = new_value[0]-new_value[0]*dist
            nG = new_value[1]-new_value[1]*dist
            nB = new_value[2]-new_value[2]*dist
            oR = self.color[0] 
            oG = self.color[1]
            oB = self.color[2]
#            R = (nR+oR)/2
#            G = (nG+oG)/2
#            B = (nB+oB)/2
            wR = (nR/255)*10+self.lb,(oR/255)*10+self.rb
            #self.logger.info(Red_w = str(wR))
#            if self.dbg != False:
#                print 'Red weight: ',wR
            wG = (nG/255)*10+self.lb,(oG/255)*10+self.rb
#            if self.dbg != False:
#                print 'Green weight: ',wG
            wB = (nB/255)*10+self.lb,(oB/255)*10+self.rb
#            if self.dbg != False:
#                print 'Blue weight: ',wB
            R = numpy.average((nR,oR),weights=wR)
            G = numpy.average((nG,oG),weights=wG)
            B = numpy.average((nB,oB),weights=wB)
            self.set_color((R,G,B))
            
#            os.system('clear')
            self.log_tile(RedW=wR,GreenW=wG,BlueW=wB,Red=R,Green=G,Blue=B)
#        self.log_tile(Red=self.color[0],Green=self.color[1],Blue=self.color[2])
#            if self.dbg != False:
#                print 'My color: ',R,'',G,'',B
#        if self.dbg != False:
#            print '**********************'
#        print 'Boo, Scarry Terry'
        pass
