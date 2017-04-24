import pygame
import random
import numpy as np

class grass():
    def __init__(self):
        if not pygame.font.get_init():
            pygame.font.init()

        self.on_init()
#        self.on_render()

    def on_init(self):
        self.f = pygame.font.SysFont ('sawasdee',10,bold=True)
        self.g = 'O' #grass symboll

    def on_render(self, surface, in_grid = False):
        if not in_grid:
            self.gr = grid()
        else:
            self.gr = in_grid
        
        self.gx = 0
        self.gy = 0

        for x in range(self.gr.w):
#            print self.gr.g[x]
            if x == 0:
                self.gx = 0
            else:
                self.gx += 10
            for y in range(self.gr.h):
                self.text = self.f.render(self.g,True,(0,self.gr.g[x][y],0))
#                print self.gx
#                print self.gy
                surface.blit(self.text,(self.gx,self.gy))
                if y == 0:
                    self.gy = 0
                else:
                    self.gy += 10 
                



class grid():
    def __init__(self):
        self.on_gen()

    def on_gen(self):
        self.w = 100 #width
        self.h = 30 #height
        self.t = 0 #type of tile
        self.g = [] #grid
        for x in range(self.w):
            self.g.append([])
            for y in range(self.h):
                self.g[x].append(self.t)
                self.t = random.randrange(100,255)

    def mandelbrot( h,w, maxit=20 ):
        """Returns an image of the Mandelbrot fractal of size (h,w)."""
        y,x = np.ogrid[ -1.4:1.4:h*1j, -2:0.8:w*1j ]
        c = x+y*1j
        z = c
        divtime = maxit + np.zeros(z.shape, dtype=int)
        
        for i in range(maxit):
                z = z**2 + c
                diverge = z*np.conj(z) > 2**2            # who is diverging
                div_now = diverge & (divtime==maxit)  # who is diverging now
                divtime[div_now] = i                  # note when
                z[diverge] = 2                        # avoid diverging too much
       
        return divtime

class grid_sample():
    def __init__(self):
        self.w = 3
        self.h = 3
        self.g = [[0,255,255],[255,0,255],[0,255,0]]

class map():
    def __inist__(self):
        pass

    def on_render(self):
        pass
