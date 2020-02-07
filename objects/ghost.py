import pygame, random

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
    def change_tile(self):
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
