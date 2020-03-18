import sys, random, math, pygame, logging, objects
from sys import exit
from pygame.locals import *
from pygame.sprite import Group, GroupSingle
from objects import observer, grid, ghost

from objects.var_depo import Depo
 
class App:
    def __init__(self):
        self._running = True
        self._display_surf = None
        self.size = self.width, self.height = 1200,750#600, 375 #1024, 368
        self.ts = 15 #tile size
        self.nerad = 0,0 #neighborhood radius
        self.net = 3 #neighborhood type from 1 to 6
        self.leb = 1 #left bias for color mix. Weight for old color 
        self.rib = 2 #right bias for color mix. Weight for new color
        self.td = 10 #time delay
        self.gc = 1 #count for genies

    def on_init(self):
#        pygame.init()
        pygame.display.init()
        pygame.font.init()
        self.surf = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption('Yep..')
        self.font = pygame.font.SysFont('sawasdee', self.ts)
        self.clock = pygame.time.Clock()
        self.herr = observer.PushModel()
        #var_depo.grid = grid.Grid(surface=self.surf,font=self.font,notifier=self.herr,tile_value='grass',neighbor_rad=self.nerad,tile_size=self.ts,nt=self.net,lb=self.leb,rb=self.rib)
        Depo.gridSingleton = grid.Grid(surface=self.surf,font=self.font,notifier=self.herr,tile_value='grass',neighbor_rad=self.nerad,tile_size=self.ts,nt=self.net,lb=self.leb,rb=self.rib)
        self.lamp = pygame.sprite.Group()
        gc = 1
        while gc <= self.gc:
            gx = random.randrange(0,self.width/self.ts)
            gy = random.randrange(0,self.height/self.ts)
            #ginie = ghost.bakemono(self.surf, self.herr,x=gx,y=gy,w=self.ts,h=self.ts)
            #ginie = ghost.virus(self.surf, self.herr,x=gx,y=gy,w=self.ts,h=self.ts)
            #ginie = ghost.yurei(self.surf, self.herr,x=gx,y=gy,w=self.ts,h=self.ts)
            #ginie = ghost.rojinbi(self.surf, self.herr,x=gx,y=gy,w=self.ts,h=self.ts)
            #ginie = ghost.Mononoke(self.surf, self.herr,x=gx,y=gy,w=self.ts,h=self.ts)
            ginie = ghost.Mononoke(self.surf, self.herr,x=gx,y=gy,w=self.ts,h=self.ts)
            self.lamp.add(ginie)
            gc += 1
            
        #self.lamp.add(ghost.bakemono(self.surf, self.herr,x=gx,y=gy,w=self.ts,h=self.ts))
        #self.lamp.add(ghost.bakemono(self.surf, self.herr,x=gx,y=gy,w=self.ts,h=self.ts))
        #self.lamp.add(ghost.Mononoke(self.surf, self.herr,x=gx,y=gy,w=self.ts,h=self.ts))

        self._running = True

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self._running = False
            if event.key == pygame.K_RIGHT or event.key == pygame.K_q:
                self.lamp.empty()
            elif event.key == pygame.K_LEFT:
                gx = random.randrange(0,self.width/self.ts)
                gy = random.randrange(0,self.height/self.ts)
                self.lamp.add(ghost.Mononoke(self.surf, self.herr,x=gx,y=gy,w=self.ts,h=self.ts))
                
            elif event.key == pygame.K_w:
                for x in range(int(self.width/self.ts)):
                    for y in range(int(self.height/self.ts)):
                        #setattr(Depo.gridSingleton.g[x][y],'color',(0,0,0))
                        self.herr.set_tile(x, y, 'color', (0,0,0))
                
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == pygame.BUTTON_LEFT:
                x,y = pygame.mouse.get_pos()
                gx = x/self.ts
                gy = y/self.ts
                self.lamp.add(ghost.Mononoke(self.surf, self.herr,x=gx,y=gy,w=self.ts,h=self.ts))
            
            if event.button == pygame.BUTTON_RIGHT:
                x,y = pygame.mouse.get_pos()
                gx = x/self.ts
                gy = y/self.ts
                for ghost_sprite in self.lamp.sprites():
                    ghost_sprite.respawn(gx, gy)
            
            if event.button == pygame.BUTTON_MIDDLE:
                pass
            

    def on_loop(self):
        self.clock.tick()
        self.fps = self.clock.get_fps()
#        print self.fps
        self.lamp.update()

    def on_render(self):
        pygame.display.flip() 
        pygame.time.delay(self.td)

    def on_cleanup(self):
        pygame.display.quit()
        pygame.quit()
        sys.exit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        while (self._running):
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
        self.on_cleanup()       


if __name__ == "__main__" :
#    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('App')
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)
#    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
#    logging.basicConfig(filename='example.log',format='%(levelname)s:%(message)s',level=logging.DEBUG)
#    logging.basicConfig(format='%(message)s',level=logging.DEBUG)
#    logging.basicConfig(filename='example.log',format='%(message)s',level=logging.DEBUG)
    theApp = App()
    theApp.on_execute()
