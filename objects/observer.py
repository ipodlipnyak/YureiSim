#-- Observer Pattern --
#http://wiki.c2.com/?ObserverPattern
#The two basic styles of notification: PushModel and PullModel.
#If the observed object does not have the necessary hooks for observers,
#the observers must rely on repeatedly polling the observed to note the changes. 
#This is a "pull" versus a "push" type of pattern. Quite appropriate for certain applications. 

import var_depo, threading, multiprocessing 
from threading import Thread
from multiprocessing import Process

#Push notifier
class PushModel(object):
    def __init__(self):
        self._observers = []

    def bind_on(self,x,y,var_name,callback):
        result = self.check_call(x,y,var_name,callback)
        if result == False:
            appe = x,y,var_name,callback
            self._observers.append(appe)

    def get_tile(self,x,y,var_name):
        return getattr(var_depo.grid.g[x][y],str(var_name))
    def set_tile(self,x,y,var_name,new_value):
        setattr(var_depo.grid.g[x][y],var_name,new_value)
        name_str = str(var_name)
        for callback in self._observers:
            if x == callback[0] and y == callback[1] and var_name == callback[2]:
                callback[3]((x,y),var_name,new_value)
#                call = callback[3]((x,y),var_name,new_value)
#                self.t = Process(target=call)
#                self.t.daemon = True
#                self.t.start()
#                self.t.join()
#                print 'callback',str(callback),'has been sent to ',x,'-',
#                pass
#            for index, number in enumerate(numbers):
#                proc = Process(target=doubler, args=(number,))
#                procs.append(proc)
#                proc.start()
#            for proc in procs:
#                proc.join()

    def check_call(self,x,y,var_name,callback):
        result = False
        for obs in self._observers:
            if x == obs[0] and y == obs[1] and var_name == obs[2] and callback == obs[3]:
                result = True
        return result
                

#class PushM_01(object):
#    def __init__(self):
#        self._observers = []
#
#    def bind_on(self,var_name,callback):
#        appe = var_name,callback
#        self._observers.append(appe)
#
#    def get(self,var_name):
#        return getattr(var_depo,var_name)
#    def set(self,var_name,new_value):
#        setattr(var_depo,var_name,new_value)
#        name_str = str(var_name)
#        for callback in self._observers:
#            call_str = str(callback[0])
#            if name_str.startswith(call_str):
#                callback[1](var_name,new_value)
#                print 'callback has been sent to ',call_str
