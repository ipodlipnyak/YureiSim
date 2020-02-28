test = 'hi'
grid = []

class Depo:
    __shared_state = {}
    def __init__(self):
        self.__dict__ = self.__shared_state

    def test(self):
        sss = 'test'
        ass = 's'
        ss