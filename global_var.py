from pathlib2 import Path


class fooPath(object):

    def __init__(self):
        self.name = Path.cwd()

    def __get__(self, instance, owner):
        return self.name

    def __set__(self, instance, value):
        print("projectPath is read-only!")


class GlobalVars(object):
    projectPath = fooPath()


globalVars = GlobalVars()
