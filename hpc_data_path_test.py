
import os

print("current working directory: ", os.getcwd())
print("path two steps back from current working directory: ", os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, "data", "data","11t51center")))