from gplace import GPlaceFinder
from LangChain import DuoLangChain
from dotenv import load_dotenv

load_dotenv()


model, opening_msg = DuoLangChain.construct()

print(opening_msg)

while True:
    a = input("Enter prompt: ")
    result, done = model.invoke(a)
    print(result)
    if done:
        g = GPlaceFinder()
        for place in model.places:
            print(g.query(place))
        print(model.mode)
        break