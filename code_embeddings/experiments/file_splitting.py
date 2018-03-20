from code_embeddings.utils import parse_subroutines_javascript

file = open("fixtures/testfile.js", "r")
data = file.read()
file.close()

funcs = parse_subroutines_javascript(data)
print(funcs)
for i, func in enumerate(funcs):
    print("\n\nFunction {}\n--".format(i))
    print(func)


