from code_embeddings.utils import (
    parse_subroutines_javascript, parse_subroutines_java
)


def split_javascript():
    file = open("fixtures/testfile.js", "r")
    data = file.read()
    file.close()

    funcs = parse_subroutines_javascript(data)
    print(funcs)
    for i, func in enumerate(funcs):
        print("\n\nFunction {}\n--".format(i))
        print(func)


def split_java():
    file = open("fixtures/testfile.java", "r")
    data = file.read()
    file.close()

    funcs = parse_subroutines_java(data)
    print(funcs)
    for i, func in enumerate(funcs):
        print("\n\nFunction {}\n--".format(i))
        print(func)


if __name__ == '__main__':
    split_java()
