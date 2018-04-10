from code_embeddings.utils import convert_snake_case
from code_embeddings.utils import convert_camel_case
from code_embeddings.utils import tokenize


def tokenize_js():
    file = open("fixtures/testfile.js", "r")
    data = file.read()
    file.close()

    tokens = tokenize(data)
    print(tokens)


def tokenize_python():
    file = open("fixtures/forest-fire.py", "r")
    data = file.read()
    file.close()

    tokens = tokenize(data)
    print(tokens)


def test_snake_case_conversion():
    for snake in ['snake_case',
                  'SNAKE_CASE']:
        print(convert_snake_case(snake))


def test_camel_case_conversion():
    for camel in ['CamelCase',
                  'CamelCamelCase',
                  'Camel2Camel2Case',
                  'getHTTPResponseCode',
                  'get2HTTPResponseCode',
                  'HTTPResponseCode']:
        print(convert_camel_case(camel))


if __name__ == '__main__':
    test_snake_case_conversion()
    tokenize_js()
    tokenize_python()
