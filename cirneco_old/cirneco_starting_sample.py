import os
import sys
import pegtree as pg
from pegtree import ParseTree
from pegtree.visitor import ParseTreeVisitor


class Transpiler(ParseTreeVisitor):
    buffer: list
    parser: object

    def __init__(self, grammar_file='cirneco.pegtree'):
        ParseTreeVisitor.__init__(self)
        self.buffer = []
        peg = pg.grammar(grammar_file)
        self.parser = pg.generate(peg)

    def push(self, s: str):
        self.buffer.append(s)

    def generate(self, source: str):
        tree = self.parser(source)
        self.buffer = []
        self.visit(tree)
        return ''.join(self.buffer)


def main(argv):
    cirneco = Transpiler()
    if len(argv) == 0:
        import readline
        try:
            while True:
                line = input('cir?>>> ')
                if line == '':
                    print('Bye')
                    sys.exit(0)
                code = cirneco.generate(line)
                print(code)
        except EOFError:
            pass
    else:
        filename = argv[0]
        with open(filename) as f:
            code = cirneco.generate(f.readlines())
        filename = filename.replace('.py', '.js')
        with open(filename, 'w') as f:
            f.write(code)
            f.write('\n')
        os.system(f'node {filename}')


if __name__ == '__main__':
    main(sys.argv[1:])
